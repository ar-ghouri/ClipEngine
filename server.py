"""
ClipEngine Studio - Local Web UI Backend
Runs on 127.0.0.1:8000 only.
Provides FastAPI endpoints, SQLite persistence, and a sequential single-GPU job queue
wrapping main.py auto-clip.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import queue
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import BASE_DIR, DB_PATH, OUTPUTS_DIR

# Ensure necessary directories exist
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
TRASH_DIR = OUTPUTS_DIR / ".trash"
TRASH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Database Schema Migrations & Helpers
# ============================================================================

def get_db() -> sqlite3.Connection:
    """Returns a SQLite connection with Row factory and busy timeout."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def sync_clip_metadata(conn: sqlite3.Connection) -> None:
    """
    Synchronizes clip metadata:
    1. Ensures review_status is not NULL (defaults to 'new').
    2. Derives stream_folder for any row where stream_folder IS NULL or stream_folder = ''.
       Uses Path(output_path).parent.name, or 'Root_Clips' if parent is 'outputs' or empty.
    """
    cur = conn.cursor()
    # Normalize review_status
    cur.execute("UPDATE clips SET review_status = 'new' WHERE review_status IS NULL OR review_status = ''")

    # Backfill stream_folder for any row missing it
    cur.execute(
        "SELECT id, output_path FROM clips WHERE (stream_folder IS NULL OR stream_folder = '') AND output_path IS NOT NULL"
    )
    rows = cur.fetchall()
    for row in rows:
        clip_id = row["id"]
        out_path = row["output_path"]
        if out_path:
            p = Path(out_path)
            parent_name = p.parent.name
            if parent_name.lower() in ("outputs", ""):
                folder_name = "Root_Clips"
            else:
                folder_name = parent_name
        else:
            folder_name = "Root_Clips"
        cur.execute("UPDATE clips SET stream_folder = ? WHERE id = ?", (folder_name, clip_id))
    conn.commit()


def init_server_db() -> None:
    """
    Safely migrates clipengine.db:
    1. Adds missing review and tracking columns to 'clips'.
    2. Backfills 'stream_folder' for existing clips.
    3. Creates the 'jobs' table for background task tracking.
    4. Resets any orphaned running jobs from previous sessions.
    """
    with get_db() as conn:
        cur = conn.cursor()

        # 1. Migrate clips table columns
        cur.execute("PRAGMA table_info(clips)")
        existing_cols = {row["name"] for row in cur.fetchall()}

        column_migrations = [
            ("review_status", "ALTER TABLE clips ADD COLUMN review_status TEXT DEFAULT 'new'"),
            ("posted_url", "ALTER TABLE clips ADD COLUMN posted_url TEXT"),
            ("platform", "ALTER TABLE clips ADD COLUMN platform TEXT"),
            ("views", "ALTER TABLE clips ADD COLUMN views INTEGER DEFAULT 0"),
            ("stream_folder", "ALTER TABLE clips ADD COLUMN stream_folder TEXT"),
            ("notes", "ALTER TABLE clips ADD COLUMN notes TEXT"),
        ]

        for col_name, sql in column_migrations:
            if col_name not in existing_cols:
                cur.execute(sql)

        # 2. Sync review_status and stream_folder metadata
        sync_clip_metadata(conn)

        # 3. Create jobs table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prompt TEXT,
            max_clips INTEGER DEFAULT 3,
            layout TEXT DEFAULT 'auto',
            cam_corner TEXT DEFAULT 'auto',
            fast_subs INTEGER DEFAULT 1,
            no_bgm INTEGER DEFAULT 0,
            status TEXT DEFAULT 'queued',
            step TEXT DEFAULT 'Waiting in queue...',
            progress INTEGER DEFAULT 0,
            error TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT
        );
        """)

        # 4. Reset orphaned running/queued jobs from previous server instances
        now_iso = datetime.now().isoformat()
        cur.execute(
            "UPDATE jobs SET status = 'failed', step = 'Server restarted', error = 'Server restarted while job was running', finished_at = ? WHERE status IN ('queued', 'running')",
            (now_iso,)
        )
        conn.commit()


def resolve_clip_file(out_path: str) -> Optional[Tuple[Path, str]]:
    """
    Resolves a clip's output_path into an existing Path on disk and its relative /outputs/ URL.
    Handles both relative ('outputs/...') and absolute ('E:\\...\\outputs\\...') paths cleanly.
    """
    p = Path(out_path)
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((BASE_DIR / p).resolve())
        candidates.append((OUTPUTS_DIR / p).resolve())
        if p.name:
            candidates.append((OUTPUTS_DIR / p.name).resolve())
            if p.parent.name:
                candidates.append((OUTPUTS_DIR / p.parent.name / p.name).resolve())

    outputs_resolved = OUTPUTS_DIR.resolve()
    for cand in candidates:
        if cand.is_file():
            try:
                rel = cand.resolve().relative_to(outputs_resolved).as_posix()
                return cand.resolve(), f"/outputs/{rel}"
            except ValueError:
                return cand.resolve(), f"/outputs/{cand.name}"
    return None


def get_clip_by_id(clip_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# ============================================================================
# Sequential Single-GPU Job Runner
# ============================================================================

ANSI_REGEX = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def clean_log_line(line: str) -> str:
    """Removes ANSI color and cursor codes from a terminal log line."""
    return ANSI_REGEX.sub("", line).strip()


class JobRunner:
    """
    Manages a background worker thread that executes jobs sequentially.
    Ensures only ONE job uses the NVIDIA RTX 4060 GPU at any time.
    """

    def __init__(self) -> None:
        self.job_queue: queue.Queue[int] = queue.Queue()
        self.current_process: Optional[subprocess.Popen[str]] = None
        self.current_job_id: Optional[int] = None
        self.cancelled_job_ids: set[int] = set()
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def submit_job(self, job_id: int) -> None:
        self.job_queue.put(job_id)

    def cancel_job(self, job_id: int) -> bool:
        """Cancels a job. If currently running, terminates the Windows process tree."""
        with self.lock:
            self.cancelled_job_ids.add(job_id)
            if self.current_job_id == job_id and self.current_process:
                pid = self.current_process.pid
                try:
                    # /F = forcefully kill, /T = terminate entire process tree including ffmpeg & yt-dlp
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=False, capture_output=True)
                except Exception as e:
                    print(f"Error terminating process PID {pid}: {e}")
                return True
            else:
                # If still queued, immediately update database record
                now_iso = datetime.now().isoformat()
                with get_db() as conn:
                    conn.execute(
                        "UPDATE jobs SET status = 'cancelled', step = 'Cancelled by user', finished_at = ? WHERE id = ? AND status = 'queued'",
                        (now_iso, job_id),
                    )
                    conn.commit()
                return True

    def _update_job(self, job_id: int, **kwargs: Any) -> None:
        with get_db() as conn:
            if "progress" in kwargs:
                cur = conn.execute("SELECT progress FROM jobs WHERE id = ?", (job_id,))
                row = cur.fetchone()
                if row and row["progress"] is not None:
                    kwargs["progress"] = max(int(row["progress"]), int(kwargs["progress"]))

            cols = []
            params = []
            for k, v in kwargs.items():
                cols.append(f"{k} = ?")
                params.append(v)
            if cols:
                params.append(job_id)
                conn.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id = ?", params)
                conn.commit()

    def _worker_loop(self) -> None:
        while True:
            job_id = self.job_queue.get()
            try:
                # Check if cancelled before starting
                with self.lock:
                    if job_id in self.cancelled_job_ids:
                        now_iso = datetime.now().isoformat()
                        self._update_job(job_id, status="cancelled", step="Cancelled before starting", finished_at=now_iso)
                        continue

                # Load job details
                with get_db() as conn:
                    cur = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
                    row = cur.fetchone()
                if not row:
                    continue
                job = dict(row)

                # Mark running
                self._update_job(job_id, status="running", step="Initializing pipeline...", progress=5)

                # Assemble command args wrapping main.py
                cmd = [
                    sys.executable,
                    "-u",
                    str(BASE_DIR / "main.py"),
                    "auto-clip",
                    job["url"],
                    "--max-clips",
                    str(job.get("max_clips", 3)),
                    "--layout",
                    job.get("layout", "auto"),
                ]
                cam_corner = job.get("cam_corner")
                if cam_corner and cam_corner != "auto":
                    cmd.extend(["--cam-corner", cam_corner])
                if job.get("prompt"):
                    cmd.extend(["--prompt", job["prompt"]])
                if job.get("fast_subs", 1):
                    cmd.append("--fast-subs")
                if job.get("no_bgm", 0):
                    cmd.append("--no-bgm")

                # Setup log file
                log_file_path = LOGS_DIR / f"job_{job_id}.log"
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                with self.lock:
                    self.current_job_id = job_id
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=env,
                        bufsize=1,
                        cwd=str(BASE_DIR),
                    )
                    self.current_process = proc

                # Stream stdout and update progress
                with open(log_file_path, "w", encoding="utf-8") as log_file:
                    if proc.stdout:
                        for raw_line in proc.stdout:
                            log_file.write(raw_line)
                            log_file.flush()

                            line = clean_log_line(raw_line)
                            if not line:
                                continue

                            # Parse progress milestones
                            if "Querying VOD metadata" in line:
                                self._update_job(job_id, step="Querying YouTube VOD metadata...", progress=10)
                            elif "Autonomous Viral Pipeline" in line:
                                self._update_job(job_id, step="VOD metadata loaded, preparing ingest...", progress=15)
                            elif "Fast-Path Subtitles Ingest" in line:
                                self._update_job(job_id, step="Extracting fast-path captions (.json3)...", progress=25)
                            elif "Step 1: Smart Audio-Only Ingest" in line:
                                self._update_job(job_id, step="Downloading audio stream (MP3)...", progress=20)
                            elif "Step 2: RTX 4060 Whisper Transcription" in line:
                                self._update_job(job_id, step="Transcribing audio with faster-whisper (CUDA)...", progress=35)
                            elif "Transcribing on RTX 4060:" in line:
                                m = re.search(r"Transcribing on RTX 4060:\s*(\d+)%", line)
                                if m:
                                    pct = int(m.group(1))
                                    overall = 35 + int(pct * 0.20)
                                    self._update_job(job_id, step=f"Transcribing audio: {pct}%...", progress=overall)
                            elif "Step 3: Gemini Viral Highlight Detection" in line:
                                self._update_job(job_id, step="Gemini AI analyzing viral hooks & retention...", progress=60)
                            elif "Step 4: Smart Targeted Ingest & GPU Compositing" in line:
                                self._update_job(job_id, step="Smart targeted ingest & NVENC compositing...", progress=70)
                            elif "Rendering Clip" in line:
                                m = re.search(r"Rendering Clip\s+(\d+)/(\d+)", line)
                                if m:
                                    clip_idx = int(m.group(1))
                                    total_c = int(m.group(2))
                                    frac = clip_idx / max(total_c, 1)
                                    prog = 70 + int(frac * 25)
                                    self._update_job(job_id, step=f"Rendering clip {clip_idx}/{total_c} (h264_nvenc)...", progress=prog)
                            elif "All clips generated successfully" in line:
                                self._update_job(job_id, step="Finalizing clips...", progress=98)

                proc.wait()

                # Process completion state
                is_cancelled = job_id in self.cancelled_job_ids
                now_iso = datetime.now().isoformat()

                with get_db() as conn:
                    sync_clip_metadata(conn)

                if is_cancelled:
                    self._update_job(job_id, status="cancelled", step="Job cancelled by user", finished_at=now_iso)
                elif proc.returncode == 0:
                    self._update_job(job_id, status="completed", step="All clips complete!", progress=100, finished_at=now_iso)
                else:
                    error_msg = f"Job exited with return code {proc.returncode}"
                    try:
                        with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                            lines = [l.strip() for l in f.readlines() if l.strip()]
                            if lines:
                                error_msg = lines[-1][:300]
                    except Exception:
                        pass
                    self._update_job(job_id, status="failed", step="Pipeline failed", error=error_msg, finished_at=now_iso)

            except Exception as exc:
                now_iso = datetime.now().isoformat()
                self._update_job(job_id, status="failed", step="Internal error", error=str(exc), finished_at=now_iso)
            finally:
                with self.lock:
                    self.current_job_id = None
                    self.current_process = None
                self.job_queue.task_done()


job_runner = JobRunner()


# ============================================================================
# FastAPI Application & Endpoints
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_server_db()
    yield

app = FastAPI(title="ClipEngine Studio", version="1.0.0", lifespan=lifespan)

# Local browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

class JobCreateRequest(BaseModel):
    url: str
    prompt: Optional[str] = None
    max_clips: int = Field(default=3, ge=1, le=10)
    layout: str = Field(default="auto")
    cam_corner: str = Field(default="auto")
    fast_subs: bool = True
    no_bgm: bool = False


class ClipUpdateRequest(BaseModel):
    review_status: Optional[str] = None
    posted_url: Optional[str] = None
    platform: Optional[str] = None
    views: Optional[int] = None
    notes: Optional[str] = None


# ----------------------------------------------------------------------------
# Job Endpoints
# ----------------------------------------------------------------------------

@app.post("/api/jobs")
def create_job(req: JobCreateRequest) -> Dict[str, Any]:
    url = req.url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="A valid YouTube URL is required")

    now_iso = datetime.now().isoformat()
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO jobs (url, prompt, max_clips, layout, cam_corner, fast_subs, no_bgm, status, step, progress, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 'Waiting in queue...', 0, ?)
            """,
            (
                url,
                req.prompt.strip() if req.prompt else None,
                req.max_clips,
                req.layout,
                req.cam_corner,
                1 if req.fast_subs else 0,
                1 if req.no_bgm else 0,
                now_iso,
            ),
        )
        conn.commit()
        job_id = cur.lastrowid
        if job_id is None:
            raise HTTPException(status_code=500, detail="Failed to initialize job record")

    job_runner.submit_job(job_id)

    with get_db() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        job = dict(cur.fetchone())

    return {"job": job}


@app.get("/api/jobs")
def list_jobs() -> Dict[str, Any]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 50")
        jobs = [dict(row) for row in cur.fetchall()]
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int) -> Dict[str, Any]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        job = dict(row)

    log_path = LOGS_DIR / f"job_{job_id}.log"
    tail_lines: List[str] = []
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                tail_lines = [clean_log_line(l) for l in all_lines[-25:] if clean_log_line(l)]
        except Exception:
            pass
    job["log_tail"] = tail_lines
    return {"job": job}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: int) -> Dict[str, Any]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        if row["status"] in ("completed", "failed", "cancelled"):
            return {"status": "ignored", "message": f"Job is already {row['status']}"}

    job_runner.cancel_job(job_id)
    return {"status": "ok", "message": f"Job {job_id} cancelled"}


# ----------------------------------------------------------------------------
# Clip Review & Management Endpoints
# ----------------------------------------------------------------------------

@app.get("/api/clips")
def list_clips_api(
    stream_folder: Optional[str] = None,
    review_status: Optional[str] = None,
    min_score: Optional[float] = None,
    include_trashed: bool = False,
) -> Dict[str, Any]:
    with get_db() as conn:
        sync_clip_metadata(conn)

        query = "SELECT * FROM clips WHERE 1=1"
        params: List[Any] = []

        if stream_folder and stream_folder != "all":
            query += " AND stream_folder = ?"
            params.append(stream_folder)

        if review_status and review_status != "all":
            query += " AND COALESCE(review_status, 'new') = ?"
            params.append(review_status)
        elif not include_trashed:
            query += " AND COALESCE(review_status, 'new') != 'trashed'"

        if min_score is not None and min_score > 0:
            query += " AND virality_score >= ?"
            params.append(min_score)

        query += " ORDER BY id DESC"
        cur = conn.execute(query, params)
        raw_clips = [dict(row) for row in cur.fetchall()]

        # Unique folders list (excluding trashed clips)
        folder_cur = conn.execute(
            """
            SELECT DISTINCT stream_folder FROM clips 
            WHERE stream_folder IS NOT NULL AND stream_folder != '' 
              AND COALESCE(review_status, 'new') != 'trashed' 
            ORDER BY stream_folder ASC
            """
        )
        folders = [r["stream_folder"] for r in folder_cur.fetchall()]

        # Aggregated stats
        stats_cur = conn.execute(
            """
            SELECT 
                COUNT(*) as total_clips,
                SUM(CASE WHEN COALESCE(review_status, 'new') != 'trashed' THEN 1 ELSE 0 END) as active_clips,
                SUM(CASE WHEN COALESCE(review_status, 'new') = 'approved' THEN 1 ELSE 0 END) as approved_count,
                SUM(CASE WHEN COALESCE(review_status, 'new') = 'posted' THEN 1 ELSE 0 END) as posted_count,
                SUM(COALESCE(views, 0)) as total_views
            FROM clips
            """
        )
        stats_row = stats_cur.fetchone()
        stats = dict(stats_row) if stats_row else {}

    # Filter out missing files on disk and prepare client URLs
    valid_clips: List[Dict[str, Any]] = []
    for c in raw_clips:
        out_path = c.get("output_path")
        if not out_path:
            continue
        resolved = resolve_clip_file(out_path)
        if not resolved:
            continue

        file_path, video_url = resolved
        c["video_url"] = video_url
        c["duration"] = round(c.get("clip_end", 0.0) - c.get("clip_start", 0.0), 1)
        valid_clips.append(c)

    return {
        "clips": valid_clips,
        "folders": folders,
        "stats": stats,
    }


@app.patch("/api/clips/{clip_id}")
def update_clip_api(clip_id: int, req: ClipUpdateRequest) -> Dict[str, Any]:
    clip = get_clip_by_id(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    allowed_statuses = {"new", "approved", "rejected", "posted", "trashed"}
    if req.review_status is not None and req.review_status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid review_status. Allowed values: {sorted(allowed_statuses)}",
        )

    updates = []
    params: List[Any] = []
    if req.review_status is not None:
        updates.append("review_status = ?")
        params.append(req.review_status)
    if req.posted_url is not None:
        updates.append("posted_url = ?")
        params.append(req.posted_url.strip())
    if req.platform is not None:
        updates.append("platform = ?")
        params.append(req.platform.strip())
    if req.views is not None:
        updates.append("views = ?")
        params.append(max(0, req.views))
    if req.notes is not None:
        updates.append("notes = ?")
        params.append(req.notes.strip())

    if updates:
        params.append(clip_id)
        with get_db() as conn:
            conn.execute(f"UPDATE clips SET {', '.join(updates)} WHERE id = ?", params)
            conn.commit()

    updated = get_clip_by_id(clip_id)
    return {"clip": updated}


@app.delete("/api/clips/{clip_id}")
def delete_clip_api(clip_id: int) -> Dict[str, Any]:
    """Soft delete: moves clip file to outputs/.trash/ and updates status."""
    clip = get_clip_by_id(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    old_path_str = clip.get("output_path")
    new_path_str = old_path_str

    if old_path_str:
        resolved = resolve_clip_file(old_path_str)
        if resolved:
            old_path, _ = resolved
            dest = TRASH_DIR / old_path.name
            if dest.exists():
                dest = TRASH_DIR / f"{clip_id}_{old_path.name}"
            try:
                shutil.move(str(old_path), str(dest))
                new_path_str = str(dest)
            except Exception as e:
                print(f"Failed to move file to trash: {e}")

    with get_db() as conn:
        conn.execute(
            "UPDATE clips SET review_status = 'trashed', output_path = ? WHERE id = ?",
            (new_path_str, clip_id),
        )
        conn.commit()

    return {"status": "ok", "message": "Clip moved to trash"}


@app.post("/api/clips/{clip_id}/open-folder")
def open_clip_folder(clip_id: int) -> Dict[str, Any]:
    """Reveals the clip file inside Windows Explorer."""
    clip = get_clip_by_id(clip_id)
    if not clip or not clip.get("output_path"):
        raise HTTPException(status_code=404, detail="Clip or video path not found")

    resolved = resolve_clip_file(clip["output_path"])
    if not resolved:
        raise HTTPException(status_code=404, detail="Video file does not exist on disk")

    file_path, _ = resolved
    norm_path = os.path.normpath(str(file_path))
    # /select,<path> opens Explorer with the specific file selected
    subprocess.Popen(f'explorer.exe /select,"{norm_path}"')
    return {"status": "ok", "path": norm_path}


# ----------------------------------------------------------------------------
# Static File & Video Serving with Strict Path-Traversal Protection
# ----------------------------------------------------------------------------

@app.get("/outputs/{filepath:path}")
def serve_output_file(filepath: str) -> FileResponse:
    file_path = (OUTPUTS_DIR / filepath).resolve()
    # Guard against directory traversal attacks like /outputs/../.env
    if not file_path.is_relative_to(OUTPUTS_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied: path outside outputs directory")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4")


@app.get("/")
def serve_index() -> FileResponse:
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Studio frontend not found in web/index.html")
    return FileResponse(index_file)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    # Strictly bound to 127.0.0.1 for local privacy and security
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
