"""
ClipEngine Overnight Batch & Source-Scouting Pipeline (Phase 1)
Autonomously scouts confirmed creator channels, filters candidate VODs, executes
single-GPU auto-clip runs under a strict 3:00 AM watchdog, and generates copy-ready
Shorts upload packs with Gemini AI.
"""

from __future__ import annotations

import argparse
import atexit
import os
import re
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from config import BASE_DIR, DB_PATH, GEMINI_API_KEY, GEMINI_MODEL
from database import (
    create_nightly_run,
    get_confirmed_sources,
    get_db_connection,
    init_db,
    is_vod_processed,
    record_processed_vod,
    update_clip_upload_pack,
    update_nightly_run,
)

# Runtime paths
LOGS_DIR: Path = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOCK_FILE: Path = BASE_DIR / "nightly.lock"

# Safety and Cutoff Limits
MAX_RUNTIME_HOURS: float = 4.0
HARD_CUTOFF_HOUR: int = 3  # 03:00 AM local time


# ============================================================================
# Lock & Process Management Helpers
# ============================================================================

def is_pid_alive(pid: int) -> bool:
    """Checks whether a process with the given PID is actively running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_nightly_lock(log_fn: Any) -> bool:
    """
    Acquires single-instance lock file with PID checking.
    Returns True if lock acquired, False if another instance is actively running.
    """
    current_pid = os.getpid()
    if LOCK_FILE.exists():
        try:
            content = LOCK_FILE.read_text(encoding="utf-8").strip()
            if content.isdigit():
                existing_pid = int(content)
                if is_pid_alive(existing_pid):
                    log_fn(f"[LOCK] Another batch run is active (PID {existing_pid}). Exiting cleanly.")
                    return False
                else:
                    log_fn(f"[LOCK] Found stale lock for inactive PID {existing_pid}. Overwriting.")
        except Exception as e:
            log_fn(f"[LOCK] Warning reading existing lock file: {e}. Overwriting.")

    try:
        LOCK_FILE.write_text(str(current_pid), encoding="utf-8")
        return True
    except Exception as e:
        log_fn(f"[LOCK] Failed to write lock file: {e}")
        return False


def release_nightly_lock() -> None:
    """Removes nightly.lock file if owned by current process."""
    try:
        if LOCK_FILE.exists():
            content = LOCK_FILE.read_text(encoding="utf-8").strip()
            if content.isdigit() and int(content) == os.getpid():
                LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def kill_process_tree(pid: int) -> None:
    """Forcefully terminates a Windows process and all its children (FFmpeg, Whisper, yt-dlp)."""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=False, capture_output=True)
    except Exception as e:
        print(f"Error terminating process PID {pid}: {e}")


def is_cutoff_reached(start_dt: datetime, max_runtime_hours: float = MAX_RUNTIME_HOURS) -> Tuple[bool, str]:
    """
    Enforces dual cutoff:
    1. Maximum runtime hours (default 4 hours)
    2. Hard 03:00 AM local time cutoff
    """
    now = datetime.now()
    # 1. Runtime elapsed check
    elapsed_seconds = (now - start_dt).total_seconds()
    if elapsed_seconds >= max_runtime_hours * 3600:
        return True, f"Max runtime of {max_runtime_hours}h reached ({elapsed_seconds / 3600:.2f}h elapsed)"

    # 2. Hard 3:00 AM cutoff check
    # If run started prior to 03:00, cutoff is today at 03:00
    # If run started at or after 03:00, cutoff is tomorrow at 03:00
    cutoff_3am = start_dt.replace(hour=HARD_CUTOFF_HOUR, minute=0, second=0, microsecond=0)
    if start_dt.hour >= HARD_CUTOFF_HOUR:
        cutoff_3am += timedelta(days=1)

    if now >= cutoff_3am:
        return True, f"Hard {HARD_CUTOFF_HOUR:02d}:00 AM local cutoff reached (Current: {now.strftime('%H:%M:%S')})"

    return False, ""


def wait_for_idle_gpu(log_fn: Any, start_dt: datetime) -> bool:
    """
    Checks clipengine.db jobs table. If any Studio UI job is 'running' or 'queued',
    waits and polls every 15s until GPU is free, or cutoff is reached.
    """
    while True:
        cutoff, reason = is_cutoff_reached(start_dt)
        if cutoff:
            log_fn(f"[WATCHDOG] Cutoff reached while waiting for GPU: {reason}")
            return False

        with get_db_connection(DB_PATH) as conn:
            cur = conn.execute("SELECT COUNT(*) as active_cnt FROM jobs WHERE status IN ('running', 'queued')")
            row = cur.fetchone()
            active_cnt = int(row["active_cnt"]) if row else 0

        if active_cnt == 0:
            return True

        log_fn(f"[GPU COORDINATION] {active_cnt} Studio UI job(s) running/queued. Waiting 15s for GPU...")
        time.sleep(15)


# ============================================================================
# YouTube RSS & Pre-Flight Video Filtering
# ============================================================================

def fetch_recent_channel_videos(rss_url: str, log_fn: Any, hours: int = 72) -> List[Dict[str, Any]]:
    """
    Fetches channel's YouTube RSS feed and filters videos published in the last `hours`.
    Returns list of dicts with video_id, title, published_iso, url.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    xml_data: Optional[bytes] = None
    last_err: Optional[Exception] = None

    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(rss_url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                xml_data = resp.read()
                break
        except Exception as e:
            last_err = e
            time.sleep(1.5)

    if not xml_data:
        log_fn(f"  [RSS ERROR] Failed to fetch feed {rss_url}: {last_err}")
        return []

    try:
        root = ET.fromstring(xml_data)
    except Exception as e:
        log_fn(f"  [RSS ERROR] Failed to parse XML feed: {e}")
        return []

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015",
    }

    now_utc = datetime.now(timezone.utc)
    cutoff_dt = now_utc - timedelta(hours=hours)
    candidate_videos: List[Dict[str, Any]] = []

    for entry in root.findall("atom:entry", ns):
        vid_elem = entry.find("yt:videoId", ns)
        title_elem = entry.find("atom:title", ns)
        pub_elem = entry.find("atom:published", ns)

        if vid_elem is None or vid_elem.text is None:
            continue
        video_id = vid_elem.text.strip()
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Untitled"
        pub_str = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else ""

        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        except Exception:
            pub_dt = now_utc

        if pub_dt >= cutoff_dt:
            candidate_videos.append({
                "video_id": video_id,
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "published_iso": pub_dt.isoformat(),
            })

    return candidate_videos


def preflight_video_filter(url: str, log_fn: Any) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Verifies video metadata via yt-dlp:
    1. Duration between 180s (3m) and 14400s (4h) - strictly skips Shorts.
    2. Live status is not 'is_upcoming' or 'is_live'.
    """
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }

    try:
        with yt_dlp.YoutubeDL(params=ydl_opts) as ydl:  # type: ignore
            raw_info = ydl.extract_info(url, download=False)
            if not raw_info:
                return False, None, "No metadata returned from yt-dlp"

            info: Dict[str, Any] = dict(raw_info)
            duration = int(info.get("duration") or 0)
            live_status = str(info.get("live_status") or "not_live")
            title = str(info.get("title") or "Unknown")

            if live_status in ("is_upcoming", "is_live"):
                return False, info, f"Stream is live or upcoming ({live_status})"

            if duration < 180:
                return False, info, f"Video duration {duration}s < 180s (skipping Short)"

            if duration > 14400:
                return False, info, f"Video duration {duration}s > 14400s (exceeds 4h max)"

            return True, {
                "id": info.get("id"),
                "title": title,
                "duration": duration,
                "channel": info.get("channel") or info.get("uploader"),
            }, "OK"

    except Exception as e:
        return False, None, f"yt-dlp preflight extraction failed: {e}"


# ============================================================================
# Resilient Upload Pack Generation (Gemini AI with Backoff Retry)
# ============================================================================

class UploadPackResult(BaseModel):
    upload_title: str = Field(..., description="High-CTR viral title under 60 chars ending with #shorts")
    upload_description: str = Field(..., description="2-sentence engaging summary + source credit + 5 niche hashtags")
    pinned_comment: str = Field(..., description="Thoughtful discussion question related to the topic. Strictly NO engagement-bait.")


def generate_upload_pack(
    clip: Dict[str, Any],
    channel_name: str,
    target_channel: str,
    log_fn: Any,
) -> Dict[str, str]:
    """
    Calls Gemini structured JSON generation with exponential backoff retry (up to 3 retries).
    Falls back to deterministic template metadata if API is unreachable or exhausted.
    """
    hook_title = clip.get("hook_title") or "Insightful Moment"
    virality_score = clip.get("virality_score") or 8.0

    prompt = f"""
Generate an optimized YouTube Shorts copy-ready upload pack for this short-form video clip.

Clip Metadata:
- Hook Title: {hook_title}
- Virality Score: {virality_score}/10
- Source Creator / Channel: {channel_name}
- Target Publishing Channel: {target_channel}

Instructions:
1. upload_title: High curiosity-gap / high CTR title, strictly under 60 characters, ending with "#shorts".
2. upload_description: Exactly 2 clear, punchy sentences summarizing the key lesson, followed by source attribution ("Credit: {channel_name}"), followed by 5 relevant niche hashtags.
3. pinned_comment: A genuine, high-value discussion question that sparks authentic replies and community conversation. Strictly NO spammy engagement bait like "comment X to win" or "type YES if you agree".
"""

    if GEMINI_API_KEY:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=UploadPackResult,
                        temperature=0.7,
                    ),
                )
                if response.parsed and isinstance(response.parsed, UploadPackResult):
                    pack = response.parsed
                    title = pack.upload_title.strip()
                    if not title.lower().endswith("#shorts"):
                        title = f"{title} #shorts"
                    if len(title) > 65:
                        title = title[:57].strip() + " #shorts"

                    return {
                        "upload_title": title,
                        "upload_description": pack.upload_description.strip(),
                        "pinned_comment": pack.pinned_comment.strip(),
                    }
            except Exception as exc:
                exc_str = str(exc)
                log_fn(f"  [GEMINI RETRY {attempt}/3] Upload pack generation error: {exc_str[:120]}")
                if "429" in exc_str or "503" in exc_str or "resource_exhausted" in exc_str.lower():
                    backoff = 2 ** attempt * 2  # 4s, 8s, 16s
                    time.sleep(backoff)
                else:
                    time.sleep(2)

    # Fallback Metadata
    log_fn("  [GEMINI FALLBACK] Using clean deterministic template for upload pack.")
    safe_title = (hook_title[:48].strip() + " #shorts") if len(hook_title) > 48 else f"{hook_title} #shorts"
    fallback_desc = (
        f"A powerful insight from {channel_name}. "
        f"Watch the full conversation for the deeper breakdown.\n\n"
        f"Credit: {channel_name}\n\n"
        f"#shorts #insights #mindset #learning #growth"
    )
    fallback_comment = f"What was your biggest takeaway from what {channel_name} shared in this clip?"

    return {
        "upload_title": safe_title,
        "upload_description": fallback_desc,
        "pinned_comment": fallback_comment,
    }


# ============================================================================
# Main Batch Pipeline Execution
# ============================================================================

def run_nightly_pipeline(
    dry_run: bool = False,
    max_clips_override: int = 10,
    log_file_path: Optional[Path] = None,
) -> int:
    """
    Executes the overnight batch clipping pipeline:
    1. Permission gating: only enabled = 1 AND permission = 'confirmed'.
    2. RSS query for videos in past 72h.
    3. Pre-flight filtering (Shorts / duration / live).
    4. Auto-clip execution with 3:00 AM watchdog.
    5. Calibration bin review status assignment (new vs low_score).
    6. Upload pack generation.
    """
    init_db(DB_PATH)
    start_dt = datetime.now()
    now_iso = start_dt.isoformat()

    if not log_file_path:
        log_file_path = LOGS_DIR / f"nightly_{start_dt.strftime('%Y%m%d_%H%M%S')}.log"

    def log(msg: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    log("=" * 70)
    log(f"ClipEngine Overnight Batch Pipeline Initiated (PID: {os.getpid()})")
    log(f"Mode: {'DRY RUN (Preview Only)' if dry_run else 'LIVE RUN'}")
    log(f"Max Nightly Clips Cap: {max_clips_override}")
    log(f"Hard Cutoff: {HARD_CUTOFF_HOUR:02d}:00 AM or {MAX_RUNTIME_HOURS}h max runtime")
    log("=" * 70)

    # 1. Acquire Lock
    if not acquire_nightly_lock(log):
        log("Execution halted: Another nightly run is currently active.")
        return 0

    atexit.register(release_nightly_lock)

    run_id = create_nightly_run(
        started_at=now_iso,
        log_path=str(log_file_path),
        status="dry_run" if dry_run else "running",
    )

    vods_checked = 0
    total_clips_created = 0
    failures = 0
    run_status = "completed"

    try:
        # 2. Permission-First Source Gating
        confirmed_sources = get_confirmed_sources(DB_PATH)
        if not confirmed_sources:
            log("No confirmed & enabled sources found in source_channels.")
            log("Review channels in Studio UI and confirm permissions before batch execution.")
            run_status = "no_sources"
            return 0

        log(f"Found {len(confirmed_sources)} confirmed & enabled source channel(s):")
        for s in confirmed_sources:
            log(f"  • {s['channel_name']} (ID: {s['youtube_channel_id']}, Target: {s.get('target_channel', 'Wealth Signals')})")

        # 3. Process Each Confirmed Channel
        for source in confirmed_sources:
            if total_clips_created >= max_clips_override:
                log(f"[CAP] Reached nightly clip quota of {max_clips_override}. Ending batch.")
                break

            cutoff, reason = is_cutoff_reached(start_dt)
            if cutoff:
                log(f"[WATCHDOG] Process-tree cutoff reached: {reason}")
                run_status = "cutoff"
                break

            channel_name = source["channel_name"]
            channel_id = source["id"]
            rss_url = source.get("rss_url") or f"https://www.youtube.com/feeds/videos.xml?channel_id={source['youtube_channel_id']}"
            target_channel = source.get("target_channel") or "Wealth Signals"
            layout = source.get("layout") or "cinema_blur"
            director_prompt = source.get("director_prompt") or ""
            max_clips_vod = min(int(source.get("max_clips_per_vod") or 2), 3, max_clips_override - total_clips_created)
            min_score = float(source.get("min_score") or 8.0)

            log("-" * 60)
            log(f"Scouting channel: {channel_name} (RSS: {rss_url})")

            recent_videos = fetch_recent_channel_videos(rss_url, log, hours=72)
            log(f"  Found {len(recent_videos)} video(s) published in last 72 hours.")

            for item in recent_videos:
                if total_clips_created >= max_clips_override:
                    break

                cutoff, reason = is_cutoff_reached(start_dt)
                if cutoff:
                    log(f"[WATCHDOG] Cutoff reached: {reason}")
                    run_status = "cutoff"
                    break

                video_id = item["video_id"]
                video_title = item["title"]
                video_url = item["url"]

                # Deduplication check
                if is_vod_processed(video_id, DB_PATH):
                    log(f"  [SKIP] Video {video_id} ('{video_title[:40]}...') already processed in previous run.")
                    continue

                vods_checked += 1
                log(f"\n  [PRE-FLIGHT] Inspecting video: {video_id} ('{video_title}')")

                # Pre-flight video filter
                passed, info, filter_reason = preflight_video_filter(video_url, log)
                if not passed:
                    log(f"  [FILTERED OUT] {filter_reason}. Skipping.")
                    record_processed_vod(
                        video_id=video_id,
                        channel_id=channel_id,
                        title=video_title,
                        url=video_url,
                        status="skipped_preflight",
                        clips_created=0,
                        error=filter_reason,
                        db_path=DB_PATH,
                    )
                    continue

                duration = info.get("duration", 0) if info else 0
                log(f"  [PASSED] Duration: {duration}s | Status: valid candidate VOD.")

                if dry_run:
                    log(f"  [DRY RUN PLAN] Would execute: main.py auto-clip {video_url} --max-clips {max_clips_vod} --layout {layout}")
                    log(f"  [DRY RUN PLAN] Target Channel: '{target_channel}' | Min Score: {min_score}")
                    continue

                # 4. Wait for GPU idle if Studio UI has active jobs
                if not wait_for_idle_gpu(log, start_dt):
                    run_status = "cutoff"
                    break

                # 5. Execute main.py auto-clip subprocess
                log(f"  [AUTO-CLIP START] Running main.py on RTX 4060 GPU...")
                cmd = [
                    sys.executable,
                    "-u",
                    str(BASE_DIR / "main.py"),
                    "auto-clip",
                    video_url,
                    "--max-clips",
                    str(max_clips_vod),
                    "--layout",
                    layout,
                    "--fast-subs",
                ]
                if director_prompt.strip():
                    cmd.extend(["--prompt", director_prompt.strip()])

                # Track pre-existing clips for this VOD to find newly created clips
                with get_db_connection(DB_PATH) as conn:
                    existing_clip_ids = {
                        r["id"] for r in conn.execute("SELECT id FROM clips WHERE vod_url = ?", (video_url,)).fetchall()
                    }

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(BASE_DIR),
                )

                subproc_error: Optional[str] = None
                try:
                    if proc.stdout:
                        for line in proc.stdout:
                            sline = line.strip()
                            if sline:
                                log(f"    [main.py] {sline}")

                            # Check cutoff while subprocess is executing
                            cutoff, reason = is_cutoff_reached(start_dt)
                            if cutoff:
                                log(f"  [WATCHDOG CRITICAL] {reason} while rendering PID {proc.pid}! Terminating process tree...")
                                kill_process_tree(proc.pid)
                                run_status = "cutoff"
                                subproc_error = f"Terminated by 3:00 AM / {MAX_RUNTIME_HOURS}h watchdog cutoff"
                                break

                    proc.wait()
                except Exception as sub_exc:
                    kill_process_tree(proc.pid)
                    subproc_error = str(sub_exc)

                if subproc_error:
                    failures += 1
                    record_processed_vod(
                        video_id=video_id,
                        channel_id=channel_id,
                        title=video_title,
                        url=video_url,
                        status="terminated_cutoff",
                        clips_created=0,
                        error=subproc_error,
                        db_path=DB_PATH,
                    )
                    break

                if proc.returncode != 0:
                    failures += 1
                    err_msg = f"main.py exited with return code {proc.returncode}"
                    log(f"  [ERROR] {err_msg}")
                    record_processed_vod(
                        video_id=video_id,
                        channel_id=channel_id,
                        title=video_title,
                        url=video_url,
                        status="failed",
                        clips_created=0,
                        error=err_msg,
                        db_path=DB_PATH,
                    )
                    continue

                # 6. Post-Process Rendered Clips: Calibration Bin & Upload Pack Generation
                with get_db_connection(DB_PATH) as conn:
                    new_clip_rows = conn.execute(
                        "SELECT * FROM clips WHERE vod_url = ? ORDER BY id ASC", (video_url,)
                    ).fetchall()
                    new_clips = [dict(r) for r in new_clip_rows if r["id"] not in existing_clip_ids]

                log(f"  [AUTO-CLIP COMPLETE] Generated {len(new_clips)} new clip(s).")
                vod_clips_count = 0

                for clip in new_clips:
                    clip_id = clip["id"]
                    virality_score = float(clip.get("virality_score") or 0.0)

                    # Score Calibration Bin: never delete or trash!
                    if virality_score >= min_score:
                        assigned_status = "new"
                        log(f"    • Clip #{clip_id} score {virality_score} >= {min_score} -> Bin: 'New' Morning Review")
                    else:
                        assigned_status = "low_score"
                        log(f"    • Clip #{clip_id} score {virality_score} < {min_score} -> Bin: 'Low Score' Calibration")

                    # Generate resilient upload pack with Gemini
                    upload_pack = generate_upload_pack(clip, channel_name, target_channel, log)

                    # Update clip record
                    update_clip_upload_pack(
                        clip_id=clip_id,
                        target_channel=target_channel,
                        upload_title=upload_pack["upload_title"],
                        upload_description=upload_pack["upload_description"],
                        pinned_comment=upload_pack["pinned_comment"],
                        source_type="nightly",
                        review_status=assigned_status,
                        db_path=DB_PATH,
                    )
                    vod_clips_count += 1
                    total_clips_created += 1

                record_processed_vod(
                    video_id=video_id,
                    channel_id=channel_id,
                    title=video_title,
                    url=video_url,
                    status="completed",
                    clips_created=vod_clips_count,
                    error=None,
                    db_path=DB_PATH,
                )

                # Small delay between VODs
                time.sleep(2)

    except Exception as run_exc:
        failures += 1
        run_status = "failed"
        log(f"[FATAL ERROR] Batch run aborted: {run_exc}")
    finally:
        finished_iso = datetime.now().isoformat()
        update_nightly_run(
            run_id=run_id,
            finished_at=finished_iso,
            status=run_status,
            vods_checked=vods_checked,
            clips_created=total_clips_created,
            failures=failures,
            log_path=str(log_file_path),
            db_path=DB_PATH,
        )
        release_nightly_lock()

        log("=" * 70)
        log(f"Overnight Batch Pipeline Concluded | Status: {run_status}")
        log(f"VODs Inspected: {vods_checked} | Clips Created: {total_clips_created} | Failures: {failures}")
        log(f"Log Output Saved: {log_file_path}")
        log("=" * 70)

    return 0


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ClipEngine Overnight Batch & Source-Scouting Pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Checks RSS feeds, filters Shorts/live streams, and prints planned work without rendering",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=10,
        help="Maximum total clips to render across all sources tonight (default 10)",
    )
    args = parser.parse_args()

    exit_code = run_nightly_pipeline(dry_run=args.dry_run, max_clips_override=args.max_clips)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
