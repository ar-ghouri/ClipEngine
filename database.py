"""
ClipEngine Database Module
Handles SQLite state tracking for processed video clips across all pipeline stages.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import DB_PATH


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Creates a database connection with dict-like row access."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Initializes the SQLite schema if it does not already exist."""
    schema = """
    CREATE TABLE IF NOT EXISTS clips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vod_url TEXT NOT NULL,
        clip_start REAL NOT NULL,
        clip_end REAL NOT NULL,
        hook_title TEXT,
        virality_score REAL,
        output_path TEXT,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        vibe TEXT,
        climax_offset REAL,
        use_dramatic_punch INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);

    CREATE TABLE IF NOT EXISTS vod_webcam_cache (
        vod_key TEXT PRIMARY KEY,
        has_corner_webcam INTEGER NOT NULL,
        cam_x REAL,
        cam_y REAL,
        cam_w REAL,
        cam_h REAL,
        detected_at TEXT NOT NULL
    );
    """
    with get_db_connection(db_path) as conn:
        conn.executescript(schema)
        # Migrate existing table if columns are missing
        existing_cols = [c["name"] for c in conn.execute("PRAGMA table_info(clips)").fetchall()]
        if "vibe" not in existing_cols:
            conn.execute("ALTER TABLE clips ADD COLUMN vibe TEXT")
        if "climax_offset" not in existing_cols:
            conn.execute("ALTER TABLE clips ADD COLUMN climax_offset REAL")
        if "use_dramatic_punch" not in existing_cols:
            conn.execute("ALTER TABLE clips ADD COLUMN use_dramatic_punch INTEGER")

        # Migrate vod_webcam_cache table if columns are missing
        existing_webcam_cols = [c["name"] for c in conn.execute("PRAGMA table_info(vod_webcam_cache)").fetchall()]
        if "layout" not in existing_webcam_cols:
            conn.execute("ALTER TABLE vod_webcam_cache ADD COLUMN layout TEXT")
        if "speaker_x" not in existing_webcam_cols:
            conn.execute("ALTER TABLE vod_webcam_cache ADD COLUMN speaker_x REAL")
        conn.commit()


def create_clip_record(
    vod_url: str,
    clip_start: float,
    clip_end: float,
    hook_title: Optional[str] = None,
    virality_score: Optional[float] = None,
    output_path: Optional[str] = None,
    status: str = "pending",
    vibe: Optional[str] = None,
    climax_offset: Optional[float] = None,
    use_dramatic_punch: Optional[bool] = None,
    db_path: Path = DB_PATH,
) -> int:
    """
    Inserts a new clip task record into the database.
    Returns the newly created primary key ID.
    """
    now_iso = datetime.now().isoformat()
    query = """
    INSERT INTO clips (vod_url, clip_start, clip_end, hook_title, virality_score, output_path, status, created_at, vibe, climax_offset, use_dramatic_punch)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            query,
            (
                vod_url,
                clip_start,
                clip_end,
                hook_title,
                virality_score,
                output_path,
                status,
                now_iso,
                vibe,
                climax_offset,
                1 if use_dramatic_punch else 0 if use_dramatic_punch is not None else None,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid or 0)


def update_clip_record(
    clip_id: int,
    status: Optional[str] = None,
    output_path: Optional[str] = None,
    hook_title: Optional[str] = None,
    virality_score: Optional[float] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Updates fields of an existing clip record by ID."""
    updates: List[str] = []
    params: List[Any] = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if output_path is not None:
        updates.append("output_path = ?")
        params.append(output_path)
    if hook_title is not None:
        updates.append("hook_title = ?")
        params.append(hook_title)
    if virality_score is not None:
        updates.append("virality_score = ?")
        params.append(virality_score)

    if not updates:
        return

    params.append(clip_id)
    query = f"UPDATE clips SET {', '.join(updates)} WHERE id = ?"
    with get_db_connection(db_path) as conn:
        conn.execute(query, tuple(params))
        conn.commit()


def get_clip_record(clip_id: int, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetches a single clip record by ID as a dictionary."""
    query = "SELECT * FROM clips WHERE id = ?"
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(query, (clip_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_clips(limit: int = 50, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves the most recent clips."""
    query = "SELECT * FROM clips ORDER BY id DESC LIMIT ?"
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(query, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_cached_webcam(
    vod_key: str, db_path: Path = DB_PATH
) -> Optional[Tuple[bool, Optional[Tuple[float, float, float, float]]]]:
    """Retrieves cached webcam detection result for a VOD from SQLite."""
    init_db(db_path)
    query = "SELECT has_corner_webcam, cam_x, cam_y, cam_w, cam_h FROM vod_webcam_cache WHERE vod_key = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(query, (vod_key,)).fetchone()
        if not row:
            return None
        has_cam = bool(row["has_corner_webcam"])
        if has_cam and row["cam_w"] is not None:
            return True, (
                float(row["cam_x"]),
                float(row["cam_y"]),
                float(row["cam_w"]),
                float(row["cam_h"]),
            )
        return False, None


def set_cached_webcam(
    vod_key: str,
    has_cam: bool,
    bbox: Optional[Tuple[float, float, float, float]],
    db_path: Path = DB_PATH,
) -> None:
    """Saves or updates cached webcam detection result for a VOD in SQLite."""
    init_db(db_path)
    now_iso = datetime.now().isoformat()
    query = """
    INSERT INTO vod_webcam_cache (vod_key, has_corner_webcam, cam_x, cam_y, cam_w, cam_h, detected_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(vod_key) DO UPDATE SET
        has_corner_webcam = excluded.has_corner_webcam,
        cam_x = excluded.cam_x,
        cam_y = excluded.cam_y,
        cam_w = excluded.cam_w,
        cam_h = excluded.cam_h,
        detected_at = excluded.detected_at
    """
    cam_x, cam_y, cam_w, cam_h = bbox if bbox else (None, None, None, None)
    with get_db_connection(db_path) as conn:
        conn.execute(query, (vod_key, 1 if has_cam else 0, cam_x, cam_y, cam_w, cam_h, now_iso))
        conn.commit()


def get_cached_layout(
    vod_key: str, db_path: Path = DB_PATH
) -> Optional[Tuple[str, Optional[Tuple[float, float, float, float]], float]]:
    """Retrieves cached layout classification (layout, cam_bbox, speaker_x) from SQLite."""
    init_db(db_path)
    query = "SELECT layout, has_corner_webcam, cam_x, cam_y, cam_w, cam_h, speaker_x FROM vod_webcam_cache WHERE vod_key = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(query, (vod_key,)).fetchone()
        if not row or not row["layout"]:
            return None
        layout = str(row["layout"])
        has_cam = bool(row["has_corner_webcam"])
        bbox = None
        if has_cam and row["cam_w"] is not None:
            bbox = (
                float(row["cam_x"]),
                float(row["cam_y"]),
                float(row["cam_w"]),
                float(row["cam_h"]),
            )
        spk_x = float(row["speaker_x"]) if row["speaker_x"] is not None else 0.5
        return layout, bbox, spk_x


def set_cached_layout(
    vod_key: str,
    layout: str,
    has_cam: bool,
    bbox: Optional[Tuple[float, float, float, float]],
    speaker_x: float = 0.5,
    db_path: Path = DB_PATH,
) -> None:
    """Saves or updates cached layout classification in SQLite."""
    init_db(db_path)
    now_iso = datetime.now().isoformat()
    query = """
    INSERT INTO vod_webcam_cache (vod_key, has_corner_webcam, cam_x, cam_y, cam_w, cam_h, layout, speaker_x, detected_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(vod_key) DO UPDATE SET
        has_corner_webcam = excluded.has_corner_webcam,
        cam_x = excluded.cam_x,
        cam_y = excluded.cam_y,
        cam_w = excluded.cam_w,
        cam_h = excluded.cam_h,
        layout = excluded.layout,
        speaker_x = excluded.speaker_x,
        detected_at = excluded.detected_at
    """
    cam_x, cam_y, cam_w, cam_h = bbox if bbox else (None, None, None, None)
    with get_db_connection(db_path) as conn:
        conn.execute(query, (vod_key, 1 if has_cam else 0, cam_x, cam_y, cam_w, cam_h, layout, speaker_x, now_iso))
        conn.commit()


