"""
ClipEngine Database Module
Handles SQLite state tracking for processed video clips across all pipeline stages.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);
    """
    with get_db_connection(db_path) as conn:
        conn.executescript(schema)
        conn.commit()


def create_clip_record(
    vod_url: str,
    clip_start: float,
    clip_end: float,
    hook_title: Optional[str] = None,
    virality_score: Optional[float] = None,
    output_path: Optional[str] = None,
    status: str = "pending",
    db_path: Path = DB_PATH,
) -> int:
    """
    Inserts a new clip task record into the database.
    Returns the newly created primary key ID.
    """
    now_iso = datetime.now().isoformat()
    query = """
    INSERT INTO clips (vod_url, clip_start, clip_end, hook_title, virality_score, output_path, status, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        conn.commit()
        return cursor.lastrowid


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

