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
    """Creates a database connection with dict-like row access, WAL mode, and busy timeout."""
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
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

    CREATE TABLE IF NOT EXISTS source_channels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_name TEXT,
        youtube_channel_id TEXT UNIQUE,
        rss_url TEXT,
        target_channel TEXT DEFAULT 'Wealth Signals',
        layout TEXT DEFAULT 'cinema_blur',
        director_prompt TEXT,
        max_clips_per_vod INTEGER DEFAULT 2,
        min_score REAL DEFAULT 8.0,
        permission TEXT DEFAULT 'unconfirmed',
        enabled INTEGER DEFAULT 0,
        created_at TEXT
    );

    CREATE TABLE IF NOT EXISTS processed_vods (
        video_id TEXT PRIMARY KEY,
        channel_id INTEGER,
        title TEXT,
        url TEXT,
        status TEXT,
        clips_created INTEGER DEFAULT 0,
        error TEXT,
        processed_at TEXT
    );

    CREATE TABLE IF NOT EXISTS nightly_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT,
        finished_at TEXT,
        status TEXT,
        vods_checked INTEGER DEFAULT 0,
        clips_created INTEGER DEFAULT 0,
        failures INTEGER DEFAULT 0,
        log_path TEXT
    );
    """
    with get_db_connection(db_path) as conn:
        conn.executescript(schema)
        # Migrate existing clips table if columns are missing
        existing_cols = {c["name"] for c in conn.execute("PRAGMA table_info(clips)").fetchall()}
        clip_migrations = [
            ("vibe", "ALTER TABLE clips ADD COLUMN vibe TEXT"),
            ("climax_offset", "ALTER TABLE clips ADD COLUMN climax_offset REAL"),
            ("use_dramatic_punch", "ALTER TABLE clips ADD COLUMN use_dramatic_punch INTEGER"),
            ("target_channel", "ALTER TABLE clips ADD COLUMN target_channel TEXT DEFAULT 'Wealth Signals'"),
            ("upload_title", "ALTER TABLE clips ADD COLUMN upload_title TEXT"),
            ("upload_description", "ALTER TABLE clips ADD COLUMN upload_description TEXT"),
            ("pinned_comment", "ALTER TABLE clips ADD COLUMN pinned_comment TEXT"),
            ("source_type", "ALTER TABLE clips ADD COLUMN source_type TEXT DEFAULT 'manual'"),
        ]
        for col_name, sql in clip_migrations:
            if col_name not in existing_cols:
                conn.execute(sql)

        # Migrate vod_webcam_cache table if columns are missing
        existing_webcam_cols = {c["name"] for c in conn.execute("PRAGMA table_info(vod_webcam_cache)").fetchall()}
        if "layout" not in existing_webcam_cols:
            conn.execute("ALTER TABLE vod_webcam_cache ADD COLUMN layout TEXT")
        if "speaker_x" not in existing_webcam_cols:
            conn.execute("ALTER TABLE vod_webcam_cache ADD COLUMN speaker_x REAL")

        # Migrate source_channels table if permission is missing
        existing_source_cols = {c["name"] for c in conn.execute("PRAGMA table_info(source_channels)").fetchall()}
        if "permission" not in existing_source_cols:
            conn.execute("ALTER TABLE source_channels ADD COLUMN permission TEXT DEFAULT 'unconfirmed'")

        # Pre-seed default channels strictly as permission = 'unconfirmed', enabled = 0
        seed_sources = [
            (
                "Chris Williamson",
                "UCEioQiyj_jPlbwA2QBjApWg",
                "https://www.youtube.com/feeds/videos.xml?channel_id=UCEioQiyj_jPlbwA2QBjApWg",
                "Wealth Signals",
                "cinema_blur",
                "Focus on high-value life advice, mental models, psychology insights, and discipline.",
                2,
                8.0,
                "unconfirmed",
                0,
            ),
            (
                "Ali Abdaal",
                "UCoOae5nYA7VqaXzerajD0lg",
                "https://www.youtube.com/feeds/videos.xml?channel_id=UCoOae5nYA7VqaXzerajD0lg",
                "Wealth Signals",
                "cinema_blur",
                "Focus on productivity hacks, building wealth, business insights, and mindset shifts.",
                2,
                8.0,
                "unconfirmed",
                0,
            ),
        ]
        now_iso = datetime.now().isoformat()
        for src in seed_sources:
            conn.execute(
                """
                INSERT OR IGNORE INTO source_channels (
                    channel_name, youtube_channel_id, rss_url, target_channel, layout,
                    director_prompt, max_clips_per_vod, min_score, permission, enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*src, now_iso),
            )
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


# ============================================================================
# Source Channels & Overnight Batch Pipeline Database Helpers
# ============================================================================

def get_confirmed_sources(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Returns only channels where enabled = 1 AND permission = 'confirmed'."""
    init_db(db_path)
    query = """
    SELECT * FROM source_channels
    WHERE enabled = 1 AND permission = 'confirmed'
    ORDER BY id ASC
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]


def list_sources(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all configured source channels."""
    init_db(db_path)
    query = "SELECT * FROM source_channels ORDER BY id ASC"
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]


def get_source_by_id(source_id: int, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves a single source channel by ID."""
    init_db(db_path)
    query = "SELECT * FROM source_channels WHERE id = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(query, (source_id,)).fetchone()
        return dict(row) if row else None


def add_source(
    channel_name: str,
    youtube_channel_id: str,
    rss_url: Optional[str] = None,
    target_channel: str = "Wealth Signals",
    layout: str = "cinema_blur",
    director_prompt: Optional[str] = None,
    max_clips_per_vod: int = 2,
    min_score: float = 8.0,
    permission: str = "unconfirmed",
    enabled: int = 0,
    db_path: Path = DB_PATH,
) -> int:
    """Adds a new source channel record. Returns the new ID."""
    init_db(db_path)
    if not rss_url:
        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={youtube_channel_id}"

    # Safety enforcement: cannot be enabled if permission is not confirmed
    if permission != "confirmed":
        enabled = 0

    now_iso = datetime.now().isoformat()
    query = """
    INSERT INTO source_channels (
        channel_name, youtube_channel_id, rss_url, target_channel, layout,
        director_prompt, max_clips_per_vod, min_score, permission, enabled, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            query,
            (
                channel_name,
                youtube_channel_id,
                rss_url,
                target_channel,
                layout,
                director_prompt,
                max_clips_per_vod,
                min_score,
                permission,
                enabled,
                now_iso,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid or 0)


def update_source(
    source_id: int,
    channel_name: Optional[str] = None,
    youtube_channel_id: Optional[str] = None,
    rss_url: Optional[str] = None,
    target_channel: Optional[str] = None,
    layout: Optional[str] = None,
    director_prompt: Optional[str] = None,
    max_clips_per_vod: Optional[int] = None,
    min_score: Optional[float] = None,
    permission: Optional[str] = None,
    enabled: Optional[int] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Updates fields of an existing source channel."""
    init_db(db_path)
    existing = get_source_by_id(source_id, db_path)
    if not existing:
        return

    # Check permission rule: if unconfirmed, enabled MUST be 0
    effective_permission = permission if permission is not None else existing.get("permission", "unconfirmed")
    effective_enabled = enabled if enabled is not None else existing.get("enabled", 0)
    if effective_permission != "confirmed":
        effective_enabled = 0

    updates: List[str] = []
    params: List[Any] = []

    if channel_name is not None:
        updates.append("channel_name = ?")
        params.append(channel_name)
    if youtube_channel_id is not None:
        updates.append("youtube_channel_id = ?")
        params.append(youtube_channel_id)
        if rss_url is None:
            updates.append("rss_url = ?")
            params.append(f"https://www.youtube.com/feeds/videos.xml?channel_id={youtube_channel_id}")
    if rss_url is not None:
        updates.append("rss_url = ?")
        params.append(rss_url)
    if target_channel is not None:
        updates.append("target_channel = ?")
        params.append(target_channel)
    if layout is not None:
        updates.append("layout = ?")
        params.append(layout)
    if director_prompt is not None:
        updates.append("director_prompt = ?")
        params.append(director_prompt)
    if max_clips_per_vod is not None:
        updates.append("max_clips_per_vod = ?")
        params.append(max(1, min(3, max_clips_per_vod)))
    if min_score is not None:
        updates.append("min_score = ?")
        params.append(min_score)

    updates.append("permission = ?")
    params.append(effective_permission)
    updates.append("enabled = ?")
    params.append(effective_enabled)

    params.append(source_id)
    query = f"UPDATE source_channels SET {', '.join(updates)} WHERE id = ?"
    with get_db_connection(db_path) as conn:
        conn.execute(query, tuple(params))
        conn.commit()


def delete_source(source_id: int, db_path: Path = DB_PATH) -> None:
    """Deletes a source channel by ID."""
    init_db(db_path)
    query = "DELETE FROM source_channels WHERE id = ?"
    with get_db_connection(db_path) as conn:
        conn.execute(query, (source_id,))
        conn.commit()


def is_vod_processed(video_id: str, db_path: Path = DB_PATH) -> bool:
    """Checks if a video has already been recorded in processed_vods."""
    init_db(db_path)
    query = "SELECT 1 FROM processed_vods WHERE video_id = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(query, (video_id,)).fetchone()
        return row is not None


def record_processed_vod(
    video_id: str,
    channel_id: Optional[int],
    title: str,
    url: str,
    status: str,
    clips_created: int = 0,
    error: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Records or updates a VOD in processed_vods."""
    init_db(db_path)
    now_iso = datetime.now().isoformat()
    query = """
    INSERT INTO processed_vods (video_id, channel_id, title, url, status, clips_created, error, processed_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(video_id) DO UPDATE SET
        channel_id = excluded.channel_id,
        title = excluded.title,
        url = excluded.url,
        status = excluded.status,
        clips_created = excluded.clips_created,
        error = excluded.error,
        processed_at = excluded.processed_at
    """
    with get_db_connection(db_path) as conn:
        conn.execute(query, (video_id, channel_id, title, url, status, clips_created, error, now_iso))
        conn.commit()


def create_nightly_run(
    started_at: str,
    log_path: Optional[str] = None,
    status: str = "running",
    db_path: Path = DB_PATH,
) -> int:
    """Inserts a new record into nightly_runs. Returns the new ID."""
    init_db(db_path)
    query = """
    INSERT INTO nightly_runs (started_at, status, log_path, vods_checked, clips_created, failures)
    VALUES (?, ?, ?, 0, 0, 0)
    """
    with get_db_connection(db_path) as conn:
        cur = conn.execute(query, (started_at, status, log_path))
        conn.commit()
        return int(cur.lastrowid or 0)


def update_nightly_run(
    run_id: int,
    finished_at: Optional[str] = None,
    status: Optional[str] = None,
    vods_checked: Optional[int] = None,
    clips_created: Optional[int] = None,
    failures: Optional[int] = None,
    log_path: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Updates fields of an existing nightly_runs record."""
    init_db(db_path)
    updates: List[str] = []
    params: List[Any] = []

    if finished_at is not None:
        updates.append("finished_at = ?")
        params.append(finished_at)
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if vods_checked is not None:
        updates.append("vods_checked = ?")
        params.append(vods_checked)
    if clips_created is not None:
        updates.append("clips_created = ?")
        params.append(clips_created)
    if failures is not None:
        updates.append("failures = ?")
        params.append(failures)
    if log_path is not None:
        updates.append("log_path = ?")
        params.append(log_path)

    if not updates:
        return

    params.append(run_id)
    query = f"UPDATE nightly_runs SET {', '.join(updates)} WHERE id = ?"
    with get_db_connection(db_path) as conn:
        conn.execute(query, tuple(params))
        conn.commit()


def get_latest_nightly_run(db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent nightly run summary."""
    init_db(db_path)
    query = "SELECT * FROM nightly_runs ORDER BY id DESC LIMIT 1"
    with get_db_connection(db_path) as conn:
        row = conn.execute(query).fetchone()
        return dict(row) if row else None


def list_nightly_runs(limit: int = 20, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent nightly runs."""
    init_db(db_path)
    query = "SELECT * FROM nightly_runs ORDER BY id DESC LIMIT ?"
    with get_db_connection(db_path) as conn:
        cur = conn.execute(query, (limit,))
        return [dict(row) for row in cur.fetchall()]


def update_clip_upload_pack(
    clip_id: int,
    target_channel: Optional[str] = None,
    upload_title: Optional[str] = None,
    upload_description: Optional[str] = None,
    pinned_comment: Optional[str] = None,
    source_type: Optional[str] = None,
    review_status: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Updates a clip's upload pack metadata and review status."""
    init_db(db_path)
    updates: List[str] = []
    params: List[Any] = []

    if target_channel is not None:
        updates.append("target_channel = ?")
        params.append(target_channel)
    if upload_title is not None:
        updates.append("upload_title = ?")
        params.append(upload_title)
    if upload_description is not None:
        updates.append("upload_description = ?")
        params.append(upload_description)
    if pinned_comment is not None:
        updates.append("pinned_comment = ?")
        params.append(pinned_comment)
    if source_type is not None:
        updates.append("source_type = ?")
        params.append(source_type)
    if review_status is not None:
        updates.append("review_status = ?")
        params.append(review_status)

    if not updates:
        return

    params.append(clip_id)
    query = f"UPDATE clips SET {', '.join(updates)} WHERE id = ?"
    with get_db_connection(db_path) as conn:
        conn.execute(query, tuple(params))
        conn.commit()


def resolve_youtube_channel(identifier: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolves a YouTube handle (@creator), custom URL, or channel URL to:
    (youtube_channel_id, channel_name, rss_url).
    """
    import re
    import urllib.request

    ident = identifier.strip()
    if not ident:
        return None, None, None

    # Direct 24-char UC channel ID
    if ident.startswith("UC") and len(ident) == 24:
        rss = f"https://www.youtube.com/feeds/videos.xml?channel_id={ident}"
        return ident, ident, rss

    url = ident
    if ident.startswith("@"):
        url = f"https://www.youtube.com/{ident}"
    elif not url.startswith("http"):
        url = f"https://www.youtube.com/@{ident}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None, None, None

    channel_id: Optional[str] = None
    for pattern in [
        r'itemprop="channelId"\s+content="([a-zA-Z0-9_-]{24})"',
        r'"externalId"\s*:\s*"([a-zA-Z0-9_-]{24})"',
        r'"channelId"\s*:\s*"([a-zA-Z0-9_-]{24})"',
        r'/channel/([a-zA-Z0-9_-]{24})',
    ]:
        m = re.search(pattern, html)
        if m:
            channel_id = m.group(1)
            break

    name: Optional[str] = None
    name_m = re.search(r'<meta\s+property="og:title"\s+content="([^"]+)"', html)
    if name_m:
        name = name_m.group(1).strip()
    else:
        title_m = re.search(r'<title>(.*?)(?: - YouTube)?</title>', html)
        if title_m:
            name = title_m.group(1).strip()

    if channel_id:
        rss = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        return channel_id, name or ident, rss

    return None, None, None




