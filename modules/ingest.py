"""
ClipEngine Ingest Module
Handles smart targeted downloading using yt-dlp.
Ensures large VODs are never downloaded in full when only specific timestamp ranges are needed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yt_dlp

from config import DOWNLOADS_DIR, FFMPEG_BIN
from modules.transcriber import TranscriptionResult, TranscriptSegment, WordTimestamp


def parse_timestamp(time_val: Union[float, int, str]) -> float:
    """
    Parses various timestamp representations into total seconds.
    Examples:
        - 125 -> 125.0
        - "125" -> 125.0
        - "02:05" -> 125.0
        - "01:02:05" -> 3725.0
        - "01:02:05.500" -> 3725.5
    """
    if isinstance(time_val, (int, float)):
        return float(time_val)

    time_str = str(time_val).strip()
    # Check if purely numeric
    try:
        return float(time_str)
    except ValueError:
        pass

    parts = time_str.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: '{time_val}'. Expected SS, MM:SS, or HH:MM:SS.")


def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS or HH:MM:SS.mmm for yt-dlp sections."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60
    if sec == int(sec):
        return f"{hours:02d}:{minutes:02d}:{int(sec):02d}"
    return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"


def sanitize_title(title: str, max_chars: int = 60) -> str:
    """
    Sanitizes a title string into a clean folder/file name containing
    only letters, numbers, and underscores, up to max_chars.
    Example: 'iShowSpeed vs Alejandro Balde EA FC 1v1!' -> 'iShowSpeed_vs_Alejandro_Balde_EA_FC_1v1'
    """
    clean = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    clean = re.sub(r"_+", "_", clean)
    if not clean:
        clean = "clip"
    return clean[:max_chars].rstrip("_")


def get_video_metadata(url: str) -> Dict[str, Any]:
    """
    Fetches video metadata (title, duration, uploader, id) without downloading media.
    """
    ydl_opts: Dict[str, Any] = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(params=ydl_opts) as ydl: # type: ignore[arg-type]
        info = ydl.extract_info(url, download=False)
        return {
            "id": info.get("id"),
            "title": info.get("title", "Unknown Title"),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", "Unknown"),
            "webpage_url": info.get("webpage_url", url),
        }


def download_target_clip(
    url: str,
    start_time: Union[float, int, str],
    end_time: Union[float, int, str],
    output_dir: Path = DOWNLOADS_DIR,
    prefix: str = "clip",
) -> Path:
    """
    Downloads ONLY the specified timestamp section using yt-dlp keyframe-accurate cuts.
    Guarantees that 5-hour VODs are never downloaded in full.

    Args:
        url: Source YouTube or VOD URL.
        start_time: Start timestamp (seconds or HH:MM:SS string).
        end_time: End timestamp (seconds or HH:MM:SS string).
        output_dir: Destination folder.
        prefix: Filename prefix.

    Returns:
        Path to the downloaded MP4 clip.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_sec = parse_timestamp(start_time)
    end_sec = parse_timestamp(end_time)

    if start_sec >= end_sec:
        raise ValueError(
            f"Invalid interval: start_time ({start_sec}s) must be strictly less than end_time ({end_sec}s)."
        )

    start_formatted = format_timestamp(start_sec)
    end_formatted = format_timestamp(end_sec)
    section_arg = f"*{start_formatted}-{end_formatted}"

    # Sanitized safe filename
    clean_start = re.sub(r"[:.]", "_", start_formatted)
    clean_end = re.sub(r"[:.]", "_", end_formatted)

    # Check if this exact section was already downloaded
    existing_section = [
        f for f in output_dir.glob(f"*_{clean_start}_{clean_end}.*")
        if f.suffix.lower() in [".mp4", ".mkv", ".webm"] and f.exists() and f.stat().st_size > 50000
    ]
    for f in existing_section:
        target_f = output_dir / f"{prefix}_{f.name.split('_', 2)[-1]}"
        if target_f != f:
            import shutil
            shutil.copy2(f, target_f)
            return target_f.resolve()
        return f.resolve()

    out_template = str(output_dir / f"{prefix}_%(id)s_{clean_start}_{clean_end}.%(ext)s")
    ffmpeg_dir = str(Path(FFMPEG_BIN).parent)

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--download-sections",
        section_arg,
        "--force-keyframes-at-cuts",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "-f",
        "bestvideo[height<=1440]+bestaudio/best[height<=1440]",
        "--merge-output-format",
        "mp4",
        "--ffmpeg-location",
        ffmpeg_dir,
        "-o",
        out_template,
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Retry with android player client fallback if 403 or format error occurred
        retry_cmd = list(cmd)
        retry_cmd.extend(["--extractor-args", "youtube:player_client=android,web"])
        result = subprocess.run(retry_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"yt-dlp targeted download failed (code {result.returncode}):\n{result.stderr}"
            )

    # Resolve resulting file on disk
    expected_pattern = f"{prefix}_*_{clean_start}_{clean_end}.*"
    matched_files = [
        f for f in output_dir.glob(expected_pattern)
        if f.suffix.lower() in [".mp4", ".mkv", ".webm"]
    ]
    if not matched_files:
        # Fallback search for recently created mp4/mkv in output_dir
        candidate_files = sorted(
            [f for f in output_dir.glob("*.*") if f.suffix.lower() in [".mp4", ".mkv", ".webm"]],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidate_files:
            return candidate_files[0].resolve()
        raise FileNotFoundError(
            f"Downloaded clip not found matching pattern '{expected_pattern}' in {output_dir}."
        )

    return matched_files[0].resolve()


def download_audio_only(
    url: str,
    start_time: Optional[Union[float, int, str]] = None,
    end_time: Optional[Union[float, int, str]] = None,
    output_dir: Path = DOWNLOADS_DIR,
    prefix: str = "audio",
) -> Path:
    """
    Downloads audio only (MP3) for transcription in Stage 2 (faster-whisper).
    Optionally clips to a target timestamp section if specified.

    Args:
        url: Source YouTube or VOD URL.
        start_time: Optional start timestamp.
        end_time: Optional end timestamp.
        output_dir: Target directory.
        prefix: Filename prefix.

    Returns:
        Path to the extracted MP3 file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_dir = str(Path(FFMPEG_BIN).parent)

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "--ffmpeg-location",
        ffmpeg_dir,
    ]

    suffix = "full"
    if start_time is not None and end_time is not None:
        start_sec = parse_timestamp(start_time)
        end_sec = parse_timestamp(end_time)
        start_formatted = format_timestamp(start_sec)
        end_formatted = format_timestamp(end_sec)
        clean_start = re.sub(r"[:.]", "_", start_formatted)
        clean_end = re.sub(r"[:.]", "_", end_formatted)
        cmd.extend(
            [
                "--download-sections",
                f"*{start_formatted}-{end_formatted}",
                "--force-keyframes-at-cuts",
            ]
        )
        suffix = f"{clean_start}_{clean_end}"

    out_template = str(output_dir / f"{prefix}_%(id)s_{suffix}.%(ext)s")
    cmd.extend(["-o", out_template, url])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp audio download failed (code {result.returncode}):\n{result.stderr}"
        )

    expected_pattern = f"{prefix}_*_{suffix}.mp3"
    matched_files = list(output_dir.glob(expected_pattern))
    if not matched_files:
        mp3_files = sorted(
            output_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if mp3_files:
            return mp3_files[0].resolve()
        raise FileNotFoundError(
            f"Downloaded audio not found matching pattern '{expected_pattern}' in {output_dir}."
        )

    return matched_files[0].resolve()


def extract_youtube_captions(
    url: str,
    output_dir: Path = DOWNLOADS_DIR,
    prefix: str = "yt_subs",
) -> Optional[TranscriptionResult]:
    """
    Fast-path subtitle extraction using yt-dlp without downloading media.
    Fetches YouTube .json3 auto-subtitles (or manual captions) with word timestamps
    and converts them into a TranscriptionResult in ~3 seconds.

    Args:
        url: Source YouTube video URL.
        output_dir: Destination folder for temporary .json3 subtitles.
        prefix: Subtitle file prefix.

    Returns:
        TranscriptionResult if captions are available, otherwise None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / f"{prefix}_%(id)s.%(ext)s")

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--write-auto-subs",
        "--write-subs",
        "--sub-format",
        "json3",
        "--sub-langs",
        "en.*,en,all",
        "--skip-download",
        "--no-warnings",
        "--quiet",
        "-o",
        out_template,
        url,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    except Exception:
        return None

    # Search for downloaded json3 files matching prefix
    sub_files = sorted(
        output_dir.glob(f"{prefix}_*.json3"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not sub_files:
        return None

    json_file = sub_files[0]
    try:
        import json

        data = json.loads(json_file.read_text(encoding="utf-8"))
        events = data.get("events", [])
        if not events:
            return None

        segments: List[TranscriptSegment] = []
        seg_id = 1
        total_duration = 0.0

        for ev in events:
            if "segs" not in ev:
                continue
            start_ms = ev.get("tStartMs", 0)
            dur_ms = ev.get("dDurationMs", 0)
            raw_segs = ev["segs"]

            words: List[WordTimestamp] = []
            text_pieces: List[str] = []

            for i, s in enumerate(raw_segs):
                w_text = s.get("utf8", "")
                clean_w = w_text.replace("\n", " ").strip()
                clean_w = re.sub(r"^>>\s*", "", clean_w).strip()
                if not clean_w:
                    continue

                t_offset = s.get("tOffsetMs", 0)
                w_start = round((start_ms + t_offset) / 1000.0, 3)

                # Estimate word end boundary
                if i + 1 < len(raw_segs):
                    next_offset = raw_segs[i + 1].get("tOffsetMs", t_offset + 300)
                    w_end = round((start_ms + next_offset) / 1000.0, 3)
                else:
                    w_end = round((start_ms + dur_ms) / 1000.0, 3)

                if w_end <= w_start:
                    w_end = round(w_start + 0.35, 3)
                elif w_end > w_start + 1.5:
                    w_end = round(w_start + 0.8, 3)

                words.append(WordTimestamp(word=clean_w, start=w_start, end=w_end, probability=1.0))
                text_pieces.append(clean_w)
                total_duration = max(total_duration, w_end)

            if not words:
                continue

            full_seg_text = " ".join(text_pieces)
            seg_start = words[0].start
            seg_end = words[-1].end
            segments.append(
                TranscriptSegment(
                    id=seg_id,
                    start=seg_start,
                    end=seg_end,
                    text=full_seg_text,
                    words=words,
                )
            )
            seg_id += 1

        if not segments:
            return None

        return TranscriptionResult(
            audio_path=f"fast_subs://{url}",
            language="en",
            duration=round(total_duration, 2),
            segments=segments,
        )
    except Exception:
        return None

