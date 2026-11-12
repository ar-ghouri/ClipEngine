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
from typing import Any, Dict, Optional, Union

import yt_dlp

from config import DOWNLOADS_DIR, FFMPEG_BIN


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
    existing_section = list(output_dir.glob(f"*_{clean_start}_{clean_end}.mp4"))
    for f in existing_section:
        if f.exists() and f.stat().st_size > 50000:
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
        "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
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
    expected_pattern = f"{prefix}_*_{clean_start}_{clean_end}.mp4"
    matched_files = list(output_dir.glob(expected_pattern))
    if not matched_files:
        # Fallback search for recently created mp4 in output_dir
        mp4_files = sorted(
            output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if mp4_files:
            return mp4_files[0].resolve()
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

