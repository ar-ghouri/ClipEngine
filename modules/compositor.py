"""
ClipEngine Compositor Module
NVIDIA GPU-accelerated video compositing engine using h264_nvenc.
Produces 1080x1920 (9:16) vertical MP4 clips for TikTok, Shorts, and Reels
with zero aspect ratio distortion.
"""

from __future__ import annotations

import json
import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from config import (
    AUDIO_BITRATE,
    AUDIO_CODEC,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEFAULT_WEBCAM_BBOX,
    FFMPEG_BIN,
    FFPROBE_BIN,
    NVENC_CODEC,
    NVENC_CQ,
    NVENC_PRESET,
    OUTPUTS_DIR,
    SPLIT_BOTTOM_AR,
    SPLIT_BOTTOM_HEIGHT,
    SPLIT_BOTTOM_WIDTH,
    SPLIT_TOP_AR,
    SPLIT_TOP_HEIGHT,
    SPLIT_TOP_WIDTH,
)

logger = logging.getLogger("clipengine.compositor")


class LayoutMode(str, Enum):
    CENTER_CROP = "center_crop"
    GAMING_SPLIT = "gaming_split"


def probe_video(video_path: Path) -> Dict[str, Any]:
    """
    Inspects input video streams using ffprobe to obtain exact geometry and duration.
    """
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{result.stderr}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise ValueError(f"No video streams detected in {video_path}")

    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])

    duration_str = stream.get("duration") or data.get("format", {}).get("duration", "0")
    try:
        duration = float(duration_str)
    except (ValueError, TypeError):
        duration = 0.0

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "aspect_ratio": width / height,
    }


def calculate_webcam_crop(
    src_w: int,
    src_h: int,
    cam_bbox: Tuple[float, float, float, float] = DEFAULT_WEBCAM_BBOX,
) -> Tuple[int, int, int, int]:
    """
    Calculates pixel-accurate crop box for the webcam slot (1080x640, AR 1.6875:1).
    Adjusts user-provided normalized bounding box (x, y, w, h) so that the crop box
    matches the 1.6875 aspect ratio prior to scaling, eliminating facial distortion.

    Returns:
        (crop_w, crop_h, crop_x, crop_y) in source pixel coordinates.
    """
    cam_x, cam_y, cam_w, cam_h = cam_bbox

    # Convert normalized bounds to pixel units
    raw_w = cam_w * src_w
    raw_h = cam_h * src_h
    raw_cx = (cam_x + cam_w / 2.0) * src_w
    raw_cy = (cam_y + cam_h / 2.0) * src_h

    # Target aspect ratio for top slot (1080 / 640 = 1.6875)
    target_ar = SPLIT_TOP_AR
    current_ar = raw_w / max(raw_h, 1.0)

    if current_ar > target_ar:
        # User box is wider than 1.6875 -> crop excess width to fit
        crop_h = raw_h
        crop_w = raw_h * target_ar
    else:
        # User box is taller than 1.6875 -> crop excess height to fit
        crop_w = raw_w
        crop_h = raw_w / target_ar

    # Position centered on the user's defined webcam box
    crop_x = raw_cx - (crop_w / 2.0)
    crop_y = raw_cy - (crop_h / 2.0)

    # Clamping within source boundaries
    crop_w = min(crop_w, float(src_w))
    crop_h = min(crop_h, float(src_h))
    crop_x = max(0.0, min(crop_x, float(src_w) - crop_w))
    crop_y = max(0.0, min(crop_y, float(src_h) - crop_h))

    # Round to even integers for YUV420p encoder compatibility
    w_even = int(round(crop_w / 2.0) * 2)
    h_even = int(round(crop_h / 2.0) * 2)
    x_even = int(round(crop_x / 2.0) * 2)
    y_even = int(round(crop_y / 2.0) * 2)

    # Ensure bounds safety after rounding
    w_even = max(2, min(w_even, src_w))
    h_even = max(2, min(h_even, src_h))
    x_even = max(0, min(x_even, src_w - w_even))
    y_even = max(0, min(y_even, src_h - h_even))

    return w_even, h_even, x_even, y_even


def calculate_gameplay_crop(src_w: int, src_h: int) -> Tuple[int, int, int, int]:
    """
    Calculates center crop for the bottom gameplay slot (1080x1280, AR 0.84375:1).
    Crops central gameplay action to match 0.84375:1 prior to scaling.

    Returns:
        (crop_w, crop_h, crop_x, crop_y) in source pixel coordinates.
    """
    target_ar = SPLIT_BOTTOM_AR
    src_ar = src_w / src_h

    if src_ar > target_ar:
        # Widescreen input (e.g. 16:9, 1.777 > 0.84375) -> crop width, keep full height
        crop_h = float(src_h)
        crop_w = src_h * target_ar
        crop_x = (src_w - crop_w) / 2.0
        crop_y = 0.0
    else:
        # Narrow input -> crop height, keep full width
        crop_w = float(src_w)
        crop_h = src_w / target_ar
        crop_x = 0.0
        crop_y = (src_h - crop_h) / 2.0

    w_even = int(round(crop_w / 2.0) * 2)
    h_even = int(round(crop_h / 2.0) * 2)
    x_even = int(round(crop_x / 2.0) * 2)
    y_even = int(round(crop_y / 2.0) * 2)

    w_even = max(2, min(w_even, src_w))
    h_even = max(2, min(h_even, src_h))
    x_even = max(0, min(x_even, src_w - w_even))
    y_even = max(0, min(y_even, src_h - h_even))

    return w_even, h_even, x_even, y_even


def escape_ffmpeg_path(path: Union[Path, str]) -> str:
    """
    Converts a file path to POSIX style and escapes Windows drive colons
    and special characters for FFmpeg filtergraph strings (e.g. libass).
    E:\\path\\file.ass -> E\\:/path/file.ass
    """
    posix_path = str(Path(path).resolve()).replace("\\", "/")
    if len(posix_path) >= 2 and posix_path[1] == ":":
        posix_path = posix_path[0] + "\\:" + posix_path[2:]
    posix_path = posix_path.replace("'", "'\\''")
    return posix_path


def build_filtergraph(
    layout: LayoutMode,
    src_w: int,
    src_h: int,
    cam_bbox: Tuple[float, float, float, float] = DEFAULT_WEBCAM_BBOX,
    ass_path: Optional[Path] = None,
) -> str:
    """
    Constructs the FFmpeg complex filtergraph for the selected layout mode.
    Guarantees square pixel SAR 1:1 and burns .ass subtitles if provided.
    """
    if layout == LayoutMode.CENTER_CROP:
        # Standard 9:16 center crop
        fg = (
            f"[0:v]crop=ih*9/16:ih:(iw-ih*9/16)/2:0,"
            f"scale={CANVAS_WIDTH}:{CANVAS_HEIGHT},"
            f"setsar=1"
        )
        if ass_path:
            fg += f",ass='{escape_ffmpeg_path(ass_path)}'"
        return fg + "[v]"

    elif layout == LayoutMode.GAMING_SPLIT:
        # Top webcam slot: 1080x640
        top_w, top_h, top_x, top_y = calculate_webcam_crop(src_w, src_h, cam_bbox)
        # Bottom gameplay slot: 1080x1280
        bot_w, bot_h, bot_x, bot_y = calculate_gameplay_crop(src_w, src_h)

        fg = (
            f"[0:v]crop={top_w}:{top_h}:{top_x}:{top_y},"
            f"scale={SPLIT_TOP_WIDTH}:{SPLIT_TOP_HEIGHT},setsar=1[top];"
            f"[0:v]crop={bot_w}:{bot_h}:{bot_x}:{bot_y},"
            f"scale={SPLIT_BOTTOM_WIDTH}:{SPLIT_BOTTOM_HEIGHT},setsar=1[bottom];"
            f"[top][bottom]vstack=inputs=2,setsar=1"
        )
        if ass_path:
            fg += f",ass='{escape_ffmpeg_path(ass_path)}'"
        return fg + "[v]"

    else:
        raise ValueError(f"Unsupported layout mode: {layout}")


def compose_clip(
    input_video: Path,
    output_path: Optional[Path] = None,
    layout: Union[LayoutMode, str] = LayoutMode.GAMING_SPLIT,
    cam_bbox: Tuple[float, float, float, float] = DEFAULT_WEBCAM_BBOX,
    ass_path: Optional[Path] = None,
    nvenc_preset: str = NVENC_PRESET,
    nvenc_cq: int = NVENC_CQ,
) -> Path:
    """
    Composites a 9:16 vertical video clip using NVIDIA NVENC hardware acceleration
    with optional single-pass burning of .ass karaoke subtitles.

    Args:
        input_video: Path to the raw downloaded video section.
        output_path: Desired output path in outputs/ (auto-generated if None).
        layout: "center_crop" or "gaming_split".
        cam_bbox: Normalized (x, y, w, h) for webcam region in gaming_split.
        ass_path: Optional path to .ass subtitle script to burn in single pass.
        nvenc_preset: NVENC preset (e.g. "p4").
        nvenc_cq: Constant Quality target (e.g. 23).

    Returns:
        Path to the finished 1080x1920 MP4 file.
    """
    input_path = Path(input_video).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if isinstance(layout, str):
        layout = LayoutMode(layout.lower())

    if output_path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_DIR / f"{input_path.stem}_{layout.value}_1080x1920.mp4"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe input dimensions
    probe_info = probe_video(input_path)
    src_w = probe_info["width"]
    src_h = probe_info["height"]

    filtergraph = build_filtergraph(layout, src_w, src_h, cam_bbox, ass_path=ass_path)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        filtergraph,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        NVENC_CODEC,
        "-preset",
        nvenc_preset,
        "-cq",
        str(nvenc_cq),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        AUDIO_CODEC,
        "-b:a",
        AUDIO_BITRATE,
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg compositing failed (exit code {result.returncode}):\n{result.stderr}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Compositing completed but output file is empty or missing: {output_path}")

    return output_path


# Alias for render_clip contract
render_clip = compose_clip

