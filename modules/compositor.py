"""
ClipEngine Compositor Module
NVIDIA GPU-accelerated video compositing engine using h264_nvenc.
Produces 1080x1920 (9:16) vertical MP4 clips for TikTok, Shorts, and Reels
with zero aspect ratio distortion.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2

from config import (
    AUDIO_BITRATE,
    AUDIO_CODEC,
    BGM_DIR,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEFAULT_WEBCAM_BBOX,
    FFMPEG_BIN,
    FFPROBE_BIN,
    NVENC_BITRATE,
    NVENC_CODEC,
    NVENC_CQ,
    NVENC_MAXRATE,
    NVENC_PIX_FMT,
    NVENC_PRESET,
    NVENC_RC,
    NVENC_TUNE,
    OUTPUTS_DIR,
    SPLIT_BOTTOM_AR,
    SPLIT_BOTTOM_HEIGHT,
    SPLIT_BOTTOM_WIDTH,
    SPLIT_TOP_AR,
    SPLIT_TOP_HEIGHT,
    SPLIT_TOP_WIDTH,
)

logger = logging.getLogger("clipengine.compositor")

# High-fidelity anti-haze sharpening & micro-contrast filter
ANTI_HAZE_FILTER: str = (
    "cas=0.6,unsharp=5:5:0.75:3:3:0.3,eq=contrast=1.05:saturation=1.10"
)


def ensure_default_bgm_tracks(bgm_dir: Path = BGM_DIR) -> Dict[str, Path]:
    """
    Ensures that default procedural background music loops exist in assets/bgm/.
    Synthesizes 3 lightweight, royalty-free ambient/beat loops via FFmpeg lavfi
    (hype_action.mp3, dramatic_tension.mp3, insightful_story.mp3, plus funny_reaction.mp3)
    if the user has not placed custom .mp3 tracks yet.
    """
    bgm_dir.mkdir(parents=True, exist_ok=True)

    presets: Dict[str, Tuple[str, str]] = {
        "hype_action": (
            "hype_action.mp3",
            "sine=frequency=130:duration=30,tremolo=f=4.5:d=0.7,flanger=delay=5:speed=2,volume=0.25",
        ),
        "dramatic_tension": (
            "dramatic_tension.mp3",
            "sine=frequency=65:duration=30,tremolo=f=1.2:d=0.5,lowpass=f=250,volume=0.25",
        ),
        "insightful_story": (
            "insightful_story.mp3",
            "sine=frequency=220:duration=30,tremolo=f=0.8:d=0.4,aecho=0.8:0.88:60:0.4,lowpass=f=800,volume=0.20",
        ),
        "funny_reaction": (
            "funny_reaction.mp3",
            "sine=frequency=330:duration=30,tremolo=f=6.0:d=0.9,highpass=f=200,volume=0.20",
        ),
    }

    result: Dict[str, Path] = {}
    for vibe, (filename, filter_expr) in presets.items():
        track_path = bgm_dir / filename
        if not track_path.exists() or track_path.stat().st_size == 0:
            logger.info("Synthesizing procedural BGM track for vibe '%s': %s", vibe, track_path.name)
            cmd = [
                FFMPEG_BIN,
                "-y",
                "-f",
                "lavfi",
                "-i",
                filter_expr,
                "-c:a",
                "mp3",
                "-b:a",
                "128k",
                str(track_path),
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except Exception as exc:
                logger.warning("Failed to synthesize BGM track %s: %s", track_path, exc)
        if track_path.exists():
            result[vibe] = track_path

    return result


def resolve_bgm_track(vibe: Optional[str] = None, bgm_dir: Path = BGM_DIR) -> Optional[Path]:
    """
    Finds the best matching BGM track for the given vibe in assets/bgm/.
    Falls back to hype_action or any existing .mp3 file in assets/bgm/.
    """
    ensure_default_bgm_tracks(bgm_dir)
    if vibe:
        vibe_key = vibe.lower()
        candidate = bgm_dir / f"{vibe_key}.mp3"
        if candidate.exists():
            return candidate

    # Fallback to hype_action
    default_track = bgm_dir / "hype_action.mp3"
    if default_track.exists():
        return default_track

    # Any mp3 in folder
    any_mp3 = list(bgm_dir.glob("*.mp3"))
    if any_mp3:
        return any_mp3[0]

    return None


class LayoutMode(str, Enum):
    CENTER_CROP = "center_crop"
    GAMING_SPLIT = "gaming_split"


def probe_video(video_path: Path) -> Dict[str, Any]:
    """
    Inspects input video streams using ffprobe to obtain exact geometry, duration,
    and presence of audio tracks.
    """
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,duration",
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
        raise ValueError(f"No media streams detected in {video_path}")

    video_stream: Optional[Dict[str, Any]] = None
    has_audio: bool = False
    for s in streams:
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio":
            has_audio = True

    if video_stream is None:
        raise ValueError(f"No video streams detected in {video_path}")

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    duration_str = video_stream.get("duration") or data.get("format", {}).get("duration", "0")
    try:
        duration = float(duration_str)
    except (ValueError, TypeError):
        duration = 0.0

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "aspect_ratio": width / height,
        "has_audio": has_audio,
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
    speaker_x: float = 0.5,
    ass_path: Optional[Path] = None,
    climax_offset: Optional[float] = None,
    use_dramatic_punch: bool = False,
) -> str:
    """
    Constructs the FFmpeg complex filtergraph for the selected layout mode.
    Guarantees square pixel SAR 1:1, Lanczos scaling, CAS 0.6 anti-haze sharpening,
    speaker-centered dynamic cropping for center_crop, optional Director FX climax zoom-punch & white-flash pop,
    and burns .ass subtitles if provided.
    """
    t_climax = max(0.0, float(climax_offset)) if climax_offset is not None else 0.0
    t_flash_end = t_climax + 0.15
    t_zoom_end = t_climax + 2.5

    if layout == LayoutMode.CENTER_CROP:
        # Standard 9:16 speaker-centered crop with Lanczos anti-haze upscaling & sharpening
        crop_expr = f"crop=ih*9/16:ih:min(max(0\\,iw*{speaker_x:.4f}-(ih*9/16)/2)\\,iw-ih*9/16):0"
        if use_dramatic_punch and climax_offset is not None:
            fg = (
                f"[0:v]{crop_expr},"
                f"scale={CANVAS_WIDTH}:{CANVAS_HEIGHT}:flags=lanczos,"
                f"{ANTI_HAZE_FILTER},setsar=1[main_base];"
                f"[main_base]split[main_pre][zsrc];"
                f"[zsrc]crop=iw/1.12:ih/1.12:(iw-iw/1.12)/2:(ih-ih/1.12)/2,"
                f"scale={CANVAS_WIDTH}:{CANVAS_HEIGHT}:flags=lanczos[zoomed];"
                f"[main_pre][zoomed]overlay=0:0:enable='between(t,{t_climax:.3f},{t_zoom_end:.3f})',"
                f"eq=brightness=0.18:enable='between(t,{t_climax:.3f},{t_flash_end:.3f})'"
            )
        else:
            fg = (
                f"[0:v]{crop_expr},"
                f"scale={CANVAS_WIDTH}:{CANVAS_HEIGHT}:flags=lanczos,"
                f"{ANTI_HAZE_FILTER},setsar=1"
            )

        if ass_path:
            fg += f",ass='{escape_ffmpeg_path(ass_path)}'"
        return fg + "[v]"

    elif layout == LayoutMode.GAMING_SPLIT:
        # Top webcam slot: 1080x640 with Lanczos + CAS 0.6 anti-haze sharpening
        top_w, top_h, top_x, top_y = calculate_webcam_crop(src_w, src_h, cam_bbox)
        # Bottom gameplay slot: 1080x1280
        bot_w, bot_h, bot_x, bot_y = calculate_gameplay_crop(src_w, src_h)

        if use_dramatic_punch and climax_offset is not None:
            # Apply 1.12x center zoom-in and 0.15s white-flash brightness pop to the gameplay layer
            fg = (
                f"[0:v]crop={top_w}:{top_h}:{top_x}:{top_y},"
                f"scale={SPLIT_TOP_WIDTH}:{SPLIT_TOP_HEIGHT}:flags=lanczos,"
                f"{ANTI_HAZE_FILTER},setsar=1[top];"
                f"[0:v]crop={bot_w}:{bot_h}:{bot_x}:{bot_y},"
                f"scale={SPLIT_BOTTOM_WIDTH}:{SPLIT_BOTTOM_HEIGHT}:flags=lanczos,"
                f"{ANTI_HAZE_FILTER},setsar=1[bot_base];"
                f"[bot_base]split[bot_main][bot_zsrc];"
                f"[bot_zsrc]crop=iw/1.12:ih/1.12:(iw-iw/1.12)/2:(ih-ih/1.12)/2,"
                f"scale={SPLIT_BOTTOM_WIDTH}:{SPLIT_BOTTOM_HEIGHT}:flags=lanczos[bot_zoomed];"
                f"[bot_main][bot_zoomed]overlay=0:0:enable='between(t,{t_climax:.3f},{t_zoom_end:.3f})',"
                f"eq=brightness=0.18:enable='between(t,{t_climax:.3f},{t_flash_end:.3f})'[bottom];"
                f"[top][bottom]vstack=inputs=2,setsar=1"
            )
        else:
            fg = (
                f"[0:v]crop={top_w}:{top_h}:{top_x}:{top_y},"
                f"scale={SPLIT_TOP_WIDTH}:{SPLIT_TOP_HEIGHT}:flags=lanczos,"
                f"{ANTI_HAZE_FILTER},setsar=1[top];"
                f"[0:v]crop={bot_w}:{bot_h}:{bot_x}:{bot_y},"
                f"scale={SPLIT_BOTTOM_WIDTH}:{SPLIT_BOTTOM_HEIGHT}:flags=lanczos,"
                f"{ANTI_HAZE_FILTER},setsar=1[bottom];"
                f"[top][bottom]vstack=inputs=2,setsar=1"
            )

        if ass_path:
            fg += f",ass='{escape_ffmpeg_path(ass_path)}'"
        return fg + "[v]"

    else:
        raise ValueError(f"Unsupported layout mode: {layout}")


def detect_facecam(
    video_path: Path,
    min_face_size: int = 60,
    min_neighbors: int = 5,
    vod_key: Optional[str] = None,
) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
    """
    Extracts sample frames (at 20%, 40%, 60%, 80% of the clip duration) and runs OpenCV
    haarcascade_frontalface_default.xml detector.

    Distinguishes corner gaming webcams from full-screen studio/couch shots:
    - gaming_split is ONLY used when a consistent face belongs to a corner gaming webcam overlay
      (average face width < 0.16 of frame width AND face horizontal center x < 0.28 or x > 0.72).
    - When face is larger (>= 0.16 width) or centered in a full-screen IRL/podcast/studio shot
      (e.g., sitting on a couch), falls back to dynamic center_crop centering on speaker_x_norm.
    - If Haar Cascade misses due to beard/hoodie/lighting, automatically queries Gemini Vision
      with per-VOD caching to lock onto streamer facecams reliably.

    Returns:
        (cam_bbox, speaker_x_norm) where cam_bbox is (x, y, w, h) or None.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        return None, 0.5

    haarcascades_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if haarcascades_dir:
        cascade_file = Path(haarcascades_dir) / "haarcascade_frontalface_default.xml"
    else:
        cascade_file = Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
    if not cascade_file.exists():
        logger.warning("Haar cascade XML not found at %s", cascade_file)
        return None, 0.5

    cascade = cv2.CascadeClassifier(str(cascade_file))
    if cascade.empty():
        logger.warning("Failed to load Haar cascade classifier")
        return None, 0.5

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("cv2.VideoCapture could not open %s", video_path)
        return None, 0.5

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0 or src_w <= 0 or src_h <= 0:
        cap.release()
        return None, 0.5

    # Sample 4 frames across clip duration: 20%, 40%, 60%, 80%
    sample_pcts = [0.20, 0.40, 0.60, 0.80]
    sample_indices = [max(0, min(total_frames - 1, int(total_frames * p))) for p in sample_pcts]

    effective_min_size = max(min_face_size, int(src_h * 0.05))
    # List of detected faces per frame: (fx, fy, fw, fh, cx, cy, norm_w, norm_h)
    frame_face_lists: List[List[Tuple[int, int, int, int, float, float, float, float]]] = []

    for f_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=min_neighbors,
            minSize=(effective_min_size, effective_min_size),
        )

        detected_faces: List[Tuple[int, int, int, int, float, float, float, float]] = []
        if len(faces) > 0:
            for (fx, fy, fw, fh) in faces:
                cx = (float(fx) + float(fw) / 2.0) / src_w
                cy = (float(fy) + float(fh) / 2.0) / src_h
                norm_w = float(fw) / src_w
                norm_h = float(fh) / src_h
                detected_faces.append((int(fx), int(fy), int(fw), int(fh), cx, cy, norm_w, norm_h))

        frame_face_lists.append(detected_faces)

    cap.release()

    # Spatial clustering across sampled frames
    clusters: List[List[Tuple[int, int, int, int, float, float, float, float]]] = []
    for face_list in frame_face_lists:
        for face in face_list:
            matched = False
            for cluster in clusters:
                avg_cx = sum(item[4] for item in cluster) / len(cluster)
                avg_cy = sum(item[5] for item in cluster) / len(cluster)
                dist = math.hypot(face[4] - avg_cx, face[5] - avg_cy)
                if dist < 0.15:  # Within 15% normalized distance
                    cluster.append(face)
                    matched = True
                    break
            if not matched:
                clusters.append([face])

    # If no faces were detected in any sampled frame, default speaker_x to 0.5 (dead center)
    if not clusters:
        return None, 0.5

    # Determine primary speaker horizontal center X from the largest face cluster
    primary_cluster = max(clusters, key=lambda c: sum(item[2] * item[3] for item in c))
    speaker_x_norm = float(sum(item[4] for item in primary_cluster) / len(primary_cluster))
    speaker_x_norm = max(0.0, min(1.0, round(speaker_x_norm, 4)))

    # Evaluate clusters for corner gaming webcam overlay:
    # Must appear consistently across >= 2 frames (or 1 if only 1 frame could be read)
    min_required_frames = 2 if len(frame_face_lists) >= 2 else 1
    consistent_clusters = [c for c in clusters if len(c) >= min_required_frames]

    corner_cam_clusters = []
    for c in consistent_clusters:
        avg_norm_w = sum(item[6] for item in c) / len(c)
        avg_cx = sum(item[4] for item in c) / len(c)
        avg_cy = sum(item[5] for item in c) / len(c)
        # Corner gaming webcam overlay condition:
        # Small overlay (average face width < 0.16) situated in an outer corner (x < 0.28 or x > 0.72, and y < 0.38 or y > 0.62).
        # Full-screen studio/couch shots (sitting at y ~ 0.50 or larger faces >= 0.16 or centered x) switch to center_crop.
        if avg_norm_w < 0.16 and (avg_cx < 0.28 or avg_cx > 0.72) and (avg_cy < 0.38 or avg_cy > 0.62):
            corner_cam_clusters.append(c)

    # If no consistent cluster satisfies corner gaming webcam, try Gemini Vision fallback
    if not corner_cam_clusters:
        try:
            from modules.brain import detect_webcam_gemini_vision

            has_vision_cam, vision_bbox = detect_webcam_gemini_vision(video_path, vod_key=vod_key)
            if has_vision_cam and vision_bbox is not None:
                logger.info("Gemini Vision detected corner webcam overlay at %s", vision_bbox)
                return vision_bbox, speaker_x_norm
        except Exception as exc:
            logger.warning("Gemini Vision fallback detection failed: %s", exc)

        return None, speaker_x_norm

    # Select the best corner webcam cluster
    best_cluster = max(corner_cam_clusters, key=lambda c: sum(item[2] * item[3] for item in c))

    avg_fx = sum(item[0] for item in best_cluster) / len(best_cluster)
    avg_fy = sum(item[1] for item in best_cluster) / len(best_cluster)
    avg_fw = sum(item[2] for item in best_cluster) / len(best_cluster)
    avg_fh = sum(item[3] for item in best_cluster) / len(best_cluster)

    # Pad generously around head and shoulders with ample headroom to prevent clipping
    headroom = avg_fh * 0.70
    chest = avg_fh * 1.10
    crop_h = avg_fh + headroom + chest
    crop_w = crop_h * SPLIT_TOP_AR

    cx_face = avg_fx + (avg_fw / 2.0)
    box_x = cx_face - (crop_w / 2.0)
    box_y = avg_fy - headroom

    # Boundary adjustments maintaining SPLIT_TOP_AR
    if box_y + crop_h > src_h:
        box_y = max(0.0, float(src_h) - crop_h)
    if box_y < 0:
        box_y = 0.0

    if box_x + crop_w > src_w:
        box_x = max(0.0, float(src_w) - crop_w)
    if box_x < 0:
        box_x = 0.0

    cam_x = round(box_x / src_w, 4)
    cam_y = round(box_y / src_h, 4)
    cam_w = round(crop_w / src_w, 4)
    cam_h = round(crop_h / src_h, 4)

    return (cam_x, cam_y, cam_w, cam_h), speaker_x_norm


def compose_clip(
    input_video: Path,
    output_path: Optional[Path] = None,
    layout: Union[LayoutMode, str] = LayoutMode.GAMING_SPLIT,
    cam_bbox: Tuple[float, float, float, float] = DEFAULT_WEBCAM_BBOX,
    speaker_x: float = 0.5,
    ass_path: Optional[Path] = None,
    auto_detect_cam: bool = True,
    vibe: Optional[str] = None,
    climax_offset: Optional[float] = None,
    use_dramatic_punch: bool = False,
    bgm_path: Optional[Path] = None,
    enable_bgm: bool = True,
    vod_key: Optional[str] = None,
    nvenc_preset: str = NVENC_PRESET,
    nvenc_cq: int = NVENC_CQ,
    nvenc_tune: str = NVENC_TUNE,
    nvenc_rc: str = NVENC_RC,
    nvenc_bitrate: str = NVENC_BITRATE,
    nvenc_maxrate: str = NVENC_MAXRATE,
    nvenc_pix_fmt: str = NVENC_PIX_FMT,
) -> Path:
    """
    Composites a 9:16 vertical video clip using NVIDIA NVENC hardware acceleration
    with optional single-pass burning of .ass karaoke subtitles, Director FX zoom-punch,
    and auto-ducked procedural background music.

    Args:
        input_video: Path to the raw downloaded video section.
        output_path: Desired output path in outputs/ (auto-generated if None).
        layout: "center_crop" or "gaming_split".
        cam_bbox: Normalized (x, y, w, h) for webcam region in gaming_split.
        speaker_x: Normalized horizontal center X (0.0 - 1.0) for center_crop speaker framing.
        ass_path: Optional path to .ass subtitle script to burn in single pass.
        auto_detect_cam: When True and cam_bbox is default, auto-detects webcam face and falls back to center_crop if faceless.
        vibe: Highlight tone for BGM selection ("hype_action", "funny_reaction", "dramatic_tension", "insightful_story").
        climax_offset: Relative seconds to apply Director FX zoom-punch.
        use_dramatic_punch: Enables 1.12x center zoom and brightness pop at climax_offset.
        bgm_path: Optional custom path to background music .mp3.
        enable_bgm: Whether to mix background music (default True).
        vod_key: Optional VOD identifier or URL for caching webcam detection across clips.
        nvenc_preset: NVENC preset (default "p6").
        nvenc_cq: Constant Quality target (default 18).
        nvenc_tune: NVENC tune profile (default "hq").
        nvenc_rc: Rate control mode (default "vbr").
        nvenc_bitrate: Target video bitrate (default "10M").
        nvenc_maxrate: Peak video bitrate (default "15M").
        nvenc_pix_fmt: Pixel format (default "yuv420p").

    Returns:
        Path to the finished 1080x1920 MP4 file.
    """
    input_path = Path(input_video).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if isinstance(layout, str):
        layout = LayoutMode(layout.lower())

    # Smart Auto-Webcam Detection & Fallback if gaming_split is requested with default cam_bbox
    if layout == LayoutMode.GAMING_SPLIT and auto_detect_cam and cam_bbox == DEFAULT_WEBCAM_BBOX:
        detected_cam, detected_speaker_x = detect_facecam(input_path, vod_key=vod_key)
        if detected_cam is not None:
            cam_bbox = detected_cam
            logger.info("Auto-detected corner webcam region at %s", cam_bbox)
        else:
            layout = LayoutMode.CENTER_CROP
            speaker_x = detected_speaker_x
            logger.info(
                "Full-screen studio/couch or faceless shot detected (speaker X: %.2f); "
                "auto-switching to speaker-centered center_crop layout.",
                speaker_x,
            )

    if output_path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_DIR / f"{input_path.stem}_{layout.value}_1080x1920.mp4"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe input dimensions and streams
    probe_info = probe_video(input_path)
    src_w = probe_info["width"]
    src_h = probe_info["height"]
    has_audio = probe_info.get("has_audio", False)

    # Build video filtergraph with anti-haze Lanczos upscaling & optional Director FX
    video_filtergraph = build_filtergraph(
        layout=layout,
        src_w=src_w,
        src_h=src_h,
        cam_bbox=cam_bbox,
        speaker_x=speaker_x,
        ass_path=ass_path,
        climax_offset=climax_offset,
        use_dramatic_punch=use_dramatic_punch,
    )

    # Resolve BGM track if enabled
    effective_bgm: Optional[Path] = None
    if enable_bgm:
        if bgm_path is not None and Path(bgm_path).exists():
            effective_bgm = Path(bgm_path).resolve()
        else:
            effective_bgm = resolve_bgm_track(vibe)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
    ]

    audio_maps: List[str] = []
    if effective_bgm is not None:
        cmd.extend(["-stream_loop", "-1", "-i", str(effective_bgm)])
        if has_audio:
            # Ducked BGM mixed under speech (-18 dB volume ducking + sidechain compressor)
            combined_filter = (
                f"{video_filtergraph};"
                f"[0:a]asplit[voice1][voice2];"
                f"[1:a]volume=0.12[bgm];"
                f"[bgm][voice1]sidechaincompress=threshold=0.08:ratio=4:attack=20:release=300[ducked];"
                f"[voice2][ducked]amix=inputs=2:duration=first:dropout_transition=2[a]"
            )
            audio_maps = ["-map", "[a]"]
        else:
            combined_filter = f"{video_filtergraph};[1:a]volume=0.20[a]"
            audio_maps = ["-map", "[a]", "-shortest"]
    else:
        combined_filter = video_filtergraph
        audio_maps = ["-map", "0:a?"]

    cmd.extend([
        "-filter_complex",
        combined_filter,
        "-map",
        "[v]",
    ])
    cmd.extend(audio_maps)
    cmd.extend([
        "-c:v",
        NVENC_CODEC,
        "-preset",
        nvenc_preset,
        "-tune",
        nvenc_tune,
        "-rc",
        nvenc_rc,
        "-cq",
        str(nvenc_cq),
        "-b:v",
        nvenc_bitrate,
        "-maxrate",
        nvenc_maxrate,
        "-pix_fmt",
        nvenc_pix_fmt,
        "-c:a",
        AUDIO_CODEC,
        "-b:a",
        AUDIO_BITRATE,
        str(output_path),
    ])

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

