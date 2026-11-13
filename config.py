"""
ClipEngine Configuration Module
Defines project paths, GPU hardware acceleration parameters,
and layout specifications.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

# Base Directory & Project Roots
BASE_DIR: Path = Path(__file__).resolve().parent
DOWNLOADS_DIR: Path = BASE_DIR / "downloads"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
ASSETS_DIR: Path = BASE_DIR / "assets"
BGM_DIR: Path = ASSETS_DIR / "bgm"
DB_PATH: Path = BASE_DIR / "clipengine.db"

# Ensure runtime directories exist
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
BGM_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# Target Canvas Specifications (9:16 Vertical Video)
CANVAS_WIDTH: int = 1080
CANVAS_HEIGHT: int = 1920

# Gaming Split Layout Specifications
# 1080x640 top slot (webcam, AR: 1.6875 = 27/16)
# 1080x1280 bottom slot (gameplay, AR: 0.84375 = 27/32)
SPLIT_TOP_WIDTH: int = 1080
SPLIT_TOP_HEIGHT: int = 640
SPLIT_TOP_AR: float = SPLIT_TOP_WIDTH / SPLIT_TOP_HEIGHT  # 1.6875

SPLIT_BOTTOM_WIDTH: int = 1080
SPLIT_BOTTOM_HEIGHT: int = 1280
SPLIT_BOTTOM_AR: float = SPLIT_BOTTOM_WIDTH / SPLIT_BOTTOM_HEIGHT  # 0.84375

# Default layout mode ('gaming_split' or 'center_crop')
DEFAULT_LAYOUT: str = "gaming_split"

# Default webcam normalized bounding box [x, y, w, h] (top-left 25% quadrant)
DEFAULT_WEBCAM_BBOX: Tuple[float, float, float, float] = (0.0, 0.0, 0.25, 0.25)

# Hardware Encoding Settings (Optimized for NVIDIA RTX 4060 8GB VRAM)
NVENC_CODEC: str = "h264_nvenc"
NVENC_PRESET: str = "p6"       # High-quality slower preset (p6)
NVENC_TUNE: str = "hq"         # High-quality tune
NVENC_RC: str = "vbr"          # Variable bitrate with CQ target
NVENC_CQ: int = 18             # Razor-sharp constant quality level
NVENC_BITRATE: str = "10M"     # Target video bitrate
NVENC_MAXRATE: str = "15M"     # Maximum peak bitrate
NVENC_PIX_FMT: str = "yuv420p" # Broad compatibility pixel format
AUDIO_CODEC: str = "aac"
AUDIO_BITRATE: str = "192k"

# Stage 2: Whisper Transcription Settings
WHISPER_MODEL: str = "large-v3-turbo"   # Fast & high-accuracy on RTX 4060
WHISPER_DEVICE: str = "cuda"
WHISPER_COMPUTE_TYPE: str = "float16"
WHISPER_VAD_FILTER: bool = True

# Stage 3: Gemini Highlight Detection Settings
GEMINI_MODEL: str = "gemini-3.1-flash-lite"


def register_cuda_dlls() -> List[str]:
    """
    On Windows, Python 3.8+ does not load DLLs from PATH automatically.
    Dynamically locates nvidia/cublas/bin and nvidia/cudnn/bin inside site-packages
    and registers them via os.add_dll_directory() so CTranslate2 / faster-whisper
    can load cublas64_12.dll and cudnn64_9.dll on the RTX 4060.
    """
    registered: List[str] = []
    site_packages = [Path(p) for p in sys.path if "site-packages" in p]
    # Also check .venv/Lib/site-packages directly if sys.path doesn't list it yet
    venv_sp = BASE_DIR / ".venv" / "Lib" / "site-packages"
    if venv_sp.exists() and venv_sp not in site_packages:
        site_packages.append(venv_sp)

    for sp in site_packages:
        for pkg in ("cublas", "cudnn", "cuda_runtime"):
            bin_dir = sp / "nvidia" / pkg / "bin"
            if bin_dir.exists():
                try:
                    os.add_dll_directory(str(bin_dir))
                    registered.append(str(bin_dir))
                except Exception:
                    pass
                if str(bin_dir) not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"

    return registered


def resolve_ffmpeg_binaries() -> Tuple[str, str]:
    """
    Locates ffmpeg and ffprobe executables, checking standard PATH first
    and falling back to local WinGet / User paths if the shell environment
    lacks them. Adds the discovered directory to PATH for subprocess calls.
    """
    ffmpeg_exe = shutil.which("ffmpeg")
    ffprobe_exe = shutil.which("ffprobe")

    if ffmpeg_exe and ffprobe_exe:
        return ffmpeg_exe, ffprobe_exe

    # Fallback paths common on Windows / WinGet installations
    candidate_dirs = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    ]
    for candidate_root in candidate_dirs:
        if candidate_root.exists():
            for bin_dir in candidate_root.glob("**/bin"):
                f_candidate = bin_dir / "ffmpeg.exe"
                p_candidate = bin_dir / "ffprobe.exe"
                if f_candidate.exists():
                    os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"
                    return str(f_candidate), str(p_candidate if p_candidate.exists() else "ffprobe")

    return "ffmpeg", "ffprobe"


FFMPEG_BIN, FFPROBE_BIN = resolve_ffmpeg_binaries()
