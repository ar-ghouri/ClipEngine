"""
ClipEngine Transcriber Module (Stage 2)
GPU-Accelerated Word-Level Audio Transcription using faster-whisper on NVIDIA RTX 4060.
Features:
- Dynamic CUDA 12 / cuDNN 9 DLL registration on Windows.
- High-performance batched transcription with word-level alignments.
- Automatic disk caching (.json) to prevent duplicate transcribing.
- Lightweight FFmpeg + NumPy audio RMS loudness peak detection.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from config import (
    FFMPEG_BIN,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
    WHISPER_VAD_FILTER,
    register_cuda_dlls,
)

logger = logging.getLogger("clipengine.transcriber")

# Register CUDA 12 / cuDNN DLLs before importing faster-whisper / ctranslate2
register_cuda_dlls()

# Lazy import placeholder
_WhisperModel = None


def _get_whisper_model_cls():
    """Lazily imports WhisperModel after CUDA DLLs have been registered."""
    global _WhisperModel
    if _WhisperModel is None:
        from faster_whisper import WhisperModel

        _WhisperModel = WhisperModel
    return _WhisperModel


# In-memory model cache to avoid reloading weights across repeated calls
_MODEL_CACHE: Dict[str, Any] = {}


class WordTimestamp(BaseModel):
    """Word-level alignment timestamp."""

    word: str
    start: float
    end: float
    probability: float = 1.0


class TranscriptSegment(BaseModel):
    """Segment-level transcription containing word-level alignments."""

    id: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """Complete transcription result from faster-whisper."""

    audio_path: str
    language: str = "en"
    duration: float
    segments: List[TranscriptSegment] = Field(default_factory=list)


class AudioPeak(BaseModel):
    """Timestamp of a loud audio spike (screams, laughter, hype moments)."""

    timestamp: float
    formatted_time: str
    rms_db: float


def compute_audio_loudness_peaks(
    audio_path: Path,
    top_k: int = 10,
    window_sec: float = 1.0,
    min_distance_sec: float = 4.0,
) -> List[AudioPeak]:
    """
    Computes 1-second audio RMS loudness peaks using FFmpeg and NumPy.
    Extracts high-energy spikes (laughter, yelling, hype) without heavy librosa dependencies.

    Args:
        audio_path: Path to audio or video media file.
        top_k: Number of loudest distinct timestamp spikes to return.
        window_sec: Window length in seconds for RMS calculation.
        min_distance_sec: Minimum spacing in seconds between detected peaks to prevent clustering.

    Returns:
        List of AudioPeak objects sorted in chronological order.
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    sample_rate = 16000
    cmd = [
        FFMPEG_BIN,
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        "pipe:1",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_pcm, stderr_data = proc.communicate()
    if proc.returncode != 0:
        logger.warning(
            f"FFmpeg PCM stream reading failed (code {proc.returncode}): {stderr_data.decode(errors='ignore')}"
        )
        return []

    if not raw_pcm:
        return []

    samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
    total_samples = len(samples)
    window_size = int(sample_rate * window_sec)
    if window_size <= 0 or total_samples < window_size:
        return []

    num_windows = total_samples // window_size
    trimmed_samples = samples[: num_windows * window_size].reshape(num_windows, window_size)

    # Compute RMS per 1-second block
    rms_values = np.sqrt(np.mean(trimmed_samples**2, axis=1))

    # Convert to dBFS (relative to full scale int16 32768)
    rms_db = 20.0 * np.log10(np.maximum(rms_values, 1e-5) / 32768.0)

    # Sort candidates by loudness (descending)
    sorted_indices = np.argsort(rms_db)[::-1]

    selected_peaks: List[Tuple[float, float]] = []
    for idx in sorted_indices:
        t = float(idx * window_sec)
        # Check distance from previously selected peaks
        if any(abs(t - prev_t) < min_distance_sec for prev_t, _ in selected_peaks):
            continue
        selected_peaks.append((t, float(rms_db[idx])))
        if len(selected_peaks) >= top_k:
            break

    # Sort selected peaks chronologically
    selected_peaks.sort(key=lambda x: x[0])

    peaks: List[AudioPeak] = []
    for t, db in selected_peaks:
        mins = int(t // 60)
        secs = int(t % 60)
        formatted = f"{mins:02d}:{secs:02d}"
        peaks.append(AudioPeak(timestamp=round(t, 2), formatted_time=formatted, rms_db=round(db, 2)))

    return peaks


def load_whisper_model(
    model_size: str = WHISPER_MODEL,
    device: str = WHISPER_DEVICE,
    compute_type: str = WHISPER_COMPUTE_TYPE,
) -> Any:
    """
    Initializes and caches the faster-whisper WhisperModel instance on GPU.
    """
    cache_key = f"{model_size}_{device}_{compute_type}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    WhisperModel = _get_whisper_model_cls()
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        logger.warning(
            f"Failed to load Whisper on {device} ({compute_type}): {exc}. Falling back to CPU/int8."
        )
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

    _MODEL_CACHE[cache_key] = model
    return model


def transcribe_audio(
    audio_path: Path,
    model_size: str = WHISPER_MODEL,
    device: str = WHISPER_DEVICE,
    compute_type: str = WHISPER_COMPUTE_TYPE,
    language: Optional[str] = None,
    word_timestamps: bool = True,
    vad_filter: bool = WHISPER_VAD_FILTER,
    force_recompute: bool = False,
    video_id: Optional[str] = None,
    cache_path: Optional[Path] = None,
) -> TranscriptionResult:
    """
    Transcribes audio into timestamped segments and aligned words using faster-whisper.
    Automatically checks and caches results as .json to avoid duplicate processing.
    Isolates cache strictly by video_id: <video_id>_whisper.transcript.json.

    Args:
        audio_path: Path to .mp3, .wav, or extracted audio.
        model_size: Whisper model variant.
        device: 'cuda' for RTX 4060 or 'cpu'.
        compute_type: 'float16' for GPU tensor cores or 'int8'.
        language: Language code (e.g. 'en') or None for auto-detection.
        word_timestamps: Extract word-level timings for karaoke subtitles.
        vad_filter: Enable Voice Activity Detection filter.
        force_recompute: Bypass disk cache if True.
        video_id: Optional YouTube video ID for strict cache isolation.
        cache_path: Optional explicit path for the transcript cache JSON.

    Returns:
        TranscriptionResult object.
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    # Determine cache path strictly by video_id if provided
    if cache_path is None:
        if video_id:
            cache_path = audio_path.parent / f"{video_id}_whisper.transcript.json"
        else:
            cache_path = audio_path.with_suffix(".transcript.json")

    # Check disk cache
    if not force_recompute and cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_res = TranscriptionResult.model_validate(cached_data)
            logger.info("Loaded transcript cache from %s", cache_path)
            from rich.progress import BarColumn, Progress, TextColumn

            tot_m = int(cached_res.duration // 60)
            tot_s = int(cached_res.duration % 60)
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=25),
                transient=False,
            ) as progress:
                desc = (
                    f"Transcribing on RTX 4060: 100% • "
                    f"{tot_m:02d}:{tot_s:02d} / {tot_m:02d}:{tot_s:02d} • speed (cached)"
                )
                progress.add_task(desc, total=int(cached_res.duration), completed=int(cached_res.duration))
            return cached_res
        except Exception as exc:
            logger.warning(f"Failed to load transcript cache from {cache_path}: {exc}. Recomputing...")

    # Load model and transcribe
    model = load_whisper_model(model_size=model_size, device=device, compute_type=compute_type)

    raw_segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
    )

    import time
    from rich.progress import BarColumn, Progress, TextColumn

    total_duration = round(info.duration, 2) if hasattr(info, "duration") and info.duration else 0.0
    start_wall_time = time.time()

    segments: List[TranscriptSegment] = []
    seg_id = 1

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=25),
        transient=False,
    ) as progress:
        tot_m = int(total_duration // 60)
        tot_s = int(total_duration % 60)
        task_id = progress.add_task(
            f"Transcribing on RTX 4060:   0% • 00:00 / {tot_m:02d}:{tot_s:02d} • speed 0.0x",
            total=total_duration if total_duration > 0 else 100.0,
        )

        for seg in raw_segments:
            words: List[WordTimestamp] = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        WordTimestamp(
                            word=w.word.strip(),
                            start=round(w.start, 3),
                            end=round(w.end, 3),
                            probability=round(w.probability, 3) if hasattr(w, "probability") else 1.0,
                        )
                    )

            segments.append(
                TranscriptSegment(
                    id=seg_id,
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    text=seg.text.strip(),
                    words=words,
                )
            )
            seg_id += 1

            # Update progress tracking
            curr_t = min(total_duration, seg.end) if total_duration > 0 else seg.end
            elapsed = max(0.001, time.time() - start_wall_time)
            speed = curr_t / elapsed
            pct = (curr_t / total_duration * 100.0) if total_duration > 0 else 0.0
            cur_m = int(curr_t // 60)
            cur_s = int(curr_t % 60)
            desc = (
                f"Transcribing on RTX 4060: {pct:2.0f}% • "
                f"{cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d} • speed {speed:.1f}x"
            )
            progress.update(task_id, completed=curr_t, description=desc)

        if total_duration > 0:
            elapsed = max(0.001, time.time() - start_wall_time)
            speed = total_duration / elapsed
            desc = (
                f"Transcribing on RTX 4060: 100% • "
                f"{tot_m:02d}:{tot_s:02d} / {tot_m:02d}:{tot_s:02d} • speed {speed:.1f}x"
            )
            progress.update(task_id, completed=total_duration, description=desc)

    result = TranscriptionResult(
        audio_path=str(audio_path),
        language=info.language if hasattr(info, "language") else "en",
        duration=round(info.duration, 2) if hasattr(info, "duration") else 0.0,
        segments=segments,
    )

    # Persist cache to disk
    export_transcript_json(result, cache_path)
    return result


def export_transcript_json(result: TranscriptionResult, output_path: Path) -> Path:
    """Serializes transcription results to JSON format."""
    output_path = Path(output_path)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return output_path
