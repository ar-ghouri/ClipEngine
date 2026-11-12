"""
ClipEngine Brain Module (Stage 3)
Gemini 2.5 Flash Viral Highlight Detection & Retention Scoring Engine.
Uses official google-genai SDK with Structured JSON Outputs.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, GEMINI_MODEL
from modules.transcriber import AudioPeak, TranscriptionResult

logger = logging.getLogger("clipengine.brain")


class ClipCandidate(BaseModel):
    """Represents an AI-selected highlight candidate with virality analysis."""

    hook_title: str = Field(
        ...,
        description="High-engagement, scroll-stopping title under 60 chars (curiosity-gap style)",
    )
    start_time: float = Field(
        ...,
        description="Clip start timestamp in seconds (aligned to natural sentence boundary)",
    )
    end_time: float = Field(
        ...,
        description="Clip end timestamp in seconds (aligned to natural sentence boundary)",
    )
    virality_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Predicted retention / virality score 0.0 to 10.0",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of retention driver (e.g. strong hook, climax, emotional payoff)",
    )
    suggested_hashtags: List[str] = Field(
        default_factory=list,
        description="Up to 3 high-relevance hashtags",
    )


class HighlightAnalysisResult(BaseModel):
    """Collection of detected clips evaluated by Gemini."""

    source_url: Optional[str] = None
    summary: str = Field(..., description="Brief synopsis of overall video content")
    candidates: List[ClipCandidate] = Field(
        default_factory=list,
        description="Top viral clip candidates (strictly 20 to 55 seconds each)",
    )


def init_gemini_client(api_key: Optional[str] = None) -> Any:
    """
    Initializes the Google GenAI SDK client using GEMINI_API_KEY.
    Uses official google-genai SDK (`from google import genai`).
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. Add GEMINI_API_KEY to .env or pass api_key parameter."
        )

    from google import genai

    return genai.Client(api_key=key)


def detect_highlights(
    transcript: TranscriptionResult,
    audio_peaks: Optional[List[AudioPeak]] = None,
    min_duration: float = 20.0,
    max_duration: float = 55.0,
    top_k: int = 3,
    api_key: Optional[str] = None,
    model_name: str = GEMINI_MODEL,
) -> HighlightAnalysisResult:
    """
    Evaluates transcript segments and audio loudness peaks using Gemini 2.5 Flash
    with structured schema output to extract viral short-form clips.

    Args:
        transcript: Word-timestamped transcription from Stage 2.
        audio_peaks: Top acoustic loudness spikes from Stage 2.
        min_duration: Minimum clip duration in seconds (strictly >= 20.0).
        max_duration: Maximum clip duration in seconds (strictly <= 55.0).
        top_k: Number of viral clips to select.
        api_key: Optional API key override.
        model_name: Gemini model name ('gemini-2.5-flash').

    Returns:
        HighlightAnalysisResult containing scored clip candidates and hook titles.
    """
    client = init_gemini_client(api_key)
    from google.genai import types

    # Format transcript lines with timestamps
    transcript_lines: List[str] = []
    for seg in transcript.segments:
        m_start = int(seg.start // 60)
        s_start = int(seg.start % 60)
        m_end = int(seg.end // 60)
        s_end = int(seg.end % 60)
        time_tag = f"[{m_start:02d}:{s_start:02d} -> {m_end:02d}:{s_end:02d}]"
        transcript_lines.append(f"{time_tag} {seg.text}")

    full_transcript_text = "\n".join(transcript_lines)

    # Format loudness energy peaks
    peak_lines: List[str] = []
    if audio_peaks:
        for peak in audio_peaks:
            peak_lines.append(f"- Timestamp: {peak.formatted_time} ({peak.timestamp:.1f}s), Loudness: {peak.rms_db:.1f} dBFS")
    peaks_text = "\n".join(peak_lines) if peak_lines else "None detected."

    total_duration_sec = transcript.duration if transcript.duration > 0 else 0.0

    prompt = f"""
You are an elite short-form video editor specializing in viral TikToks, YouTube Shorts, and Instagram Reels.
Your mission is to find the top {top_k} most engaging, high-retention moments from this stream/video.

### CRITICAL RULES:
1. VIDEO BOUNDS: Total video duration is {total_duration_sec:.1f} seconds. Both `start_time` and `end_time` MUST be between 0.0 and {total_duration_sec:.1f}. NEVER exceed {total_duration_sec:.1f}s!
2. DURATION: Each clip must be STRICTLY between {min_duration:.0f} and {max_duration:.0f} seconds long (duration = end_time - start_time).
3. BOUNDARIES: `start_time` and `end_time` MUST align with natural sentence boundaries from the transcript. NEVER cut mid-sentence or mid-thought.
4. AUDIO PEAKS: Use the acoustic loudness spikes below as clues. Streamer screams, laughter, gasps, and loud hype moments often make great climaxes for clips.
5. HOOK TITLE: Generate an intriguing, curiosity-gap title under 60 characters (e.g. "He didn't realize the mic was on...", "The 1 in a million shot that broke him"). NO cringe hashtag spam.
6. INDEPENDENT VALUE: Every clip must make sense on its own with clear setup, action, and resolution/punchline.
7. NO OVERLAPS: Candidate clips must not overlap each other.

### AUDIO LOUDNESS SPIKES (Screams, Laughter, Hype):
{peaks_text}

### FULL TRANSCRIPT WITH TIMESTAMPS:
{full_transcript_text}
"""

    candidate_models = [model_name]
    for fallback in ("gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash"):
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    response = None
    last_error: Optional[Exception] = None

    for m in candidate_models:
        try:
            logger.info(f"Attempting highlight detection using Gemini model: {m}")
            response = client.models.generate_content(
                model=m,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=HighlightAnalysisResult,
                    temperature=0.2,
                ),
            )
            if response:
                break
        except Exception as exc:
            logger.warning(f"Model {m} failed: {exc}. Trying next fallback candidate...")
            last_error = exc

    if response is None:
        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

    result: Optional[HighlightAnalysisResult] = None
    if getattr(response, "parsed", None) is not None and isinstance(response.parsed, HighlightAnalysisResult):
        result = response.parsed
    elif response.text:
        result = HighlightAnalysisResult.model_validate_json(response.text)

    if result is None:
        raise RuntimeError("Failed to obtain structured response from Gemini API.")

    # Post-validation: ensure timestamps within video bounds and valid duration
    valid_candidates: List[ClipCandidate] = []
    max_vod_duration = transcript.duration if transcript.duration > 0 else 999999.0

    for cand in result.candidates:
        if cand.start_time >= max_vod_duration:
            logger.warning(
                f"Candidate '{cand.hook_title}' starts after video ends ({cand.start_time:.1f}s >= {max_vod_duration:.1f}s), skipping."
            )
            continue

        cand.end_time = min(cand.end_time, max_vod_duration)
        duration = cand.end_time - cand.start_time
        if duration < 10.0:
            logger.warning(f"Candidate '{cand.hook_title}' too short ({duration:.1f}s), skipping.")
            continue
        valid_candidates.append(cand)

    result.candidates = valid_candidates[:top_k]
    return result


def generate_hook_titles(
    clip_transcript: str,
    num_variants: int = 3,
    api_key: Optional[str] = None,
    model_name: str = GEMINI_MODEL,
) -> List[str]:
    """
    Generates high-CTR viral hook titles and captions for a specific clip.
    """
    client = init_gemini_client(api_key)

    prompt = f"""
Given this short-form video clip transcript, write {num_variants} high-CTR curiosity-gap hook titles
for TikTok and YouTube Shorts (under 60 characters each, no hashtags):

Transcript:
"{clip_transcript}"
"""
    candidate_models = [model_name]
    for fallback in ("gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash"):
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    response = None
    for m in candidate_models:
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt,
            )
            if response:
                break
        except Exception:
            continue

    if response is None:
        return []

    lines = [line.strip().lstrip("0123456789.- ") for line in (response.text or "").split("\n") if line.strip()]
    return lines[:num_variants]
