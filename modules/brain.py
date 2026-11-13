"""
ClipEngine Brain Module (Stage 3)
Gemini 2.5 Flash Viral Highlight Detection & Retention Scoring Engine.
Uses official google-genai SDK with Structured JSON Outputs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, GEMINI_MODEL
from modules.transcriber import AudioPeak, TranscriptionResult, WordTimestamp

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
    vibe: Literal["hype_action", "funny_reaction", "dramatic_tension", "insightful_story"] = Field(
        default="hype_action",
        description="Vibe/category of the highlight for visual pacing and background music selection",
    )
    climax_offset: float = Field(
        default=0.0,
        description="Seconds relative to start_time where the biggest peak/scream/punchline occurs",
    )
    use_dramatic_punch: bool = Field(
        default=False,
        description="True only when a zoom-punch / flash transition at climax_offset genuinely elevates the moment",
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


def _generate_time_windows(
    total_duration: float,
    chunk_size: float = 1800.0,
    overlap: float = 30.0,
) -> List[Tuple[float, float]]:
    """
    Splits total duration into 30-minute chunks with 30-second overlap.
    Guarantees boundary events are not cut in half across chunk edges.
    """
    if total_duration <= chunk_size:
        return [(0.0, total_duration)]

    windows: List[Tuple[float, float]] = []
    start = 0.0
    step = chunk_size - overlap
    while start < total_duration:
        end = min(total_duration, start + chunk_size)
        if end >= total_duration:
            if windows and (end - start) < 60.0:
                prev_s, _ = windows[-1]
                windows[-1] = (prev_s, end)
            else:
                windows.append((start, end))
            break
        windows.append((start, end))
        start += step
    return windows


def _detect_highlights_window(
    client: Any,
    segments: List[Any],
    audio_peaks: Optional[List[AudioPeak]],
    window_start: float,
    window_end: float,
    min_duration: float,
    max_duration: float,
    num_candidates: int,
    model_name: str,
) -> HighlightAnalysisResult:
    """
    Analyzes a single time window [window_start, window_end] with Gemini structured outputs.
    """
    from google.genai import types

    # Format transcript lines with timestamps showing BOTH raw seconds and MM:SS
    transcript_lines: List[str] = []
    for seg in segments:
        m_start = int(seg.start // 60)
        s_start = int(seg.start % 60)
        m_end = int(seg.end // 60)
        s_end = int(seg.end % 60)
        time_tag = f"[{seg.start:.1f}s ({m_start:02d}:{s_start:02d}) -> {seg.end:.1f}s ({m_end:02d}:{s_end:02d})]"
        transcript_lines.append(f"{time_tag} {seg.text}")

    full_transcript_text = "\n".join(transcript_lines) if transcript_lines else "No speech recorded."

    # Format loudness energy peaks inside this window
    peak_lines: List[str] = []
    if audio_peaks:
        for peak in audio_peaks:
            if window_start <= peak.timestamp <= window_end:
                peak_lines.append(
                    f"- Timestamp: {peak.formatted_time} ({peak.timestamp:.1f}s), Loudness: {peak.rms_db:.1f} dBFS"
                )
    peaks_text = "\n".join(peak_lines) if peak_lines else "None detected in this segment."

    prompt = f"""
You are an elite short-form video editor specializing in viral TikToks, YouTube Shorts, and Instagram Reels.
Your mission is to find the top {num_candidates} most engaging, high-retention moments from this stream segment [{window_start:.1f}s -> {window_end:.1f}s].

### CRITICAL RULES:
1. QUANTITY: You MUST return EXACTLY {num_candidates} candidate clips (or up to {num_candidates} distinct non-overlapping moments if content allows). Do NOT return only 1 clip when multiple highlight moments exist!
2. RAW SECONDS ONLY: All `start_time` and `end_time` fields in your JSON response MUST be raw seconds as floating-point numbers strictly within [{window_start:.1f}, {window_end:.1f}].
   CRITICAL NOTE: Do NOT confuse minutes with hundreds of seconds! Always use the raw seconds values shown before the parentheses in the transcript (e.g. 127.8s)!
3. DURATION: Each clip must be STRICTLY between {min_duration:.0f} and {max_duration:.0f} seconds long (duration = end_time - start_time).
4. BOUNDARIES: `start_time` and `end_time` MUST align with natural sentence boundaries from the transcript. NEVER cut mid-sentence or mid-thought.
5. AUDIO PEAKS: Use the acoustic loudness spikes below as clues. Streamer screams, laughter, gasps, and loud hype moments often make great climaxes for clips.
6. HOOK TITLE: Generate an intriguing, curiosity-gap title under 60 characters (e.g. "He didn't realize the mic was on...", "The 1 in a million shot that broke him"). NO cringe hashtag spam.
7. INDEPENDENT VALUE: Every clip must make sense on its own with clear setup, action, and resolution/punchline.
8. NO OVERLAPS: Candidate clips must not overlap each other.
9. VIBE CLASSIFICATION: Classify each candidate's `vibe` into exactly one of:
   - "hype_action": fast-paced gameplay action, scream/rage, clutch win, celebration
   - "funny_reaction": comedic fail, hilarious banter, unexpected meme moment, laughing fit
   - "dramatic_tension": intense suspense, high stakes, close call, nail-biting buildup
   - "insightful_story": storytelling, intriguing life advice, podcast discussion, deep commentary
10. CLIMAX TIMING (`climax_offset`): Pinpoint the exact second of maximum impact/punchline/scream, expressed in relative seconds from `start_time` (e.g., if clip starts at 42.0s and the peak occurs at 54.5s, `climax_offset` = 12.5). Use the Audio Loudness Spikes to help pinpoint the exact peak.
11. DRAMATIC PUNCH (`use_dramatic_punch`): Set to `true` ONLY if a sudden 1.12x zoom-punch and subtle white-flash pop at `climax_offset` will heighten viewer engagement (e.g. explosive screams, sudden clutch kills, shocking fails). Set to `false` for calm, thoughtful, or conversational clips.

### AUDIO LOUDNESS SPIKES (Screams, Laughter, Hype):
{peaks_text}

### TRANSCRIPT FOR THIS SEGMENT [{window_start:.1f}s -> {window_end:.1f}s]:
{full_transcript_text}
"""

    candidate_models = [model_name]
    for fallback in ("gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"):
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

    # Post-validation: ensure timestamps within window bounds and valid duration
    valid_candidates: List[ClipCandidate] = []
    for cand in result.candidates:
        if cand.start_time >= window_end or cand.end_time <= window_start:
            logger.warning(
                f"Candidate '{cand.hook_title}' out of window bounds [{window_start:.1f}s -> {window_end:.1f}s], skipping."
            )
            continue

        cand.start_time = max(window_start, cand.start_time)
        cand.end_time = min(cand.end_time, window_end)
        duration = cand.end_time - cand.start_time
        if duration < 10.0:
            logger.warning(f"Candidate '{cand.hook_title}' too short ({duration:.1f}s), skipping.")
            continue

        # Sanitize climax_offset bounds
        if cand.climax_offset < 0.0 or cand.climax_offset > duration:
            cand.climax_offset = round(duration * 0.65, 2)
        else:
            cand.climax_offset = round(cand.climax_offset, 2)

        valid_candidates.append(cand)

    result.candidates = valid_candidates[:num_candidates]
    return result


def snap_to_word_boundaries(
    candidates: List[ClipCandidate],
    transcript: TranscriptionResult,
) -> List[ClipCandidate]:
    """
    Snaps candidate clip start and end timestamps to exact sentence boundaries
    and word alignments from the transcript, with smooth audio padding.

    - Start snap: Finds sentence starters within ±2.5s of candidate.start_time
      (first word of transcript, word following .?!, or word after a pause > 0.35s).
      Applies -0.25s padding (max 0.0) for a clean audio lead-in.
    - End snap: Finds sentence enders within ±2.5s of candidate.end_time
      (word ending in .?!, last word of transcript, or word before a pause > 0.35s).
      Applies +0.45s padding (min total_duration) for a natural breathing finish.
    - Fallback: Snaps to nearest word boundary if no sentence boundary is in range.
    """
    all_words: List[WordTimestamp] = []
    for seg in transcript.segments:
        all_words.extend(seg.words)

    if not all_words:
        return candidates

    total_duration = transcript.duration if transcript.duration > 0 else 0.0

    # Identify sentence starters:
    # 1. First word of transcript
    # 2. Word immediately following sentence-ending punctuation (. ? !)
    # 3. Word preceded by an inter-word pause > 0.35s
    starters: List[WordTimestamp] = []
    for i, w in enumerate(all_words):
        if i == 0:
            starters.append(w)
        elif re.search(r"[.?!][\"']?$", all_words[i - 1].word.strip()):
            starters.append(w)
        elif (w.start - all_words[i - 1].end) > 0.35:
            starters.append(w)

    # Identify sentence enders:
    # 1. Word ending in sentence-ending punctuation (. ? !)
    # 2. Last word of transcript
    # 3. Word followed by an inter-word pause > 0.35s
    enders: List[WordTimestamp] = []
    for i, w in enumerate(all_words):
        if i == len(all_words) - 1:
            enders.append(w)
        elif re.search(r"[.?!][\"']?$", w.word.strip()):
            enders.append(w)
        elif (i + 1 < len(all_words)) and ((all_words[i + 1].start - w.end) > 0.35):
            enders.append(w)

    for cand in candidates:
        orig_start = cand.start_time
        orig_end = cand.end_time

        # Start snap within ±2.5s
        matching_starters = [w for w in starters if abs(w.start - orig_start) <= 2.5]
        if matching_starters:
            best_starter = min(matching_starters, key=lambda w: abs(w.start - orig_start))
            snapped_start = max(0.0, best_starter.start - 0.25)
        else:
            # Fallback to nearest word boundary
            best_word = min(all_words, key=lambda w: abs(w.start - orig_start))
            snapped_start = max(0.0, best_word.start - 0.25)

        # End snap within ±2.5s
        matching_enders = [w for w in enders if abs(w.end - orig_end) <= 2.5]
        max_end_bound = total_duration if total_duration > 0 else (orig_end + 10.0)
        if matching_enders:
            best_ender = min(matching_enders, key=lambda w: abs(w.end - orig_end))
            snapped_end = min(max_end_bound, best_ender.end + 0.45)
        else:
            # Fallback to nearest word boundary
            best_word = min(all_words, key=lambda w: abs(w.end - orig_end))
            snapped_end = min(max_end_bound, best_word.end + 0.45)

        # Ensure valid window duration
        if snapped_end > snapped_start + 5.0:
            orig_climax_abs = orig_start + cand.climax_offset
            cand.start_time = round(snapped_start, 2)
            cand.end_time = round(snapped_end, 2)
            new_dur = cand.end_time - cand.start_time
            cand.climax_offset = round(
                max(0.0, min(max(0.0, new_dur - 0.5), orig_climax_abs - cand.start_time)), 2
            )
            logger.info(
                f"Snapped '{cand.hook_title}': [{orig_start:.2f}s -> {orig_end:.2f}s] to "
                f"[{cand.start_time:.2f}s -> {cand.end_time:.2f}s] "
                f"(duration: {new_dur:.2f}s, climax: {cand.climax_offset:.2f}s)"
            )
        else:
            logger.warning(
                f"Snapping resulted in invalid duration for '{cand.hook_title}' "
                f"[{snapped_start:.2f}s -> {snapped_end:.2f}s], retaining original."
            )

    return candidates


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
    Evaluates transcript segments and audio loudness peaks using Gemini Flash
    with structured schema output to extract viral short-form clips.

    For streams exceeding 1800s (30 minutes), splits into 30-minute chunks with 30s overlap,
    runs Gemini per chunk, merges all candidates, deduplicates overlapping timestamps,
    sorts by virality_score descending, and returns the global top k clips.
    """
    client = init_gemini_client(api_key)
    total_duration = transcript.duration if transcript.duration > 0 else 0.0

    windows = _generate_time_windows(total_duration, chunk_size=1800.0, overlap=30.0)
    logger.info(
        f"Highlight detection initialized for duration {total_duration:.1f}s across {len(windows)} window(s)."
    )

    if len(windows) == 1:
        # Single window fast-path
        result = _detect_highlights_window(
            client=client,
            segments=transcript.segments,
            audio_peaks=audio_peaks,
            window_start=0.0,
            window_end=total_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            num_candidates=top_k,
            model_name=model_name,
        )
        result.candidates = snap_to_word_boundaries(result.candidates, transcript)
        return result

    # Multi-window processing for long streams (> 30 min)
    all_candidates: List[ClipCandidate] = []
    candidates_per_window = max(2, min(top_k, 4))

    for w_idx, (w_start, w_end) in enumerate(windows, 1):
        window_segs = [s for s in transcript.segments if s.end > w_start and s.start < w_end]
        if not window_segs:
            continue

        logger.info(
            f"Processing chunk {w_idx}/{len(windows)}: [{w_start:.1f}s -> {w_end:.1f}s] ({len(window_segs)} segments)..."
        )
        try:
            window_result = _detect_highlights_window(
                client=client,
                segments=window_segs,
                audio_peaks=audio_peaks,
                window_start=w_start,
                window_end=w_end,
                min_duration=min_duration,
                max_duration=max_duration,
                num_candidates=candidates_per_window,
                model_name=model_name,
            )
            all_candidates.extend(window_result.candidates)
        except Exception as exc:
            logger.warning(f"Highlight detection failed on window {w_idx} [{w_start:.1f}s -> {w_end:.1f}s]: {exc}")

    if not all_candidates:
        return HighlightAnalysisResult(
            summary=f"Processed {len(windows)} windows across {total_duration / 60:.1f} minutes; no candidates found.",
            candidates=[],
        )

    # Global ranking by virality_score descending
    all_candidates.sort(key=lambda c: c.virality_score, reverse=True)

    # Deduplicate overlapping timestamps
    deduped: List[ClipCandidate] = []
    for cand in all_candidates:
        overlaps = False
        for chosen in deduped:
            # Overlap exists if max(start1, start2) < min(end1, end2)
            if max(cand.start_time, chosen.start_time) < min(cand.end_time, chosen.end_time):
                overlaps = True
                break
        if not overlaps:
            deduped.append(cand)
        if len(deduped) >= top_k:
            break

    snapped_candidates = snap_to_word_boundaries(deduped[:top_k], transcript)

    return HighlightAnalysisResult(
        summary=f"Multi-window viral analysis across {len(windows)} chunks ({total_duration / 60:.1f} mins).",
        candidates=snapped_candidates,
    )


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
    for fallback in ("gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"):
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
