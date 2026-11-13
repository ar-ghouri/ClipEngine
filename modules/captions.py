"""
ClipEngine Captions Module (Stage 4)
Advanced SubStation Alpha (.ass) zero-flicker dynamic karaoke captions.
Renders TikTok / Shorts / Reels style animated word-by-word highlighted captions.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from modules.transcriber import WordTimestamp


class CaptionStyle(BaseModel):
    """Configuration for .ass subtitle typography, positioning, and animation."""

    font_name: str = "Arial Black"
    font_size: int = 76
    primary_color: str = "&H00FFFFFF"      # Crisp White
    highlight_color: str = "&H0000FFFF"    # Bright Viral Yellow (AABBGGRR in ASS)
    outline_color: str = "&H00000000"      # Pure Black
    shadow_color: str = "&H80000000"       # Semi-transparent Black
    outline_width: int = 6
    shadow_width: int = 3
    alignment: int = 5                     # 5 = Middle-Center, 2 = Bottom-Center
    margin_v: int = 640                    # Vertical anchor


def format_ass_time(seconds: float) -> str:
    """
    Formats floating seconds into ASS time format: H:MM:SS.cs (centiseconds).
    Example: 75.32 -> 0:01:15.32
    """
    secs = max(0.0, seconds)
    hours = int(secs // 3600)
    minutes = int((secs % 3600) // 60)
    remainder = secs % 60
    s = int(remainder)
    cs = int(round((remainder - s) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{hours}:{minutes:02d}:{s:02d}.{cs:02d}"


def clean_caption_text(text: str) -> str:
    """Cleans word text, removing bracketed sound tags, ASS curly brace overrides, stripping stray leading apostrophes and trailing commas, and capitalizing."""
    cleaned = re.sub(r"\[.*?\]", "", text).strip()
    cleaned = re.sub(r"[{}]", "", cleaned).strip()
    cleaned = re.sub(r"^['’]+", "", cleaned)
    cleaned = re.sub(r",+$", "", cleaned).strip()
    return cleaned.upper()


def strip_bracketed_tags(words: List[WordTimestamp]) -> List[WordTimestamp]:
    """
    Strips bracketed sound tags like [SCREAMING], [LAUGHTER], [MUSIC], [APPLAUSE], [ __ ]
    from the WordTimestamp stream so they never appear in subtitles.
    """
    result: List[WordTimestamp] = []
    for w in words:
        cleaned_word = re.sub(r"\[.*?\]", "", w.word).strip()
        if cleaned_word:
            result.append(
                WordTimestamp(
                    word=cleaned_word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
            )
    return result


def collapse_repetitions(words: List[WordTimestamp]) -> List[WordTimestamp]:
    """
    Collapses 2+ consecutive identical words (e.g. 'HOUSE HOUSE' -> 'HOUSE' or 'I I' -> 'I')
    into a single word spanning the full timestamp range.
    """
    if not words:
        return []

    result: List[WordTimestamp] = []
    i = 0
    n = len(words)
    while i < n:
        base_norm = re.sub(r"[^A-Za-z0-9']+", "", words[i].word).upper()
        if not base_norm:
            result.append(words[i])
            i += 1
            continue

        j = i + 1
        while j < n and re.sub(r"[^A-Za-z0-9']+", "", words[j].word).upper() == base_norm:
            j += 1

        count = j - i
        if count >= 2:
            last_word_raw = words[j - 1].word.strip()
            raw_clean = words[i].word.strip().rstrip(",.?!")
            if last_word_raw.endswith("?"):
                collapsed_word = f"{raw_clean}?"
            elif last_word_raw.endswith("!"):
                collapsed_word = f"{raw_clean}!"
            elif last_word_raw.endswith("."):
                collapsed_word = f"{raw_clean}."
            else:
                collapsed_word = raw_clean

            result.append(
                WordTimestamp(
                    word=collapsed_word,
                    start=words[i].start,
                    end=words[j - 1].end,
                    probability=max(w.probability for w in words[i:j]),
                )
            )
            i = j
        else:
            for k in range(i, j):
                result.append(words[k])
            i = j

    return result


def normalize_contractions(words: List[WordTimestamp]) -> List[WordTimestamp]:
    """
    Normalizes word list by merging any token that starts with an apostrophe (' or ’)
    onto the end of the previous word (combining their start/end timestamps), and
    strips any stray leading apostrophes from standalone words so chunks never
    render split contractions like 'FOR ALL Y' or ''ALL OVERWATCH'.
    """
    if not words:
        return []

    normalized: List[WordTimestamp] = []
    for w in words:
        raw_text = w.word.strip()
        if not raw_text:
            continue

        if raw_text.startswith(("'", "’")):
            if normalized:
                prev = normalized[-1]
                prev_text = prev.word.rstrip()
                prev.word = f"{prev_text}{raw_text}"
                prev.end = max(prev.end, w.end)
                prev.probability = min(prev.probability, w.probability)
            else:
                stripped = re.sub(r"^['’]+", "", raw_text)
                if stripped:
                    normalized.append(
                        WordTimestamp(
                            word=stripped,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )
        else:
            normalized.append(
                WordTimestamp(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
            )

    return normalized


def chunk_words(
    words: List[WordTimestamp],
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
    words_per_group: int = 3,
    max_pause_sec: float = 0.6,
) -> List[List[Tuple[float, float, str]]]:
    """
    Filters words overlapping [clip_start, clip_end], shifts timestamps relative to clip start,
    and groups words into punchy 3-word chunks. Chunks break early on pauses > 0.6s
    or sentence-ending punctuation.

    Returns:
        List of chunks, where each chunk is a list of (start_rel, end_rel, clean_word).
    """
    if not words:
        return []

    # Preprocess words: strip sound tags, merge contractions, and collapse 2+ repetitions
    words = strip_bracketed_tags(words)
    words = normalize_contractions(words)
    words = collapse_repetitions(words)

    end_bound = clip_end if clip_end is not None else (max(w.end for w in words) + 1.0)

    # 1. Filter and shift timestamps relative to clip start
    shifted_words: List[Tuple[float, float, str, str]] = []
    for w in words:
        if w.end <= clip_start or w.start >= end_bound:
            continue
        start_rel = max(0.0, w.start - clip_start)
        end_rel = max(start_rel + 0.05, min(end_bound - clip_start, w.end - clip_start))
        word_clean = clean_caption_text(w.word)
        if not word_clean:
            continue
        shifted_words.append((start_rel, end_rel, word_clean, w.word.strip()))

    if not shifted_words:
        return []

    # 2. Group into chunks
    chunks: List[List[Tuple[float, float, str]]] = []
    current_chunk: List[Tuple[float, float, str]] = []

    for i, (s, e, clean_w, raw_w) in enumerate(shifted_words):
        current_chunk.append((s, e, clean_w))

        # Check chunk boundary conditions
        has_reached_max = len(current_chunk) >= words_per_group
        has_punctuation = bool(re.search(r"[.?!]$", raw_w))
        has_long_pause = False
        if i + 1 < len(shifted_words):
            next_start = shifted_words[i + 1][0]
            if next_start - e > max_pause_sec:
                has_long_pause = True

        if has_reached_max or has_punctuation or has_long_pause:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def generate_ass_subtitles(
    words: List[WordTimestamp],
    output_path: Path,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
    layout: str = "gaming_split",
    hook_title: Optional[str] = None,
    style: Optional[CaptionStyle] = None,
    words_per_group: int = 3,
) -> Path:
    """
    Builds an Advanced SubStation Alpha (.ass) file with zero-flicker word-by-word
    karaoke highlighting, viral 3.5s hook banner overlay, and layout-aware vertical positioning.

    Args:
        words: List of WordTimestamp from Stage 2 transcription.
        output_path: Target .ass file destination.
        clip_start: Absolute timestamp in VOD where the clip begins.
        clip_end: Absolute timestamp in VOD where the clip ends.
        layout: 'gaming_split' (centered on 640px seam) or 'center_crop' (lower third).
        hook_title: Optional headline hook title to overlay for the first 3.5 seconds.
        style: Optional typography override.
        words_per_group: Number of words per phrase group (default 3).

    Returns:
        Path to the generated .ass file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if style is None:
        style = CaptionStyle()

    # Layout-aware alignment and vertical anchor
    is_gaming_split = "gaming_split" in layout.lower()
    if is_gaming_split:
        alignment = 5       # Middle-Center
        margin_v = 640      # Exactly on the 640px split seam
        pos_tag = "{\\pos(540,640)}"
    else:
        alignment = 2       # Bottom-Center
        margin_v = 420      # Lower-third safe zone
        pos_tag = ""

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{style.font_size},{style.primary_color},&H000000FF,{style.outline_color},{style.shadow_color},-1,0,0,0,100,100,0,0,1,{style.outline_width},{style.shadow_width},{alignment},40,40,{margin_v},1
Style: HookBanner,Arial Black,46,&H00FFFFFF,&H000000FF,&H00000000,&H20000000,-1,0,0,0,100,100,0,0,3,14,0,5,40,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    chunks = chunk_words(
        words,
        clip_start=clip_start,
        clip_end=clip_end,
        words_per_group=words_per_group,
    )

    events: List[str] = []

    # 3.5s Viral Hook Title Overlay Banner (Word-wrapped to prevent horizontal overflow past 1080px)
    if hook_title:
        clean_hook = re.sub(r"[{}]", "", hook_title).strip().upper()
        # Wrap at word boundaries at width=26, cap at 2 lines max, join with \N
        wrapped_lines = textwrap.wrap(clean_hook, width=26)
        if len(wrapped_lines) > 2:
            wrapped_text = r"\N".join(wrapped_lines[:2])
        else:
            wrapped_text = r"\N".join(wrapped_lines) if wrapped_lines else clean_hook

        # For gaming_split: \pos(540,735) (top of gameplay zone)
        # For center_crop: \pos(540,260) (upper third safe zone)
        banner_pos = r"\pos(540,735)" if is_gaming_split else r"\pos(540,260)"
        events.append(
            f"Dialogue: 1,0:00:00.00,0:00:03.50,HookBanner,,0,0,0,,{{{banner_pos}\\fad(0,250)}}{wrapped_text}"
        )

    for chunk_idx, chunk in enumerate(chunks):
        num_words = len(chunk)
        for i, (w_start, w_end, _) in enumerate(chunk):
            event_start = w_start

            # Anti-Flicker Rule:
            # Set this word's End timestamp equal to the Start of the next word in the chunk.
            # The phrase never disappears during tiny inter-word gaps.
            if i < num_words - 1:
                event_end = chunk[i + 1][0]
            else:
                # Last word in chunk: hold for a clean finish
                hold_end = w_end + 0.18
                if chunk_idx + 1 < len(chunks):
                    next_chunk_start = chunks[chunk_idx + 1][0][0]
                    hold_end = min(hold_end, next_chunk_start)
                event_end = max(event_start + 0.1, hold_end)

            # Build line text: active word pop-scales 110% in viral yellow; inactive words remain white
            word_elements: List[str] = []
            for j, (_, _, word_text) in enumerate(chunk):
                if j == i:
                    word_elements.append(
                        f"{{\\c{style.highlight_color}&\\fscx110\\fscy110}}{word_text}{{\\fscx100\\fscy100\\c{style.primary_color}&}}"
                    )
                else:
                    word_elements.append(word_text)

            line_text = f"{pos_tag}{' '.join(word_elements)}"
            events.append(
                f"Dialogue: 0,{format_ass_time(event_start)},{format_ass_time(event_end)},Default,,0,0,0,,{line_text}"
            )

    full_ass_content = header + "\n".join(events) + "\n"
    output_path.write_text(full_ass_content, encoding="utf-8")
    return output_path


def burn_captions_to_video(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
    nvenc: bool = True,
) -> Path:
    """
    Burns .ass subtitles into an existing video using FFmpeg.
    """
    from modules.compositor import compose_clip

    return compose_clip(
        input_video=video_path,
        output_path=output_path,
        ass_path=ass_path,
    )
