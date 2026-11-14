"""
ClipEngine CLI Entry Point
A local, zero-cost video clipping engine optimized for NVIDIA RTX 4060 on Windows.
Supports manual clipping (clip), audio ingestion (audio), inspection (info),
database inspection (list), and full autonomous viral clipping (auto-clip).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import (
    DEFAULT_WEBCAM_BBOX,
    DOWNLOADS_DIR,
    OUTPUTS_DIR,
    WHISPER_MODEL,
    register_cuda_dlls,
    resolve_ffmpeg_binaries,
)
from database import (
    create_clip_record,
    get_clip_record,
    init_db,
    list_clips,
    update_clip_record,
)
from modules.brain import detect_highlights
from modules.captions import generate_ass_subtitles
from modules.compositor import (
    LayoutMode,
    compose_clip,
    detect_facecam,
    probe_video,
)
from modules.ingest import (
    download_audio_only,
    download_target_clip,
    extract_video_id,
    extract_youtube_captions,
    get_video_metadata,
    parse_timestamp,
    sanitize_title,
)
from modules.transcriber import compute_audio_loudness_peaks, transcribe_audio

# Initialize Typer App and Rich Console
app = typer.Typer(
    name="clipengine",
    help="ClipEngine: Local, zero-cost video clipping engine powered by NVIDIA NVENC & Gemini.",
    add_completion=False,
)
console = Console()


def parse_bbox_string(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parses 'x,y,w,h' normalized coordinates into a float tuple."""
    try:
        parts = [float(p.strip()) for p in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        x, y, w, h = parts
        if not all(0.0 <= val <= 1.0 for val in (x, y, w, h)):
            raise ValueError
        return (x, y, w, h)
    except Exception:
        console.print(
            f"[yellow]Warning: Invalid cam_bbox '{bbox_str}'. Falling back to default {DEFAULT_WEBCAM_BBOX}.[/yellow]"
        )
        return DEFAULT_WEBCAM_BBOX


CORNER_MAP: Dict[str, Tuple[float, float, float, float]] = {
    "top-left": (0.0, 0.0, 0.32, 0.34),
    "top-right": (0.68, 0.0, 0.32, 0.34),
    "bottom-left": (0.0, 0.66, 0.32, 0.34),
    "bottom-right": (0.68, 0.66, 0.32, 0.34),
}


@app.callback()
def main_callback():
    """Ensure database, CUDA DLLs, and ffmpeg paths are ready before any command executes."""
    init_db()
    register_cuda_dlls()
    resolve_ffmpeg_binaries()


@app.command("clip")
def clip_video(
    url: str = typer.Argument(..., help="YouTube video or stream VOD URL"),
    start_time: str = typer.Option(
        ...,
        "--start-time",
        "-s",
        help="Clip start timestamp (e.g. '00:01:15', '75', or '1:15')",
    ),
    end_time: str = typer.Option(
        ...,
        "--end-time",
        "-e",
        help="Clip end timestamp (e.g. '00:02:00', '120', or '2:00')",
    ),
    layout: LayoutMode = typer.Option(
        LayoutMode.AUTO,
        "--layout",
        "-l",
        help="Output layout: 'auto' (universal AI router), 'cinema_blur' (slides/wide), 'gaming_split' (webcam + gameplay), or 'center_crop' (9:16 crop)",
    ),
    cam_bbox: str = typer.Option(
        "0.0,0.0,0.25,0.25",
        "--cam-bbox",
        "--cam-box",
        help="Normalized webcam box 'x,y,w,h' for gaming_split (default top-left quadrant)",
    ),
    cam_corner: Optional[str] = typer.Option(
        None,
        "--cam-corner",
        help="Webcam corner override: 'auto', 'top-left', 'bottom-left', 'top-right', 'bottom-right', 'none'",
    ),
    hook_title: Optional[str] = typer.Option(
        None,
        "--hook-title",
        "-t",
        help="Optional title / hook for metadata tracking",
    ),
    virality_score: Optional[float] = typer.Option(
        None,
        "--virality-score",
        "-v",
        help="Virality score (0-10)",
    ),
    captions: bool = typer.Option(
        True,
        "--captions/--no-captions",
        help="Burn dynamic word-by-word karaoke subtitles",
    ),
    fast_subs: bool = typer.Option(
        False,
        "--fast-subs",
        help="Fast-path: extract YouTube auto-subtitles instead of running Whisper",
    ),
    bgm: bool = typer.Option(
        False,
        "--bgm/--no-bgm",
        help="Mix auto-ducked procedural background music",
    ),
):
    """
    Downloads targeted timestamp range with yt-dlp, renders 9:16 vertical video via NVENC,
    and logs state to SQLite.
    """
    console.print(
        Panel.fit(
            f"[bold cyan]ClipEngine Pipeline Execution[/bold cyan]\n"
            f"VOD URL:      [green]{url}[/green]\n"
            f"Range:        [yellow]{start_time}[/yellow] -> [yellow]{end_time}[/yellow]\n"
            f"Layout:       [magenta]{layout.value}[/magenta]",
            title="ClipEngine v1.0",
        )
    )

    # 1. Parse and validate timestamps
    try:
        start_sec = parse_timestamp(start_time)
        end_sec = parse_timestamp(end_time)
    except Exception as exc:
        console.print(f"[bold red]Timestamp Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    duration = end_sec - start_sec
    if duration <= 0:
        console.print("[bold red]Error:[/bold red] end_time must be greater than start_time.")
        raise typer.Exit(code=1)

    # 2. Register job in SQLite
    clip_id = create_clip_record(
        vod_url=url,
        clip_start=start_sec,
        clip_end=end_sec,
        hook_title=hook_title,
        virality_score=virality_score,
        status="ingesting",
    )
    console.print(f"[dim]Initialized SQLite record [bold]#{clip_id}[/bold][/dim]")

    # 3. Targeted Smart Ingest
    vod_id = extract_video_id(url)
    vod_downloads_dir = DOWNLOADS_DIR / vod_id
    vod_downloads_dir.mkdir(parents=True, exist_ok=True)

    with console.status(f"[bold green]Downloading section ({duration:.1f}s) via yt-dlp..."):
        try:
            raw_path = download_target_clip(
                url=url,
                start_time=start_sec,
                end_time=end_sec,
                output_dir=vod_downloads_dir,
                prefix=f"clip_{clip_id}",
            )
        except Exception as exc:
            update_clip_record(clip_id, status="failed_ingest")
            console.print(f"[bold red]Ingest Failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    console.print(f"[green]✓ Ingest complete:[/green] {raw_path.name}")
    update_clip_record(clip_id, status="compositing")

    # 4. Resolve Layout and Auto-Webcam Detection
    user_overrode_bbox = (cam_bbox != "0.0,0.0,0.25,0.25")
    parsed_bbox = parse_bbox_string(cam_bbox)
    effective_layout = layout
    effective_bbox = parsed_bbox
    effective_speaker_x = 0.5
    norm_corner = cam_corner.lower().strip() if cam_corner else None

    if norm_corner == "none":
        effective_layout = LayoutMode.CENTER_CROP
        console.print("[yellow]--cam-corner none specified -> forcing center_crop layout.[/yellow]")
    elif norm_corner in CORNER_MAP:
        effective_layout = LayoutMode.GAMING_SPLIT
        effective_bbox = CORNER_MAP[norm_corner]
        console.print(f"[green]✓ Manual webcam corner override ({norm_corner}):[/green] {effective_bbox} -> gaming_split")
    elif layout == LayoutMode.AUTO:
        from modules.brain import classify_video_layout_vision
        with console.status("[bold blue]Universal AI Layout Router analyzing video framing..."):
            try:
                classified_l, det_cam, det_spk = classify_video_layout_vision(raw_path, vod_key=vod_id)
                effective_layout = LayoutMode(classified_l)
                if det_cam:
                    effective_bbox = det_cam
                effective_speaker_x = det_spk
                console.print(f"[green]✓ Universal AI Layout Router:[/green] selected '{effective_layout.value}' (speaker_x={effective_speaker_x:.2f})")
            except Exception as exc:
                console.print(f"[yellow]Layout classification failed ({exc}); falling back to cinema_blur[/yellow]")
                effective_layout = LayoutMode.CINEMA_BLUR
    elif layout == LayoutMode.GAMING_SPLIT and not user_overrode_bbox:
        with console.status("[bold blue]Analyzing video framing and webcam/speaker position..."):
            detected_bbox, detected_speaker_x = detect_facecam(raw_path, vod_key=vod_id)
            if detected_bbox is not None:
                effective_bbox = detected_bbox
                console.print(f"[green]✓ Corner webcam face detected at normalized coordinates:[/green] {effective_bbox}")
            else:
                effective_layout = LayoutMode.CENTER_CROP
                effective_speaker_x = detected_speaker_x
                console.print(
                    f"[yellow]Full-screen studio/couch shot detected (speaker X: {effective_speaker_x:.2f}). "
                    "Automatically switching layout to speaker-centered 'center_crop' and lower-third captions.[/yellow]"
                )

    # 5. Generate Karaoke Subtitles if enabled
    ass_file: Optional[Path] = None
    if captions:
        try:
            console.print("[bold magenta]Transcribing section via faster-whisper on RTX 4060...[/bold magenta]")
            clip_transcript = transcribe_audio(raw_path)

            clip_words = [w for seg in clip_transcript.segments for w in seg.words]
            if clip_words:
                ass_file = vod_downloads_dir / f"clip_{clip_id}.ass"
                generate_ass_subtitles(
                    words=clip_words,
                    output_path=ass_file,
                    clip_start=0.0,
                    clip_end=duration,
                    layout=effective_layout.value,
                    hook_title=hook_title,
                )
                console.print(f"[green]✓ Generated subtitles:[/green] {ass_file.name}")
        except Exception as exc:
            console.print(f"[yellow]Warning: Subtitle generation skipped: {exc}[/yellow]")

    # 6. Target Output Path (Organized subfolder if title available)
    manual_output_file: Optional[Path] = None
    if hook_title:
        subfolder_name = sanitize_title(hook_title, max_chars=60)
        clip_dir = OUTPUTS_DIR / subfolder_name
        clip_dir.mkdir(parents=True, exist_ok=True)
        manual_output_file = clip_dir / f"01_{subfolder_name}.mp4"

    # 7. GPU-Accelerated Compositing (Single-pass with NVENC)
    with console.status(f"[bold magenta]Rendering 1080x1920 using NVIDIA NVENC ({effective_layout.value})..."):
        try:
            output_file = compose_clip(
                input_video=raw_path,
                output_path=manual_output_file,
                layout=effective_layout,
                cam_bbox=effective_bbox,
                speaker_x=effective_speaker_x,
                ass_path=ass_file,
                auto_detect_cam=False,
                enable_bgm=bgm,
            )
        except Exception as exc:
            update_clip_record(clip_id, status="failed_composite")
            console.print(f"[bold red]Compositing Failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    # 8. Finalize database state
    update_clip_record(
        clip_id,
        status="completed",
        output_path=str(output_file),
    )

    # Probe final output for summary
    final_info = probe_video(output_file)

    # Output Presentation
    table = Table(title="Render Summary", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Database ID", f"#{clip_id}")
    table.add_row("Status", "[green]COMPLETED[/green]")
    table.add_row("Resolution", f"{final_info['width']}x{final_info['height']} (9:16)")
    table.add_row("Duration", f"{final_info['duration']:.2f}s")
    table.add_row("Encoder", "h264_nvenc (NVIDIA RTX 4060)")
    table.add_row("Layout", effective_layout.value)
    table.add_row("Raw Section", str(raw_path))
    table.add_row("Final Output", f"[cyan]{output_file}[/cyan]")
    console.print(table)


@app.command("auto-clip")
def auto_clip_vod(
    url: str = typer.Argument(..., help="YouTube video or stream VOD URL"),
    max_clips: int = typer.Option(
        3,
        "--max-clips",
        "-k",
        help="Maximum viral clips to generate (default 3)",
    ),
    layout: LayoutMode = typer.Option(
        LayoutMode.AUTO,
        "--layout",
        "-l",
        help="Output layout: 'auto' (universal AI router), 'cinema_blur' (slides/wide), 'gaming_split' (webcam + gameplay), or 'center_crop' (9:16 crop)",
    ),
    cam_bbox: str = typer.Option(
        "0.0,0.0,0.25,0.25",
        "--cam-bbox",
        "--cam-box",
        help="Normalized webcam coordinates 'x,y,w,h' for gaming_split (default top-left quadrant)",
    ),
    cam_corner: Optional[str] = typer.Option(
        None,
        "--cam-corner",
        help="Webcam corner override: 'auto', 'top-left', 'bottom-left', 'top-right', 'bottom-right', 'none'",
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Custom steering directive for Gemini highlight detection (e.g. 'Find moments where CaseOh gets jumpscared')",
    ),
    whisper_model: Optional[str] = typer.Option(
        None,
        "--whisper-model",
        help="Whisper model variant (default 'large-v3-turbo', or 'base.en', 'distil-large-v3')",
    ),
    min_duration: float = typer.Option(
        20.0,
        "--min-duration",
        help="Minimum clip duration in seconds (strictly >= 20.0)",
    ),
    max_duration: float = typer.Option(
        55.0,
        "--max-duration",
        help="Maximum clip duration in seconds (strictly <= 55.0)",
    ),
    captions: bool = typer.Option(
        True,
        "--captions/--no-captions",
        help="Burn dynamic word-by-word karaoke subtitles",
    ),
    fast_subs: bool = typer.Option(
        False,
        "--fast-subs",
        help="Fast-path: extract YouTube auto-subtitles instead of running Whisper",
    ),
    bgm: bool = typer.Option(
        True,
        "--bgm/--no-bgm",
        help="Auto-ducked background music based on clip vibe",
    ),
    force_whisper: bool = typer.Option(
        False,
        "--force-whisper",
        help="Bypass disk cache and force faster-whisper re-transcription",
    ),
):
    """
    End-to-End Autonomous Viral Clipping:
    1. Downloads audio-only (never the full video).
    2. Transcribes with faster-whisper on RTX 4060 + detects audio loudness peaks.
    3. Detects top viral highlights using Gemini 2.5 Flash Structured Outputs.
    4. Downloads ONLY those targeted video sections and renders via h264_nvenc into 9:16 MP4s.
    5. Logs each clip with hook_title and virality_score into SQLite.
    """
    # 0. Query metadata
    with console.status("[bold blue]Querying VOD metadata..."):
        try:
            meta = get_video_metadata(url)
        except Exception as exc:
            console.print(f"[bold red]Metadata Query Failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]ClipEngine Autonomous Viral Pipeline[/bold cyan]\n"
            f"Title:        [bold]{meta['title']}[/bold]\n"
            f"Uploader:     {meta['uploader']}\n"
            f"Duration:     {meta['duration']}s\n"
            f"Target Clips: [yellow]{max_clips}[/yellow] (Layout: [magenta]{layout.value}[/magenta])",
            title="ClipEngine Auto-Clip",
        )
    )

    # Step 1 & 2: Subtitles & Audio Analysis
    transcript = None
    audio_path = None
    vod_id = meta.get("id") or extract_video_id(url)
    vod_downloads_dir = DOWNLOADS_DIR / vod_id
    vod_downloads_dir.mkdir(parents=True, exist_ok=True)

    if fast_subs:
        console.rule("[bold cyan]Fast-Path Subtitles Ingest (--fast-subs)[/bold cyan]")
        with console.status("[bold green]Checking & extracting YouTube auto-subtitles (.json3)..."):
            fast_transcript = extract_youtube_captions(url, output_dir=vod_downloads_dir, video_id=vod_id)
        if fast_transcript:
            transcript = fast_transcript
            console.print(
                f"[bold green]✓ Fast-Path:[/bold green] Extracted {len(transcript.segments)} segments "
                f"across {transcript.duration:.1f}s in ~3 seconds!"
            )
        else:
            console.print(
                "[yellow]Notice: YouTube auto-subtitles not found. Falling back to audio ingest + Whisper transcription.[/yellow]"
            )

    if transcript is None:
        # Step 1: Download Audio Only
        console.rule("[bold cyan]Step 1: Smart Audio-Only Ingest[/bold cyan]")
        with console.status("[bold green]Downloading audio stream (MP3)..."):
            try:
                audio_path = download_audio_only(
                    url=url,
                    output_dir=vod_downloads_dir,
                    prefix=f"{vod_id}_audio",
                )
                audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
                console.print(
                    f"[green]✓ Audio ingest complete:[/green] {audio_path.name} [dim]({audio_size_mb:.2f} MB)[/dim]"
                )
            except Exception as exc:
                console.print(f"[bold red]Audio Ingest Failed:[/bold red] {exc}")
                raise typer.Exit(code=1)

        # Step 2: GPU Transcription
        console.rule("[bold cyan]Step 2: RTX 4060 Whisper Transcription & Loudness Analysis[/bold cyan]")
        w_model = whisper_model or WHISPER_MODEL
        console.print(f"[bold magenta]Transcribing audio via faster-whisper ({w_model} on CUDA)...[/bold magenta]")
        try:
            whisper_cache = vod_downloads_dir / f"{vod_id}_whisper.transcript.json"
            transcript = transcribe_audio(
                audio_path,
                model_size=w_model,
                force_recompute=force_whisper,
                video_id=vod_id,
                cache_path=whisper_cache,
            )
        except Exception as exc:
            console.print(f"[bold red]Transcription Failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

        console.print(
            f"[green]✓ Transcribed {len(transcript.segments)} segments across {transcript.duration:.1f}s.[/green]"
        )

    # Compute audio loudness peaks if audio file is available
    audio_peaks = None
    if audio_path and audio_path.exists():
        with console.status("[bold blue]Analyzing audio loudness peaks..."):
            try:
                audio_peaks = compute_audio_loudness_peaks(audio_path, top_k=10)
            except Exception as exc:
                console.print(f"[yellow]Audio loudness peak analysis skipped: {exc}[/yellow]")
        if audio_peaks:
            peak_summary = ", ".join(f"{p.formatted_time} ({p.rms_db:.1f}dB)" for p in audio_peaks[:5])
            console.print(f"[dim]Top energy spikes: {peak_summary}[/dim]")

    # Step 3: Gemini Viral Highlight Detection
    console.rule("[bold cyan]Step 3: Gemini Viral Highlight Detection[/bold cyan]")
    if prompt:
        console.print(f"[dim cyan]User Prompt Steering Directive:[/dim cyan] [bold cyan]\"{prompt}\"[/bold cyan]")
    with console.status("[bold yellow]Analyzing retention drivers with Gemini..."):
        try:
            analysis = detect_highlights(
                transcript=transcript,
                audio_peaks=audio_peaks,
                min_duration=min_duration,
                max_duration=max_duration,
                top_k=max_clips,
                user_prompt=prompt,
            )
        except Exception as exc:
            console.print(f"[bold red]Gemini Highlight Detection Failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    if not analysis.candidates:
        console.print("[yellow]No viral candidates found matching duration criteria.[/yellow]")
        return

    console.print(f"[bold green]✓ Gemini identified {len(analysis.candidates)} viral clip moments:[/bold green]")
    for idx, cand in enumerate(analysis.candidates, 1):
        dur = cand.end_time - cand.start_time
        punch_str = f" [bold red][⚡ PUNCH @ +{cand.climax_offset:.1f}s][/bold red]" if cand.use_dramatic_punch else ""
        console.print(
            f"  [cyan]{idx}.[/cyan] [bold]'{cand.hook_title}'[/bold]{punch_str} "
            f"({cand.start_time:.1f}s -> {cand.end_time:.1f}s, {dur:.1f}s, Score: [bold yellow]{cand.virality_score}/10.0[/bold yellow], Vibe: [magenta]{cand.vibe}[/magenta])\n"
            f"     [dim]Reason: {cand.reasoning}[/dim]"
        )

    # Step 4: Targeted Video Ingest & NVENC Compositing
    console.rule("[bold cyan]Step 4: Smart Targeted Ingest & GPU Compositing[/bold cyan]")
    user_overrode_bbox = (cam_bbox != "0.0,0.0,0.25,0.25")
    parsed_bbox = parse_bbox_string(cam_bbox)
    completed_clips: List[Dict[str, Any]] = []

    # Ensure organized output subfolder exists: outputs/<sanitized_vod_title>/
    vod_folder_name = sanitize_title(meta.get("title", "vod_clips"), max_chars=60)
    vod_output_dir = OUTPUTS_DIR / vod_folder_name
    vod_output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Output subfolder: [cyan]{vod_output_dir}[/cyan][/dim]")

    vod_id = meta.get("id") or url
    norm_corner = cam_corner.lower().strip() if cam_corner else None

    for idx, cand in enumerate(analysis.candidates, 1):
        dur = cand.end_time - cand.start_time
        console.print(
            f"\n[bold]Rendering Clip {idx}/{len(analysis.candidates)}:[/bold] [yellow]'{cand.hook_title}'[/yellow] "
            f"[dim]({cand.start_time:.1f}s -> {cand.end_time:.1f}s, Vibe: {cand.vibe})[/dim]"
        )

        # 4a. Register record in SQLite
        clip_id = create_clip_record(
            vod_url=url,
            clip_start=cand.start_time,
            clip_end=cand.end_time,
            hook_title=cand.hook_title,
            virality_score=cand.virality_score,
            status="ingesting",
            vibe=cand.vibe,
            climax_offset=cand.climax_offset,
            use_dramatic_punch=cand.use_dramatic_punch,
        )

        # 4b. Targeted section download ONLY
        try:
            raw_clip = download_target_clip(
                url=url,
                start_time=cand.start_time,
                end_time=cand.end_time,
                output_dir=vod_downloads_dir,
                prefix=f"clip_{clip_id}",
            )
            console.print(f"  [green]✓ Ingest:[/green] {raw_clip.name} ({dur:.1f}s)")
        except Exception as exc:
            update_clip_record(clip_id, status="failed_ingest")
            console.print(f"  [red]✗ Download failed:[/red] {exc}")
            continue

        update_clip_record(clip_id, status="compositing")

        # Resolve webcam & layout mode with --cam-corner override, AI Universal Router & Gemini Vision fallback
        clip_layout = layout
        clip_bbox = parsed_bbox
        clip_speaker_x = 0.5

        if norm_corner == "none":
            clip_layout = LayoutMode.CENTER_CROP
            console.print("  [yellow]--cam-corner none specified -> forcing center_crop layout[/yellow]")
        elif norm_corner in CORNER_MAP:
            clip_layout = LayoutMode.GAMING_SPLIT
            clip_bbox = CORNER_MAP[norm_corner]
            console.print(f"  [green]✓ Webcam Corner Override ({norm_corner}):[/green] {clip_bbox} -> gaming_split")
        elif layout == LayoutMode.AUTO:
            from modules.brain import classify_video_layout_vision
            try:
                classified_l, det_cam, det_spk = classify_video_layout_vision(raw_clip, vod_key=vod_id)
                clip_layout = LayoutMode(classified_l)
                if det_cam:
                    clip_bbox = det_cam
                clip_speaker_x = det_spk
                console.print(f"  [green]✓ Universal AI Layout Router:[/green] selected '{clip_layout.value}' (speaker_x={clip_speaker_x:.2f})")
            except Exception as exc:
                console.print(f"  [yellow]Layout classification failed ({exc}); falling back to cinema_blur[/yellow]")
                clip_layout = LayoutMode.CINEMA_BLUR
        elif layout == LayoutMode.GAMING_SPLIT and not user_overrode_bbox:
            detected_bbox, detected_speaker_x = detect_facecam(raw_clip, vod_key=vod_id)
            if detected_bbox is not None:
                clip_bbox = detected_bbox
                clip_layout = LayoutMode.GAMING_SPLIT
                console.print(f"  [green]✓ Corner Facecam:[/green] detected at {clip_bbox} -> gaming_split")
            else:
                clip_layout = LayoutMode.CENTER_CROP
                clip_speaker_x = detected_speaker_x
                console.print(
                    f"  [yellow]✓ Studio/Couch Framing:[/yellow] speaker X={clip_speaker_x:.2f} -> dynamic center_crop"
                )

        # 4c. Generate zero-flicker karaoke subtitles + 3.5s hook banner
        # Hybrid Fast-Discovery Workflow:
        # When --fast-subs is used, YouTube transcript found viral timestamps in seconds (Step 2 & 3).
        # In Step 4, run local RTX 4060 faster-whisper on the 40-50s clip audio (~1.5s) for pristine, uncensored word timestamps!
        ass_path: Optional[Path] = None
        if captions:
            try:
                if fast_subs:
                    console.print("  [bold magenta]Running local GPU Whisper on clip audio (~1.5s precision)...[/bold magenta]")
                    clip_trans = transcribe_audio(raw_clip)
                    clip_words = [w for seg in clip_trans.segments for w in seg.words]
                    c_start = 0.0
                    c_end = dur
                elif transcript and transcript.segments:
                    clip_words = [
                        w
                        for seg in transcript.segments
                        for w in seg.words
                        if w.end > cand.start_time and w.start < cand.end_time
                    ]
                    c_start = cand.start_time
                    c_end = cand.end_time
                else:
                    clip_words = []
                    c_start = 0.0
                    c_end = dur

                if clip_words:
                    ass_file = vod_downloads_dir / f"clip_{clip_id}.ass"
                    generate_ass_subtitles(
                        words=clip_words,
                        output_path=ass_file,
                        clip_start=c_start,
                        clip_end=c_end,
                        layout=clip_layout.value,
                        hook_title=cand.hook_title,
                    )
                    ass_path = ass_file
                    console.print(f"  [green]✓ Captions:[/green] {ass_file.name} (uncensored GPU karaoke + hook)")
            except Exception as exc:
                console.print(f"  [yellow]Warning: Subtitle generation failed: {exc}[/yellow]")

        # Target output path: outputs/<sanitized_vod_title>/<index:02d>_<sanitized_hook_title>.mp4
        hook_clean = sanitize_title(cand.hook_title, max_chars=60)
        target_clip_output = vod_output_dir / f"{idx:02d}_{hook_clean}.mp4"

        # 4d. Single-pass GPU NVENC Compositing with Anti-Haze, Director FX & Ducked BGM
        try:
            output_video = compose_clip(
                input_video=raw_clip,
                output_path=target_clip_output,
                layout=clip_layout,
                cam_bbox=clip_bbox,
                speaker_x=clip_speaker_x,
                ass_path=ass_path,
                auto_detect_cam=False,
                vibe=cand.vibe,
                climax_offset=cand.climax_offset,
                use_dramatic_punch=cand.use_dramatic_punch,
                enable_bgm=bgm,
                vod_key=vod_id,
            )
            console.print(f"  [green]✓ Rendered:[/green] {output_video.name}")
        except Exception as exc:
            update_clip_record(clip_id, status="failed_composite")
            console.print(f"  [red]✗ Compositing failed:[/red] {exc}")
            continue

        update_clip_record(clip_id, status="completed", output_path=str(output_video))
        director_fx_str = (
            f"Punch + Flash @ {cand.climax_offset:.1f}s"
            if cand.use_dramatic_punch
            else "Standard"
        )
        bgm_str = f"{cand.vibe} (-18dB)" if bgm else "Disabled"
        completed_clips.append(
            {
                "id": clip_id,
                "hook_title": cand.hook_title,
                "score": cand.virality_score,
                "duration": dur,
                "layout": clip_layout.value,
                "captions": "Burned" if ass_path else "None",
                "vibe": cand.vibe,
                "director_fx": director_fx_str,
                "bgm": bgm_str,
                "path": output_video,
            }
        )

    # Summary
    console.print("\n")
    table = Table(
        title="Auto-Clip Execution Summary",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("ID", style="bold", width=4)
    table.add_column("Hook Title", width=30)
    table.add_column("Score", justify="center", width=7)
    table.add_column("Duration", justify="center", width=9)
    table.add_column("Layout", justify="center", width=13)
    table.add_column("Vibe", justify="center", width=15)
    table.add_column("Director FX", justify="center", width=22)
    table.add_column("BGM", justify="center", width=18)
    table.add_column("Captions", justify="center", width=9)
    table.add_column("Output File Path")

    for c in completed_clips:
        table.add_row(
            f"#{c['id']}",
            c["hook_title"],
            f"{c['score']:.1f}/10",
            f"{c['duration']:.1f}s",
            c["layout"],
            f"[magenta]{c['vibe']}[/magenta]",
            f"[bold yellow]{c['director_fx']}[/bold yellow]" if "Punch" in c['director_fx'] else "[dim]Standard[/dim]",
            f"[cyan]{c['bgm']}[/cyan]" if c['bgm'] != "Disabled" else "[dim]Disabled[/dim]",
            f"[green]{c['captions']}[/green]",
            str(c["path"]),
        )

    console.print(table)


@app.command("audio")
def ingest_audio(
    url: str = typer.Argument(..., help="YouTube video or stream VOD URL"),
    start_time: Optional[str] = typer.Option(
        None,
        "--start-time",
        "-s",
        help="Optional start timestamp",
    ),
    end_time: Optional[str] = typer.Option(
        None,
        "--end-time",
        "-e",
        help="Optional end timestamp",
    ),
):
    """Downloads audio-only (MP3) for Stage 2 Whisper transcription."""
    vod_id = extract_video_id(url)
    vod_downloads_dir = DOWNLOADS_DIR / vod_id
    vod_downloads_dir.mkdir(parents=True, exist_ok=True)
    with console.status("[bold green]Downloading MP3 audio..."):
        try:
            audio_path = download_audio_only(
                url=url,
                start_time=start_time,
                end_time=end_time,
                output_dir=vod_downloads_dir,
                prefix=f"{vod_id}_audio",
            )
            console.print(f"[green]✓ Audio saved to:[/green] {audio_path}")
        except Exception as exc:
            console.print(f"[bold red]Audio download failed:[/bold red] {exc}")
            raise typer.Exit(code=1)


@app.command("list")
def list_records(limit: int = typer.Option(20, "--limit", "-n", help="Number of records to show")):
    """Lists recent clips stored in the SQLite database."""
    records = list_clips(limit=limit)
    if not records:
        console.print("[dim]No clip records found in SQLite database.[/dim]")
        return

    table = Table(title="ClipEngine Database Records", show_header=True, header_style="bold blue")
    table.add_column("ID", style="bold", width=5)
    table.add_column("Status", width=12)
    table.add_column("Time Range", width=18)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Hook Title", width=30)
    table.add_column("Output Path")

    for rec in records:
        status_color = "green" if rec["status"] == "completed" else "yellow"
        if "failed" in rec["status"]:
            status_color = "red"
        time_range = f"{rec['clip_start']:.1f}s - {rec['clip_end']:.1f}s"
        out = rec["output_path"] or "-"
        score_str = f"{rec['virality_score']:.1f}" if rec["virality_score"] is not None else "-"
        table.add_row(
            str(rec["id"]),
            f"[{status_color}]{rec['status']}[/{status_color}]",
            time_range,
            score_str,
            rec["hook_title"] or "-",
            str(out),
        )

    console.print(table)


@app.command("info")
def inspect_url(url: str = typer.Argument(..., help="YouTube URL to inspect")):
    """Inspects a video's metadata without downloading."""
    with console.status("[bold blue]Querying video info..."):
        try:
            meta = get_video_metadata(url)
            console.print(
                Panel.fit(
                    f"Title:    [bold]{meta['title']}[/bold]\n"
                    f"Uploader: {meta['uploader']}\n"
                    f"Duration: {meta['duration']}s\n"
                    f"Video ID: {meta['id']}",
                    title="VOD Metadata",
                )
            )
        except Exception as exc:
            console.print(f"[bold red]Metadata lookup failed:[/bold red] {exc}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
