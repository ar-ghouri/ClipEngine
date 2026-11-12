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
from modules.compositor import LayoutMode, compose_clip, probe_video
from modules.ingest import (
    download_audio_only,
    download_target_clip,
    get_video_metadata,
    parse_timestamp,
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
        LayoutMode.GAMING_SPLIT,
        "--layout",
        "-l",
        help="Output layout: 'gaming_split' (webcam + gameplay) or 'center_crop' (9:16 crop)",
    ),
    cam_bbox: str = typer.Option(
        "0.0,0.0,0.25,0.25",
        "--cam-bbox",
        help="Normalized webcam box 'x,y,w,h' for gaming_split (default top-left quadrant)",
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
    with console.status(f"[bold green]Downloading section ({duration:.1f}s) via yt-dlp..."):
        try:
            raw_path = download_target_clip(
                url=url,
                start_time=start_sec,
                end_time=end_sec,
                output_dir=DOWNLOADS_DIR,
                prefix=f"clip_{clip_id}",
            )
        except Exception as exc:
            update_clip_record(clip_id, status="failed_ingest")
            console.print(f"[bold red]Ingest Failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    console.print(f"[green]✓ Ingest complete:[/green] {raw_path.name}")
    update_clip_record(clip_id, status="compositing")

    # 4. GPU-Accelerated Compositing
    parsed_bbox = parse_bbox_string(cam_bbox)
    with console.status(f"[bold magenta]Rendering 1080x1920 using NVIDIA NVENC ({layout.value})..."):
        try:
            output_file = compose_clip(
                input_video=raw_path,
                layout=layout,
                cam_bbox=parsed_bbox,
            )
        except Exception as exc:
            update_clip_record(clip_id, status="failed_composite")
            console.print(f"[bold red]Compositing Failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    # 5. Finalize database state
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
    table.add_row("Layout", layout.value)
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
        LayoutMode.GAMING_SPLIT,
        "--layout",
        "-l",
        help="Output layout: 'gaming_split' (webcam + gameplay) or 'center_crop' (9:16 crop)",
    ),
    cam_bbox: str = typer.Option(
        "0.0,0.0,0.25,0.25",
        "--cam-bbox",
        help="Normalized webcam coordinates 'x,y,w,h' for gaming_split (default top-left quadrant)",
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

    # Step 1: Download Audio Only
    console.rule("[bold cyan]Step 1: Smart Audio-Only Ingest[/bold cyan]")
    with console.status("[bold green]Downloading audio stream (MP3)..."):
        try:
            audio_path = download_audio_only(
                url=url,
                output_dir=DOWNLOADS_DIR,
                prefix=f"vod_audio_{meta['id']}",
            )
            audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
            console.print(
                f"[green]✓ Audio ingest complete:[/green] {audio_path.name} [dim]({audio_size_mb:.2f} MB)[/dim]"
            )
        except Exception as exc:
            console.print(f"[bold red]Audio Ingest Failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    # Step 2: GPU Transcription & Audio Peak Analysis
    console.rule("[bold cyan]Step 2: RTX 4060 Whisper Transcription & Loudness Analysis[/bold cyan]")
    w_model = whisper_model or WHISPER_MODEL
    with console.status(f"[bold magenta]Transcribing audio via faster-whisper ({w_model} on CUDA)..."):
        try:
            transcript = transcribe_audio(audio_path, model_size=w_model)
            audio_peaks = compute_audio_loudness_peaks(audio_path, top_k=10)
        except Exception as exc:
            console.print(f"[bold red]Transcription / Peak Analysis Failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    console.print(
        f"[green]✓ Transcribed {len(transcript.segments)} segments across {transcript.duration:.1f}s.[/green]"
    )
    if audio_peaks:
        peak_summary = ", ".join(f"{p.formatted_time} ({p.rms_db:.1f}dB)" for p in audio_peaks[:5])
        console.print(f"[dim]Top energy spikes: {peak_summary}[/dim]")

    # Step 3: Gemini Viral Highlight Detection
    console.rule("[bold cyan]Step 3: Gemini 2.5 Flash Viral Highlight Detection[/bold cyan]")
    with console.status("[bold yellow]Analyzing retention drivers with Gemini 2.5 Flash..."):
        try:
            analysis = detect_highlights(
                transcript=transcript,
                audio_peaks=audio_peaks,
                min_duration=min_duration,
                max_duration=max_duration,
                top_k=max_clips,
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
        console.print(
            f"  [cyan]{idx}.[/cyan] [bold]'{cand.hook_title}'[/bold] "
            f"({cand.start_time:.1f}s -> {cand.end_time:.1f}s, {dur:.1f}s, Score: [bold yellow]{cand.virality_score}/10.0[/bold yellow])\n"
            f"     [dim]Reason: {cand.reasoning}[/dim]"
        )

    # Step 4: Targeted Video Ingest & NVENC Compositing
    console.rule("[bold cyan]Step 4: Smart Targeted Ingest & GPU Compositing[/bold cyan]")
    parsed_bbox = parse_bbox_string(cam_bbox)
    completed_clips: List[Dict[str, Any]] = []

    for idx, cand in enumerate(analysis.candidates, 1):
        dur = cand.end_time - cand.start_time
        console.print(
            f"\n[bold]Rendering Clip {idx}/{len(analysis.candidates)}:[/bold] [yellow]'{cand.hook_title}'[/yellow]"
        )

        # 4a. Register record in SQLite
        clip_id = create_clip_record(
            vod_url=url,
            clip_start=cand.start_time,
            clip_end=cand.end_time,
            hook_title=cand.hook_title,
            virality_score=cand.virality_score,
            status="ingesting",
        )

        # 4b. Targeted section download ONLY
        with console.status(f"  [cyan]Targeted download of section ({cand.start_time:.1f}s -> {cand.end_time:.1f}s)..."):
            try:
                raw_clip = download_target_clip(
                    url=url,
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                    output_dir=DOWNLOADS_DIR,
                    prefix=f"clip_{clip_id}",
                )
            except Exception as exc:
                update_clip_record(clip_id, status="failed_ingest")
                console.print(f"  [red]Download failed:[/red] {exc}")
                continue

        update_clip_record(clip_id, status="compositing")

        # 4c. NVENC Compositing
        with console.status(f"  [magenta]Compositing via h264_nvenc ({layout.value})..."):
            try:
                output_video = compose_clip(
                    input_video=raw_clip,
                    layout=layout,
                    cam_bbox=parsed_bbox,
                )
            except Exception as exc:
                update_clip_record(clip_id, status="failed_composite")
                console.print(f"  [red]Compositing failed:[/red] {exc}")
                continue

        update_clip_record(clip_id, status="completed", output_path=str(output_video))
        completed_clips.append(
            {
                "id": clip_id,
                "hook_title": cand.hook_title,
                "score": cand.virality_score,
                "duration": dur,
                "path": output_video,
            }
        )
        console.print(f"  [green]✓ Output ready:[/green] {output_video.name}")

    # Summary
    console.print("\n")
    table = Table(
        title="Auto-Clip Execution Summary",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("ID", style="bold", width=5)
    table.add_column("Hook Title", width=35)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Duration", justify="center", width=10)
    table.add_column("Output File Path")

    for c in completed_clips:
        table.add_row(
            f"#{c['id']}",
            c["hook_title"],
            f"{c['score']:.1f}/10",
            f"{c['duration']:.1f}s",
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
    with console.status("[bold green]Downloading MP3 audio..."):
        try:
            audio_path = download_audio_only(
                url=url,
                start_time=start_time,
                end_time=end_time,
                output_dir=DOWNLOADS_DIR,
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
