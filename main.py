"""
ClipEngine CLI Entry Point
A local, zero-cost video clipping engine optimized for NVIDIA RTX 4060 on Windows.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import (
    DEFAULT_WEBCAM_BBOX,
    DOWNLOADS_DIR,
    OUTPUTS_DIR,
    resolve_ffmpeg_binaries,
)
from database import (
    create_clip_record,
    get_clip_record,
    init_db,
    list_clips,
    update_clip_record,
)
from modules.compositor import LayoutMode, compose_clip, probe_video
from modules.ingest import (
    download_audio_only,
    download_target_clip,
    get_video_metadata,
    parse_timestamp,
)

# Initialize Typer App and Rich Console
app = typer.Typer(
    name="clipengine",
    help="ClipEngine: Local, zero-cost video clipping engine powered by NVIDIA NVENC.",
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
    """Ensure database and ffmpeg paths are ready before any command executes."""
    init_db()
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
        help="Virality score (0-100)",
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
    table.add_column("Hook Title", width=25)
    table.add_column("Output Path")

    for rec in records:
        status_color = "green" if rec["status"] == "completed" else "yellow"
        if "failed" in rec["status"]:
            status_color = "red"
        time_range = f"{rec['clip_start']:.1f}s - {rec['clip_end']:.1f}s"
        out = rec["output_path"] or "-"
        table.add_row(
            str(rec["id"]),
            f"[{status_color}]{rec['status']}[/{status_color}]",
            time_range,
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

