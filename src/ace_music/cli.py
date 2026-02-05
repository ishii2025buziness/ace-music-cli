"""Interactive CLI for ACE-Step music generation."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.shortcuts import confirm
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text

from ace_music.app import create_agent, create_backend
from ace_music.config import AppConfig, load_config
from ace_music.models import GenerationParams, GenerationRequest, JobState
from ace_music.player import detect_player, play_audio

app = typer.Typer(add_completion=False, help="ACE-Step Music Generator CLI")
console = Console()


async def _interactive_session(config: AppConfig) -> None:
    backend = create_backend(config)
    agent = create_agent(config)
    output_dir = Path(config.output.directory)

    console.print(
        Panel.fit(
            "[bold cyan]ACE-Step Music Generator[/bold cyan]\n"
            f"Backend: [green]{config.backend.type}[/green]  "
            f"Agent: [green]{config.agent.type}[/green]",
            border_style="cyan",
        )
    )

    # Connect to backend
    with console.status("[bold yellow]Connecting to backend..."):
        try:
            await backend.connect()
        except (ConnectionError, Exception) as e:
            console.print(f"[bold red]Connection failed:[/bold red] {e}")
            return

    console.print("[green]Connected.[/green]\n")

    try:
        while True:
            request = await _build_request(agent, config)
            if request is None:
                break

            audio_path = await _generate_and_download(backend, request, output_dir)
            if audio_path is None:
                continue

            await _post_generation(audio_path, config)

            # Next action
            action = _ask_next_action()
            if action == "quit":
                break
            elif action == "retry":
                continue
            # "new" falls through to next loop iteration

    finally:
        with console.status("[yellow]Disconnecting..."):
            await backend.disconnect()
        console.print("[dim]Disconnected.[/dim]")


async def _build_request(agent, config: AppConfig) -> GenerationRequest | None:
    """Interactive flow to build a generation request with AI assistance."""

    # Step 1: Get user's idea
    console.print("[bold]What kind of music do you want to create?[/bold]")
    user_input = pt_prompt("  > ").strip()
    if not user_input:
        return None

    # Step 2: AI generates prompt tags
    with console.status("[cyan]Generating style tags..."):
        try:
            prompt_tags = await agent.assist_prompt(user_input)
        except Exception as e:
            console.print(f"[yellow]Agent error: {e}[/yellow]")
            prompt_tags = user_input

    console.print(f"\n[bold]Prompt:[/bold] {prompt_tags}")

    if confirm("  Edit prompt?", default=False):
        prompt_tags = pt_prompt("  > ", default=prompt_tags).strip()

    # Step 3: Lyrics
    wants_lyrics = confirm("\n  Add lyrics?", default=True)
    lyrics = ""
    if wants_lyrics:
        console.print("[dim]  Describe the lyrics theme (or type lyrics directly):[/dim]")
        lyrics_input = pt_prompt("  > ").strip()

        if any(tag in lyrics_input for tag in ["[Verse]", "[Chorus]", "[Intro]"]):
            lyrics = lyrics_input
        else:
            with console.status("[cyan]Writing lyrics..."):
                try:
                    lyrics = await agent.assist_lyrics(lyrics_input, prompt_tags)
                except Exception as e:
                    console.print(f"[yellow]Agent error: {e}[/yellow]")
                    lyrics = lyrics_input

            console.print(f"\n[bold]Lyrics:[/bold]\n{lyrics}")
            if confirm("  Edit lyrics?", default=False):
                console.print("[dim]  (Enter multi-line lyrics, press Ctrl+D when done)[/dim]")
                lines = []
                try:
                    while True:
                        lines.append(pt_prompt("  "))
                except EOFError:
                    pass
                if lines:
                    lyrics = "\n".join(lines)

    # Step 4: Parameters
    console.print()
    with console.status("[cyan]Suggesting parameters..."):
        try:
            params = await agent.suggest_params(prompt_tags)
        except Exception:
            params = GenerationParams()

    _show_params(params)

    if confirm("  Adjust parameters?", default=False):
        params = _edit_params(params)

    return GenerationRequest(
        prompt=prompt_tags,
        lyrics=lyrics,
        params=params,
    )


async def _generate_and_download(
    backend, request: GenerationRequest, output_dir: Path
) -> Path | None:
    """Submit generation, poll with progress bar, download result."""

    console.print()

    # Queue
    with console.status("[bold yellow]Submitting to backend..."):
        try:
            job_id = await backend.generate(request)
        except Exception as e:
            console.print(f"[bold red]Generation failed:[/bold red] {e}")
            return None

    # Poll with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=100)

        while True:
            status = await backend.poll_status(job_id)

            if status.state == JobState.COMPLETED:
                progress.update(task, completed=100)
                break
            elif status.state == JobState.ERROR:
                progress.stop()
                console.print(f"[bold red]Error:[/bold red] {status.error_message}")
                return None
            else:
                progress.update(task, completed=int(status.progress * 100))

            await asyncio.sleep(3)

    # Download
    with console.status("[bold green]Downloading audio..."):
        try:
            audio_path = await backend.download(job_id, output_dir)
        except Exception as e:
            console.print(f"[bold red]Download failed:[/bold red] {e}")
            return None

    # Terminal bell notification
    console.bell()
    console.print(f"\n[bold green]Done![/bold green] Saved to: {audio_path}")
    return audio_path


async def _post_generation(audio_path: Path, config: AppConfig) -> None:
    """Offer to play the generated audio."""
    player_cmd = config.player.command

    if detect_player(player_cmd) is None:
        console.print("[dim]No audio player found. Install ffplay, mpv, or aplay.[/dim]")
        return

    if confirm("  Play audio?", default=True):
        console.print("[dim]  Playing... (wait for completion or Ctrl+C to stop)[/dim]")
        try:
            await play_audio(audio_path, player_cmd)
        except KeyboardInterrupt:
            pass


def _ask_next_action() -> str:
    """Ask what to do next."""
    console.print("\n[bold]Next action:[/bold]")
    console.print("  [1] New song")
    console.print("  [2] Retry with different params")
    console.print("  [3] Quit")

    choice = pt_prompt("  > ", default="1").strip()
    return {"1": "new", "2": "retry", "3": "quit"}.get(choice, "new")


def _show_params(params: GenerationParams) -> None:
    """Display current generation parameters."""
    table = Table(title="Generation Parameters", show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Duration", f"{params.duration}s")
    table.add_row("Steps", str(params.steps))
    table.add_row("Guidance", str(params.guidance_scale))
    table.add_row("Scheduler", params.scheduler.value)
    table.add_row("Seed", str(params.seed))
    console.print(table)


def _edit_params(params: GenerationParams) -> GenerationParams:
    """Let user edit specific parameters."""
    raw = pt_prompt("  Duration (s): ", default=str(params.duration)).strip()
    params.duration = float(raw)
    raw = pt_prompt("  Steps: ", default=str(params.steps)).strip()
    params.steps = int(raw)
    raw = pt_prompt("  Seed (-1=random): ", default=str(params.seed)).strip()
    params.seed = int(raw)
    return params


@app.command()
def generate(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.toml"
    ),
) -> None:
    """Start an interactive music generation session."""
    config = load_config(config_path)
    asyncio.run(_interactive_session(config))


@app.command()
def version() -> None:
    """Show version."""
    console.print("ace-music-cli 0.1.0")


if __name__ == "__main__":
    app()
