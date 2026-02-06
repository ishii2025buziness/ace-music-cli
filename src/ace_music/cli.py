"""Interactive CLI for ACE-Step music generation."""

from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from prompt_toolkit import prompt as pt_prompt
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text

from ace_music.app import create_agent, create_backend
from ace_music.config import (
    AppConfig,
    get_config_path,
    load_config,
    save_config,
    AnthropicAgentConfig,
    OpenAIAgentConfig,
    LocalAgentConfig,
    VastAIConfig,
    RunPodConfig,
    LocalConfig,
)
from ace_music.models import GenerationParams, GenerationRequest, JobState
from ace_music.player import detect_player, play_audio

_MAIN_HELP = """\
ACE-Step Music Generator CLI - Generate music with AI on cloud GPUs.

[bold]Workflow:[/bold]
  1. backend search  → Find available GPU instances
  2. backend create  → Create and setup instance (ComfyUI + ACE-Step)
  3. backend connect → Start ComfyUI and establish connection
  4. generate        → Create music interactively

[bold]Quick start:[/bold]
  ace-music backend search --vram 12 --max-price 0.1
  ace-music backend create <OFFER_ID>
  ace-music backend connect
  ace-music generate

[bold]For automation (--json flag):[/bold]
  ace-music backend status --json
  ace-music backend create <ID> --json
  ace-music backend connect --json
"""

_BACKEND_HELP = """\
Manage vast.ai GPU instances for music generation.

[bold]Commands:[/bold]
  search   → Find available GPU offers (use --json for automation)
  create   → Create instance from offer ID (auto-installs ComfyUI + ACE-Step)
  status   → Check instance state (running/stopped)
  start    → Start a stopped instance
  stop     → Stop a running instance
  connect  → Start ComfyUI and verify connection

[bold]Typical flow:[/bold]
  search → create → (wait for setup) → connect → ready for generate
"""

_CONFIG_HELP = """\
Manage CLI configuration (backend, agent, output settings).

[bold]Commands:[/bold]
  show  → Display current configuration
  set   → Interactive configuration wizard
  init  → Create default config file

[bold]Config location:[/bold] ~/.config/ace-music/config.toml
"""

app = typer.Typer(
    add_completion=False,
    help=_MAIN_HELP,
    rich_markup_mode="rich",
)
config_app = typer.Typer(help=_CONFIG_HELP, rich_markup_mode="rich")
backend_app = typer.Typer(help=_BACKEND_HELP, rich_markup_mode="rich")
app.add_typer(config_app, name="config")
app.add_typer(backend_app, name="backend")
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

    if typer.confirm("  Edit prompt?", default=False):
        prompt_tags = pt_prompt("  > ", default=prompt_tags).strip()

    # Step 3: Lyrics
    wants_lyrics = typer.confirm("\n  Add lyrics?", default=True)
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
            if typer.confirm("  Edit lyrics?", default=False):
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

    if typer.confirm("  Adjust parameters?", default=False):
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

    if typer.confirm("  Play audio?", default=True):
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
    """Start an interactive music generation session.

    Requires: backend must be connected (run 'backend connect' first).

    Flow: describe music → AI generates style tags → add lyrics → generate audio.
    """
    config = load_config(config_path)
    asyncio.run(_interactive_session(config))


@app.command()
def version() -> None:
    """Show version."""
    console.print("ace-music-cli 0.1.0")


@app.command()
def doctor(
    fix: bool = typer.Option(False, "--fix", help="Attempt to install missing dependencies"),
) -> None:
    """Check dependencies and show setup instructions.

    Verifies: vastai CLI, API keys, audio player.
    Use --fix to auto-install missing Python packages.
    """
    import shutil
    import os

    all_ok = True

    console.print("[bold]Checking dependencies...[/bold]\n")

    # 1. vastai CLI
    vastai_path = shutil.which("vastai")
    if vastai_path:
        console.print("  [green]✓[/green] vastai CLI found")
    else:
        console.print("  [red]✗[/red] vastai CLI not found")
        console.print("    [dim]Fix: pip install vastai[/dim]")
        all_ok = False
        if fix:
            console.print("    [cyan]Installing vastai...[/cyan]")
            subprocess.run(["pip", "install", "vastai"], capture_output=True)
            if shutil.which("vastai"):
                console.print("    [green]✓ Installed[/green]")
            else:
                console.print("    [red]✗ Install failed[/red]")

    # 2. vastai API key
    vastai_key_path = Path.home() / ".config" / "vastai" / "vast_api_key"
    if vastai_key_path.exists():
        console.print("  [green]✓[/green] vastai API key configured")
    else:
        console.print("  [red]✗[/red] vastai API key not found")
        console.print("    [dim]Fix: vastai set api-key <YOUR_API_KEY>[/dim]")
        console.print("    [dim]Get key: https://console.vast.ai/account[/dim]")
        all_ok = False

    # 3. ANTHROPIC_API_KEY
    if os.environ.get("ANTHROPIC_API_KEY"):
        console.print("  [green]✓[/green] ANTHROPIC_API_KEY set")
    else:
        # Check config file
        config = load_config()
        if config.agent.anthropic.api_key:
            console.print("  [green]✓[/green] ANTHROPIC_API_KEY in config")
        else:
            console.print("  [yellow]![/yellow] ANTHROPIC_API_KEY not set")
            console.print("    [dim]Fix: export ANTHROPIC_API_KEY=<YOUR_KEY>[/dim]")
            console.print("    [dim]Or: ace-music config set (to save in config)[/dim]")
            all_ok = False

    # 4. Audio player (optional)
    player_found = None
    for player in ["ffplay", "mpv", "aplay"]:
        if shutil.which(player):
            player_found = player
            break

    if player_found:
        console.print(f"  [green]✓[/green] Audio player found ({player_found})")
    else:
        console.print("  [yellow]![/yellow] No audio player found (optional)")
        console.print("    [dim]Fix: apt install ffmpeg (for ffplay)[/dim]")

    # Summary
    console.print()
    if all_ok:
        console.print("[bold green]All dependencies OK![/bold green]")
        console.print("[dim]Run 'ace-music backend search' to get started.[/dim]")
    else:
        console.print("[bold yellow]Some dependencies missing.[/bold yellow]")
        console.print("[dim]Follow the fix instructions above.[/dim]")


# ============================================================================
# Config subcommands
# ============================================================================


@config_app.command("show")
def config_show(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.toml"
    ),
) -> None:
    """Show current configuration."""
    config = load_config(config_path)
    used_path = config_path or get_config_path()

    console.print(f"[dim]Config file: {used_path}[/dim]\n")

    # Backend section
    backend_table = Table(title="Backend", show_header=False, box=None, padding=(0, 2))
    backend_table.add_column(style="bold cyan")
    backend_table.add_column()

    backend_table.add_row("Type", f"[green]{config.backend.type}[/green]")

    if config.backend.type == "vastai":
        cfg = config.backend.vastai
        backend_table.add_row("Instance ID", cfg.instance_id or "[dim]not set[/dim]")
        backend_table.add_row("SSH Host", f"{cfg.ssh_user}@{cfg.ssh_host}:{cfg.ssh_port}")
        backend_table.add_row("ComfyUI Port", str(cfg.comfyui_port))
    elif config.backend.type == "runpod":
        cfg = config.backend.runpod
        backend_table.add_row("Pod ID", cfg.pod_id or "[dim]not set[/dim]")
        backend_table.add_row("API Key", "***" if cfg.api_key else "[dim]not set[/dim]")
    elif config.backend.type == "local":
        backend_table.add_row("ComfyUI URL", config.backend.local.comfyui_url)

    console.print(backend_table)
    console.print()

    # Agent section
    agent_table = Table(title="Agent", show_header=False, box=None, padding=(0, 2))
    agent_table.add_column(style="bold cyan")
    agent_table.add_column()

    agent_table.add_row("Type", f"[green]{config.agent.type}[/green]")

    if config.agent.type == "anthropic":
        agent_table.add_row("Model", config.agent.anthropic.model)
        has_key = bool(config.agent.anthropic.api_key)
        agent_table.add_row("API Key", "[green]set[/green]" if has_key else "[dim]not set (check ANTHROPIC_API_KEY)[/dim]")
    elif config.agent.type == "openai":
        agent_table.add_row("Model", config.agent.openai.model)
        has_key = bool(config.agent.openai.api_key)
        agent_table.add_row("API Key", "[green]set[/green]" if has_key else "[dim]not set[/dim]")
    elif config.agent.type == "local":
        agent_table.add_row("Model", config.agent.local.model)
        agent_table.add_row("Endpoint", config.agent.local.endpoint)

    console.print(agent_table)
    console.print()

    # Output section
    output_table = Table(title="Output", show_header=False, box=None, padding=(0, 2))
    output_table.add_column(style="bold cyan")
    output_table.add_column()
    output_table.add_row("Directory", config.output.directory)
    output_table.add_row("Format", config.output.format)
    output_table.add_row("Player", config.player.command)
    console.print(output_table)


@config_app.command("set")
def config_set(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.toml (default: ~/.config/ace-music/config.toml)"
    ),
) -> None:
    """Interactively configure backend and agent settings."""
    config = load_config(config_path)
    target_path = config_path or get_config_path()

    console.print(
        Panel.fit(
            "[bold cyan]ACE-Music Configuration Wizard[/bold cyan]",
            border_style="cyan",
        )
    )

    # --- Backend selection ---
    console.print("\n[bold]Select backend:[/bold]")
    console.print("  [1] vastai  - vast.ai GPU instance via SSH")
    console.print("  [2] local   - Local ComfyUI")
    console.print("  [3] runpod  - RunPod (not implemented yet)")

    backend_choice = pt_prompt(
        "  > ",
        default={"vastai": "1", "local": "2", "runpod": "3"}.get(config.backend.type, "1"),
    ).strip()

    backend_type = {"1": "vastai", "2": "local", "3": "runpod"}.get(backend_choice, "vastai")
    config.backend.type = backend_type  # type: ignore

    if backend_type == "vastai":
        console.print("\n[bold cyan]vast.ai settings:[/bold cyan]")
        config.backend.vastai.instance_id = pt_prompt(
            "  Instance ID: ", default=config.backend.vastai.instance_id
        ).strip()
        config.backend.vastai.ssh_host = pt_prompt(
            "  SSH Host: ", default=config.backend.vastai.ssh_host or "sshX.vast.ai"
        ).strip()
        config.backend.vastai.ssh_port = int(
            pt_prompt("  SSH Port: ", default=str(config.backend.vastai.ssh_port)).strip()
        )
        config.backend.vastai.ssh_user = pt_prompt(
            "  SSH User: ", default=config.backend.vastai.ssh_user
        ).strip()
        config.backend.vastai.comfyui_port = int(
            pt_prompt("  ComfyUI Port: ", default=str(config.backend.vastai.comfyui_port)).strip()
        )

    elif backend_type == "local":
        console.print("\n[bold cyan]Local ComfyUI settings:[/bold cyan]")
        config.backend.local.comfyui_url = pt_prompt(
            "  ComfyUI URL: ", default=config.backend.local.comfyui_url
        ).strip()

    elif backend_type == "runpod":
        console.print("\n[bold cyan]RunPod settings:[/bold cyan]")
        config.backend.runpod.pod_id = pt_prompt(
            "  Pod ID: ", default=config.backend.runpod.pod_id
        ).strip()
        api_key = pt_prompt(
            "  API Key: ", default="" if not config.backend.runpod.api_key else "***"
        ).strip()
        if api_key and api_key != "***":
            config.backend.runpod.api_key = api_key

    # --- Agent selection ---
    console.print("\n[bold]Select AI agent:[/bold]")
    console.print("  [1] anthropic - Claude (Anthropic)")
    console.print("  [2] openai    - GPT-4 (OpenAI)")
    console.print("  [3] local     - Local LLM (Ollama)")

    agent_choice = pt_prompt(
        "  > ",
        default={"anthropic": "1", "openai": "2", "local": "3"}.get(config.agent.type, "1"),
    ).strip()

    agent_type = {"1": "anthropic", "2": "openai", "3": "local"}.get(agent_choice, "anthropic")
    config.agent.type = agent_type  # type: ignore

    if agent_type == "anthropic":
        console.print("\n[bold cyan]Anthropic settings:[/bold cyan]")
        config.agent.anthropic.model = pt_prompt(
            "  Model: ", default=config.agent.anthropic.model
        ).strip()
        console.print("  [dim]API Key: Set ANTHROPIC_API_KEY env var or enter below[/dim]")
        api_key = pt_prompt(
            "  API Key (leave blank to use env): ",
            default="" if not config.agent.anthropic.api_key else "***",
        ).strip()
        if api_key and api_key != "***":
            config.agent.anthropic.api_key = api_key

    elif agent_type == "openai":
        console.print("\n[bold cyan]OpenAI settings:[/bold cyan]")
        config.agent.openai.model = pt_prompt(
            "  Model: ", default=config.agent.openai.model
        ).strip()
        api_key = pt_prompt(
            "  API Key: ", default="" if not config.agent.openai.api_key else "***"
        ).strip()
        if api_key and api_key != "***":
            config.agent.openai.api_key = api_key

    elif agent_type == "local":
        console.print("\n[bold cyan]Local LLM settings:[/bold cyan]")
        config.agent.local.model = pt_prompt(
            "  Model: ", default=config.agent.local.model
        ).strip()
        config.agent.local.endpoint = pt_prompt(
            "  Endpoint: ", default=config.agent.local.endpoint
        ).strip()

    # --- Output settings ---
    console.print("\n[bold]Output settings:[/bold]")
    config.output.directory = pt_prompt(
        "  Output directory: ", default=config.output.directory
    ).strip()

    # --- Save ---
    console.print()
    if typer.confirm(f"Save to {target_path}?", default=True):
        saved_path = save_config(config, target_path)
        console.print(f"\n[green]Configuration saved to {saved_path}[/green]")
    else:
        console.print("[yellow]Configuration not saved.[/yellow]")


@config_app.command("init")
def config_init(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to create config.toml"
    ),
) -> None:
    """Create a new config file with defaults."""
    target_path = config_path or get_config_path()

    if target_path.exists():
        if not typer.confirm(f"{target_path} already exists. Overwrite?", default=False):
            console.print("[yellow]Aborted.[/yellow]")
            return

    config = AppConfig()
    saved_path = save_config(config, target_path)
    console.print(f"[green]Created {saved_path}[/green]")
    console.print("[dim]Run 'ace-music config set' to configure.[/dim]")


# ============================================================================
# Backend subcommands
# ============================================================================


def _run_vastai(args: list[str], raw: bool = True) -> dict | list | str:
    """Run vastai CLI command and return parsed output."""
    cmd = ["vastai"] + args
    if raw:
        cmd.append("--raw")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"vastai command failed: {result.stderr}")

    if raw:
        return json.loads(result.stdout)
    return result.stdout


def _get_instance_info(instance_id: str) -> dict | None:
    """Get info for a specific instance."""
    instances = _run_vastai(["show", "instances"])
    if not isinstance(instances, list):
        return None

    for inst in instances:
        if str(inst.get("id")) == instance_id:
            return inst
    return None


@backend_app.command("status")
def backend_status(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.toml"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show backend instance status."""
    config = load_config(config_path)

    if config.backend.type != "vastai":
        if json_output:
            print(json.dumps({"error": f"Backend type '{config.backend.type}' not supported"}))
        else:
            console.print(f"[yellow]Backend type '{config.backend.type}' does not support status command.[/yellow]")
        return

    instance_id = config.backend.vastai.instance_id
    if not instance_id:
        if json_output:
            print(json.dumps({"error": "No instance_id configured"}))
        else:
            console.print("[red]No instance_id configured. Run 'ace-music config set' first.[/red]")
        return

    if not json_output:
        with console.status("[cyan]Fetching instance status..."):
            info = _get_instance_info(instance_id)
    else:
        info = _get_instance_info(instance_id)

    if info is None:
        if json_output:
            print(json.dumps({"error": "Instance not found", "instance_id": instance_id}))
        else:
            console.print(f"[red]Instance {instance_id} not found.[/red]")
        return

    state = info.get("cur_state", "unknown")

    if json_output:
        print(json.dumps({
            "instance_id": instance_id,
            "state": state,
            "gpu": info.get("gpu_name"),
            "gpu_ram_gb": info.get("gpu_ram", 0) // 1024,
            "cost_per_hour": round(info.get("dph_total", 0), 4),
            "ssh_host": info.get("ssh_host"),
            "ssh_port": info.get("ssh_port"),
        }))
        return

    # Build status table
    state_color = {"running": "green", "stopped": "yellow", "exited": "red"}.get(state, "white")

    table = Table(title=f"Instance {instance_id}", show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row("Status", f"[{state_color}]{state}[/{state_color}]")
    table.add_row("GPU", f"{info.get('gpu_name', 'N/A')} ({info.get('gpu_ram', 0) // 1024}GB)")
    table.add_row("Location", info.get("geolocation", "N/A"))
    table.add_row("Cost", f"${info.get('dph_total', 0):.3f}/hr")
    table.add_row("Disk", f"{info.get('disk_usage', 0):.1f} / {info.get('disk_space', 0):.1f} GB")
    table.add_row("SSH", f"ssh -p {info.get('ssh_port')} root@{info.get('ssh_host')}")

    console.print(table)

    if state == "stopped":
        console.print("\n[dim]Run 'ace-music backend start' to start the instance.[/dim]")


@backend_app.command("start")
def backend_start(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.toml"
    ),
) -> None:
    """Start the backend instance."""
    config = load_config(config_path)

    if config.backend.type != "vastai":
        console.print(f"[yellow]Backend type '{config.backend.type}' does not support start command.[/yellow]")
        return

    instance_id = config.backend.vastai.instance_id
    if not instance_id:
        console.print("[red]No instance_id configured.[/red]")
        return

    with console.status(f"[cyan]Starting instance {instance_id}..."):
        try:
            result = _run_vastai(["start", "instance", instance_id], raw=False)
            console.print(f"[green]Instance {instance_id} starting...[/green]")
            console.print(f"[dim]{result.strip()}[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to start: {e}[/red]")
            return

    console.print("\n[dim]Use 'ace-music backend status' to check when it's running.[/dim]")


@backend_app.command("stop")
def backend_stop(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.toml"
    ),
) -> None:
    """Stop the backend instance."""
    config = load_config(config_path)

    if config.backend.type != "vastai":
        console.print(f"[yellow]Backend type '{config.backend.type}' does not support stop command.[/yellow]")
        return

    instance_id = config.backend.vastai.instance_id
    if not instance_id:
        console.print("[red]No instance_id configured.[/red]")
        return

    if not typer.confirm(f"Stop instance {instance_id}?", default=False):
        console.print("[yellow]Aborted.[/yellow]")
        return

    with console.status(f"[cyan]Stopping instance {instance_id}..."):
        try:
            result = _run_vastai(["stop", "instance", instance_id], raw=False)
            console.print(f"[green]Instance {instance_id} stopped.[/green]")
            console.print(f"[dim]{result.strip()}[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to stop: {e}[/red]")


async def _wait_for_instance_running(instance_id: str, timeout: int = 120) -> bool:
    """Wait for instance to reach 'running' state."""
    import time
    start = time.time()
    while time.time() - start < timeout:
        info = _get_instance_info(instance_id)
        if info and info.get("cur_state") == "running":
            return True
        await asyncio.sleep(5)
    return False


async def _start_comfyui_via_ssh(cfg) -> bool:
    """Start ComfyUI on the remote instance via SSH."""
    # Command to start ComfyUI in background
    comfyui_cmd = (
        "cd /root/ComfyUI && "
        "nohup python3 main.py --listen 0.0.0.0 --port 8188 "
        "> /root/comfyui.log 2>&1 &"
    )

    proc = await asyncio.create_subprocess_exec(
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-p", str(cfg.ssh_port),
        f"{cfg.ssh_user}@{cfg.ssh_host}",
        comfyui_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()
    return proc.returncode == 0


async def _check_comfyui_ready(cfg, timeout: int = 60) -> bool:
    """Check if ComfyUI is responding via SSH tunnel."""
    import httpx

    # Start SSH tunnel
    local_port = cfg.comfyui_port
    tunnel_proc = await asyncio.create_subprocess_exec(
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-N",
        "-L", f"{local_port}:localhost:{cfg.comfyui_port}",
        "-p", str(cfg.ssh_port),
        f"{cfg.ssh_user}@{cfg.ssh_host}",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    try:
        async with httpx.AsyncClient() as client:
            for _ in range(timeout // 2):
                try:
                    resp = await client.get(f"http://localhost:{local_port}/", timeout=5.0)
                    if resp.status_code == 200:
                        return True
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                await asyncio.sleep(2)
        return False
    finally:
        # Kill the test tunnel
        if tunnel_proc.returncode is None:
            tunnel_proc.terminate()
            await tunnel_proc.wait()


async def _connect_backend(config, quiet: bool = False) -> dict:
    """Full connection flow: start instance, launch ComfyUI, verify.

    Returns dict with status info for JSON output.
    """
    cfg = config.backend.vastai
    instance_id = cfg.instance_id

    # Step 1: Check instance state
    if not quiet:
        console.print("[bold]Step 1:[/bold] Checking instance status...")
    info = _get_instance_info(instance_id)

    if info is None:
        if not quiet:
            console.print(f"[red]Instance {instance_id} not found.[/red]")
        return {"status": "error", "error": "Instance not found"}

    state = info.get("cur_state", "unknown")

    if state == "stopped":
        if not quiet:
            console.print(f"  Instance is [yellow]stopped[/yellow]. Starting...")
        _run_vastai(["start", "instance", instance_id], raw=False)

        if not quiet:
            with console.status("[cyan]  Waiting for instance to start..."):
                started = await _wait_for_instance_running(instance_id, timeout=120)
        else:
            started = await _wait_for_instance_running(instance_id, timeout=120)

        if not started:
            if not quiet:
                console.print("[red]  Timeout waiting for instance to start.[/red]")
            return {"status": "error", "error": "Timeout waiting for instance"}

        if not quiet:
            console.print("  Instance is now [green]running[/green].")
        await asyncio.sleep(5)

    elif state == "running":
        if not quiet:
            console.print(f"  Instance is already [green]running[/green].")
    else:
        if not quiet:
            console.print(f"  Instance state: [yellow]{state}[/yellow]. Waiting...")
            with console.status("[cyan]  Waiting for instance..."):
                started = await _wait_for_instance_running(instance_id, timeout=120)
        else:
            started = await _wait_for_instance_running(instance_id, timeout=120)

        if not started:
            if not quiet:
                console.print(f"[red]  Instance stuck in '{state}' state.[/red]")
            return {"status": "error", "error": f"Instance stuck in {state}"}

    # Step 2: Start ComfyUI
    if not quiet:
        console.print("\n[bold]Step 2:[/bold] Starting ComfyUI on remote instance...")
        with console.status("[cyan]  Launching ComfyUI via SSH..."):
            ssh_ok = await _start_comfyui_via_ssh(cfg)
    else:
        ssh_ok = await _start_comfyui_via_ssh(cfg)

    if not quiet:
        if ssh_ok:
            console.print("  ComfyUI launch command sent.")
        else:
            console.print("[yellow]  Warning: SSH command may have failed.[/yellow]")

    # Step 3: Wait for ComfyUI to be ready
    if not quiet:
        console.print("\n[bold]Step 3:[/bold] Waiting for ComfyUI to respond...")
        with console.status("[cyan]  Checking ComfyUI via SSH tunnel..."):
            comfyui_ready = await _check_comfyui_ready(cfg, timeout=90)
    else:
        comfyui_ready = await _check_comfyui_ready(cfg, timeout=90)

    if not comfyui_ready:
        if not quiet:
            console.print("[red]  ComfyUI did not respond in time.[/red]")
            console.print(f"[dim]  Check logs: ssh -p {cfg.ssh_port} {cfg.ssh_user}@{cfg.ssh_host} 'tail /root/comfyui.log'[/dim]")
        return {"status": "error", "error": "ComfyUI not responding"}

    if not quiet:
        console.print("  ComfyUI is [green]ready[/green]!")
        console.print()
        console.print(Panel.fit(
            "[bold green]Backend connected![/bold green]\n"
            f"Instance: {instance_id}\n"
            f"ComfyUI: http://localhost:{cfg.comfyui_port}\n\n"
            "[dim]Run 'ace-music generate' to create music.[/dim]",
            border_style="green",
        ))

    return {
        "status": "connected",
        "instance_id": instance_id,
        "comfyui_url": f"http://localhost:{cfg.comfyui_port}",
        "ssh_host": cfg.ssh_host,
        "ssh_port": cfg.ssh_port,
    }


@backend_app.command("connect")
def backend_connect(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.toml"
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Start instance if stopped, launch ComfyUI, and verify connection.

    Establishes SSH tunnel to ComfyUI on port 8188.
    Run this before 'generate'.

    Next step: generate
    """
    config = load_config(config_path)

    if config.backend.type != "vastai":
        if json_output:
            print(json.dumps({"error": f"Backend type '{config.backend.type}' not supported"}))
        else:
            console.print(f"[yellow]Backend type '{config.backend.type}' does not support connect command.[/yellow]")
            console.print("[dim]For local backend, just ensure ComfyUI is running.[/dim]")
        return

    instance_id = config.backend.vastai.instance_id
    if not instance_id:
        if json_output:
            print(json.dumps({"error": "No instance_id configured"}))
        else:
            console.print("[red]No instance_id configured. Run 'ace-music config set' first.[/red]")
        return

    result = asyncio.run(_connect_backend(config, quiet=json_output))
    if json_output:
        print(json.dumps(result))


# Onstart script for ComfyUI + ACE-Step setup
# Creates progress markers in /root/.ace-setup/ for tracking
_COMFYUI_ONSTART_SCRIPT = """\
#!/bin/bash
set -e

PROGRESS_DIR=/root/.ace-setup
mkdir -p $PROGRESS_DIR
echo "started" > $PROGRESS_DIR/status

# Step 1: Install dependencies
echo "step1_deps" > $PROGRESS_DIR/status
apt-get update && apt-get install -y git wget curl python3 python3-pip python3-venv ffmpeg

# Step 2: Clone ComfyUI
echo "step2_comfyui" > $PROGRESS_DIR/status
if [ ! -d /root/ComfyUI ]; then
    cd /root
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    pip install -r requirements.txt
fi

# Step 3: Install ACE-Step custom node
echo "step3_customnode" > $PROGRESS_DIR/status
if [ ! -d /root/ComfyUI/custom_nodes/ComfyUI_ACE-Step ]; then
    cd /root/ComfyUI/custom_nodes
    git clone https://github.com/billwuhao/ComfyUI_ACE-Step.git
    cd ComfyUI_ACE-Step
    pip install -r requirements.txt
fi

# Step 4: Download ACE-Step models
echo "step4_models" > $PROGRESS_DIR/status
MODELS_DIR=/root/ComfyUI/models/ace_step
if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A $MODELS_DIR 2>/dev/null)" ]; then
    mkdir -p "$MODELS_DIR"
    cd "$MODELS_DIR"
    pip install huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ACE-Step/ACE-Step-v1-3.5B', local_dir='.', local_dir_use_symlinks=False)
"
fi

# Done
echo "complete" > $PROGRESS_DIR/status
echo "Setup complete!"
"""

# Setup status descriptions
_SETUP_STATUS_DESC = {
    "started": "Starting setup...",
    "step1_deps": "Installing system dependencies...",
    "step2_comfyui": "Installing ComfyUI...",
    "step3_customnode": "Installing ACE-Step custom node...",
    "step4_models": "Downloading ACE-Step models (this may take a while)...",
    "complete": "Setup complete!",
}


async def _check_setup_status_ssh(cfg) -> str | None:
    """Check setup progress via SSH. Returns status string or None if not accessible."""
    proc = await asyncio.create_subprocess_exec(
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        "-p", str(cfg.ssh_port),
        f"{cfg.ssh_user}@{cfg.ssh_host}",
        "cat /root/.ace-setup/status 2>/dev/null || echo 'not_started'",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode == 0:
        return stdout.decode().strip()
    return None


async def _wait_for_setup_complete(cfg, timeout: int = 1200, quiet: bool = False) -> bool:
    """Wait for setup to complete, showing progress. Returns True if successful."""
    import time
    start = time.time()
    last_status = ""

    if not quiet:
        console.print("\n[bold]Waiting for setup to complete...[/bold]")

    while time.time() - start < timeout:
        # First check if instance is running
        info = _get_instance_info(cfg.instance_id)
        if not info:
            if not quiet:
                console.print("  [yellow]Instance not found, waiting...[/yellow]")
            await asyncio.sleep(10)
            continue

        state = info.get("cur_state", "unknown")
        if state != "running":
            if not quiet:
                console.print(f"  [yellow]Instance state: {state}, waiting...[/yellow]")
            await asyncio.sleep(10)
            continue

        # Check setup status via SSH
        status = await _check_setup_status_ssh(cfg)

        if status is None:
            if not quiet:
                console.print("  [dim]Waiting for SSH access...[/dim]")
            await asyncio.sleep(10)
            continue

        if status != last_status:
            if status == "complete":
                if not quiet:
                    desc = _SETUP_STATUS_DESC.get(status, f"Status: {status}")
                    console.print(f"  [green]{desc}[/green]")
                return True
            else:
                if not quiet:
                    desc = _SETUP_STATUS_DESC.get(status, f"Status: {status}")
                    console.print(f"  [cyan]{desc}[/cyan]")
            last_status = status

        await asyncio.sleep(15)

    if not quiet:
        console.print("[red]Setup timed out.[/red]")
    return False


@backend_app.command("search")
def backend_search(
    min_vram: int = typer.Option(12, "--vram", help="Minimum GPU VRAM in GB"),
    max_price: float = typer.Option(0.2, "--max-price", help="Maximum price in $/hr"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
) -> None:
    """Search for available GPU instances on vast.ai.

    Returns offer IDs that can be used with 'backend create'.
    ACE-Step requires at least 12GB VRAM (RTX 3060 or better).

    Next step: backend create <OFFER_ID>
    """
    # Fetch more results and filter in Python (vastai query has issues with some filters)
    query = "num_gpus=1 rentable=true"

    with console.status("[cyan]Searching for offers..."):
        try:
            all_offers = _run_vastai(
                ["search", "offers", query, "-o", "dph_total", "--limit", "100"]
            )
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
            return

    # Filter by VRAM and price
    min_vram_mb = min_vram * 1024
    offers = [
        o for o in all_offers
        if o.get("gpu_ram", 0) >= min_vram_mb
        and o.get("dph_total", 999) <= max_price
        and o.get("verification") == "verified"
    ][:limit]

    if not offers:
        console.print(f"[yellow]No offers found (VRAM >= {min_vram}GB, <= ${max_price}/hr).[/yellow]")
        console.print("[dim]Try increasing --max-price or decreasing --vram.[/dim]")
        return

    table = Table(title=f"Available Offers (VRAM >= {min_vram}GB, <= ${max_price}/hr)")
    table.add_column("ID", style="cyan")
    table.add_column("GPU")
    table.add_column("VRAM")
    table.add_column("$/hr", justify="right")
    table.add_column("Location")
    table.add_column("Reliability")

    for offer in offers:
        table.add_row(
            str(offer["id"]),
            offer.get("gpu_name", "N/A"),
            f"{offer.get('gpu_ram', 0) // 1024}GB",
            f"${offer.get('dph_total', 0):.3f}",
            offer.get("geolocation", "N/A")[:20],
            f"{offer.get('reliability2', 0) * 100:.1f}%",
        )

    console.print(table)
    console.print("\n[dim]Use 'ace-music backend create <ID>' to create an instance.[/dim]")


@backend_app.command("create")
def backend_create(
    offer_id: int = typer.Argument(..., help="Offer ID from 'backend search'"),
    disk: int = typer.Option(50, "--disk", help="Disk size in GB"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for setup to complete"),
    json_output: bool = typer.Option(False, "--json", help="JSON output (skips confirmation)"),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.toml"
    ),
) -> None:
    """Create a new vast.ai instance and auto-install ComfyUI + ACE-Step.

    Uses OFFER_ID from 'backend search'. By default waits for setup (10-20 min).
    Config is auto-updated with new instance details.

    Next step: backend connect
    """
    config = load_config(config_path)

    # Confirm before creating (costs money) - skip for JSON mode
    if not json_output:
        console.print(f"[bold]Creating instance from offer {offer_id}[/bold]")
        console.print(f"  Disk: {disk}GB")
        console.print(f"  Image: nvidia/cuda:12.4.1-base-ubuntu22.04")
        console.print()

        if not typer.confirm("Create this instance? (will incur charges)", default=False):
            console.print("[yellow]Aborted.[/yellow]")
            return

    # Create instance
    if not json_output:
        with console.status("[cyan]Creating instance..."):
            result = _create_instance(offer_id, disk)
    else:
        result = _create_instance(offer_id, disk)

    if result.get("error"):
        if json_output:
            print(json.dumps({"error": result["error"]}))
        else:
            console.print(f"[red]Failed to create instance: {result['error']}[/red]")
        return

    new_instance_id = result["instance_id"]

    if not json_output:
        console.print(f"[green]Instance created![/green] ID: {new_instance_id}")

    # Get instance details (SSH info)
    info = None
    for _ in range(10):
        info = _get_instance_info(str(new_instance_id))
        if info and info.get("ssh_host"):
            break
        import time
        time.sleep(3)

    if info:
        # Update config with new instance info
        config.backend.type = "vastai"
        config.backend.vastai.instance_id = str(new_instance_id)
        config.backend.vastai.ssh_host = info.get("ssh_host", "")
        config.backend.vastai.ssh_port = info.get("ssh_port", 22)

        target_path = config_path or get_config_path()
        save_config(config, target_path)

        if not json_output:
            console.print(f"\n[green]Config updated:[/green] {target_path}")
            console.print(f"  instance_id: {new_instance_id}")
            console.print(f"  ssh: ssh -p {info.get('ssh_port')} root@{info.get('ssh_host')}")

        # Wait for setup if requested
        if wait:
            setup_ok = asyncio.run(_wait_for_setup_complete(config.backend.vastai, quiet=json_output))

            if json_output:
                print(json.dumps({
                    "status": "ready" if setup_ok else "setup_timeout",
                    "instance_id": str(new_instance_id),
                    "ssh_host": info.get("ssh_host"),
                    "ssh_port": info.get("ssh_port"),
                }))
            else:
                if setup_ok:
                    console.print("\n[green]Instance is ready![/green]")
                    console.print("[dim]Run 'ace-music backend connect' to start ComfyUI.[/dim]")
                else:
                    console.print("\n[yellow]Setup may still be in progress.[/yellow]")
                    console.print("[dim]Run 'ace-music backend connect' to check.[/dim]")
        else:
            if json_output:
                print(json.dumps({
                    "status": "created",
                    "instance_id": str(new_instance_id),
                    "ssh_host": info.get("ssh_host"),
                    "ssh_port": info.get("ssh_port"),
                }))
            else:
                console.print()
                console.print("[bold yellow]Note:[/bold yellow] ComfyUI + ACE-Step setup will run automatically.")
                console.print("This may take 10-20 minutes for model download.")
                console.print("\n[dim]Run 'ace-music backend status' to check instance state.[/dim]")
                console.print("[dim]Run 'ace-music backend connect' once setup is complete.[/dim]")


def _create_instance(offer_id: int, disk: int) -> dict:
    """Create a vast.ai instance. Returns {"instance_id": ...} or {"error": ...}."""
    try:
        result = subprocess.run(
            [
                "vastai", "create", "instance", str(offer_id),
                "--image", "nvidia/cuda:12.4.1-base-ubuntu22.04",
                "--disk", str(disk),
                "--ssh",
                "--direct",
                "--env", "-p 8188:8188",
                "--onstart-cmd", _COMFYUI_ONSTART_SCRIPT,
                "--raw",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {"error": result.stderr}

        data = json.loads(result.stdout)
        return {"instance_id": data.get("new_contract")}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    app()
