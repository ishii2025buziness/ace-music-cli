"""Audio playback with automatic player detection."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path


_PLAYER_COMMANDS: list[tuple[str, list[str]]] = [
    ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ("mpv", ["mpv", "--no-video"]),
    ("aplay", ["aplay"]),
    ("paplay", ["paplay"]),
]


def detect_player(preferred: str = "auto") -> list[str] | None:
    """Find an available audio player. Returns the command list or None."""
    if preferred != "auto":
        if shutil.which(preferred):
            # Use the preferred player with the path as last arg
            return [preferred]
        return None

    for name, cmd in _PLAYER_COMMANDS:
        if shutil.which(name):
            return cmd

    return None


async def play_audio(file_path: Path, player_cmd: str = "auto") -> bool:
    """Play an audio file. Returns True if playback succeeded."""
    cmd = detect_player(player_cmd)
    if cmd is None:
        return False

    proc = await asyncio.create_subprocess_exec(
        *cmd, str(file_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode == 0
