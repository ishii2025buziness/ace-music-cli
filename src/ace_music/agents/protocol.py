"""Agent protocol — the contract all AI assistants must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ace_music.models import GenerationParams


@runtime_checkable
class MusicAgent(Protocol):
    """Abstract interface for an AI agent that assists with music creation."""

    async def assist_prompt(self, user_input: str) -> str:
        """Turn a natural-language description into ACE-Step style tags."""
        ...

    async def assist_lyrics(self, user_input: str, style: str) -> str:
        """Generate or refine lyrics with structure tags ([Verse], [Chorus], etc.)."""
        ...

    async def suggest_params(self, prompt: str) -> GenerationParams:
        """Suggest generation parameters based on the music style."""
        ...
