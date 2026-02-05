"""Local LLM agent (Ollama, etc.) — stub for future implementation."""

from __future__ import annotations

from ace_music.config import LocalAgentConfig
from ace_music.models import GenerationParams


class LocalAgent:
    """Local LLM agent via Ollama (not yet implemented)."""

    def __init__(self, config: LocalAgentConfig) -> None:
        self._cfg = config

    async def assist_prompt(self, user_input: str) -> str:
        raise NotImplementedError("Local agent is not yet implemented")

    async def assist_lyrics(self, user_input: str, style: str) -> str:
        raise NotImplementedError

    async def suggest_params(self, prompt: str) -> GenerationParams:
        raise NotImplementedError
