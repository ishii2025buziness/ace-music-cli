"""OpenAI agent — stub for future implementation."""

from __future__ import annotations

from ace_music.config import OpenAIAgentConfig
from ace_music.models import GenerationParams


class OpenAIAgent:
    """OpenAI-based agent (not yet implemented)."""

    def __init__(self, config: OpenAIAgentConfig) -> None:
        self._cfg = config

    async def assist_prompt(self, user_input: str) -> str:
        raise NotImplementedError("OpenAI agent is not yet implemented")

    async def assist_lyrics(self, user_input: str, style: str) -> str:
        raise NotImplementedError

    async def suggest_params(self, prompt: str) -> GenerationParams:
        raise NotImplementedError
