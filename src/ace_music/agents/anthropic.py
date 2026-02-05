"""Anthropic Claude agent for music prompt assistance."""

from __future__ import annotations

import json

import anthropic

from ace_music.config import AnthropicAgentConfig
from ace_music.models import GenerationParams

_SYSTEM = """\
You are a music production assistant for ACE-Step, a text-to-music AI model.
Your job is to help users create effective prompts and lyrics.

ACE-Step prompt format: comma-separated style tags describing genre, tempo, \
instruments, mood, vocal characteristics.
Example: "pop, 120 BPM, female voice, bright, synth, upbeat, catchy melody"

Lyrics format: use structure tags [Intro], [Verse], [Pre-Chorus], [Chorus], \
[Bridge], [Outro], [Instrumental]. Each section on its own line.

Respond ONLY with the requested content — no explanations or commentary."""


class AnthropicAgent:
    """Uses Claude to assist with prompt/lyrics generation."""

    def __init__(self, config: AnthropicAgentConfig) -> None:
        self._model = config.model
        self._client = anthropic.AsyncAnthropic(
            api_key=config.api_key or None  # falls back to env var
        )

    async def assist_prompt(self, user_input: str) -> str:
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Convert this description into ACE-Step style tags "
                        f"(comma-separated, one line): {user_input}"
                    ),
                }
            ],
        )
        return msg.content[0].text.strip()

    async def assist_lyrics(self, user_input: str, style: str) -> str:
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Write song lyrics for: {user_input}\n"
                        f"Style: {style}\n"
                        f"Include [Verse], [Chorus], and optionally [Bridge]/[Outro] tags. "
                        f"Keep it concise (under 200 words)."
                    ),
                }
            ],
        )
        return msg.content[0].text.strip()

    async def suggest_params(self, prompt: str) -> GenerationParams:
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Given this music style: {prompt}\n"
                        f"Suggest generation parameters as JSON with these fields:\n"
                        f"duration (float, 10-120), steps (int, 27 or 60), "
                        f"guidance_scale (float, 5-20), scheduler ('euler'), "
                        f"cfg_type ('apg')\n"
                        f"Return ONLY valid JSON, nothing else."
                    ),
                }
            ],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        return GenerationParams.model_validate(data)
