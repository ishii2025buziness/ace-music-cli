"""Application factory — wires up backends and agents from config."""

from __future__ import annotations

from ace_music.config import AppConfig
from ace_music.backends.protocol import MusicBackend
from ace_music.backends.vastai import VastAIBackend
from ace_music.backends.local import LocalBackend
from ace_music.backends.runpod import RunPodBackend
from ace_music.agents.protocol import MusicAgent
from ace_music.agents.anthropic import AnthropicAgent
from ace_music.agents.openai_agent import OpenAIAgent
from ace_music.agents.local import LocalAgent


def create_backend(config: AppConfig) -> MusicBackend:
    """Instantiate the configured backend."""
    match config.backend.type:
        case "vastai":
            return VastAIBackend(config.backend.vastai)
        case "local":
            return LocalBackend(config.backend.local)
        case "runpod":
            return RunPodBackend(config.backend.runpod)
        case _:
            raise ValueError(f"Unknown backend type: {config.backend.type}")


def create_agent(config: AppConfig) -> MusicAgent:
    """Instantiate the configured AI agent."""
    match config.agent.type:
        case "anthropic":
            return AnthropicAgent(config.agent.anthropic)
        case "openai":
            return OpenAIAgent(config.agent.openai)
        case "local":
            return LocalAgent(config.agent.local)
        case _:
            raise ValueError(f"Unknown agent type: {config.agent.type}")
