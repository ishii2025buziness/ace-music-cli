"""Pluggable AI agent providers for prompt assistance."""

from .protocol import MusicAgent
from .anthropic import AnthropicAgent

__all__ = ["MusicAgent", "AnthropicAgent"]
