"""Pluggable music generation backends."""

from .protocol import MusicBackend
from .vastai import VastAIBackend
from .local import LocalBackend

__all__ = ["MusicBackend", "VastAIBackend", "LocalBackend"]
