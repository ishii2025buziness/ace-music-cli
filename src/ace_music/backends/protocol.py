"""Backend protocol — the contract all music backends must satisfy."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ace_music.models import GenerationRequest, JobStatus


@runtime_checkable
class MusicBackend(Protocol):
    """Abstract interface for a music generation backend."""

    async def connect(self) -> None:
        """Establish connection to the backend (SSH tunnel, health check, etc.)."""
        ...

    async def generate(self, request: GenerationRequest) -> str:
        """Submit a generation job. Returns a job ID for tracking."""
        ...

    async def poll_status(self, job_id: str) -> JobStatus:
        """Check the current status of a generation job."""
        ...

    async def download(self, job_id: str, dest: Path) -> Path:
        """Download the completed audio to *dest* directory. Returns the file path."""
        ...

    async def disconnect(self) -> None:
        """Tear down connection resources."""
        ...
