"""RunPod backend — stub for future implementation."""

from __future__ import annotations

from pathlib import Path

from ace_music.config import RunPodConfig
from ace_music.models import GenerationRequest, JobStatus


class RunPodBackend:
    """RunPod backend (not yet implemented)."""

    def __init__(self, config: RunPodConfig) -> None:
        self._cfg = config

    async def connect(self) -> None:
        raise NotImplementedError("RunPod backend is not yet implemented")

    async def generate(self, request: GenerationRequest) -> str:
        raise NotImplementedError

    async def poll_status(self, job_id: str) -> JobStatus:
        raise NotImplementedError

    async def download(self, job_id: str, dest: Path) -> Path:
        raise NotImplementedError

    async def disconnect(self) -> None:
        pass
