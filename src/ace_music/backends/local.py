"""Local backend — connects directly to a ComfyUI instance on localhost."""

from __future__ import annotations

from pathlib import Path

from ace_music.config import LocalConfig
from ace_music.models import GenerationRequest, JobStatus

from .comfyui_api import ComfyUIClient


class LocalBackend:
    """Connects to a local ComfyUI instance (no tunnel needed)."""

    def __init__(self, config: LocalConfig) -> None:
        self._cfg = config
        self._client: ComfyUIClient | None = None

    async def connect(self) -> None:
        self._client = ComfyUIClient(self._cfg.comfyui_url)
        if not await self._client.health_check():
            raise ConnectionError(
                f"ComfyUI is not reachable at {self._cfg.comfyui_url}. "
                "Make sure ComfyUI is running locally."
            )

    async def generate(self, request: GenerationRequest) -> str:
        assert self._client is not None
        return await self._client.queue_prompt(request)

    async def poll_status(self, job_id: str) -> JobStatus:
        assert self._client is not None
        return await self._client.get_status(job_id)

    async def download(self, job_id: str, dest: Path) -> Path:
        assert self._client is not None
        status = await self._client.get_status(job_id)
        if not status.result_filename:
            raise RuntimeError("No audio file available for download")

        audio_bytes = await self._client.download_audio(status.result_filename)
        dest.mkdir(parents=True, exist_ok=True)
        out_path = dest / status.result_filename
        out_path.write_bytes(audio_bytes)
        return out_path

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
