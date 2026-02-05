"""vast.ai backend — manages SSH tunnel and delegates to ComfyUI API."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path

from ace_music.config import VastAIConfig
from ace_music.models import GenerationRequest, JobStatus

from .comfyui_api import ComfyUIClient


class VastAIBackend:
    """Connects to a vast.ai instance via SSH tunnel, runs ComfyUI remotely."""

    def __init__(self, config: VastAIConfig) -> None:
        self._cfg = config
        self._local_port = config.comfyui_port
        self._tunnel_proc: asyncio.subprocess.Process | None = None
        self._client: ComfyUIClient | None = None

    async def connect(self) -> None:
        # Kill any stale tunnel on the same port
        await self._kill_existing_tunnel()

        self._tunnel_proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-N",
            "-L", f"{self._local_port}:localhost:{self._cfg.comfyui_port}",
            "-p", str(self._cfg.ssh_port),
            f"{self._cfg.ssh_user}@{self._cfg.ssh_host}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        self._client = ComfyUIClient(f"http://localhost:{self._local_port}")

        # Wait for tunnel + ComfyUI readiness
        for _ in range(30):
            if await self._client.health_check():
                return
            await asyncio.sleep(2)

        raise ConnectionError(
            f"Could not reach ComfyUI via tunnel to {self._cfg.ssh_host}:{self._cfg.ssh_port}. "
            "Is the vast.ai instance running and ComfyUI started?"
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
        if self._tunnel_proc and self._tunnel_proc.returncode is None:
            self._tunnel_proc.send_signal(signal.SIGTERM)
            await self._tunnel_proc.wait()
            self._tunnel_proc = None

    async def _kill_existing_tunnel(self) -> None:
        proc = await asyncio.create_subprocess_shell(
            f"pkill -f 'ssh.*{self._local_port}.*{self._cfg.ssh_host}' 2>/dev/null || true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
