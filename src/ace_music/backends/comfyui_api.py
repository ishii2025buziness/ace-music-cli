"""ComfyUI HTTP API client — shared by all backends that host ComfyUI."""

from __future__ import annotations

import random

import httpx

from ace_music.models import GenerationRequest, JobState, JobStatus


def _build_prompt_payload(request: GenerationRequest) -> dict:
    """Convert a GenerationRequest into a ComfyUI /prompt payload."""
    p = request.params
    seed = p.seed if p.seed >= 0 else random.randint(0, 2**32)

    return {
        "prompt": {
            "38": {
                "class_type": "ACEModelLoader",
                "inputs": {
                    "dcae_checkpoint": "music_dcae_f8c8",
                    "vocoder_checkpoint": "music_vocoder",
                    "ace_step_checkpoint": "ace_step_transformer",
                    "text_encoder_checkpoint": "umt5-base",
                    "cpu_offload": True,
                    "torch_compile": False,
                },
            },
            "84": {
                "class_type": "MultiLineLyrics",
                "inputs": {"multi_line_prompt": request.lyrics or "[Instrumental]"},
            },
            "85": {
                "class_type": "GenerationParameters",
                "inputs": {
                    "audio_duration": p.duration,
                    "infer_step": p.steps,
                    "guidance_scale": p.guidance_scale,
                    "scheduler_type": p.scheduler.value,
                    "cfg_type": p.cfg_type.value,
                    "omega_scale": p.omega_scale,
                    "seed": seed,
                    "guidance_interval": p.guidance_interval,
                    "guidance_interval_decay": p.guidance_interval_decay,
                    "min_guidance_scale": p.min_guidance_scale,
                    "use_erg_tag": p.use_erg_tag,
                    "use_erg_lyric": p.use_erg_lyric,
                    "use_erg_diffusion": p.use_erg_diffusion,
                    "oss_steps": p.oss_steps,
                    "guidance_scale_text": p.guidance_scale_text,
                    "guidance_scale_lyric": p.guidance_scale_lyric,
                },
            },
            "86": {
                "class_type": "MultiLinePromptACES",
                "inputs": {"multi_line_prompt": request.prompt},
            },
            "87": {
                "class_type": "ACEStepGen",
                "inputs": {
                    "models": ["38", 0],
                    "prompt": ["86", 0],
                    "lyrics": ["84", 0],
                    "parameters": ["85", 0],
                    "negative_prompt": request.negative_prompt,
                    "ref_audio_strength": 0.5,
                    "overlapped_decode": False,
                    "delicious_song": "None",
                },
            },
            "11": {
                "class_type": "PreviewAudio",
                "inputs": {"audio": ["87", 0]},
            },
        }
    }


class ComfyUIClient:
    """Thin async wrapper around the ComfyUI REST API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    async def queue_prompt(self, request: GenerationRequest) -> str:
        """Queue a generation job. Returns the prompt_id."""
        payload = _build_prompt_payload(request)
        resp = await self._client.post("/prompt", json=payload)
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    async def get_status(self, prompt_id: str) -> JobStatus:
        """Poll the history endpoint for job status."""
        resp = await self._client.get(f"/history/{prompt_id}")
        resp.raise_for_status()
        data = resp.json()

        if prompt_id not in data:
            return JobStatus(state=JobState.QUEUED)

        entry = data[prompt_id]
        outputs = entry.get("outputs", {})
        status_info = entry.get("status", {})

        if status_info.get("status_str") == "error":
            return JobStatus(
                state=JobState.ERROR,
                error_message=str(status_info.get("messages", "Unknown error")),
            )

        audio_list = outputs.get("11", {}).get("audio", [])
        if audio_list:
            return JobStatus(
                state=JobState.COMPLETED,
                progress=1.0,
                result_filename=audio_list[0].get("filename"),
            )

        return JobStatus(state=JobState.RUNNING, progress=0.5)

    async def download_audio(self, filename: str) -> bytes:
        """Download a generated audio file."""
        resp = await self._client.get(
            "/view", params={"filename": filename, "type": "temp"}, timeout=120.0
        )
        resp.raise_for_status()
        return resp.content

    async def close(self) -> None:
        await self._client.aclose()
