"""Data models for music generation."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Scheduler(str, Enum):
    EULER = "euler"
    HEUN = "heun"
    PINGPONG = "pingpong"


class CfgType(str, Enum):
    CFG = "cfg"
    APG = "apg"
    CFG_STAR = "cfg_star"


class GenerationParams(BaseModel):
    """Parameters controlling the diffusion process."""

    duration: float = Field(30.0, ge=1.0, le=240.0, description="Audio duration in seconds")
    steps: int = Field(60, ge=1, le=200, description="Inference steps (27=fast, 60=standard)")
    guidance_scale: float = Field(15.0, ge=0.0, le=200.0)
    scheduler: Scheduler = Scheduler.EULER
    cfg_type: CfgType = CfgType.APG
    omega_scale: float = Field(10.0, ge=-100.0, le=100.0)
    seed: int = Field(-1, description="-1 for random")
    guidance_interval: float = Field(0.5, ge=0.0, le=1.0)
    guidance_interval_decay: float = Field(0.0, ge=0.0, le=1.0)
    min_guidance_scale: int = Field(3, ge=0, le=200)
    use_erg_tag: bool = True
    use_erg_lyric: bool = False
    use_erg_diffusion: bool = True
    oss_steps: str = ""
    guidance_scale_text: float = Field(0.0, ge=0.0, le=10.0)
    guidance_scale_lyric: float = Field(0.0, ge=0.0, le=10.0)


class GenerationRequest(BaseModel):
    """A complete request for music generation."""

    prompt: str = Field(..., description="Style tags (e.g. 'pop, 120 BPM, female voice')")
    lyrics: str = Field("", description="Lyrics with [Verse]/[Chorus] tags")
    params: GenerationParams = Field(default_factory=GenerationParams)
    negative_prompt: str = ""


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class JobStatus(BaseModel):
    """Status of a generation job."""

    state: JobState
    progress: float = Field(0.0, ge=0.0, le=1.0)
    result_filename: str | None = None
    error_message: str | None = None
