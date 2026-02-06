"""Configuration management via TOML files and environment variables."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore


class VastAIConfig(BaseModel):
    instance_id: str = ""
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = "root"
    comfyui_port: int = 8188


class RunPodConfig(BaseModel):
    pod_id: str = ""
    api_key: str = ""
    comfyui_port: int = 8188


class LocalConfig(BaseModel):
    comfyui_url: str = "http://localhost:8188"


class BackendConfig(BaseModel):
    type: Literal["vastai", "runpod", "local"] = "vastai"
    vastai: VastAIConfig = Field(default_factory=VastAIConfig)
    runpod: RunPodConfig = Field(default_factory=RunPodConfig)
    local: LocalConfig = Field(default_factory=LocalConfig)


class AnthropicAgentConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    api_key: str = Field(default="", description="Falls back to ANTHROPIC_API_KEY env var")


class OpenAIAgentConfig(BaseModel):
    model: str = "gpt-4o"
    api_key: str = ""


class LocalAgentConfig(BaseModel):
    model: str = "llama3"
    endpoint: str = "http://localhost:11434"


class AgentConfig(BaseModel):
    type: Literal["anthropic", "openai", "local"] = "anthropic"
    anthropic: AnthropicAgentConfig = Field(default_factory=AnthropicAgentConfig)
    openai: OpenAIAgentConfig = Field(default_factory=OpenAIAgentConfig)
    local: LocalAgentConfig = Field(default_factory=LocalAgentConfig)


class OutputConfig(BaseModel):
    directory: str = "./output"
    format: str = "flac"


class PlayerConfig(BaseModel):
    command: str = "auto"


class AppConfig(BaseModel):
    backend: BackendConfig = Field(default_factory=BackendConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    player: PlayerConfig = Field(default_factory=PlayerConfig)


_CONFIG_SEARCH_PATHS = [
    Path("config.toml"),
    Path.home() / ".config" / "ace-music" / "config.toml",
]


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from TOML file, falling back to defaults."""
    if path and path.exists():
        return _parse_config(path)

    for candidate in _CONFIG_SEARCH_PATHS:
        if candidate.exists():
            return _parse_config(candidate)

    return AppConfig()


def _parse_config(path: Path) -> AppConfig:
    with open(path, "rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    cfg = AppConfig.model_validate(data)

    # Allow env var overrides
    if not cfg.agent.anthropic.api_key:
        cfg.agent.anthropic.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    return cfg


def get_config_path() -> Path:
    """Return the user config path (~/.config/ace-music/config.toml)."""
    return Path.home() / ".config" / "ace-music" / "config.toml"


def save_config(config: AppConfig, path: Path | None = None) -> Path:
    """Save configuration to TOML file.

    Args:
        config: The configuration to save
        path: Target path. Defaults to ~/.config/ace-music/config.toml

    Returns:
        The path where config was saved

    Raises:
        ImportError: If tomli_w is not installed
    """
    if tomli_w is None:
        raise ImportError("tomli_w is required to save config. Install with: pip install tomli-w")

    if path is None:
        path = get_config_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, excluding defaults where possible for cleaner output
    data = config.model_dump(mode="json")

    with open(path, "wb") as f:
        tomli_w.dump(data, f)

    return path
