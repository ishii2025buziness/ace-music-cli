# ace-music-cli

Interactive CLI for [ACE-Step](https://github.com/ace-step/ACE-Step) music generation with pluggable backends and AI-assisted prompt creation.

## Features

- **Interactive session** — describe what you want in natural language, AI generates style tags and lyrics
- **Push notification** — terminal bell + playback prompt when generation completes
- **Pluggable backends** — vast.ai, local ComfyUI, RunPod (planned)
- **Pluggable AI agents** — Anthropic Claude, OpenAI (planned), local LLM (planned)
- **Auto audio playback** — detects ffplay, mpv, or aplay

## Install

```bash
uv venv && uv pip install -e .
```

## Setup

```bash
cp config.example.toml config.toml
# Edit config.toml with your backend and agent settings
```

### Backend: vast.ai

Requires a running vast.ai instance with ComfyUI + ACE-Step deployed.
See [ACE-STEP-API.md](../ACE-STEP-API.md) for setup instructions.

```toml
[backend]
type = "vastai"

[backend.vastai]
instance_id = "30971522"
ssh_host = "ssh8.vast.ai"
ssh_port = 11522
```

### Backend: Local

Requires ComfyUI running locally with the [ComfyUI_ACE-Step](https://github.com/billwuhao/ComfyUI_ACE-Step) custom node.

```toml
[backend]
type = "local"

[backend.local]
comfyui_url = "http://localhost:8188"
```

### AI Agent

Set `ANTHROPIC_API_KEY` environment variable, or put the key in config.toml.

```toml
[agent]
type = "anthropic"

[agent.anthropic]
model = "claude-sonnet-4-20250514"
```

## Usage

```bash
# Interactive session
ace-music generate

# With custom config
ace-music generate -c /path/to/config.toml
```

### Session flow

```
$ ace-music generate

╭──────────────────────────────────╮
│ ACE-Step Music Generator         │
│ Backend: vastai  Agent: anthropic│
╰──────────────────────────────────╯
Connected.

What kind of music do you want to create?
  > 明るいポップな曲

Prompt: pop, bright, female voice, 120 BPM, synth, upbeat
  Edit prompt? [y/N]

  Add lyrics? [Y/n]
  Describe the lyrics theme:
  > 朝の散歩で気分が良い

Lyrics:
  [Verse]
  Morning light through the window...
  [Chorus]
  Walking on sunshine...
  Edit lyrics? [y/N]

Generation Parameters
  Duration  30.0s
  Steps     60
  Guidance  15.0
  Seed      -1
  Adjust parameters? [y/N]

⠋ Generating... ████████████████ 100%

Done! Saved to: output/ComfyUI_temp_xxxxx_00001_.flac
  Play audio? [Y/n]

Next action:
  [1] New song
  [2] Retry with different params
  [3] Quit
  >
```

## Architecture

```
ace_music/
├── cli.py          ← Interactive UI (typer + rich + prompt_toolkit)
├── app.py          ← DI factory (wires backends & agents from config)
├── models.py       ← Pydantic data models
├── config.py       ← TOML configuration
├── player.py       ← Audio playback
├── backends/
│   ├── protocol.py    ← Backend Protocol (interface)
│   ├── comfyui_api.py ← ComfyUI HTTP API (shared logic)
│   ├── vastai.py      ← SSH tunnel + ComfyUI
│   ├── local.py       ← Direct localhost ComfyUI
│   └── runpod.py      ← Stub
└── agents/
    ├── protocol.py    ← Agent Protocol (interface)
    ├── anthropic.py   ← Claude API
    ├── openai_agent.py← Stub
    └── local.py       ← Stub
```

### Adding a new backend

Implement the `MusicBackend` protocol:

```python
class MusicBackend(Protocol):
    async def connect(self) -> None: ...
    async def generate(self, request: GenerationRequest) -> str: ...
    async def poll_status(self, job_id: str) -> JobStatus: ...
    async def download(self, job_id: str, dest: Path) -> Path: ...
    async def disconnect(self) -> None: ...
```

### Adding a new AI agent

Implement the `MusicAgent` protocol:

```python
class MusicAgent(Protocol):
    async def assist_prompt(self, user_input: str) -> str: ...
    async def assist_lyrics(self, user_input: str, style: str) -> str: ...
    async def suggest_params(self, prompt: str) -> GenerationParams: ...
```

## License

MIT
