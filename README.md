# ace-music-cli

[ACE-Step](https://github.com/ace-step/ACE-Step) を使った対話型音楽生成CLIツール。バックエンドとAIエージェントを差し替え可能な設計。

## 特徴

- **対話型セッション** — 自然言語で説明するだけで、AIがスタイルタグと歌詞を生成
- **生成完了通知** — 生成が終わるとターミナルベルで通知し、再生するかをプロンプト
- **バックエンド差し替え** — vast.ai / ローカルComfyUI / RunPod（予定）
- **AIエージェント差し替え** — Anthropic Claude / OpenAI（予定）/ ローカルLLM（予定）
- **自動音声再生** — ffplay, mpv, aplay を自動検出

## インストール

```bash
uv venv && uv pip install -e .
```

## セットアップ

```bash
cp config.example.toml config.toml
# config.toml をバックエンドとエージェントの設定に合わせて編集
```

### バックエンド: vast.ai

ComfyUI + ACE-Step がデプロイ済みの vast.ai インスタンスが必要。
セットアップ手順は [ACE-STEP-API.md](../ACE-STEP-API.md) を参照。

```toml
[backend]
type = "vastai"

[backend.vastai]
instance_id = "30971522"
ssh_host = "ssh8.vast.ai"
ssh_port = 11522
```

### バックエンド: ローカル

ローカルで ComfyUI + [ComfyUI_ACE-Step](https://github.com/billwuhao/ComfyUI_ACE-Step) カスタムノードが動作している必要あり。

```toml
[backend]
type = "local"

[backend.local]
comfyui_url = "http://localhost:8188"
```

### AIエージェント

環境変数 `ANTHROPIC_API_KEY` を設定するか、config.toml に記載。

```toml
[agent]
type = "anthropic"

[agent.anthropic]
model = "claude-sonnet-4-20250514"
```

## 使い方

```bash
# 対話型セッション
ace-music generate

# 設定ファイル指定
ace-music generate -c /path/to/config.toml
```

### セッションの流れ

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
  朝の光が差し込んで...
  [Chorus]
  歩き出そう 今日も...
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

## アーキテクチャ

```
ace_music/
├── cli.py          ← 対話型UI (typer + rich + prompt_toolkit)
├── app.py          ← DIファクトリ (設定からbackend/agentを生成)
├── models.py       ← pydanticデータモデル
├── config.py       ← TOML設定管理
├── player.py       ← 音声再生
├── backends/
│   ├── protocol.py    ← Backend Protocol (インターフェース)
│   ├── comfyui_api.py ← ComfyUI HTTP API (共通ロジック)
│   ├── vastai.py      ← SSHトンネル + ComfyUI
│   ├── local.py       ← ローカルComfyUI直接接続
│   └── runpod.py      ← スタブ
└── agents/
    ├── protocol.py    ← Agent Protocol (インターフェース)
    ├── anthropic.py   ← Claude API
    ├── openai_agent.py← スタブ
    └── local.py       ← スタブ
```

### バックエンドの追加

`MusicBackend` プロトコルを実装する:

```python
class MusicBackend(Protocol):
    async def connect(self) -> None: ...
    async def generate(self, request: GenerationRequest) -> str: ...
    async def poll_status(self, job_id: str) -> JobStatus: ...
    async def download(self, job_id: str, dest: Path) -> Path: ...
    async def disconnect(self) -> None: ...
```

### AIエージェントの追加

`MusicAgent` プロトコルを実装する:

```python
class MusicAgent(Protocol):
    async def assist_prompt(self, user_input: str) -> str: ...
    async def assist_lyrics(self, user_input: str, style: str) -> str: ...
    async def suggest_params(self, prompt: str) -> GenerationParams: ...
```

## ライセンス

MIT
