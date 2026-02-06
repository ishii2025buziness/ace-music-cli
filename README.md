# ace-music-cli

[ACE-Step](https://github.com/ace-step/ACE-Step) を使った対話型音楽生成CLIツール。クラウドGPU (vast.ai) で音楽を生成。

## インストール

```bash
# GitHubから直接インストール
pip install git+https://github.com/ishii2025buziness/ace-music-cli.git

# または開発用
git clone https://github.com/ishii2025buziness/ace-music-cli.git
cd ace-music-cli
pip install -e .
```

## セットアップ

```bash
# 依存関係をチェック
ace-music doctor

# 不足があれば以下を実行:

# 1. vast.ai CLI
pip install vastai
vastai set api-key <YOUR_VASTAI_API_KEY>
# APIキー取得: https://console.vast.ai/account

# 2. Anthropic API キー
export ANTHROPIC_API_KEY=<YOUR_KEY>
# または ace-music config set で設定ファイルに保存
```

## クイックスタート

```bash
# 1. GPUインスタンスを検索 (12GB VRAM以上, $0.1/hr以下)
ace-music backend search --vram 12 --max-price 0.1

# 2. インスタンス作成 (ComfyUI + ACE-Step自動セットアップ)
ace-music backend create <OFFER_ID>
# → セットアップ完了まで10-20分待機

# 3. 接続
ace-music backend connect

# 4. 音楽生成
ace-music generate
```

## コマンド一覧

```
ace-music --help              # 全体のヘルプとワークフロー

ace-music doctor              # 依存関係チェック
ace-music generate            # 対話型音楽生成

ace-music config show         # 設定表示
ace-music config set          # 対話的に設定変更
ace-music config init         # デフォルト設定ファイル作成

ace-music backend search      # GPUオファー検索
ace-music backend create      # インスタンス作成
ace-music backend status      # インスタンス状態確認
ace-music backend start       # インスタンス起動
ace-music backend stop        # インスタンス停止
ace-music backend connect     # ComfyUI起動 + 接続
```

## 自動化 (エージェント向け)

`--json` フラグで機械可読な出力:

```bash
ace-music backend status --json
# → {"instance_id": "123", "state": "running", ...}

ace-music backend create <ID> --json
# → {"status": "ready", "instance_id": "123", ...}

ace-music backend connect --json
# → {"status": "connected", "comfyui_url": "http://localhost:8188", ...}
```

## 特徴

- **対話型セッション** — 自然言語で説明するだけで、AIがスタイルタグと歌詞を生成
- **ワンコマンドセットアップ** — `backend create` でComfyUI + ACE-Stepを自動インストール
- **バックエンド差し替え** — vast.ai / ローカルComfyUI / RunPod（予定）
- **AIエージェント差し替え** — Anthropic Claude / OpenAI（予定）/ ローカルLLM（予定）
- **自動音声再生** — ffplay, mpv, aplay を自動検出

## 設定ファイル

`~/.config/ace-music/config.toml` に保存されます。

```toml
[backend]
type = "vastai"

[backend.vastai]
instance_id = "12345678"
ssh_host = "ssh5.vast.ai"
ssh_port = 12345

[agent]
type = "anthropic"

[agent.anthropic]
model = "claude-sonnet-4-20250514"

[output]
directory = "./output"
```

## アーキテクチャ

```
ace_music/
├── cli.py          ← CLI (typer + rich)
├── app.py          ← DIファクトリ
├── models.py       ← pydanticデータモデル
├── config.py       ← TOML設定管理
├── player.py       ← 音声再生
├── backends/
│   ├── protocol.py    ← Backend Protocol
│   ├── comfyui_api.py ← ComfyUI HTTP API
│   ├── vastai.py      ← SSHトンネル + ComfyUI
│   └── local.py       ← ローカルComfyUI
└── agents/
    ├── protocol.py    ← Agent Protocol
    └── anthropic.py   ← Claude API
```

## ライセンス

MIT
