# TODO

## 次に実装するもの

### SSHトンネル自動管理
- 接続切れ時の自動再接続
- `ace-music backend connect` でComfyUI起動 + SSHトンネル確立を自動化

### ワークフローテンプレート管理
- text2music以外にedit, extend, repaint対応
- `--workflow` オプションでテンプレート切り替え

## 未実装のスタブ

- `backends/runpod.py` — RunPodバックエンド
- `agents/openai_agent.py` — OpenAIエージェント
- `agents/local.py` — Ollama等ローカルLLMエージェント

## 改善案

- WebSocket対応（ComfyUI WebSocket APIでリアルタイム進捗取得）
- 生成履歴の保存と再利用
- テスト追加 (pytest)

## 実装済み

### 設定管理コマンド (`ace-music config`) - 2025-02-06
- `ace-music config show` — 現在の設定表示
- `ace-music config set` — バックエンド/エージェントの対話的切り替え
- `ace-music config init` — デフォルト設定ファイル作成
- config.toml への保存 (`~/.config/ace-music/config.toml`)

### バックエンド管理 (`ace-music backend`) - 2025-02-06
- `ace-music backend status` — vast.aiインスタンス状態確認
- `ace-music backend start` — インスタンス起動
- `ace-music backend stop` — インスタンス停止

## 現在の環境情報

### vast.aiインスタンス
- **ID**: 30971522
- **状態**: 停止中（`ace-music backend start` で再開）
- **GPU**: RTX 3060 (12GB VRAM), Quebec/Canada
- **料金**: $0.064/hr
- **SSH**: `ssh -p 11522 root@ssh8.vast.ai`
- **ComfyUI**: ポート8188、ACE-Step-v1-3.5Bモデル設置済み
- **再開手順**:
  1. `ace-music backend start`
  2. SSHでComfyUI起動: `ssh -p 11522 root@ssh8.vast.ai 'cd /root/ComfyUI && nohup python3 main.py --listen 0.0.0.0 --port 8188 > /root/comfyui.log 2>&1 &'`
  3. SSHトンネル: `ssh -f -N -L 8188:localhost:8188 -p 11522 root@ssh8.vast.ai`
