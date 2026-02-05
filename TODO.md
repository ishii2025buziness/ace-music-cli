# TODO

## 次に実装するもの

### CLI内での設定管理コマンド (`ace-music config`)
- バックエンド選択 (vastai / local / runpod) を対話的に切り替え
- AIエージェント選択 (anthropic / openai / local) を対話的に切り替え
- 各バックエンド/エージェントの接続情報を対話的に入力
- config.toml への保存
- 現在の設定表示 (`ace-music config show`)

### バックエンド管理
- vast.aiインスタンスの起動/停止 (`ace-music backend start/stop`)
- インスタンス状態確認 (`ace-music backend status`)
- SSHトンネルの自動管理（接続切れ時の再接続）

## 未実装のスタブ

- `backends/runpod.py` — RunPodバックエンド
- `agents/openai_agent.py` — OpenAIエージェント
- `agents/local.py` — Ollama等ローカルLLMエージェント

## 改善案

- WebSocket対応（ComfyUI WebSocket APIでリアルタイム進捗取得）
- 生成履歴の保存と再利用
- ワークフローテンプレートの管理（text2music以外にedit, extend, repaint対応）
- テスト追加 (pytest)

## 現在の環境情報

### vast.aiインスタンス
- **ID**: 30971522
- **状態**: 停止中（`vastai start instance 30971522` で再開）
- **GPU**: RTX 3060 (12GB VRAM), Quebec/Canada
- **料金**: $0.064/hr
- **SSH**: `ssh -p 11522 root@ssh8.vast.ai`
- **ComfyUI**: ポート8188、ACE-Step-v1-3.5Bモデル設置済み
- **再開手順**:
  1. `vastai start instance 30971522`
  2. SSHでComfyUI起動: `ssh -p 11522 root@ssh8.vast.ai 'cd /root/ComfyUI && nohup python3 main.py --listen 0.0.0.0 --port 8188 > /root/comfyui.log 2>&1 &'`
  3. SSHトンネル: `ssh -f -N -L 8188:localhost:8188 -p 11522 root@ssh8.vast.ai`
