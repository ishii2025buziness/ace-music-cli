# CLAUDE.md

## ルール

### 破壊的操作は必ず確認を取る

以下の操作は**実行前に必ずユーザーに確認**すること：

- `vastai destroy instance` — インスタンス削除
- `rm -rf`, `rm -r` — ファイル/ディレクトリ削除
- `git reset --hard`, `git clean -f` — 変更の破棄
- `DROP TABLE`, `DELETE FROM` — DB操作
- その他、取り返しのつかない操作全般

**理由**: 2025-02-06にvast.aiインスタンスを確認なしに削除し、ComfyUI + ACE-Stepのセットアップ済み環境を失った。再セットアップに時間がかかる。
