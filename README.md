# Telegram Group Bot (`telegram-group-bot`, `bot_api` mode)

Autonomous Telegram group & admin-DM agent powered by **Telegram Bot API** and **Antigravity SDK/CLI**, featuring role-based model routing, token quota budgets, secret sanitization, `botops` non-root isolation, an autonomous `SelfEvolutionManager` with `Adversarial Judge` + root gatekeeper canary rollback, and a WireGuard kill-switch template.

> Note: This repository is standalone for the **Bot API agent** (`mode: "bot_api"`). The MTProto Telethon userbot lives in the separate `telegram-mtproto-userbot` repository.

## Key Features
- **Strict Tool-Profile Isolation**: Group chats and proactive loops run with `tool_profile="chat"` (`search_web`, `read_url_content` only); `ops` / `self_dev` tools are gated strictly to `admin_user_ids` in private DMs and the staging git worktree.
- **Quota & Model Router (`QuotaBudgetManager`)**: SQLite `usage_history` tracking, per-model/bucket cooldowns, cross-bucket fallback, automatic economy mode, and `/quota` admin command.
- **Autonomous Self-Evolution (`SelfEvolutionManager`)**: Runtime error triage + anti-churn limits, `Adversarial Judge` falsification gate, model escalation, staging worktree commits, and root gatekeeper (`deploy/bot-self-deploy.sh` + `telegram-self-deploy.path`) with `ruff`/`pytest`/`--dry-run` validation and automatic canary rollback.
- **Security & Privacy**: `SecretSanitizer` + `SanitizingLogFilter` for outbound messages and logs, non-root `User=botops` systemd unit, and ready-to-enable WireGuard client template (`deploy/wireguard/wg0.conf.example`).
- **Hot SQLite Memory Backup**: Automated daily WAL-safe snapshots (`src/bot/memory/backup.py`) to `data/backups/` with gzip compression, integrity verification, retention pruning, `/backup` admin DM command, and `backup-memory` / `restore-memory` CLI commands.

## Quick Start
1. Copy templates and fill in your credentials/IDs locally (never commit `.env`, `config/bot.yaml`, `config/friends.yaml`, or `wg0.conf`):
   ```bash
   cp .env.example .env
   cp config/bot.example.yaml config/bot.yaml
   cp config/friends.example.yaml config/friends.yaml
   ```
2. Install & verify:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e ".[dev]"
   .venv/bin/ruff check .
   .venv/bin/python -m pytest -q
   .venv/bin/telegram-group-bot run --config config/bot.yaml --dry-run
   ```

## AI Assistant Documentation
See [AGENTS.md](AGENTS.md) for full architecture contracts, security invariants, and verification workflows.
