# Telegram Group Bot (`telegram-group-bot`, `bot_api` mode)

Autonomous Telegram group & admin-DM agent powered by **Telegram Bot API** and **Antigravity SDK/CLI**, featuring role-based model routing, token quota budgets, secret sanitization, `botops` non-root isolation, an autonomous `SelfEvolutionManager` with `Adversarial Judge` + root gatekeeper canary rollback, and a WireGuard kill-switch template.

> Note: This repository is standalone for the **Bot API agent** (`mode: "bot_api"`). The MTProto Telethon userbot lives in the separate `telegram-mtproto-userbot` repository.

## Key Features
- **Toggleable Extended Tool Mode (`/ops on | off | admin | status`)**: By default (`antigravity.ops_tools_default_mode: "on"`), permitted groups and DMs can use the non-root `ops` Antigravity toolkit (shell commands, file read/write/edit, subagents/workflows/tasks, plus `/evolve`, `/quota`, `/backup`) under `botops`. Administrators (`admin_user_ids`) can switch to safe `chat`-only mode anytime via `/ops off` (persisted in SQLite `bot_settings`), restrict to admin DMs via `/ops admin`, or re-enable via `/ops on` (aliases: `/tools`, `/safemode`). Proactive loops and compaction always stay on `tool_profile="chat"`.
- **Native Image Generation (`/imagine`, `/img`, `[IMG:{...}]`)**: Explicit `/imagine <prompt> [--ar 16:9]` (alias `/img`, supporting text/photo replies and photo captions) and gated natural-language `[IMG:{...}]` generation via isolated `tool_profile="image"` + fail-closed `PreToolUse` hook firewall, configurable via `image_generation:` in `bot.yaml`, with per-user/chat rate limits, `upload_photo` keepalive, automatic `send_photo` / >9.5 MB `send_document` delivery, and text-only SQLite memory logging.
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
