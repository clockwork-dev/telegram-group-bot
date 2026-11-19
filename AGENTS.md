# AGENTS.md — AI Assistant Guide (`telegram-group-bot`, `bot_api` mode)

## 1. Repository Scope
This repository contains the **classic Telegram Bot API agent** (`mode: "bot_api"`), deployed on production servers under `/opt/telegram-group-bot` (`telegram-userbot.service`, `User=botops`, `Group=botops`).

**Strictly out of scope for this repo:**
- The sibling **MTProto Userbot** (`telegram-mtproto-userbot`) lives in a separate repository and runs isolated under `/opt/telegram-mtproto-userbot` (`telegram-mtproto-userbot.service`). Never modify, reference, or couple code/scripts to `/opt/telegram-mtproto-userbot`.
- Never commit `.env`, `config/bot.yaml`, `config/friends.yaml`, `data/`, `*.sqlite*`, `*.session*`, `wg0.conf`, `*.key`, or real OAuth tokens. Use `config/bot.example.yaml`, `config/friends.example.yaml`, `.env.example`, and `deploy/wireguard/wg0.conf.example`.

## 2. Architecture & Key Modules
- **CLI & Orchestrator**: `src/bot/cli.py`, `src/bot/main.py` (`GroupBotOrchestrator`)
- **Configuration (Hot-Reloadable)**: `src/bot/config/models.py` (`BotConfig`, `TelegramConfig`, `QuotaConfig`, `ModelRouterConfig`, `SelfEvolutionConfig`), `src/bot/config/manager.py` (`ConfigManager`)
- **Telegram Bot API Transport**: `src/bot/telegram/bot_api_client.py` (`TelegramBotAPIClient`), `src/bot/telegram/handlers.py` (`IncomingMessage`), `src/bot/telegram/heuristics.py` (`ReplyHeuristics`)
- **Antigravity Provider & Tool Isolation**:
  - `src/bot/antigravity/__init__.py` (`AntigravityProvider`, per-model circuit breaker)
  - `src/bot/antigravity/cli_client.py` (`AntigravityCLIClient`, ACP + stream-json, `watch_log_for_errors`)
  - `src/bot/antigravity/stream.py` (`AntigravityStreamParser`, `ToolProfile`: `"chat"`, `"ops"`, `"self_dev"`)
  - `src/bot/antigravity/errors.py` (`classify_error`, `extract_retry_after`, `parse_quota_warning`)
  - `src/bot/antigravity/auth.py` (isolated `data/antigravity` home, `build_clean_child_env`)
- **Security**: `src/bot/security/sanitizer.py` (`SecretSanitizer`, `SanitizingLogFilter`)
- **Quota & Model Routing**: `src/bot/quota/manager.py` (`QuotaBudgetManager` — `usage_history` SQLite tracking, per-model/bucket cooldowns, economy mode, `/quota` admin command)
- **Self-Evolution & Gatekeeper**:
  - `src/bot/evolution/manager.py` (`SelfEvolutionManager` — runtime error signatures, anti-churn cooldowns/daily cap, `Adversarial Judge`, escalation chain, atomic `request.json` trigger)
  - `deploy/bot-self-deploy.sh` (root-owned gatekeeper validating staging diffs, running `ruff`/`pytest`/`--dry-run`, applying & canary-checking `telegram-userbot.service` with auto-rollback)
  - `deploy/telegram-self-deploy.path` & `deploy/telegram-self-deploy.service`
- **Memory, Backups, Media & Proactive**: `src/bot/memory/` (including `backup.py` hot SQLite WAL backup/restore, `backup-memory`/`restore-memory` CLI, scheduled `data/backups/` retention, and `/backup` admin DM command), `src/bot/media/` (Whisper STT + Edge TTS), `src/bot/proactive/` (cron digests & banter), `src/bot/documents/` (PDF/DOCX/XLSX/PPTX)
- **WireGuard Template**: `deploy/wireguard/wg0.conf.example`, `deploy/wireguard/killswitch.nft.example`, `deploy/wireguard/README.md`

## 3. Critical Invariants (Do Not Break)
1. **Tool Profile Fail-Closed Boundary**:
   - Group chats (`chat_id < 0`), proactive loops, compaction, and non-admin DMs MUST strictly use `tool_profile="chat"` (`CHAT_ALLOWED_TOOLS = {"search_web", "read_url_content"}`).
   - `ops` and `self_dev` are granted ONLY for verified `admin_user_ids` in private DMs or internal `SelfEvolutionManager` turns targeting `self_evolution.staging_dir`.
2. **Secret Sanitization**:
   - All outbound Telegram replies, document captions, admin alerts, and Python logs pass through `SecretSanitizer` / `SanitizingLogFilter`.
3. **Self-Evolution Safety**:
   - Never bypass `Adversarial Judge`, anti-churn limits, `protected_paths`, or the staging worktree gatekeeper.
   - Never run the bot process as `root` in production (`systemd/telegram-userbot.service` enforces `User=botops`, `Group=botops`, `NoNewPrivileges=true`).

## 4. Verification & Development Commands
```bash
# Lint
.venv/bin/ruff check .

# Full unit + integration + offline E2E test suite
.venv/bin/python -m pytest -q

# Offline dry-run smoke test
.venv/bin/telegram-group-bot run --config config/bot.yaml --dry-run

# Bash syntax check on deploy scripts
bash -n deploy/*.sh
```

## 5. Deployment Workflow
- Deploy to production `bot_api` target via `REMOTE_HOST=user@host ./deploy/deploy_bot_api.sh` (uses SSH keys or `SSH_PASSWORD` env var; never hardcode credentials).
- For autonomous server updates, `botops` writes `data/self_deploy/request.json`, which triggers `telegram-self-deploy.path` -> `/usr/local/sbin/bot-self-deploy.sh`.
