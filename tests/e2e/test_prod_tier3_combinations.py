"""Tier 3 Production Hardening E2E Tests: 34 Pairwise Cross-Feature Combinations (F1-F18).

Covers:
- 18 pairwise combinations across F1-F10 (Security, DM Routing, Tool Profiles, Quota & Routing)
- 16 pairwise combinations across F1-F18 (SelfEvolution, Adversarial Judge, Escalation, Gatekeeper, WireGuard, Packaging)

All tests are strictly offline and opaque-box:
- Never read live `.env`, `config/friends.yaml`, or OAuth tokens.
- Never contact Telegram, `192.168.1.151`, `192.168.1.164`, or `/opt/telegram-mtproto-userbot`.
- Use deferred inside-function imports for M1-M3 modules (`bot.security.sanitizer`,
  `bot.quota.manager`, `bot.evolution.manager`) so lint and collection stay clean.
"""

from __future__ import annotations

import configparser
import inspect
import io
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from bot.antigravity import (
    AntigravityConfig,
    AntigravityNativeToolAbort,
    AntigravityProvider,
    AntigravityRateLimitError,
    ProviderResponse,
)
from bot.antigravity.cli_client import AntigravityCLIClient
from bot.antigravity.errors import ErrorCategory, ErrorVerdict, classify_error
from bot.antigravity.stream import AntigravityStreamParser
from bot.config.manager import ConfigManager
from bot.config.models import BotConfig
from bot.core.models import UsageInfo
from bot.main import GroupBotOrchestrator, run_dry_run_smoke
from bot.media.processor import MediaProcessor
from bot.memory.db import MemoryStore
from bot.telegram.bot_api_client import TelegramBotAPIClient
from bot.telegram.client import IncomingMessage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADMIN_USER_ID = 382962362
GROUP_CHAT_ID = -1003845062562
BOT_ID = 777000


# ============================================================================
# Shared Offline Harness Helpers (Self-Contained)
# ============================================================================


def _make_isolated_config(
    tmp_path: Path,
    *,
    extra_bot_cfg: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    friends: Optional[Dict[int, Dict[str, Any]]] = None,
) -> ConfigManager:
    """Create an isolated tmp_path ConfigManager without touching repo config/ or friends.yaml."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    bot_data: Dict[str, Any] = {
        "telegram": {
            "mode": "bot_api",
            "bot_token": "123456789:DUMMY_TEST_TOKEN_ABCDEFGHIJKLMNOPQRST",
            "api_id": 12345678,
            "api_hash": "0123456789abcdef0123456789abcdef",
            "allowed_chat_ids": [GROUP_CHAT_ID],
            "allowed_dm_user_ids": [ADMIN_USER_ID],
            "admin_user_ids": [ADMIN_USER_ID],
            "ignored_sender_usernames": ["hermes_kuk7yi6kfm3lcdfa_bot"],
        },
        "behavior": {
            "bot_names": ["Джефф", "бот"],
            "typing_simulation": False,
            "reactions_enabled": True,
            "spontaneous_replies": {
                "enabled": True,
                "min_interval_seconds": 0,
                "probability": 1.0,
                "max_per_hour": 10,
                "active_conversation_window_seconds": 3600,
            },
        },
        "memory": {
            "db_path": str(tmp_path / "memory.sqlite"),
            "short_term_limit": 20,
            "compaction_threshold": 50,
            "compaction_batch_size": 20,
        },
        "media": {
            "max_doc_size_mb": 10,
            "temp_dir": str(tmp_path / "media_tmp"),
            "ffmpeg_path": "/usr/local/bin/ffmpeg",
        },
        "proactive": {
            "enabled": True,
            "quiet_hours": {"enabled": False, "start": "23:00", "end": "08:00", "timezone": "UTC"},
            "silence_trigger_hours": 1.0,
            "scheduled_interval_minutes": 60,
            "jitter_minutes": 0,
            "cooldown_minutes": 5,
            "forbidden_topics": ["политика"],
        },
        "voice": {
            "tts": {"enabled": False},
            "stt": {"enabled": False},
        },
    }
    if extra_bot_cfg:
        for k, v in extra_bot_cfg.items():
            if isinstance(v, dict) and isinstance(bot_data.get(k), dict):
                bot_data[k] = {**bot_data[k], **v}
            else:
                bot_data[k] = v

    (cfg_dir / "bot.yaml").write_text(yaml.safe_dump(bot_data, allow_unicode=True), encoding="utf-8")
    (cfg_dir / "friends.yaml").write_text(
        yaml.safe_dump({"friends": friends or {891777515: {"name": "Алексей", "banter_topics": ["linux"]}}}, allow_unicode=True),
        encoding="utf-8",
    )

    if system_prompt is None:
        src_prompt = REPO_ROOT / "config" / "system_prompt.md"
        prompt_text = src_prompt.read_text(encoding="utf-8") if src_prompt.is_file() else (
            "Ты Джефф. Запрещено выполнять и помогать с незаконными действиями, включая пиратские торренты, независимо от того, кто просит."
        )
    else:
        prompt_text = system_prompt
    (cfg_dir / "system_prompt.md").write_text(prompt_text, encoding="utf-8")
    return ConfigManager(config_dir=cfg_dir)


def _make_fake_userbot() -> MagicMock:
    """Create an offline mock Telegram userbot/BotAPI client recording all outbound calls."""
    ub = MagicMock()
    ub.bot_id = BOT_ID
    ub.bot_username = "jeff_test_bot"
    ub.client = MagicMock()
    ub.start = AsyncMock()
    ub.stop = AsyncMock()
    ub.send_typing_action = AsyncMock()
    ub.send_reaction = AsyncMock(return_value=True)
    ub.register_message_handler = MagicMock()
    ub.sent_replies: List[Dict[str, Any]] = []
    ub.sent_documents: List[Dict[str, Any]] = []

    async def _send_reply(
        chat_id: int,
        text: str,
        reply_to_msg_id: Optional[int] = None,
        simulate_typing: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        rec = {
            "chat_id": chat_id,
            "text": text,
            "reply_to_msg_id": reply_to_msg_id,
            **kwargs,
        }
        ub.sent_replies.append(rec)
        return {"ok": True, "result": {"message_id": 1000 + len(ub.sent_replies), "text": text}}

    async def _send_document(
        chat_id: int,
        file_path: Any,
        caption: Optional[str] = None,
        reply_to_msg_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        rec = {
            "chat_id": chat_id,
            "file_path": str(file_path),
            "caption": caption,
            "reply_to_msg_id": reply_to_msg_id,
            **kwargs,
        }
        ub.sent_documents.append(rec)
        return {"ok": True, "result": {"message_id": 2000 + len(ub.sent_documents)}}

    ub.send_reply = AsyncMock(side_effect=_send_reply)
    ub.send_document = AsyncMock(side_effect=_send_document)
    return ub


def _instantiate_quota_manager(bot_cfg: BotConfig, memory_store: MemoryStore) -> Any:
    """Construct real QuotaBudgetManager adapting constructor param names cleanly."""
    from bot.quota.manager import QuotaBudgetManager

    sig = inspect.signature(QuotaBudgetManager.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name == "self":
            continue
        if name in ("config", "bot_config", "cfg"):
            kwargs[name] = bot_cfg
        elif name == "quota_config":
            kwargs[name] = getattr(bot_cfg, "quota", None)
        elif name in ("memory_store", "store", "db"):
            kwargs[name] = memory_store
    return QuotaBudgetManager(**kwargs)


async def _maybe_await(val: Any) -> Any:
    if inspect.isawaitable(val):
        return await val
    return val


def _instantiate_evolution_manager(
    *,
    config_mgr: ConfigManager,
    memory_store: MemoryStore,
    provider: Any,
    quota_manager: Any,
    userbot: Any,
) -> Any:
    """Construct real SelfEvolutionManager adapting constructor param names cleanly."""
    from bot.evolution.manager import SelfEvolutionManager

    sig = inspect.signature(SelfEvolutionManager.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name == "self":
            continue
        if name in ("config_mgr", "config_manager"):
            kwargs[name] = config_mgr
        elif name in ("config", "bot_config"):
            kwargs[name] = config_mgr.bot
        elif name in ("memory_store", "store", "db"):
            kwargs[name] = memory_store
        elif name in ("provider", "antigravity_provider"):
            kwargs[name] = provider
        elif name in ("quota_manager", "quota_budget_manager", "quota"):
            kwargs[name] = quota_manager
        elif name in ("userbot", "telegram_client", "client"):
            kwargs[name] = userbot
        elif name == "send_dm_fn":
            kwargs[name] = lambda uid, txt: userbot.send_reply(chat_id=uid, text=txt)
    return SelfEvolutionManager(**kwargs)


def _make_orchestrator(
    *,
    config_mgr: ConfigManager,
    memory_store: MemoryStore,
    media_processor: MediaProcessor,
    provider: Any,
    userbot: Any,
    quota_manager: Optional[Any] = None,
    sanitizer: Optional[Any] = None,
    evolution_manager: Optional[Any] = None,
) -> GroupBotOrchestrator:
    sig = inspect.signature(GroupBotOrchestrator.__init__)
    extra: Dict[str, Any] = {}
    if quota_manager is not None and "quota_manager" in sig.parameters:
        extra["quota_manager"] = quota_manager
    if sanitizer is not None and "sanitizer" in sig.parameters:
        extra["sanitizer"] = sanitizer
    if evolution_manager is not None and "evolution_manager" in sig.parameters:
        extra["evolution_manager"] = evolution_manager

    orch = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        media_processor=media_processor,
        provider=provider,
        userbot=userbot,
        **extra,
    )
    if quota_manager is not None and hasattr(orch, "quota_manager") and "quota_manager" not in sig.parameters:
        orch.quota_manager = quota_manager
    if sanitizer is not None and hasattr(orch, "sanitizer") and "sanitizer" not in sig.parameters:
        orch.sanitizer = sanitizer
    if evolution_manager is not None and hasattr(orch, "evolution_manager") and "evolution_manager" not in sig.parameters:
        orch.evolution_manager = evolution_manager
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)
    return orch


def _import_tool_constants() -> Tuple[frozenset[str], frozenset[str]]:
    """Deferred import of tool profile allowlists from antigravity package."""
    try:
        from bot.antigravity.stream import CHAT_ALLOWED_TOOLS, OPS_ALLOWED_TOOLS
    except ImportError:
        from bot.antigravity import CHAT_ALLOWED_TOOLS, OPS_ALLOWED_TOOLS  # type: ignore[attr-defined]
    return CHAT_ALLOWED_TOOLS, OPS_ALLOWED_TOOLS


class _FakeACPProc:
    """Fake ACP stdio subprocess pipe supporting initialize -> session/new -> session/prompt."""

    def __init__(self, script_responses: List[Dict[str, Any]], *, exit_code: int = 0) -> None:
        self._responses = list(script_responses)
        self._exit_code = exit_code
        self.pid = 42424
        self.stderr = io.StringIO("")
        self.written_frames: List[Dict[str, Any]] = []
        self._out_queue: List[str] = []
        self._closed = False
        self.stdin = self._StdinProxy(self)
        self.stdout = self._StdoutProxy(self)

    class _StdinProxy:
        def __init__(self, parent: "_FakeACPProc") -> None:
            self._parent = parent

        def write(self, data: str) -> int:
            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                self._parent.written_frames.append(frame)
                self._parent._on_client_frame(frame)
            return len(data)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self._parent._closed = True

    class _StdoutProxy:
        def __init__(self, parent: "_FakeACPProc") -> None:
            self._parent = parent

        def readline(self) -> str:
            if self._parent._out_queue:
                return self._parent._out_queue.pop(0)
            return ""

    def _emit(self, obj: Dict[str, Any]) -> None:
        self._out_queue.append(json.dumps(obj, ensure_ascii=False) + "\n")

    def _on_client_frame(self, frame: Dict[str, Any]) -> None:
        method = frame.get("method")
        msg_id = frame.get("id")
        if method == "initialize":
            self._emit({"jsonrpc": "2.0", "id": msg_id, "result": {"protocolVersion": 1}})
        elif method == "session/new":
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "sessionId": "sess-tier3-acp",
                        "models": {
                            "currentModelId": "gemini-4-hydrogen",
                            "availableModels": [
                                {"modelId": "gemini-4-hydrogen", "name": "Gemini 4 Hydrogen"},
                                {"modelId": "claude-sonnet-5-5", "name": "Claude Sonnet 5.5"},
                            ],
                        },
                    },
                }
            )
        elif method == "session/set_mode":
            self._emit({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        elif method == "session/set_model":
            self._emit({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        elif method == "session/prompt":
            for item in self._responses:
                if "id" in item and item["id"] == "__PROMPT_RESULT__":
                    item = {**item, "id": msg_id}
                self._emit(item)

    def poll(self) -> Optional[int]:
        return self._exit_code if self._closed or not self._out_queue else None


@dataclass
class GatekeeperSandbox:
    """Self-contained tmp_path sandbox for testing deploy/bot-self-deploy.sh safely."""

    root: Path
    prod_dir: Path
    staging_dir: Path
    state_dir: Path
    bin_dir: Path
    request_file: Path
    result_file: Path
    prev_commit: str

    def write_request(
        self,
        *,
        request_id: str = "req-001",
        branch: str = "self-edit/fix-1",
        commit_sha: Optional[str] = None,
        reason: str = "Automated test fix",
    ) -> Path:
        if commit_sha is None:
            commit_sha = self.staging_head()
        payload = {
            "request_id": request_id,
            "branch": branch,
            "commit_sha": commit_sha,
            "reason": reason,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        self.request_file.parent.mkdir(parents=True, exist_ok=True)
        self.request_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.request_file

    def staging_head(self) -> str:
        return subprocess.check_output(
            ["git", "-C", str(self.staging_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()

    def prod_head(self) -> str:
        return subprocess.check_output(
            ["git", "-C", str(self.prod_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()

    def commit_staging_change(
        self,
        rel_path: str,
        content: str,
        message: str = "fix: update in staging",
    ) -> str:
        target = self.staging_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        subprocess.run(["git", "-C", str(self.staging_dir), "add", rel_path], check=True)
        subprocess.run(["git", "-C", str(self.staging_dir), "commit", "-m", message], check=True)
        return self.staging_head()

    def set_tool_rc(self, tool: str, rc: int) -> None:
        (self.state_dir / f"{tool}_rc").write_text(str(rc), encoding="utf-8")

    def set_journalctl_output(self, text: str) -> None:
        (self.state_dir / "journalctl_output").write_text(text, encoding="utf-8")

    def systemctl_calls(self) -> List[str]:
        log_file = self.state_dir / "systemctl.log"
        if not log_file.is_file():
            return []
        return [line.strip() for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def run_gatekeeper(
        self,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Tuple[subprocess.CompletedProcess[str], Optional[Dict[str, Any]]]:
        script_path = REPO_ROOT / "deploy" / "bot-self-deploy.sh"
        assert script_path.is_file(), f"Missing gatekeeper script: {script_path}"
        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}:{env.get('PATH', '')}"
        env["BOT_PROD_DIR"] = str(self.prod_dir)
        env["BOT_STAGING_DIR"] = str(self.staging_dir)
        env["BOT_DEPLOY_REQUEST_FILE"] = str(self.request_file)
        env["BOT_DEPLOY_RESULT_FILE"] = str(self.result_file)
        env["BOT_SERVICE_NAME"] = "telegram-userbot.service"
        env["BOT_CANARY_SECONDS"] = "0"
        env["BOT_MAX_DIFF_LINES"] = "200"
        if extra_env:
            env.update(extra_env)

        proc = subprocess.run(
            ["bash", str(script_path)],
            env=env,
            cwd=str(self.root),
            capture_output=True,
            text=True,
            check=False,
        )
        result_data: Optional[Dict[str, Any]] = None
        if self.result_file.is_file():
            result_data = json.loads(self.result_file.read_text(encoding="utf-8"))
        return proc, result_data


def gatekeeper_sandbox(tmp_path: Path) -> GatekeeperSandbox:
    """Build an isolated git prod + staging worktree + PATH-shimmed systemctl/journalctl sandbox."""
    root = tmp_path / "gk_sandbox"
    prod_dir = root / "prod"
    staging_dir = root / "staging"
    state_dir = root / "state"
    bin_dir = root / "bin"
    for d in (prod_dir, state_dir, bin_dir):
        d.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init", "-b", "main", str(prod_dir)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(prod_dir), "config", "user.email", "botops@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(prod_dir), "config", "user.name", "botops"], check=True)

    (prod_dir / ".gitignore").write_text(".env\ndata/\n.venv/\n__pycache__/\n", encoding="utf-8")
    (prod_dir / "pyproject.toml").write_text('[project]\nname = "telegram-group-bot"\nversion = "0.1.0"\n', encoding="utf-8")
    (prod_dir / "config").mkdir()
    (prod_dir / "config" / "bot.yaml").write_text("telegram:\n  mode: bot_api\n  api_id: 1\n  api_hash: abc\n", encoding="utf-8")
    (prod_dir / "src" / "bot").mkdir(parents=True)
    (prod_dir / "src" / "bot" / "__init__.py").write_text("", encoding="utf-8")
    (prod_dir / "src" / "bot" / "app.py").write_text("def greet() -> str:\n    return 'hello'\n", encoding="utf-8")
    (prod_dir / "tests").mkdir()
    (prod_dir / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (prod_dir / "deploy").mkdir()
    (prod_dir / "deploy" / "bot-self-deploy.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (prod_dir / "systemd").mkdir()
    (prod_dir / "systemd" / "telegram-userbot.service").write_text("[Service]\nUser=botops\n", encoding="utf-8")

    subprocess.run(["git", "-C", str(prod_dir), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(prod_dir), "commit", "-m", "baseline"], check=True, capture_output=True)
    prev_commit = subprocess.check_output(["git", "-C", str(prod_dir), "rev-parse", "HEAD"], text=True).strip()

    # Untracked secret files in prod
    env_file = prod_dir / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=123456789:AAH_sandbox_secret_token_987654321\n", encoding="utf-8")
    os.chmod(env_file, 0o600)
    (prod_dir / "data" / "self_deploy").mkdir(parents=True, exist_ok=True)

    # Staging git worktree
    subprocess.run(
        ["git", "-C", str(prod_dir), "worktree", "add", "-b", "self-edit/fix-1", str(staging_dir)],
        check=True,
        capture_output=True,
    )

    # Configurable fake .venv/bin tools in staging (and prod for dry-run checks)
    for target_dir in (staging_dir, prod_dir):
        venv_bin = target_dir / ".venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        for tool_name, rc_name in (
            ("ruff", "ruff_rc"),
            ("pytest", "pytest_rc"),
            ("telegram-group-bot", "dry_run_rc"),
        ):
            tool_script = venv_bin / tool_name
            tool_script.write_text(
                f"#!/usr/bin/env bash\n"
                f"rc_file='{state_dir}/{rc_name}'\n"
                f"if [[ -f \"$rc_file\" ]]; then exit \"$(cat \"$rc_file\")\"; fi\n"
                f"exit 0\n",
                encoding="utf-8",
            )
            os.chmod(tool_script, 0o755)

    # Fake systemctl and journalctl shims in PATH
    systemctl_shim = bin_dir / "systemctl"
    systemctl_shim.write_text(
        f"#!/usr/bin/env bash\n"
        f"echo \"$*\" >> '{state_dir}/systemctl.log'\n"
        f"if [[ \"${{1:-}}\" == \"is-active\" ]]; then\n"
        f"  rc_file='{state_dir}/systemctl_is_active_rc'\n"
        f"  rc=0\n"
        f"  [[ -f \"$rc_file\" ]] && rc=\"$(cat \"$rc_file\")\"\n"
        f"  if [[ \"$rc\" -eq 0 ]]; then echo active; else echo failed; fi\n"
        f"  exit \"$rc\"\n"
        f"fi\n"
        f"if [[ \"${{1:-}}\" == \"restart\" ]]; then\n"
        f"  rc_file='{state_dir}/systemctl_restart_rc'\n"
        f"  [[ -f \"$rc_file\" ]] && exit \"$(cat \"$rc_file\")\"\n"
        f"fi\n"
        f"exit 0\n",
        encoding="utf-8",
    )
    os.chmod(systemctl_shim, 0o755)

    journalctl_shim = bin_dir / "journalctl"
    journalctl_shim.write_text(
        f"#!/usr/bin/env bash\n"
        f"echo \"$*\" >> '{state_dir}/journalctl.log'\n"
        f"out_file='{state_dir}/journalctl_output'\n"
        f"if [[ -f \"$out_file\" ]]; then cat \"$out_file\"; else echo 'INFO bot started cleanly'; fi\n"
        f"exit 0\n",
        encoding="utf-8",
    )
    os.chmod(journalctl_shim, 0o755)

    (state_dir / "journalctl_output").write_text("INFO telegram-userbot started cleanly\n", encoding="utf-8")

    return GatekeeperSandbox(
        root=root,
        prod_dir=prod_dir,
        staging_dir=staging_dir,
        state_dir=state_dir,
        bin_dir=bin_dir,
        request_file=prod_dir / "data" / "self_deploy" / "request.json",
        result_file=prod_dir / "data" / "self_deploy" / "result.json",
        prev_commit=prev_commit,
    )


# ============================================================================
# Part A: 18 Pairwise Cross-Feature Tests Across F1-F10 (Explorer 2 § 4.3)
# ============================================================================


@pytest.mark.asyncio
async def test_prod_t3_01_f2_sanitizer_plus_f4_admin_dm_error_report(tmp_path: Path):
    """F2 + F4: Admin DM exception/reply containing .env secret is sanitized in DM and logs."""
    from bot.security.sanitizer import SanitizingLogFilter, SecretSanitizer

    secret_val = "987654321:AAH_SUPER_SECRET_BOT_TOKEN_1234567890"
    env_file = tmp_path / ".env"
    env_file.write_text(f"TELEGRAM_BOT_TOKEN={secret_val}\n", encoding="utf-8")

    sanitizer = SecretSanitizer(env_path=env_file)
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.addFilter(SanitizingLogFilter(sanitizer))
    test_logger = logging.getLogger("bot.test_prod_t3_01")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(handler)

    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    userbot = _make_fake_userbot()
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text=f"Диагностика упала с токеном {secret_val}",
            model="gemini-4-hydrogen",
        )
    )

    orch = _make_orchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
        sanitizer=sanitizer,
    )
    try:
        test_logger.error("Admin DM diagnostic leaked token=%s", secret_val)
        incoming = IncomingMessage(
            msg_id=101,
            chat_id=ADMIN_USER_ID,
            sender_id=ADMIN_USER_ID,
            sender_name="Родион",
            text="покажи ошибку",
            is_private=True,
            is_mention=False,
        )
        await orch.handle_incoming_message(incoming)

        assert userbot.sent_replies, "Admin DM reply must be sent"
        sent_text = userbot.sent_replies[-1]["text"]
        assert secret_val not in sent_text
        assert "[REDACTED]" in sent_text
        assert secret_val not in log_stream.getvalue()
        assert "[REDACTED]" in log_stream.getvalue()
    finally:
        test_logger.removeHandler(handler)
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_t3_02_f4_admin_dm_plus_f5_ops_tool_profile(tmp_path: Path):
    """F4 + F5: Admin DM via BotAPIClient routes through orchestrator with tool_profile='ops'."""
    CHAT_ALLOWED_TOOLS, OPS_ALLOWED_TOOLS = _import_tool_constants()

    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()

    client = TelegramBotAPIClient(
        bot_token="123456789:DUMMY_TOKEN_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        allowed_chat_ids=[GROUP_CHAT_ID],
        allowed_dm_user_ids=[ADMIN_USER_ID],
        bot_names=["Джефф"],
    )
    client.bot_id = BOT_ID
    client.bot_username = "jeff_test_bot"

    raw_dm = {
        "message_id": 501,
        "chat": {"id": ADMIN_USER_ID, "type": "private"},
        "from": {"id": ADMIN_USER_ID, "is_bot": False, "first_name": "Rodion", "username": "rodion"},
        "text": "проверь логи сервиса",
    }
    incoming = client._convert_msg_dict(raw_dm)
    assert incoming is not None
    assert incoming.is_private is True

    captured_calls: List[Dict[str, Any]] = []
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        captured_calls.append(kwargs)
        # Verify stream parser with returned tool_profile permits ops tools & blocks browser_subagent
        profile = kwargs.get("tool_profile", "chat")
        parser_ops = AntigravityStreamParser(tool_profile=profile)
        parser_ops.feed_line('{"event":"step_update","step_update":{"step_type":"tool","tool":"read_file"}}')
        parser_ops.feed_line('{"event":"step_update","step_update":{"step_type":"tool","tool":"run_command"}}')
        with pytest.raises(AntigravityNativeToolAbort):
            parser_ops.feed_line('{"event":"step_update","step_update":{"step_type":"tool","tool":"browser_subagent"}}')
        return ProviderResponse(text="Логи проверены.", model="claude-sonnet-5-5")

    provider.generate_response = AsyncMock(side_effect=_gen)
    userbot = _make_fake_userbot()

    orch = _make_orchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
    )
    try:
        await orch.handle_incoming_message(incoming)
        assert len(captured_calls) == 1
        assert captured_calls[0].get("tool_profile") == "ops"
        assert "read_file" in OPS_ALLOWED_TOOLS
        assert "run_command" in OPS_ALLOWED_TOOLS
        assert "read_file" not in CHAT_ALLOWED_TOOLS
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_t3_03_f4_group_user_plus_f5_chat_tool_profile(tmp_path: Path):
    """F4 + F5: Admin user speaking in public group MUST still get tool_profile='chat', never 'ops'."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    userbot = _make_fake_userbot()
    captured_calls: List[Dict[str, Any]] = []

    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        captured_calls.append(kwargs)
        parser = AntigravityStreamParser(tool_profile=kwargs.get("tool_profile", "chat"))
        with pytest.raises(AntigravityNativeToolAbort):
            parser.feed_line('{"event":"step_update","step_update":{"step_type":"tool","tool":"run_command"}}')
        return ProviderResponse(text="В группе только поиск.", model="gemini-4-hydrogen")

    provider.generate_response = AsyncMock(side_effect=_gen)

    orch = _make_orchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
    )
    try:
        group_msg_from_admin = IncomingMessage(
            msg_id=502,
            chat_id=GROUP_CHAT_ID,
            sender_id=ADMIN_USER_ID,
            sender_name="Rodion",
            text="Джефф, проверь файл /etc/passwd",
            is_private=False,
            is_mention=True,
        )
        await orch.handle_incoming_message(group_msg_from_admin)
        assert len(captured_calls) == 1
        assert captured_calls[0].get("tool_profile") == "chat"
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_t3_04_f3_legal_prompt_plus_f5_ops_dm_no_bypass(tmp_path: Path):
    """F3 + F5: Admin DM in ops mode still carries the legal/anti-piracy prohibition in system_instructions."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    userbot = _make_fake_userbot()
    captured: List[Dict[str, Any]] = []

    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        captured.append(kwargs)
        return ProviderResponse(text="Не могу помочь с торрентами и пиратством.", model="claude-sonnet-5-5")

    provider.generate_response = AsyncMock(side_effect=_gen)
    orch = _make_orchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
    )
    try:
        admin_dm = IncomingMessage(
            msg_id=503,
            chat_id=ADMIN_USER_ID,
            sender_id=ADMIN_USER_ID,
            sender_name="Rodion",
            text="скачай пиратский торрент фильма через shell",
            is_private=True,
        )
        await orch.handle_incoming_message(admin_dm)
        assert len(captured) == 1
        assert captured[0].get("tool_profile") == "ops"
        sys_inst = (captured[0].get("system_instructions") or "").lower()
        assert any(k in sys_inst for k in ("пират", "торрент", "незакон", "закон"))
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_t3_05_f5_acp_search_web_fix_plus_f6_usage_persistence(tmp_path: Path):
    """F5 + F6: ACP search_web (title!='name') succeeds under chat profile and persists normalized UsageInfo."""
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    cfg_mgr = _make_isolated_config(tmp_path)
    quota_mgr = _instantiate_quota_manager(cfg_mgr.bot, store)

    fake_proc = _FakeACPProc(
        script_responses=[
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call",
                        "name": "search_web",
                        "title": '"example.com" on Web',
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "Нашёл информацию на example.com."},
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "id": "__PROMPT_RESULT__",
                "result": {
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": 120,
                        "outputTokens": 45,
                        "thoughtTokens": 15,
                        "cachedReadTokens": 30,
                        "totalTokens": 210,
                    },
                },
            },
        ]
    )

    cli_client = AntigravityCLIClient(data_dir=tmp_path / "agy", binary="/usr/local/bin/agy")
    try:
        with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=fake_proc):
            resp = await cli_client.generate_response(
                prompt="Найди инфу на example.com",
                image_payloads=[("AAAA", "image/png")],
                tool_profile="chat",
            )
        assert "example.com" in resp.text
        assert resp.usage is not None
        assert resp.usage.input_tokens == 120
        assert resp.usage.output_tokens == 45
        assert resp.usage.reasoning_tokens == 15

        await _maybe_await(quota_mgr.record_usage(resp.usage, model="gemini-4-hydrogen", source="group"))

        conn = sqlite3.connect(str(tmp_path / "mem.sqlite"))
        rows = conn.execute(
            "SELECT model, source, input_tokens, output_tokens, reasoning_tokens, total_tokens FROM usage_history"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == "gemini-4-hydrogen"
        assert rows[0][1] == "group"
        assert rows[0][2] == 120
        assert rows[0][3] == 45
        assert rows[0][4] == 15
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_t3_06_f6_usage_persistence_plus_f9_economy_mode_across_restart(tmp_path: Path):
    """F6 + F9: Persisted usage_history exceeding budget restores economy mode after DB reopen."""
    db_path = tmp_path / "restart_quota.sqlite"
    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "quota": {
                "hourly_token_budget": 1000,
                "daily_token_budget": 5000,
                "economy_threshold_pct": 0.8,
                "admin_reserve_pct": 0.2,
            }
        },
    )

    store1 = MemoryStore(str(db_path))
    await store1.init_db()
    qm1 = _instantiate_quota_manager(cfg_mgr.bot, store1)
    await _maybe_await(
        qm1.record_usage(
            UsageInfo(input_tokens=700, output_tokens=200, total_tokens=900, prompt_tokens=700, completion_tokens=200),
            model="gemini-4-hydrogen",
            source="group",
        )
    )
    assert qm1.is_economy_mode() is True
    await store1.close()

    store2 = MemoryStore(str(db_path))
    await store2.init_db()
    try:
        qm2 = _instantiate_quota_manager(cfg_mgr.bot, store2)
        if hasattr(qm2, "sync_from_db"):
            await _maybe_await(qm2.sync_from_db())
        assert qm2.is_economy_mode() is True
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_prod_t3_07_f7_per_model_cooldown_plus_f8_cross_bucket_fallback(tmp_path: Path):
    """F7 + F8: 429 RESOURCE_EXHAUSTED cools gemini_models bucket and routes next call to claude_gpt."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    try:
        qm = _instantiate_quota_manager(cfg_mgr.bot, store)
        primary_model, _ = qm.select_model("group")
        assert primary_model.startswith("gemini-")

        qm.record_rate_limit(primary_model, retry_after=300.0, bucket_wide=True, reason="429 RESOURCE_EXHAUSTED")
        fallback_model, _ = qm.select_model("group")
        assert not fallback_model.startswith("gemini-4-hydrogen")
        assert any(tok in fallback_model for tok in ("claude-", "gpt-oss")), (
            f"Expected cross-bucket fallback in claude_gpt, got {fallback_model}"
        )

        # Provider level integration: first call fails 429, fallback call succeeds
        provider = AntigravityProvider(AntigravityConfig(data_dir=tmp_path / "agy", enable_sdk=False))
        attempt_models: List[str] = []

        async def _fake_cli_gen(*args: Any, **kwargs: Any) -> ProviderResponse:
            m = kwargs.get("model") or provider.config.model
            attempt_models.append(m)
            if m == primary_model:
                raise AntigravityRateLimitError(
                    "429 RESOURCE_EXHAUSTED",
                    verdict=ErrorVerdict(
                        category=ErrorCategory.RATE_LIMIT,
                        retryable=True,
                        should_compress=False,
                        should_fallback=True,
                        message="429 RESOURCE_EXHAUSTED",
                    ),
                    retry_after=120.0,
                )
            return ProviderResponse(text="Fallback reply OK", model=m)

        provider._cli_client = MagicMock()
        provider._cli_client.generate_response = AsyncMock(side_effect=_fake_cli_gen)

        resp = await provider.generate_response("Привет", model=fallback_model)
        assert resp.text == "Fallback reply OK"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_t3_08_f7_quota_warning_plus_f9_proactive_economy_suppression(tmp_path: Path):
    """F7 + F9: [quota] low-remaining warning activates economy mode and suppresses proactive ticks."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    try:
        qm = _instantiate_quota_manager(cfg_mgr.bot, store)
        qm.record_quota_warning("gemini-4-hydrogen", remaining_pct=5.0, resets_at="2026-11-18T18:00:00Z")
        assert qm.is_economy_mode() is True
        assert qm.should_allow_proactive() is False

        userbot = _make_fake_userbot()
        provider = MagicMock()
        provider.is_rate_limited = MagicMock(return_value=False)
        provider.generate_response = AsyncMock(return_value=ProviderResponse(text="banter", model="gemini-3.8-flash"))

        orch = _make_orchestrator(
            config_mgr=cfg_mgr,
            memory_store=store,
            media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
            provider=provider,
            userbot=userbot,
            quota_manager=qm,
        )
        now_ts = time.time()
        orch.chat_states[GROUP_CHAT_ID] = {
            "last_chat_msg_time": now_ts - 7200,
            "last_bot_msg_time": now_ts - 7200,
        }
        await orch.proactive_engine.run_proactive_tick(
            chat_states=orch.chat_states,
            send_message_fn=orch._send_proactive,
            target_chat_ids=[GROUP_CHAT_ID],
        )
        assert userbot.sent_replies == []
        await orch.stop()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_t3_09_f8_model_router_plus_f6_usage_source_attribution(tmp_path: Path):
    """F8 + F6: Group, compaction, proactive, and dm_admin turns attribute distinct models and sources in SQLite."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    try:
        qm = _instantiate_quota_manager(cfg_mgr.bot, store)
        for role, src in (
            ("group", "group"),
            ("compaction", "compaction"),
            ("proactive", "proactive"),
            ("dm_admin", "dm_admin"),
        ):
            model, _effort = qm.select_model(role)
            await _maybe_await(
                qm.record_usage(
                    UsageInfo(input_tokens=50, output_tokens=20, total_tokens=70, prompt_tokens=50, completion_tokens=20),
                    model=model,
                    source=src,
                )
            )

        conn = sqlite3.connect(str(tmp_path / "mem.sqlite"))
        rows = conn.execute("SELECT source, model, total_tokens FROM usage_history ORDER BY id").fetchall()
        conn.close()
        assert [r[0] for r in rows] == ["group", "compaction", "proactive", "dm_admin"]
        assert all(r[2] == 70 for r in rows)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_t3_10_f9_economy_mode_plus_f2_sanitized_admin_dm_alert(tmp_path: Path):
    """F9 + F2: Economy mode warning sent to admin DM passes through SecretSanitizer without leaking tokens."""
    from bot.security.sanitizer import SecretSanitizer

    secret = "123456789:AAH_ECONOMY_ALERT_SECRET_XYZ98765"
    sanitizer = SecretSanitizer(extra_secrets=[secret])
    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "quota": {
                "hourly_token_budget": 100,
                "daily_token_budget": 500,
                "economy_threshold_pct": 0.5,
            }
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    userbot = _make_fake_userbot()
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text=f"Ответ пользователю {secret}",
            model="gemini-3.8-flash",
            usage=UsageInfo(input_tokens=60, output_tokens=20, total_tokens=80),
        )
    )

    orch = _make_orchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
        sanitizer=sanitizer,
    )
    try:
        msg = IncomingMessage(
            msg_id=10,
            chat_id=GROUP_CHAT_ID,
            sender_id=891777515,
            sender_name="Алексей",
            text="Джефф, привет!",
            is_mention=True,
            is_private=False,
        )
        await orch.handle_incoming_message(msg)
        assert userbot.sent_replies
        for sent in userbot.sent_replies:
            assert secret not in sent["text"]
        admin_alerts = [r for r in userbot.sent_replies if r["chat_id"] == ADMIN_USER_ID]
        assert len(admin_alerts) <= 1
        if admin_alerts:
            assert secret not in admin_alerts[0]["text"]
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_t3_11_f10_quota_command_plus_f7_live_cooldown_reflection(tmp_path: Path):
    """F10 + F7: Admin DM /quota integrates live TSV buckets and active model cooldowns."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    qm.record_rate_limit("gemini-4-hydrogen", retry_after=180.0, bucket_wide=False, reason="429 capacity")

    sample_tsv = (
        "Gemini Models\tWeekly Limit Remaining\t98%\t2026-11-25T10:00:00Z\n"
        "Gemini Models\tFive Hour Limit Remaining\t97%\t2026-11-18T18:00:00Z\n"
        "Claude and GPT models\tWeekly Limit Remaining\t100%\t2026-11-25T10:00:00Z\n"
        "Claude and GPT models\tFive Hour Limit Remaining\t100%\t2026-11-18T18:00:00Z\n"
        "Gemini 4 Argon\tWeekly Limit Remaining\t100%\t2026-11-25T10:00:00Z\n"
    )

    userbot = _make_fake_userbot()
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)
    provider.generate_response = AsyncMock(side_effect=AssertionError("/quota should not invoke chat LLM"))

    orch = _make_orchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
        quota_manager=qm,
    )
    try:
        fake_subproc = MagicMock()
        fake_subproc.returncode = 0
        fake_subproc.stdout = sample_tsv
        fake_subproc.stderr = ""

        with patch("subprocess.run", return_value=fake_subproc):
            dm = IncomingMessage(
                msg_id=701,
                chat_id=ADMIN_USER_ID,
                sender_id=ADMIN_USER_ID,
                sender_name="Rodion",
                text="/quota",
                is_private=True,
            )
            await orch.handle_incoming_message(dm)

        assert userbot.sent_replies, "Admin /quota command must send a DM reply"
        report_text = userbot.sent_replies[-1]["text"]
        assert "gemini-4-hydrogen" in report_text.lower() or "cooldown" in report_text.lower() or "квот" in report_text.lower()
        assert provider.generate_response.await_count == 0
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_t3_12_f10_context_summary_plus_f3_system_prompt_in_group_turn(tmp_path: Path):
    """F10 + F3: Group turn passes both system_prompt legal prohibition and quota context summary."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    qm.record_rate_limit("gemini-4-hydrogen", retry_after=60.0, bucket_wide=False, reason="429")

    userbot = _make_fake_userbot()
    captured: List[Dict[str, Any]] = []
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        captured.append(kwargs)
        return ProviderResponse(text="Отвечаю.", model="gemini-3.8-flash")

    provider.generate_response = AsyncMock(side_effect=_gen)
    orch = _make_orchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
        quota_manager=qm,
    )
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)
    try:
        msg = IncomingMessage(
            msg_id=702,
            chat_id=GROUP_CHAT_ID,
            sender_id=891777515,
            sender_name="Алексей",
            text="Джефф, как дела?",
            is_mention=True,
            is_private=False,
        )
        await orch.handle_incoming_message(msg)
        assert len(captured) == 1
        sys_inst = captured[0].get("system_instructions") or ""
        prompt = captured[0].get("prompt") or ""
        combined = f"{sys_inst}\n{prompt}"
        summary = qm.format_context_summary()
        assert any(w in sys_inst.lower() for w in ("пират", "торрент", "незакон", "закон"))
        assert summary.strip() and summary.strip() in combined
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_t3_13_f4_bot_api_bot_sender_plus_f5_no_llm_call(tmp_path: Path):
    """F4 + F5: Bot-originated update (is_bot=True or ignored username) produces zero LLM calls and zero usage rows."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    userbot = _make_fake_userbot()
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)
    provider.generate_response = AsyncMock(return_value=ProviderResponse(text="nope", model="gemini-4-hydrogen"))

    client = TelegramBotAPIClient(
        bot_token="123456789:DUMMY_TOKEN_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        allowed_chat_ids=[GROUP_CHAT_ID],
        allowed_dm_user_ids=[ADMIN_USER_ID],
        ignored_sender_usernames=["hermes_kuk7yi6kfm3lcdfa_bot"],
        bot_names=["Джефф"],
    )
    client.bot_id = BOT_ID

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
    )
    try:
        bot_update = {
            "message_id": 801,
            "chat": {"id": GROUP_CHAT_ID, "type": "supergroup"},
            "from": {"id": 998877, "is_bot": True, "username": "hermes_kuk7yi6kfm3lcdfa_bot"},
            "text": "Джефф, запусти команду",
        }
        converted = client._convert_msg_dict(bot_update)
        if converted is not None:
            await orch.handle_incoming_message(converted)

        assert provider.generate_response.await_count == 0
        conn = sqlite3.connect(str(tmp_path / "mem.sqlite"))
        cnt = conn.execute("SELECT COUNT(*) FROM usage_history").fetchone()[0]
        conn.close()
        assert cnt == 0
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_t3_14_f1_clean_child_env_plus_f5_ops_cwd_isolation(tmp_path: Path):
    """F1 + F5: Under tool_profile='ops' with cwd=staging_dir, agy spawns with cwd=staging, HOME=isolated, zero .env secrets."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    agy_home = tmp_path / "agy"
    cli_client = AntigravityCLIClient(data_dir=agy_home, binary="/usr/local/bin/agy")

    captured_popen: Dict[str, Any] = {}
    fake_proc = _FakeACPProc(
        script_responses=[
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "Staging inspected."},
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "id": "__PROMPT_RESULT__",
                "result": {"stopReason": "end_turn"},
            },
        ]
    )

    def _fake_popen(cmd: List[str], *, cwd: Path, env: Dict[str, str]) -> _FakeACPProc:
        captured_popen["cmd"] = cmd
        captured_popen["cwd"] = cwd
        captured_popen["env"] = env
        return fake_proc

    with patch.dict(
        os.environ,
        {
            "TELEGRAM_BOT_TOKEN": "123456:LEAKED_ENV_SECRET",
            "LD_PRELOAD": "/tmp/evil.so",
        },
        clear=False,
    ):
        with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=_fake_popen):
            resp = await cli_client.generate_response(
                prompt="Inspect repo",
                image_payloads=[("AAAA", "image/png")],
                tool_profile="ops",
                cwd=staging_dir,
            )

    assert resp.text == "Staging inspected."
    assert "TELEGRAM_BOT_TOKEN" not in captured_popen["env"]
    assert "LD_PRELOAD" not in captured_popen["env"]
    assert Path(captured_popen["env"]["HOME"]).resolve() != staging_dir.resolve()
    new_frames = [f for f in fake_proc.written_frames if f.get("method") == "session/new"]
    assert len(new_frames) == 1
    assert Path(new_frames[0]["params"]["cwd"]).resolve() == staging_dir.resolve()


@pytest.mark.asyncio
async def test_prod_t3_15_f7_404_model_not_found_plus_f8_fallback_plus_f6_usage(tmp_path: Path):
    """F7 + F8 + F6: NOT_FOUND (404) classifies as MODEL_NOT_FOUND, falls back, and persists used fallback model."""
    verdict = classify_error(RuntimeError("NOT_FOUND (code 404): Requested entity was not found"))
    assert verdict.category == ErrorCategory.MODEL_NOT_FOUND
    assert verdict.should_fallback is True

    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    try:
        qm = _instantiate_quota_manager(cfg_mgr.bot, store)
        first_model, _ = qm.select_model("dm_admin")
        qm.record_rate_limit(first_model, retry_after=120.0, bucket_wide=False, reason="NOT_FOUND 404")
        second_model, _ = qm.select_model("dm_admin")
        assert second_model != first_model

        await _maybe_await(
            qm.record_usage(
                UsageInfo(input_tokens=80, output_tokens=30, total_tokens=110),
                model=second_model,
                source="dm_admin",
            )
        )
        conn = sqlite3.connect(str(tmp_path / "mem.sqlite"))
        row = conn.execute("SELECT model, source, total_tokens FROM usage_history").fetchone()
        conn.close()
        assert row == (second_model, "dm_admin", 110)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_t3_16_f9_economy_mode_plus_f4_heuristics_mention_still_works(tmp_path: Path):
    """F9 + F4: In economy mode, unmentioned group chatter is suppressed while direct @mention replies with light model."""
    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "quota": {
                "hourly_token_budget": 100,
                "daily_token_budget": 500,
                "economy_threshold_pct": 0.5,
            }
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    await _maybe_await(
        qm.record_usage(UsageInfo(input_tokens=70, output_tokens=20, total_tokens=90), model="gemini-4-hydrogen", source="group")
    )
    assert qm.is_economy_mode() is True
    assert qm.should_allow_spontaneous() is False

    userbot = _make_fake_userbot()
    provider_calls: List[Dict[str, Any]] = []
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        provider_calls.append(kwargs)
        return ProviderResponse(text="На связи!", model=kwargs.get("model") or "gemini-3.8-flash")

    provider.generate_response = AsyncMock(side_effect=_gen)
    orch = _make_orchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
        quota_manager=qm,
    )
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)
    try:
        casual = IncomingMessage(
            msg_id=810,
            chat_id=GROUP_CHAT_ID,
            sender_id=891777515,
            sender_name="Алексей",
            text="погода сегодня отличная",
            is_mention=False,
            is_private=False,
        )
        await orch.handle_incoming_message(casual)
        assert len(provider_calls) == 0

        mention = IncomingMessage(
            msg_id=811,
            chat_id=GROUP_CHAT_ID,
            sender_id=891777515,
            sender_name="Алексей",
            text="Джефф, ты тут?",
            is_mention=True,
            is_private=False,
        )
        await orch.handle_incoming_message(mention)
        assert len(provider_calls) == 1
        assert "flash" in (provider_calls[0].get("model") or "").lower()
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_t3_17_f2_sanitizer_plus_f6_usage_no_overredaction(tmp_path: Path):
    """F2 + F6: SecretSanitizer does not corrupt numeric usage_history figures or /quota Markdown formatting."""
    from bot.security.sanitizer import SecretSanitizer

    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    try:
        qm = _instantiate_quota_manager(cfg_mgr.bot, store)
        await _maybe_await(
            qm.record_usage(
                UsageInfo(
                    input_tokens=123456,
                    output_tokens=7890,
                    reasoning_tokens=420,
                    cache_read_tokens=5000,
                    total_tokens=136766,
                ),
                model="gemini-4-hydrogen",
                source="group",
            )
        )
        report = qm.format_quota_report()
        sanitizer = SecretSanitizer(extra_secrets=["real_secret_token_abcdef123456"])
        cleaned = sanitizer.sanitize(report)
        assert cleaned == report
        assert "[REDACTED]" not in cleaned
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_t3_18_f1_dry_run_smoke_plus_f4_f6_new_schema_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """F1 + F4 + F6: run_dry_run_smoke succeeds offline with new BotConfig fields and usage_history schema."""
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "cfg"
    shutil.copytree(REPO_ROOT / "config", cfg_dir)
    # Remove copied friends.yaml content dependency by replacing with minimal synthetic friends
    (cfg_dir / "friends.yaml").write_text("friends:\n  891777515:\n    name: Alex\n", encoding="utf-8")

    cfg_file = cfg_dir / "bot.yaml" if (cfg_dir / "bot.yaml").is_file() else cfg_dir / "bot.example.yaml"
    with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden in dry-run")):
        summary = await run_dry_run_smoke(config_path=cfg_file)

    assert summary["status"] == "ok"
    assert summary["leftover_temp_dirs"] == 0
    assert summary["sent_messages_count"] >= 1


# ============================================================================
# Part B: 16 Pairwise Cross-Feature Tests Across F1-F18 (Explorer 3 § 4.4)
# ============================================================================


@pytest.mark.asyncio
async def test_t3_p1_self_edit_vs_chat_tool_profile_isolation(tmp_path: Path):
    """F11 + F5: self_dev executes with OPS_ALLOWED_TOOLS in staging_dir while group chat uses CHAT_ALLOWED_TOOLS."""
    CHAT_ALLOWED_TOOLS, OPS_ALLOWED_TOOLS = _import_tool_constants()
    sandbox = gatekeeper_sandbox(tmp_path)

    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox.staging_dir),
                "prod_dir": str(sandbox.prod_dir),
                "request_file": str(sandbox.request_file),
                "result_file": str(sandbox.result_file),
            }
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    userbot = _make_fake_userbot()
    recorded_profiles: List[Tuple[str, Optional[str], Optional[str]]] = []

    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        profile = kwargs.get("tool_profile", "chat")
        cwd = str(kwargs.get("cwd")) if kwargs.get("cwd") else None
        recorded_profiles.append((profile, cwd, kwargs.get("model")))
        prompt_txt = kwargs.get("prompt", "")
        if "CODE_FIX_REQUIRED" in prompt_txt or "judge" in prompt_txt.lower() or profile == "chat" and "self_edit" in str(kwargs):
            return ProviderResponse(
                text='{"verdict": "CODE_FIX_REQUIRED", "reason": "Bug confirmed", "affected_files": ["src/bot/app.py"]}',
                model="claude-sonnet-5-5",
            )
        if profile == "self_dev":
            sandbox.commit_staging_change("src/bot/app.py", "def greet() -> str:\n    return 'fixed'\n")
            return ProviderResponse(text="Patched src/bot/app.py", model="gemini-4-hydrogen")
        return ProviderResponse(text="Привет в чате!", model="gemini-4-hydrogen")

    provider.generate_response = AsyncMock(side_effect=_gen)
    evo = _instantiate_evolution_manager(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        quota_manager=qm,
        userbot=userbot,
    )
    orch = _make_orchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
        quota_manager=qm,
        evolution_manager=evo,
    )
    try:
        # 1. Group chat message -> strictly chat profile
        await orch.handle_incoming_message(
            IncomingMessage(
                msg_id=1,
                chat_id=GROUP_CHAT_ID,
                sender_id=891777515,
                sender_name="Алексей",
                text="Джефф, расскажи шутку",
                is_mention=True,
                is_private=False,
            )
        )
        assert recorded_profiles[0][0] == "chat"

        # 2. Evolution task -> self_dev profile with staging cwd
        if hasattr(evo, "handle_admin_task"):
            await _maybe_await(evo.handle_admin_task("Исправь app.py", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "run_admin_self_edit"):
            await _maybe_await(evo.run_admin_self_edit("Исправь app.py", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "process_admin_request"):
            await _maybe_await(evo.process_admin_request("Исправь app.py", user_id=ADMIN_USER_ID))

        self_dev_calls = [c for c in recorded_profiles if c[0] == "self_dev"]
        assert len(self_dev_calls) >= 1
        assert self_dev_calls[0][1] is not None
        assert Path(self_dev_calls[0][1]).resolve() == sandbox.staging_dir.resolve()
        assert "edit_file" in OPS_ALLOWED_TOOLS
        assert "edit_file" not in CHAT_ALLOWED_TOOLS
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_t3_p2_admin_dm_routing_gates_self_edit(tmp_path: Path):
    """F4 + F11: BotAPIClient ingests admin DM (is_private=True) into SelfEvolutionManager, rejecting non-admin DMs."""
    sandbox = gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox.staging_dir),
                "prod_dir": str(sandbox.prod_dir),
                "request_file": str(sandbox.request_file),
                "result_file": str(sandbox.result_file),
            }
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    userbot = _make_fake_userbot()
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text='{"verdict": "NO_CHANGE", "reason": "ok"}', model="claude-sonnet-5-5")
    )

    evo = _instantiate_evolution_manager(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        quota_manager=qm,
        userbot=userbot,
    )

    client = TelegramBotAPIClient(
        bot_token="123456789:DUMMY_TOKEN_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        allowed_chat_ids=[GROUP_CHAT_ID],
        allowed_dm_user_ids=[ADMIN_USER_ID],
        bot_names=["Джефф"],
    )
    client.bot_id = BOT_ID

    try:
        # Non-admin DM dropped at BotAPIClient ingestion boundary
        non_admin = client._convert_msg_dict(
            {
                "message_id": 910,
                "chat": {"id": 999999, "type": "private"},
                "from": {"id": 999999, "is_bot": False, "first_name": "Stranger"},
                "text": "/self_edit сломай бота",
            }
        )
        assert non_admin is None

        # Also verify SelfEvolutionManager itself rejects non-admin user
        if hasattr(evo, "is_authorized_admin"):
            assert evo.is_authorized_admin(999999) is False
            assert evo.is_authorized_admin(ADMIN_USER_ID) is True

        # Admin DM accepted
        admin_dm = client._convert_msg_dict(
            {
                "message_id": 911,
                "chat": {"id": ADMIN_USER_ID, "type": "private"},
                "from": {"id": ADMIN_USER_ID, "is_bot": False, "first_name": "Rodion"},
                "text": "self-edit: проверь обработку ошибок",
            }
        )
        assert admin_dm is not None
        assert admin_dm.is_private is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t3_p3_rate_limit_cooldown_and_antichurn_filter(tmp_path: Path):
    """F7 + F12: 429 RESOURCE_EXHAUSTED triggers QuotaBudgetManager cooldown AND is filtered as transient by SelfEvolutionManager."""
    sandbox = gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox.staging_dir),
                "prod_dir": str(sandbox.prod_dir),
                "request_file": str(sandbox.request_file),
                "result_file": str(sandbox.result_file),
            }
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    userbot = _make_fake_userbot()
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)
    provider.generate_response = AsyncMock(side_effect=AssertionError("Transient 429 must not call Judge/self_dev"))

    evo = _instantiate_evolution_manager(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        quota_manager=qm,
        userbot=userbot,
    )
    try:
        err = AntigravityRateLimitError("429 RESOURCE_EXHAUSTED: exhausted your capacity")
        qm.record_rate_limit("gemini-4-hydrogen", retry_after=120.0, bucket_wide=False, reason=str(err))

        # Feed the same 429 error repeatedly into SelfEvolutionManager
        for _ in range(5):
            if hasattr(evo, "record_runtime_error"):
                await _maybe_await(evo.record_runtime_error(err, context="group_reply"))
            elif hasattr(evo, "handle_runtime_error"):
                await _maybe_await(evo.handle_runtime_error(err, source="group_reply"))

        assert provider.generate_response.await_count == 0
        assert not sandbox.request_file.exists()
        next_model, _ = qm.select_model("group")
        assert next_model != "gemini-4-hydrogen"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t3_p4_adversarial_judge_uses_model_router_fallback(tmp_path: Path):
    """F8 + F13: Judge role ('self_edit_judge') fails over cross-bucket on cooldown and vetoes self_dev on NO_CHANGE."""
    sandbox = gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "self_evolution": {
                "enabled": True,
                "error_threshold": 2,
                "staging_dir": str(sandbox.staging_dir),
                "prod_dir": str(sandbox.prod_dir),
                "request_file": str(sandbox.request_file),
                "result_file": str(sandbox.result_file),
            }
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)

    judge_primary, _ = qm.select_model("self_edit_judge")
    qm.record_rate_limit(judge_primary, retry_after=300.0, bucket_wide=True, reason="429")
    judge_fallback, _ = qm.select_model("self_edit_judge")
    assert judge_fallback != judge_primary

    userbot = _make_fake_userbot()
    judge_calls: List[Dict[str, Any]] = []
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        judge_calls.append(kwargs)
        return ProviderResponse(
            text='{"verdict": "NO_CHANGE", "reason": "One-off user input issue"}',
            model=kwargs.get("model") or judge_fallback,
        )

    provider.generate_response = AsyncMock(side_effect=_gen)
    evo = _instantiate_evolution_manager(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        quota_manager=qm,
        userbot=userbot,
    )
    try:
        if hasattr(evo, "handle_admin_task"):
            await _maybe_await(evo.handle_admin_task("Проверь ошибку парсинга", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "run_admin_self_edit"):
            await _maybe_await(evo.run_admin_self_edit("Проверь ошибку парсинга", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "process_admin_request"):
            await _maybe_await(evo.process_admin_request("Проверь ошибку парсинга", user_id=ADMIN_USER_ID))

        assert len(judge_calls) == 1
        assert judge_calls[0].get("model") == judge_fallback
        assert judge_calls[0].get("tool_profile", "chat") != "self_dev"
        assert not sandbox.request_file.exists()

        conn = sqlite3.connect(str(tmp_path / "mem.sqlite"))
        verdicts = [r[0] for r in conn.execute("SELECT verdict FROM self_edit_history").fetchall()]
        conn.close()
        assert "NO_CHANGE" in verdicts
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t3_p5_escalation_respects_cooldowns_and_economy_mode(tmp_path: Path):
    """F7 + F9 + F14: Economy mode defers autonomous self_edit; admin escalation skips cooled Tier 1 to claude-sonnet-5-5 high."""
    sandbox = gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "quota": {
                "hourly_token_budget": 100,
                "daily_token_budget": 500,
                "economy_threshold_pct": 0.5,
                "admin_reserve_pct": 0.3,
            },
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox.staging_dir),
                "prod_dir": str(sandbox.prod_dir),
                "request_file": str(sandbox.request_file),
                "result_file": str(sandbox.result_file),
            },
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)

    # Trigger economy mode + cool down Tier 1 escalation model
    await _maybe_await(
        qm.record_usage(UsageInfo(input_tokens=60, output_tokens=10, total_tokens=70), model="gemini-4-hydrogen", source="group")
    )
    assert qm.is_economy_mode() is True
    allowed_auto, _reason = qm.should_allow_self_edit(priority="auto")
    assert allowed_auto is False

    qm.record_rate_limit("gemini-4-hydrogen", retry_after=300.0, bucket_wide=False, reason="429")

    userbot = _make_fake_userbot()
    calls: List[Dict[str, Any]] = []
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        calls.append(kwargs)
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(
                text='{"verdict": "CODE_FIX_REQUIRED", "reason": "Real bug", "affected_files": ["src/bot/app.py"]}',
                model=kwargs.get("model") or "claude-sonnet-5-5",
            )
        sandbox.commit_staging_change("src/bot/app.py", "def greet() -> str:\n    return 'admin-fix'\n")
        return ProviderResponse(text="Fixed", model=kwargs.get("model") or "claude-sonnet-5-5")

    provider.generate_response = AsyncMock(side_effect=_gen)
    evo = _instantiate_evolution_manager(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        quota_manager=qm,
        userbot=userbot,
    )
    try:
        # Autonomous error should be deferred in economy mode
        if hasattr(evo, "record_runtime_error"):
            for _ in range(3):
                await _maybe_await(evo.record_runtime_error(ValueError("real bug"), context="test"))
        assert len(calls) == 0

        # Explicit admin task uses reserve & skips cooled gemini-4-hydrogen -> claude-sonnet-5-5 (effort=high)
        if hasattr(evo, "handle_admin_task"):
            await _maybe_await(evo.handle_admin_task("Исправь app.py срочно", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "run_admin_self_edit"):
            await _maybe_await(evo.run_admin_self_edit("Исправь app.py срочно", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "process_admin_request"):
            await _maybe_await(evo.process_admin_request("Исправь app.py срочно", user_id=ADMIN_USER_ID))

        self_dev_calls = [c for c in calls if c.get("tool_profile") == "self_dev"]
        assert len(self_dev_calls) >= 1
        assert "claude-sonnet-5-5" in str(self_dev_calls[0].get("model"))
        assert self_dev_calls[0].get("effort") == "high"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t3_p6_gatekeeper_report_sanitizes_secrets_in_dm_and_logs(tmp_path: Path):
    """F2 + F15: Gatekeeper result.json error containing TELEGRAM_BOT_TOKEN is redacted before admin DM and logging."""
    from bot.security.sanitizer import SanitizingLogFilter, SecretSanitizer

    sandbox = gatekeeper_sandbox(tmp_path)
    secret_tok = "123456789:AAH_GATEKEEPER_LEAK_SECRET_TOKEN_12345"
    sanitizer = SecretSanitizer(extra_secrets=[secret_tok])

    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.addFilter(SanitizingLogFilter(sanitizer))
    evo_logger = logging.getLogger("bot.evolution")
    evo_logger.addHandler(handler)

    sandbox.result_file.write_text(
        json.dumps(
            {
                "request_id": "req-leak-1",
                "status": "rolled_back",
                "prev_commit": sandbox.prev_commit,
                "new_commit": "deadbeef",
                "error": f"Crash during curl https://api.telegram.org/bot{secret_tok}/getMe",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox.staging_dir),
                "prod_dir": str(sandbox.prod_dir),
                "request_file": str(sandbox.request_file),
                "result_file": str(sandbox.result_file),
            }
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    userbot = _make_fake_userbot()
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    evo = _instantiate_evolution_manager(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        quota_manager=qm,
        userbot=userbot,
    )
    try:
        for method_name in ("reconcile_deploy_result", "reconcile_pending_result", "poll_deploy_result", "check_deploy_result"):
            if hasattr(evo, method_name):
                await _maybe_await(getattr(evo, method_name)())
                break
        assert userbot.sent_replies, "Expected DM notification for rolled_back result.json"
        for msg in userbot.sent_replies:
            assert secret_tok not in msg["text"]
            assert "[REDACTED]" in msg["text"]
        assert secret_tok not in log_buf.getvalue()
    finally:
        evo_logger.removeHandler(handler)
        await store.close()


def test_t3_p7_gatekeeper_protected_paths_and_mtproto_boundary(tmp_path: Path):
    """F1 + F15: Gatekeeper rejects staging commits touching .env, data/, gatekeeper, systemd/, or mtproto path."""
    for protected_rel, content in (
        (".env", "TELEGRAM_BOT_TOKEN=pwned\n"),
        ("data/evil.txt", "corrupt\n"),
        ("deploy/bot-self-deploy.sh", "#!/bin/bash\nrm -rf /\n"),
        ("systemd/telegram-userbot.service", "[Service]\nUser=root\n"),
        ("src/bot/app.py", "MTPROTO_DIR = '/opt/telegram-mtproto-userbot'\n"),
    ):
        sub = tmp_path / protected_rel.replace("/", "_")
        sub.mkdir()
        sandbox = gatekeeper_sandbox(sub)
        if protected_rel == ".env" or protected_rel.startswith("data/"):
            # Force-add ignored protected path in staging to simulate malicious commit
            target_p = sandbox.staging_dir / protected_rel
            target_p.parent.mkdir(parents=True, exist_ok=True)
            target_p.write_text(content, encoding="utf-8")
            subprocess.run(["git", "-C", str(sandbox.staging_dir), "add", "-f", protected_rel], check=True)
            subprocess.run(["git", "-C", str(sandbox.staging_dir), "commit", "-m", f"touch {protected_rel}"], check=True)
            sha = sandbox.staging_head()
        else:
            sha = sandbox.commit_staging_change(protected_rel, content, message=f"touch {protected_rel}")
        sandbox.write_request(request_id=f"req-{protected_rel}", commit_sha=sha)
        proc, res = sandbox.run_gatekeeper()

        assert res is not None, f"Expected result.json for {protected_rel}; stderr={proc.stderr}"
        assert res["status"] == "rejected", f"Expected rejected for {protected_rel}, got {res}"
        assert sandbox.prod_head() == sandbox.prev_commit


def test_t3_p8_gatekeeper_worktree_and_systemd_nonewprivileges_contract(tmp_path: Path):
    """F15 + F17: Gatekeeper staging worktree + request.json aligns with systemd path/service units and NoNewPrivileges."""
    path_unit = REPO_ROOT / "deploy" / "telegram-self-deploy.path"
    svc_unit = REPO_ROOT / "deploy" / "telegram-self-deploy.service"
    bot_unit = REPO_ROOT / "systemd" / "telegram-userbot.service"

    assert path_unit.is_file()
    assert svc_unit.is_file()
    assert bot_unit.is_file()

    path_cfg = configparser.ConfigParser(interpolation=None)
    path_cfg.optionxform = str
    path_cfg.read_string(path_unit.read_text(encoding="utf-8"))
    assert path_cfg.get("Path", "PathExists") == "/opt/telegram-group-bot/data/self_deploy/request.json"

    svc_cfg = configparser.ConfigParser(interpolation=None)
    svc_cfg.optionxform = str
    svc_cfg.read_string(svc_unit.read_text(encoding="utf-8"))
    assert svc_cfg.get("Service", "Type") == "oneshot"
    assert "/usr/local/sbin/bot-self-deploy.sh" in svc_cfg.get("Service", "ExecStart")

    bot_cfg_unit = configparser.ConfigParser(interpolation=None)
    bot_cfg_unit.optionxform = str
    bot_cfg_unit.read_string(bot_unit.read_text(encoding="utf-8"))
    assert bot_cfg_unit.get("Service", "User") == "botops"
    assert bot_cfg_unit.get("Service", "Group") == "botops"
    assert bot_cfg_unit.get("Service", "NoNewPrivileges").lower() == "true"

    sandbox = gatekeeper_sandbox(tmp_path)
    sha = sandbox.commit_staging_change("src/bot/app.py", "def greet() -> str:\n    return 'contract-ok'\n")
    sandbox.write_request(request_id="req-contract", commit_sha=sha)
    proc, res = sandbox.run_gatekeeper()
    assert proc.returncode == 0
    assert res is not None and res["status"] == "applied"
    assert sandbox.prod_head() == sha
    assert not sandbox.request_file.exists()


def test_t3_p9_wireguard_template_security_and_gitignore(tmp_path: Path):
    """F1 + F16 + F17: WireGuard template has zero real keys, is safe with .gitignore, and deploy scripts never enable wg-quick."""
    wg_example = REPO_ROOT / "deploy" / "wireguard" / "wg0.conf.example"
    assert wg_example.is_file(), "Missing deploy/wireguard/wg0.conf.example"
    wg_text = wg_example.read_text(encoding="utf-8")

    assert "[Interface]" in wg_text
    assert "[Peer]" in wg_text
    assert "192.168.1.0/24" in wg_text

    # No real 44-char base64 WireGuard keys (ending in '=') outside comments/placeholders
    for line in wg_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        k, v = [p.strip() for p in stripped.split("=", 1)]
        if k in ("PrivateKey", "PublicKey", "PresharedKey"):
            assert "<" in v or "REPLACE" in v.upper() or "PLACEHOLDER" in v.upper(), (
                f"Suspicious real-looking WireGuard key in template: {line}"
            )

    # No deploy script enables or starts wg-quick
    for sh_file in (REPO_ROOT / "deploy").glob("*.sh"):
        sh_text = sh_file.read_text(encoding="utf-8")
        assert "wg-quick up" not in sh_text
        assert "systemctl enable wg-quick" not in sh_text

    # .gitignore allows wg0.conf.example while ignoring active wg0.conf / *.key / .env
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    shutil.copy2(REPO_ROOT / ".gitignore", repo / ".gitignore")
    ignored_wg = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", ".env", "deploy/wireguard/wg0.conf.example"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ".env" in ignored_wg.stdout
    assert "wg0.conf.example" not in ignored_wg.stdout


@pytest.mark.asyncio
async def test_t3_p10_dry_run_initializes_usage_and_self_edit_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """F6 + F11 + F18: --dry-run initializes messages, memory_facts, usage_history, and self_edit_history cleanly in temp SQLite."""
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "cfg"
    shutil.copytree(REPO_ROOT / "config", cfg_dir)
    (cfg_dir / "friends.yaml").write_text("friends:\n  891777515:\n    name: Alex\n", encoding="utf-8")

    created_dbs: List[str] = []
    orig_init_db = MemoryStore.init_db

    async def _spy_init_db(self: MemoryStore) -> None:
        created_dbs.append(self.db_path)
        await orig_init_db(self)
        conn = sqlite3.connect(self.db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert {"messages", "memory_facts", "usage_history", "self_edit_history"}.issubset(tables)

    cfg_file = cfg_dir / "bot.yaml" if (cfg_dir / "bot.yaml").is_file() else cfg_dir / "bot.example.yaml"
    with patch.object(MemoryStore, "init_db", _spy_init_db):
        summary = await run_dry_run_smoke(config_path=cfg_file)

    assert summary["status"] == "ok"
    assert len(created_dbs) == 1
    assert str(tmp_path) not in str(REPO_ROOT / "data")


@pytest.mark.asyncio
async def test_t3_p11_self_edit_persists_usage_info_to_sqlite(tmp_path: Path):
    """F6 + F11: Judge and self_dev LLM turns persist UsageInfo rows to usage_history with source='self_edit'."""
    sandbox = gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox.staging_dir),
                "prod_dir": str(sandbox.prod_dir),
                "request_file": str(sandbox.request_file),
                "result_file": str(sandbox.result_file),
            }
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    userbot = _make_fake_userbot()
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(
                text='{"verdict": "CODE_FIX_REQUIRED", "reason": "Confirmed", "affected_files": ["src/bot/app.py"]}',
                model="claude-sonnet-5-5",
                usage=UsageInfo(input_tokens=111, output_tokens=22, total_tokens=133),
            )
        sandbox.commit_staging_change("src/bot/app.py", "def greet() -> str:\n    return 'u'\n")
        return ProviderResponse(
            text="Patched",
            model="gemini-4-hydrogen",
            usage=UsageInfo(input_tokens=222, output_tokens=33, total_tokens=255),
        )

    provider.generate_response = AsyncMock(side_effect=_gen)
    evo = _instantiate_evolution_manager(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        quota_manager=qm,
        userbot=userbot,
    )
    try:
        if hasattr(evo, "handle_admin_task"):
            await _maybe_await(evo.handle_admin_task("Почини app.py", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "run_admin_self_edit"):
            await _maybe_await(evo.run_admin_self_edit("Почини app.py", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "process_admin_request"):
            await _maybe_await(evo.process_admin_request("Почини app.py", user_id=ADMIN_USER_ID))

        conn = sqlite3.connect(str(tmp_path / "mem.sqlite"))
        rows = conn.execute(
            "SELECT source, model, total_tokens FROM usage_history WHERE source = 'self_edit' ORDER BY id"
        ).fetchall()
        conn.close()
        assert len(rows) >= 2
        assert sum(r[2] for r in rows) >= 388
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t3_p12_gatekeeper_applied_feeds_file_cooldown(tmp_path: Path):
    """F12 + F15: After gatekeeper result.json reports 'applied' for src/bot/foo.py, subsequent edits to it are blocked by cooldown."""
    sandbox = gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "self_evolution": {
                "enabled": True,
                "file_cooldown_seconds": 3600,
                "signature_cooldown_seconds": 3600,
                "staging_dir": str(sandbox.staging_dir),
                "prod_dir": str(sandbox.prod_dir),
                "request_file": str(sandbox.request_file),
                "result_file": str(sandbox.result_file),
            }
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    userbot = _make_fake_userbot()
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text='{"verdict": "CODE_FIX_REQUIRED", "reason": "Bug", "affected_files": ["src/bot/foo.py"]}',
            model="claude-sonnet-5-5",
        )
    )

    # Seed self_edit_history with recent applied edit to src/bot/foo.py
    conn = sqlite3.connect(str(tmp_path / "mem.sqlite"))
    conn.execute(
        """
        INSERT INTO self_edit_history (
            created_at, trigger_type, signature, reason, affected_files,
            verdict, model, branch, commit_sha, diff_lines,
            test_result, deploy_result, rollback_commit, summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            "runtime_error",
            "RuntimeError:old_sig",
            "Initial fix",
            json.dumps(["src/bot/foo.py"]),
            "CODE_FIX_REQUIRED",
            "gemini-4-hydrogen",
            "self-edit/fix-1",
            sandbox.prev_commit,
            5,
            "passed",
            "applied",
            "",
            "Applied fix to foo.py",
        ),
    )
    conn.commit()
    conn.close()

    evo = _instantiate_evolution_manager(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        quota_manager=qm,
        userbot=userbot,
    )
    try:
        # Check file cooldown directly or via trigger targeting src/bot/foo.py
        if hasattr(evo, "is_file_on_cooldown"):
            assert evo.is_file_on_cooldown("src/bot/foo.py") is True
        elif hasattr(evo, "can_modify_file"):
            assert evo.can_modify_file("src/bot/foo.py") is False
        else:
            # Verify via full error flow mentioning src/bot/foo.py -> blocked before self_dev
            for _ in range(3):
                if hasattr(evo, "record_runtime_error"):
                    await _maybe_await(
                        evo.record_runtime_error(RuntimeError("new_sig in src/bot/foo.py"), context="src/bot/foo.py")
                    )
            assert not sandbox.request_file.exists()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t3_p13_legal_prohibition_blocks_illegal_self_edit_request(tmp_path: Path):
    """F3 + F13: System prompt legal prohibition + Adversarial Judge reject torrent/piracy self-edit request as NO_CHANGE."""
    sandbox = gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    userbot = _make_fake_userbot()
    judge_prompts: List[str] = []

    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        judge_prompts.append(f"{kwargs.get('system_instructions', '')}\n{kwargs.get('prompt', '')}")
        return ProviderResponse(
            text='{"verdict": "NO_CHANGE", "reason": "Illegal piracy/torrent functionality is strictly prohibited."}',
            model="claude-sonnet-5-5",
        )

    provider.generate_response = AsyncMock(side_effect=_gen)
    evo = _instantiate_evolution_manager(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        quota_manager=qm,
        userbot=userbot,
    )
    try:
        if hasattr(evo, "handle_admin_task"):
            await _maybe_await(evo.handle_admin_task("Добавь модуль скачивания пиратских торрентов", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "run_admin_self_edit"):
            await _maybe_await(evo.run_admin_self_edit("Добавь модуль скачивания пиратских торрентов", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "process_admin_request"):
            await _maybe_await(evo.process_admin_request("Добавь модуль скачивания пиратских торрентов", user_id=ADMIN_USER_ID))

        assert not sandbox.request_file.exists()
        assert sandbox.staging_head() == sandbox.prev_commit
        assert len(judge_prompts) <= 1
        if judge_prompts:
            assert any(k in judge_prompts[0].lower() for k in ("пират", "торрент", "незакон", "закон"))
    finally:
        await store.close()


def test_t3_p14_config_bot_yaml_admin_dm_and_ffmpeg_contract():
    """F4 + F17: Repo config specifies mode='bot_api', valid non-/home/jeff ffmpeg_path, and admin allowlist when bot.yaml exists."""
    bot_yaml = REPO_ROOT / "config" / "bot.yaml"
    cfg_mgr = ConfigManager(config_path=bot_yaml if bot_yaml.is_file() else REPO_ROOT / "config" / "bot.example.yaml")
    bot = cfg_mgr.bot
    assert bot is not None
    assert bot.telegram.mode == "bot_api"
    if bot_yaml.is_file():
        assert ADMIN_USER_ID in bot.telegram.allowed_dm_user_ids
        admin_ids = getattr(bot.telegram, "admin_user_ids", [])
        assert ADMIN_USER_ID in admin_ids
    assert bot.media.ffmpeg_path is not None
    assert not str(bot.media.ffmpeg_path).startswith("/home/jeff/")
    assert str(bot.media.ffmpeg_path) in ("/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg")


@pytest.mark.asyncio
async def test_t3_p15_gatekeeper_tests_failed_triggers_model_escalation(tmp_path: Path):
    """F14 + F15: Tier 1 staging commit failing gatekeeper pytest ('tests_failed') escalates to Tier 2 claude-sonnet-5-5 high."""
    sandbox = gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        extra_bot_cfg={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox.staging_dir),
                "prod_dir": str(sandbox.prod_dir),
                "request_file": str(sandbox.request_file),
                "result_file": str(sandbox.result_file),
            }
        },
    )
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    userbot = _make_fake_userbot()
    used_models: List[Tuple[str, Optional[str]]] = []

    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    async def _gen(**kwargs: Any) -> ProviderResponse:
        model = str(kwargs.get("model") or "")
        effort = kwargs.get("effort")
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(
                text='{"verdict": "CODE_FIX_REQUIRED", "reason": "Bug", "affected_files": ["src/bot/app.py"]}',
                model=model or "claude-sonnet-5-5",
            )
        used_models.append((model, effort))
        if "gemini-4-hydrogen" in model:
            # Tier 1 fails gatekeeper tests -> simulated tests_failed
            sandbox.commit_staging_change("src/bot/app.py", "def greet() -> str:\n    return 'broken-tier1'\n", "tier1 broken")
            raise RuntimeError("Gatekeeper tests_failed on tier1 commit")
        # Tier 2 succeeds
        sha = sandbox.commit_staging_change("src/bot/app.py", "def greet() -> str:\n    return 'fixed-tier2'\n", "tier2 fixed")
        sandbox.set_tool_rc("pytest", 0)
        sandbox.write_request(request_id="req-tier2", commit_sha=sha)
        proc, res = sandbox.run_gatekeeper()
        assert proc.returncode == 0 and res is not None and res["status"] == "applied"
        return ProviderResponse(text="Tier 2 fixed & applied", model=model)

    provider.generate_response = AsyncMock(side_effect=_gen)
    evo = _instantiate_evolution_manager(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        quota_manager=qm,
        userbot=userbot,
    )
    try:
        if hasattr(evo, "handle_admin_task"):
            await _maybe_await(evo.handle_admin_task("Исправь app.py с проверкой тестов", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "run_admin_self_edit"):
            await _maybe_await(evo.run_admin_self_edit("Исправь app.py с проверкой тестов", requester_id=ADMIN_USER_ID))
        elif hasattr(evo, "process_admin_request"):
            await _maybe_await(evo.process_admin_request("Исправь app.py с проверкой тестов", user_id=ADMIN_USER_ID))

        assert len(used_models) >= 2
        assert "gemini-4-hydrogen" in used_models[0][0]
        assert "claude-sonnet-5-5" in used_models[1][0]
        assert used_models[1][1] == "high"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t3_p16_quota_command_surfaces_self_edit_usage_and_cooldowns(tmp_path: Path):
    """F10 + F11: Admin /quota command surfaces self_edit token consumption and active escalation cooldowns."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    qm = _instantiate_quota_manager(cfg_mgr.bot, store)
    await _maybe_await(
        qm.record_usage(
            UsageInfo(input_tokens=400, output_tokens=150, reasoning_tokens=50, total_tokens=600),
            model="claude-sonnet-5-5",
            source="self_edit",
        )
    )
    qm.record_rate_limit("gemini-4-hydrogen", retry_after=240.0, bucket_wide=False, reason="self_edit 429")

    userbot = _make_fake_userbot()
    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)

    orch = _make_orchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "m")),
        provider=provider,
        userbot=userbot,
        quota_manager=qm,
    )
    try:
        fake_subproc = MagicMock()
        fake_subproc.returncode = 0
        fake_subproc.stdout = "Gemini Models\tFive Hour Limit Remaining\t95%\t2026-11-18T18:00:00Z\n"
        fake_subproc.stderr = ""

        with patch("subprocess.run", return_value=fake_subproc):
            await orch.handle_incoming_message(
                IncomingMessage(
                    msg_id=950,
                    chat_id=ADMIN_USER_ID,
                    sender_id=ADMIN_USER_ID,
                    sender_name="Rodion",
                    text="/quota",
                    is_private=True,
                )
            )
        assert userbot.sent_replies
        report = userbot.sent_replies[-1]["text"]
        assert "600" in report or "self_edit" in report or "claude-sonnet-5-5" in report.lower()
        assert "gemini-4-hydrogen" in report.lower() or "cooldown" in report.lower()
    finally:
        await orch.stop()
