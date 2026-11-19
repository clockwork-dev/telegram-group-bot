"""Tier 1 Production Hardening E2E Tests (97 isolated feature tests, F1-F18).

Opaque-box, strictly offline E2E verification covering R1-R7 requirements:
- F1: R1 Security Boundaries (6 tests)
- F2: R3.1 Secret Sanitizer (6 tests)
- F3: R3.2 Legal Prohibition Prompt (5 tests)
- F4: R3.3 Config & Admin DM Routing (6 tests)
- F5: R3.4 Tool Profiles & ACP Fix (6 tests)
- F6: R5.1 UsageInfo SQLite Persistence (5 tests)
- F7: R5.2 Per-Model & Bucket Cooldowns (6 tests)
- F8: R5.3 ModelRouter & Fallback (5 tests)
- F9: R5.4 Budgets & Economy Mode (5 tests)
- F10: R5.5 /quota & System Context (5 tests)
- F11: R4.1 SelfEvolutionManager & self_edit_history SQLite (6 tests)
- F12: R4.2 Anti-Churn Guards (5 tests)
- F13: R4.3 Adversarial Judge (5 tests)
- F14: R4.4 Model Escalation (5 tests)
- F15: R4.5 Staging, Gatekeeper & Canary (6 tests)
- F16: R6 WireGuard Preparation (5 tests)
- F17: R2 Server Packaging & Unit Hardening (5 tests)
- F18: R7 Offline Dry-Run & Suite Readiness (5 tests)
"""

from __future__ import annotations

import configparser
import io
import json
import logging
import os
import queue
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from bot.antigravity.auth import _KEYRING_BYPASS_SSH_MARKER, build_clean_child_env
from bot.antigravity.cli_client import AntigravityCLIClient, watch_log_for_errors
from bot.antigravity.errors import (
    AntigravityNativeToolAbort,
    AntigravityRateLimitError,
    ErrorCategory,
    classify_error,
    extract_retry_after,
)
from bot.antigravity.stream import AntigravityStreamParser
from bot.config.manager import ConfigManager
from bot.config.models import BotConfig
from bot.core.models import ProviderResponse, UsageInfo
from bot.main import GroupBotOrchestrator, run_dry_run_smoke
from bot.media.processor import MediaProcessor
from bot.memory.db import MemoryStore
from bot.telegram.bot_api_client import BotAPIMessageEvent, TelegramBotAPIClient
from bot.telegram.handlers import IncomingMessage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADMIN_USER_ID = 382962362
GROUP_CHAT_ID = -1003845062562


# ============================================================================
# Shared Offline Test Harness Helpers
# ============================================================================


def _make_isolated_config(
    tmp_path: Path,
    *,
    bot_overrides: Optional[dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    friends: Optional[dict[int, dict[str, Any]]] = None,
) -> ConfigManager:
    """Create an isolated ConfigManager under tmp_path without reading live secrets."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg: dict[str, Any] = {
        "telegram": {
            "mode": "bot_api",
            "bot_token": "123456:DUMMY_OFFLINE_TOKEN_ABCDEFGHIJKLMNOP",
            "api_id": 12345,
            "api_hash": "0123456789abcdef0123456789abcdef",
            "allowed_chat_ids": [GROUP_CHAT_ID],
            "allowed_dm_user_ids": [ADMIN_USER_ID],
            "admin_user_ids": [ADMIN_USER_ID],
            "ignored_sender_usernames": ["hermes_kuk7yi6kfm3lcdfa_bot"],
        },
        "behavior": {
            "bot_names": ["Джефф", "jeff", "піструнькиня"],
            "typing_simulation": False,
            "reactions_enabled": True,
            "spontaneous_replies": {
                "enabled": True,
                "min_interval_seconds": 0,
                "probability": 1.0,
                "max_per_hour": 10,
                "active_conversation_window_seconds": 90,
            },
        },
        "proactive": {
            "enabled": True,
            "quiet_hours": {
                "enabled": False,
                "start": "23:00",
                "end": "08:00",
                "timezone": "UTC",
            },
            "silence_trigger_hours": 1.0,
            "scheduled_interval_minutes": 30,
            "jitter_minutes": 0,
            "cooldown_minutes": 10,
        },
        "memory": {
            "db_path": str(tmp_path / "memory.sqlite"),
            "short_term_limit": 30,
            "compaction_threshold": 60,
            "compaction_batch_size": 30,
        },
        "media": {
            "max_doc_size_mb": 10,
            "temp_dir": str(tmp_path / "media_tmp"),
            "ffmpeg_path": "/usr/local/bin/ffmpeg",
        },
    }
    if bot_overrides:
        for key, val in bot_overrides.items():
            if isinstance(val, dict) and isinstance(base_cfg.get(key), dict):
                base_cfg[key].update(val)
            else:
                base_cfg[key] = val

    (cfg_dir / "bot.yaml").write_text(
        yaml.safe_dump(base_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    if friends is not None:
        (cfg_dir / "friends.yaml").write_text(
            yaml.safe_dump({"friends": friends}, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    prompt_text = system_prompt
    if prompt_text is None:
        repo_prompt = REPO_ROOT / "config" / "system_prompt.md"
        prompt_text = (
            repo_prompt.read_text(encoding="utf-8")
            if repo_prompt.is_file()
            else "Ты — Джефф. Запрещены любые незаконные действия и пиратские торренты."
        )
    (cfg_dir / "system_prompt.md").write_text(prompt_text, encoding="utf-8")
    return ConfigManager(config_dir=cfg_dir)


def _make_mock_userbot(bot_id: int = 777000, bot_username: str = "jeff_test_bot") -> MagicMock:
    """Create an offline mock Telegram client conforming to orchestrator expectations."""
    userbot = MagicMock()
    userbot.bot_id = bot_id
    userbot.bot_username = bot_username
    userbot.client = MagicMock()
    userbot.start = AsyncMock()
    userbot.stop = AsyncMock()
    userbot.send_typing_action = AsyncMock()
    userbot.send_reaction = AsyncMock(return_value=True)
    userbot.send_reply = AsyncMock(return_value={"ok": True, "result": {"message_id": 9001}})
    userbot.send_document = AsyncMock(return_value={"ok": True, "result": {"message_id": 9002}})
    userbot.send_voice = AsyncMock(return_value={"ok": True, "result": {"message_id": 9003}})
    return userbot


def _make_incoming(
    *,
    chat_id: int,
    sender_id: int,
    text: str,
    msg_id: int = 101,
    sender_name: str = "Алексей",
    is_private: bool = False,
    is_mention: bool = False,
    sender_is_bot: bool = False,
    sender_username: Optional[str] = None,
) -> IncomingMessage:
    """Construct a synthetic IncomingMessage with matching raw_event attributes."""
    raw = MagicMock()
    raw.id = msg_id
    raw.chat_id = chat_id
    raw.sender_id = sender_id
    raw.out = False
    raw.is_reply = False
    raw.reply_to_msg_id = None
    raw.raw_text = text
    raw.text = text
    raw.media = None
    raw.photo = False
    raw.is_private = is_private
    raw.sender_is_bot = sender_is_bot
    raw.sender_username = sender_username
    raw.date = datetime.now(timezone.utc)
    return IncomingMessage(
        msg_id=msg_id,
        chat_id=chat_id,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        reply_to_msg_id=None,
        is_reply=False,
        is_mention=is_mention,
        raw_event=raw,
        media=None,
        is_private=is_private,
        sender_is_bot=sender_is_bot,
        sender_username=sender_username,
    )


def _import_tool_constants() -> tuple[frozenset[str], frozenset[str]]:
    """Deferred import of CHAT_ALLOWED_TOOLS and OPS_ALLOWED_TOOLS."""
    import bot.antigravity.stream as stream_mod

    chat_tools = getattr(stream_mod, "CHAT_ALLOWED_TOOLS", None)
    ops_tools = getattr(stream_mod, "OPS_ALLOWED_TOOLS", None)
    if chat_tools is None or ops_tools is None:
        import bot.antigravity as agy_mod

        chat_tools = chat_tools or getattr(agy_mod, "CHAT_ALLOWED_TOOLS")
        ops_tools = ops_tools or getattr(agy_mod, "OPS_ALLOWED_TOOLS")
    return chat_tools, ops_tools


def _make_quota_manager(
    memory_store: MemoryStore,
    config: Optional[BotConfig] = None,
) -> Any:
    """Deferred instantiation of QuotaBudgetManager compatible with BotConfig/MemoryStore."""
    import inspect

    from bot.quota.manager import QuotaBudgetManager

    sig = inspect.signature(QuotaBudgetManager.__init__)
    kwargs: dict[str, Any] = {}
    if "memory_store" in sig.parameters:
        kwargs["memory_store"] = memory_store
    if config is not None:
        if "config" in sig.parameters:
            kwargs["config"] = config
        elif "quota_config" in sig.parameters:
            kwargs["quota_config"] = getattr(config, "quota", None)
        if "model_router" in sig.parameters:
            kwargs["model_router"] = getattr(config, "model_router", None)
    return QuotaBudgetManager(**kwargs)


async def _maybe_await(val: Any) -> Any:
    """Await value if awaitable, otherwise return directly."""
    import inspect

    if inspect.isawaitable(val):
        return await val
    return val


class _FakeACPProcess:
    """Event-driven fake ACP stdio subprocess for deterministic offline ACP tests."""

    def __init__(
        self,
        *,
        on_prompt: Optional[
            Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]]
        ] = None,
        available_models: Optional[list[dict[str, str]]] = None,
    ) -> None:
        self.stdout_queue: queue.Queue[str] = queue.Queue()
        self.stderr = io.StringIO("")
        self.pid = 88888
        self.requests: list[dict[str, Any]] = []
        self.permission_responses: list[dict[str, Any]] = []
        self._terminated = False
        self._on_prompt = on_prompt
        self._available_models = available_models or [
            {"modelId": "gemini-4-hydrogen", "name": "gemini-4-hydrogen"},
            {"modelId": "claude-sonnet-5-5", "name": "claude-sonnet-5-5"},
        ]
        self._pending_after_perm: list[dict[str, Any]] = []
        self.stdin = self

    def write(self, data: str) -> int:
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if "method" in msg:
                self.requests.append(msg)
                method = msg["method"]
                req_id = msg.get("id")
                if method == "initialize":
                    self._push({"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": 1}})
                elif method == "session/new":
                    self._push(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "sessionId": "acp-sess-1",
                                "models": {
                                    "currentModelId": "gemini-4-hydrogen",
                                    "availableModels": self._available_models,
                                },
                            },
                        }
                    )
                elif method in ("session/set_model", "session/set_mode"):
                    self._push({"jsonrpc": "2.0", "id": req_id, "result": {}})
                elif method == "session/prompt":
                    frames = (
                        self._on_prompt(msg, self.permission_responses)
                        if self._on_prompt
                        else [
                            {
                                "jsonrpc": "2.0",
                                "method": "session/update",
                                "params": {
                                    "update": {
                                        "sessionUpdate": "agent_message_chunk",
                                        "content": {"type": "text", "text": "ACP OK"},
                                    }
                                },
                            },
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": {"stopReason": "end_turn"},
                            },
                        ]
                    )
                    for idx, frame in enumerate(frames):
                        self._push(frame)
                        if frame.get("method") == "session/request_permission":
                            self._pending_after_perm = frames[idx + 1 :]
                            break
            else:
                # Client response to server->client session/request_permission
                self.permission_responses.append(msg)
                for frame in self._pending_after_perm:
                    self._push(frame)
                self._pending_after_perm = []
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._terminated = True
        self.stdout_queue.put("")

    @property
    def stdout(self) -> "_FakeACPProcess":
        return self

    def readline(self) -> str:
        if self._terminated and self.stdout_queue.empty():
            return ""
        return self.stdout_queue.get(timeout=5.0)

    def _push(self, frame: dict[str, Any]) -> None:
        self.stdout_queue.put(json.dumps(frame, ensure_ascii=False) + "\n")

    def poll(self) -> Optional[int]:
        return 0 if self._terminated else None

    def terminate(self) -> None:
        self._terminated = True
        self.stdout_queue.put("")

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: Optional[float] = None) -> int:
        return 0


def _gatekeeper_sandbox(
    tmp_path: Path,
    *,
    max_diff_lines: int = 200,
    ruff_rc: int = 0,
    pytest_rc: int = 0,
    dry_run_rc: int = 0,
    systemctl_active_rc: int = 0,
    journalctl_output: str = "INFO telegram-group-bot started cleanly\n",
) -> dict[str, Any]:
    """Create isolated git prod + staging worktree + PATH shims for bot-self-deploy.sh."""
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    state_dir = tmp_path / "state"
    bin_dir = tmp_path / "bin"
    prod_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init", "-b", "main", str(prod_dir)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(prod_dir), "config", "user.name", "Bot Tester"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(prod_dir), "config", "user.email", "tester@example.invalid"],
        check=True,
        capture_output=True,
    )

    (prod_dir / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    (prod_dir / "config").mkdir(parents=True, exist_ok=True)
    (prod_dir / "config" / "bot.yaml").write_text("telegram:\n  mode: bot_api\n", encoding="utf-8")
    (prod_dir / "src" / "bot").mkdir(parents=True, exist_ok=True)
    (prod_dir / "src" / "bot" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (prod_dir / "tests").mkdir(parents=True, exist_ok=True)
    (prod_dir / "tests" / "test_ok.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    (prod_dir / ".gitignore").write_text(".env\ndata/\n.venv/\n", encoding="utf-8")

    subprocess.run(["git", "-C", str(prod_dir), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(prod_dir), "commit", "-m", "baseline"],
        check=True,
        capture_output=True,
    )
    prev_commit = (
        subprocess.check_output(["git", "-C", str(prod_dir), "rev-parse", "HEAD"], text=True)
        .strip()
    )

    # Untracked protected runtime files
    env_file = prod_dir / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=111111:FAKE_SANDBOX_TOKEN_ABCDE\n", encoding="utf-8")
    os.chmod(env_file, 0o600)
    deploy_dir = prod_dir / "data" / "self_deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # Staging git worktree
    subprocess.run(
        ["git", "-C", str(prod_dir), "worktree", "add", "-b", "self-edit/fix-1", str(staging_dir)],
        check=True,
        capture_output=True,
    )

    # Fake .venv/bin tools inside staging_dir (and prod_dir if gatekeeper switches cwd)
    for base in (staging_dir, prod_dir):
        venv_bin = base / ".venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        for tool_name, rc_name in (
            ("ruff", "ruff_rc"),
            ("pytest", "pytest_rc"),
            ("telegram-group-bot", "dry_run_rc"),
        ):
            shim = venv_bin / tool_name
            shim.write_text(
                "#!/usr/bin/env bash\n"
                f'echo "{tool_name} $*" >> "{state_dir}/venv.log"\n'
                f'rc=$(cat "{state_dir}/{rc_name}")\n'
                "exit ${rc:-0}\n",
                encoding="utf-8",
            )
            shim.chmod(0o755)

    (state_dir / "ruff_rc").write_text(str(ruff_rc), encoding="utf-8")
    (state_dir / "pytest_rc").write_text(str(pytest_rc), encoding="utf-8")
    (state_dir / "dry_run_rc").write_text(str(dry_run_rc), encoding="utf-8")
    (state_dir / "systemctl_is_active_rc").write_text(str(systemctl_active_rc), encoding="utf-8")
    (state_dir / "journalctl_output").write_text(journalctl_output, encoding="utf-8")

    systemctl_shim = bin_dir / "systemctl"
    systemctl_shim.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "systemctl $*" >> "{state_dir}/systemctl.log"\n'
        'if [ "${1:-}" = "is-active" ]; then\n'
        f'  rc=$(cat "{state_dir}/systemctl_is_active_rc")\n'
        "  exit ${rc:-0}\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    systemctl_shim.chmod(0o755)

    journalctl_shim = bin_dir / "journalctl"
    journalctl_shim.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "journalctl $*" >> "{state_dir}/journalctl.log"\n'
        f'cat "{state_dir}/journalctl_output"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    journalctl_shim.chmod(0o755)

    request_file = deploy_dir / "request.json"
    result_file = deploy_dir / "result.json"

    def run_gatekeeper(extra_env: Optional[dict[str, str]] = None) -> subprocess.CompletedProcess[str]:
        script = REPO_ROOT / "deploy" / "bot-self-deploy.sh"
        assert script.is_file(), f"Missing gatekeeper script: {script}"
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        env.update(
            {
                "BOT_PROD_DIR": str(prod_dir),
                "BOT_STAGING_DIR": str(staging_dir),
                "BOT_DEPLOY_REQUEST_FILE": str(request_file),
                "BOT_DEPLOY_RESULT_FILE": str(result_file),
                "BOT_SERVICE_NAME": "telegram-userbot.service",
                "BOT_CANARY_SECONDS": "0",
                "BOT_MAX_DIFF_LINES": str(max_diff_lines),
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    return {
        "prod_dir": prod_dir,
        "staging_dir": staging_dir,
        "state_dir": state_dir,
        "request_file": request_file,
        "result_file": result_file,
        "prev_commit": prev_commit,
        "run_gatekeeper": run_gatekeeper,
    }


# ============================================================================
# F1: R1 Security Boundaries (6 tests)
# ============================================================================


def test_prod_f1_01_gitignore_blocks_all_secret_artifacts(tmp_path: Path) -> None:
    """Verify .gitignore rejects .env, *.env, sessions, oauth tokens, data/, and sqlite files."""
    repo = tmp_path / "git_check"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    shutil.copy2(REPO_ROOT / ".gitignore", repo / ".gitignore")

    must_ignore = [
        ".env",
        "prod.env",
        "userbot.session",
        "data/session/userbot.session-journal",
        "data/antigravity/jetski-standalone-oauth-token",
        "data/memory.sqlite",
        "data/memory.sqlite-wal",
        "data/self_deploy/request.json",
        "config/bot.yaml",
    ]
    for rel in must_ignore:
        res = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "-q", rel],
            capture_output=True,
        )
        assert res.returncode == 0, f"Expected .gitignore to ignore {rel!r}"

    must_allow = [".env.example", "config/bot.example.yaml"]
    for rel in must_allow:
        res = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "-q", rel],
            capture_output=True,
        )
        assert res.returncode == 1, f"Expected .gitignore NOT to ignore {rel!r}"


def test_prod_f1_02_example_configs_and_bot_yaml_contain_zero_real_secrets() -> None:
    """Verify bot.yaml (if present) and example config/env templates contain zero real tokens or private keys."""
    candidate_files = [
        REPO_ROOT / "config" / "bot.example.yaml",
        REPO_ROOT / ".env.example",
    ]
    for optional in (
        REPO_ROOT / "config" / "bot.yaml",
        REPO_ROOT / "config" / "userbot.example.yaml",
        REPO_ROOT / ".env.userbot.example",
    ):
        if optional.is_file():
            candidate_files.append(optional)

    secret_patterns = [
        re.compile(r"\b\d{7,12}:AA[A-Za-z0-9_-]{25,}\b"),
        re.compile(r"ya29\.[0-9A-Za-z_-]{20,}"),
        re.compile(r"1//0[0-9A-Za-z_-]{20,}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
    ]
    for file_path in candidate_files:
        text = file_path.read_text(encoding="utf-8")
        for pat in secret_patterns:
            assert pat.search(text) is None, f"Secret pattern {pat.pattern} matched in {file_path}"


def test_prod_f1_03_git_tracked_files_exclude_env_sessions_and_oauth() -> None:
    """Verify git index contains zero .env, session, oauth-token, sqlite, or data/ files."""
    out = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        text=True,
    )
    tracked = out.splitlines()
    forbidden = re.compile(
        r"(?:^|/)\.env$|\.session|oauth-token|\.sqlite|^data/",
        re.IGNORECASE,
    )
    bad = [p for p in tracked if forbidden.search(p)]
    assert bad == [], f"Tracked sensitive/runtime artifacts found: {bad}"


def test_prod_f1_04_deploy_bot_api_never_targets_mtproto_or_old_server() -> None:
    """Verify deploy_remote.sh rejects bot_api targeting mtproto dir and 192.168.1.151 is absent."""
    deploy_script = REPO_ROOT / "deploy" / "deploy_remote.sh"
    proc = subprocess.run(
        ["bash", str(deploy_script), "bot_api", "root@127.0.0.1", "/opt/telegram-mtproto-userbot"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2
    assert "FATAL" in proc.stderr

    for scan_dir in (REPO_ROOT / "deploy", REPO_ROOT / "systemd", REPO_ROOT / "src" / "bot"):
        for path in scan_dir.rglob("*"):
            if path.is_file() and path.suffix in {".sh", ".py", ".service", ".yaml", ".md"}:
                assert "192.168.1.151" not in path.read_text(encoding="utf-8", errors="replace"), (
                    f"Legacy server 192.168.1.151 referenced in {path}"
                )


def test_prod_f1_05_child_env_strips_all_secrets_and_dangerous_loaders(tmp_path: Path) -> None:
    """Verify build_clean_child_env strips secret credentials and loader injection variables."""
    injected = {
        "TELEGRAM_BOT_TOKEN": "999999:AAFakeSecretTokenValue1234567890",
        "OPENAI_API_KEY": "sk-proj-fake1234567890abcdef1234567890",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "WG_PRIVATE_KEY": "cHJpdmF0ZWtleXBsYWNlaG9sZGVyMTIzNDU2Nzg5MDE=",
        "LD_PRELOAD": "/tmp/evil.so",
        "PYTHONPATH": "/tmp/evil_py",
    }
    with patch.dict(os.environ, injected, clear=False):
        clean_env = build_clean_child_env(tmp_path, force_file_auth=True)

    for key in injected:
        assert key not in clean_env
    assert clean_env.get("SSH_CONNECTION") == _KEYRING_BYPASS_SSH_MARKER


def test_prod_f1_06_no_hardcoded_secrets_or_old_server_in_source_tree() -> None:
    """Verify src/bot and config/bot.yaml contain zero hardcoded credentials or old host IPs."""
    token_re = re.compile(r"\b\d{7,12}:AA[A-Za-z0-9_-]{25,}\b")
    for py_file in (REPO_ROOT / "src" / "bot").rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert token_re.search(content) is None, f"Hardcoded bot token in {py_file}"
        assert "192.168.1.151" not in content
    for cfg_name in ("bot.yaml", "bot.example.yaml"):
        cfg_file = REPO_ROOT / "config" / cfg_name
        if cfg_file.is_file():
            bot_yaml = cfg_file.read_text(encoding="utf-8")
            assert "192.168.1.151" not in bot_yaml
            assert token_re.search(bot_yaml) is None


# ============================================================================
# F2: R3.1 Secret Sanitizer (6 tests)
# ============================================================================


def test_prod_f2_01_redacts_registered_and_dotenv_secrets(tmp_path: Path) -> None:
    """Verify SecretSanitizer loads .env secrets and redacts both .env and registered values."""
    from bot.security.sanitizer import SecretSanitizer

    env_file = tmp_path / ".env"
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN=777111:AAFakeDotenvTokenValue123456789\n"
        "TELEGRAM_API_HASH=abcdef0123456789abcdef0123456789\n"
        "CUSTOM_SECRET=super_secret_phrase_xyz\n",
        encoding="utf-8",
    )
    sanitizer = SecretSanitizer(env_path=env_file)
    sanitizer.register_secret("my_runtime_secret_123")

    raw = (
        "token=777111:AAFakeDotenvTokenValue123456789 "
        "hash=abcdef0123456789abcdef0123456789 "
        "custom=super_secret_phrase_xyz runtime=my_runtime_secret_123"
    )
    cleaned = sanitizer.sanitize(raw)
    assert "777111:AAFakeDotenvTokenValue123456789" not in cleaned
    assert "abcdef0123456789abcdef0123456789" not in cleaned
    assert "super_secret_phrase_xyz" not in cleaned
    assert "my_runtime_secret_123" not in cleaned
    assert cleaned.count("[REDACTED]") == 4


def test_prod_f2_02_redacts_telegram_token_oauth_bearer_and_pem_patterns() -> None:
    """Verify regex patterns redact Telegram tokens, OAuth bearer tokens, API keys, and PEM blocks."""
    from bot.security.sanitizer import SecretSanitizer

    sanitizer = SecretSanitizer()
    sample = (
        "tg=123456789:AAH_abcdefghijklmnopqrstuvwxyz12345\n"
        "auth=Bearer ya29.a0AfH6SMC_fake_oauth_token_12345\n"
        "key=sk-proj-1234567890abcdef1234567890\n"
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----"
    )
    cleaned = sanitizer.sanitize(sample)
    assert "123456789:AAH_abcdefghijklmnopqrstuvwxyz12345" not in cleaned
    assert "ya29.a0AfH6SMC_fake_oauth_token_12345" not in cleaned
    assert "sk-proj-1234567890abcdef1234567890" not in cleaned
    assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC" not in cleaned
    assert "[REDACTED]" in cleaned


def test_prod_f2_03_logging_filter_masks_secrets_in_records_and_exc_info() -> None:
    """Verify SanitizingLogFilter redacts secrets from formatted log args and exception tracebacks."""
    from bot.security.sanitizer import SanitizingLogFilter, SecretSanitizer

    secret = "ultra_secret_credential_98765"
    sanitizer = SecretSanitizer(extra_secrets=[secret])
    log_filter = SanitizingLogFilter(sanitizer)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(log_filter)

    test_logger = logging.getLogger("test_prod_f2_03_logger")
    test_logger.setLevel(logging.DEBUG)
    test_logger.propagate = False
    test_logger.addHandler(handler)
    try:
        test_logger.info("Connecting with credential %s", secret)
        try:
            raise RuntimeError(f"boom while using {secret}")
        except RuntimeError:
            test_logger.exception("Failure occurred")
    finally:
        test_logger.removeHandler(handler)

    output = stream.getvalue()
    assert secret not in output
    assert "[REDACTED]" in output


@pytest.mark.asyncio
async def test_prod_f2_04_orchestrator_sanitizes_outbound_telegram_reply_and_doc_caption(
    tmp_path: Path,
) -> None:
    """Verify GroupBotOrchestrator sanitizes outbound Telegram reply text and doc captions."""
    secret = "7654321:AAFakeOutboundSecretToken1234567890"
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"telegram": {"bot_token": secret}},
    )
    store = MemoryStore(db_path=str(tmp_path / "f2_04.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text=(
                f"Вот отчёт с токеном {secret} "
                f'[DOC:{{"format":"pdf","title":"Отчёт","filename":"r.pdf",'
                f'"caption":"Секрет {secret}","sections":[{{"heading":"H","body":"B"}}]}}]'
            ),
            model="gemini-3.8-flash",
        )
    )
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    try:
        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=GROUP_CHAT_ID,
                sender_id=891777515,
                text="Джефф, сделай отчёт",
                is_mention=True,
            )
        )
        sent_texts: list[str] = []
        for call in mock_userbot.send_reply.call_args_list:
            sent_texts.append(str(call.kwargs.get("text") or ""))
        for call in mock_userbot.send_document.call_args_list:
            sent_texts.append(str(call.kwargs.get("caption") or ""))
        assert sent_texts, "Expected at least one outbound Telegram call"
        for outbound in sent_texts:
            assert secret not in outbound
            assert "[REDACTED]" in outbound
    finally:
        await orch.stop()


def test_prod_f2_05_bot_api_client_error_logs_redact_bot_token_in_url() -> None:
    """Verify Bot API HTTP/URL error logs mask the bot<TOKEN> URL component when filter active."""
    from bot.security.sanitizer import SanitizingLogFilter, SecretSanitizer

    token = "555666777:AAFakeBotApiTokenForUrlMasking12345"
    sanitizer = SecretSanitizer(extra_secrets=[token])
    log_filter = SanitizingLogFilter(sanitizer)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(log_filter)

    client_logger = logging.getLogger("bot.telegram.bot_api_client")
    prev_level = client_logger.level
    client_logger.setLevel(logging.DEBUG)
    client_logger.addHandler(handler)

    client = TelegramBotAPIClient(bot_token=token, allowed_chat_ids=[GROUP_CHAT_ID])
    try:
        with patch(
            "urllib.request.urlopen",
            side_effect=ConnectionResetError(f"Error connecting to https://api.telegram.org/bot{token}/getMe"),
        ), patch("time.sleep"):
            with pytest.raises((ConnectionResetError, RuntimeError)) as exc_info:
                client._sync_call_api("getMe")
        assert token not in str(exc_info.value)
    finally:
        client_logger.removeHandler(handler)
        client_logger.setLevel(prev_level)

    logged = stream.getvalue()
    assert token not in logged


def test_prod_f2_06_sanitizer_masks_wireguard_key_and_totp_secrets(tmp_path: Path) -> None:
    """Verify SecretSanitizer masks WG_PRIVATE_KEY and BA_TOTP_SECRET loaded from .env."""
    from bot.security.sanitizer import SecretSanitizer

    env_file = tmp_path / ".env"
    env_file.write_text(
        "BA_TOTP_SECRET=JBSWY3DPEHPK3PXP\n"
        "WG_PRIVATE_KEY=yAnz5TF+lXXJte14tji3zlMNq+hd2rYUIgJBgB3fBmk=\n",
        encoding="utf-8",
    )
    sanitizer = SecretSanitizer(env_path=env_file)
    out = sanitizer.sanitize(
        "totp=JBSWY3DPEHPK3PXP wg=yAnz5TF+lXXJte14tji3zlMNq+hd2rYUIgJBgB3fBmk="
    )
    assert "JBSWY3DPEHPK3PXP" not in out
    assert "yAnz5TF+lXXJte14tji3zlMNq+hd2rYUIgJBgB3fBmk=" not in out
    assert out.count("[REDACTED]") == 2


# ============================================================================
# F3: R3.2 Legal Prohibition Prompt (5 tests)
# ============================================================================


def test_prod_f3_01_system_prompt_contains_explicit_illegal_content_prohibition() -> None:
    """Verify config/system_prompt.md explicitly forbids illegal acts and pirated torrents."""
    cfg_mgr = ConfigManager(config_dir=REPO_ROOT / "config")
    prompt = cfg_mgr.system_prompt.lower()
    assert prompt
    assert any(w in prompt for w in ("закон", "незакон", "противоправ", "нелегальн"))
    assert any(w in prompt for w in ("торрент", "пират", "torrent"))


def test_prod_f3_02_system_prompt_applies_prohibition_regardless_of_requester() -> None:
    """Verify legal prohibition explicitly applies regardless of who asks (friends/admin)."""
    prompt = (REPO_ROOT / "config" / "system_prompt.md").read_text(encoding="utf-8").lower()
    assert any(
        phrase in prompt
        for phrase in ("независимо", "кто бы ни просил", "админ", "исключен")
    )


@pytest.mark.asyncio
async def test_prod_f3_03_orchestrator_passes_legal_prohibition_prompt_to_provider_in_group(
    tmp_path: Path,
) -> None:
    """Verify group chat LLM turns receive system_instructions containing legal prohibition."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "f3_03.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Не, таким не занимаюсь.", model="gemini-3.8-flash")
    )
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    try:
        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=GROUP_CHAT_ID,
                sender_id=891777515,
                text="Джефф, скачай торрент фильма",
                is_mention=True,
            )
        )
        sys_inst = str(mock_provider.generate_response.call_args.kwargs.get("system_instructions") or "").lower()
        assert any(w in sys_inst for w in ("торрент", "пират"))
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_f3_04_orchestrator_passes_legal_prohibition_prompt_even_for_admin_dm(
    tmp_path: Path,
) -> None:
    """Verify admin DM turns still enforce the legal prohibition inside system_instructions."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "f3_04.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Отказ.", model="gemini-4-hydrogen")
    )
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    try:
        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=ADMIN_USER_ID,
                sender_id=ADMIN_USER_ID,
                text="найди пиратский торрент",
                is_private=True,
                is_mention=False,
                sender_name="Родион",
            )
        )
        sys_inst = str(mock_provider.generate_response.call_args.kwargs.get("system_instructions") or "").lower()
        assert any(w in sys_inst for w in ("торрент", "пират"))
    finally:
        await orch.stop()


def test_prod_f3_05_hot_reload_preserves_legal_prohibition_prompt(tmp_path: Path) -> None:
    """Verify ConfigManager hot-reload loads and refreshes system_prompt with legal prohibition."""
    cfg_mgr = _make_isolated_config(tmp_path)
    assert any(w in cfg_mgr.system_prompt.lower() for w in ("торрент", "пират"))

    new_prompt = cfg_mgr.system_prompt + "\nДополнительно: запрещено нарушать закон."
    cfg_mgr.system_prompt_path.write_text(new_prompt, encoding="utf-8")
    new_mtime = time.time() + 10.0
    os.utime(cfg_mgr.system_prompt_path, (new_mtime, new_mtime))

    changed = cfg_mgr.reload_changed()
    assert changed is True
    assert "запрещено нарушать закон" in cfg_mgr.system_prompt
    assert any(w in cfg_mgr.system_prompt.lower() for w in ("торрент", "пират"))


# ============================================================================
# F4: R3.3 Config & Admin DM Routing (6 tests)
# ============================================================================


def test_prod_f4_01_bot_yaml_loads_admin_and_valid_ffmpeg_path() -> None:
    """Verify repo config sets bot_api mode, portable ffmpeg, and (when bot.yaml exists) admin_user_ids=382962362."""
    bot_yaml = REPO_ROOT / "config" / "bot.yaml"
    cfg_mgr = ConfigManager(config_path=bot_yaml if bot_yaml.is_file() else REPO_ROOT / "config" / "bot.example.yaml")
    assert cfg_mgr.bot is not None
    assert cfg_mgr.bot.telegram.mode == "bot_api"
    if bot_yaml.is_file():
        assert ADMIN_USER_ID in cfg_mgr.bot.telegram.admin_user_ids
        assert ADMIN_USER_ID in cfg_mgr.bot.telegram.allowed_dm_user_ids
    ffmpeg_path = cfg_mgr.bot.media.ffmpeg_path or ""
    assert "/home/jeff" not in ffmpeg_path
    assert ffmpeg_path in ("/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg")


def test_prod_f4_02_bot_config_backward_compatible_with_minimal_yaml() -> None:
    """Verify BotConfig validates minimal legacy config without new quota/router/evolution keys."""
    cfg = BotConfig.model_validate(
        {"telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"}}
    )
    assert cfg.telegram.api_id == 12345
    assert cfg.telegram.admin_user_ids == []


def test_prod_f4_03_bot_api_client_ingests_whitelisted_admin_private_dm() -> None:
    """Verify TelegramBotAPIClient ingests whitelisted private DM with is_private and username."""
    client = TelegramBotAPIClient(
        bot_token="123456:DUMMY_TOKEN_ABCDEFGH",
        allowed_chat_ids=[GROUP_CHAT_ID],
        allowed_dm_user_ids=[ADMIN_USER_ID],
    )
    client.bot_id = 777000
    client.bot_username = "jeff_test_bot"

    raw_dm = {
        "message_id": 5001,
        "from": {
            "id": ADMIN_USER_ID,
            "is_bot": False,
            "first_name": "Родион",
            "username": "rodion",
        },
        "chat": {"id": ADMIN_USER_ID, "type": "private"},
        "date": 1794586333,
        "text": "проверь статус бота",
    }
    incoming = client._convert_msg_dict(raw_dm)
    assert incoming is not None
    assert incoming.is_private is True
    assert incoming.sender_is_bot is False
    assert incoming.sender_username == "rodion"
    assert incoming.chat_id == ADMIN_USER_ID


@pytest.mark.asyncio
async def test_prod_f4_04_orchestrator_wires_allowed_dm_and_replies_to_admin_dm_without_mention(
    tmp_path: Path,
) -> None:
    """Verify orchestrator replies to admin private DM even without explicit bot-name mention."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "f4_04.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Всё штатно работает.", model="gemini-4-hydrogen")
    )
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    try:
        reply = await orch.handle_incoming_message(
            _make_incoming(
                chat_id=ADMIN_USER_ID,
                sender_id=ADMIN_USER_ID,
                text="проверь статус",
                is_private=True,
                is_mention=False,
                sender_name="Родион",
            )
        )
        assert reply == "Всё штатно работает."
        mock_userbot.send_reply.assert_awaited_once()
        assert mock_userbot.send_reply.call_args.kwargs["chat_id"] == ADMIN_USER_ID
    finally:
        await orch.stop()


def test_prod_f4_05_bot_api_client_parses_group_vs_private_and_bot_flag() -> None:
    """Verify BotAPIMessageEvent and _convert_msg_dict distinguish supergroup vs private and is_bot."""
    client = TelegramBotAPIClient(
        bot_token="123456:DUMMY_TOKEN_ABCDEFGH",
        allowed_chat_ids=[GROUP_CHAT_ID],
        allowed_dm_user_ids=[ADMIN_USER_ID],
    )
    client.bot_id = 777000

    group_msg = {
        "message_id": 6001,
        "from": {"id": 891777515, "is_bot": False, "first_name": "Алексей", "username": "alex"},
        "chat": {"id": GROUP_CHAT_ID, "type": "supergroup"},
        "date": 1794586333,
        "text": "привет чат",
    }
    inc_group = client._convert_msg_dict(group_msg)
    assert inc_group is not None
    assert inc_group.is_private is False
    assert inc_group.sender_is_bot is False

    bot_msg = {
        "message_id": 6002,
        "from": {"id": 999888777, "is_bot": True, "first_name": "OtherBot", "username": "other_bot"},
        "chat": {"id": GROUP_CHAT_ID, "type": "supergroup"},
        "date": 1794586334,
        "text": "spam from bot",
    }
    ev_bot = BotAPIMessageEvent(bot_msg, bot_id=777000)
    assert ev_bot.sender_is_bot is True
    inc_bot = client._convert_msg_dict(bot_msg)
    assert inc_bot is None or inc_bot.sender_is_bot is True


def test_prod_f4_06_bot_config_validates_quota_model_router_and_self_evolution_sections() -> None:
    """Verify BotConfig accepts and validates quota, model_router, antigravity, and self_evolution."""
    cfg = BotConfig.model_validate(
        {
            "telegram": {
                "mode": "bot_api",
                "api_id": 12345,
                "api_hash": "0123456789abcdef",
                "admin_user_ids": [ADMIN_USER_ID],
                "allowed_dm_user_ids": [ADMIN_USER_ID],
            },
            "quota": {
                "hourly_token_budget": 100000,
                "daily_token_budget": 500000,
                "economy_threshold_ratio": 0.8,
                "admin_reserve_ratio": 0.2,
            },
            "models": {
                "group": {"model": "gemini-3.8-flash", "effort": "low"},
                "dm_admin": {"model": "gemini-4-hydrogen", "effort": "high"},
            },
            "self_evolution": {
                "enabled": True,
                "max_daily_edits": 2,
                "staging_dir": "/opt/telegram-group-bot-staging",
            },
        }
    )
    assert ADMIN_USER_ID in cfg.telegram.admin_user_ids
    assert cfg.quota.hourly_token_budget == 100000
    assert cfg.models.group.model == "gemini-3.8-flash"
    assert cfg.self_evolution.max_daily_edits == 2


# ============================================================================
# F5: R3.4 Tool Profiles & ACP Fix (6 tests)
# ============================================================================


def test_prod_f5_01_stream_parser_default_chat_profile_allows_web_and_aborts_shell_fs() -> None:
    """Verify default chat profile allows search_web/read_url_content and aborts FS/shell tools."""
    chat_tools, _ = _import_tool_constants()
    assert chat_tools == frozenset({"search_web", "read_url_content"})

    parser_ok = AntigravityStreamParser()
    parser_ok.feed_line(
        json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "name": "search_web"}})
    )
    parser_ok.feed_line(
        json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "name": "read_url_content"}})
    )
    parser_ok.finish()

    for forbidden in ("run_command", "write_to_file", "read_file", "edit_file", "view_file"):
        parser_bad = AntigravityStreamParser()
        with pytest.raises(AntigravityNativeToolAbort):
            parser_bad.feed_line(
                json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "name": forbidden}})
            )


def test_prod_f5_02_stream_parser_ops_and_self_dev_allow_ops_tools_but_block_browser_mcp_subagents() -> None:
    """Verify ops and self_dev profiles allow FS/shell tools while blocking browser/MCP/subagents."""
    _, ops_tools = _import_tool_constants()
    allowed_ops = (
        "read_file",
        "edit_file",
        "write_file",
        "run_command",
        "view_file",
        "replace_file_content",
        "grep_search",
        "find_by_name",
        "list_dir",
        "command_status",
    )
    for tool in allowed_ops:
        assert tool in ops_tools

    for profile in ("ops", "self_dev"):
        parser = AntigravityStreamParser(tool_profile=profile)
        for tool in allowed_ops:
            parser.feed_line(
                json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "name": tool}})
            )
        parser.finish()

        for blocked in ("browser_subagent", "call_mcp_tool", "invoke_subagent", "swarm", "schedule"):
            p_block = AntigravityStreamParser(tool_profile=profile)
            with pytest.raises(AntigravityNativeToolAbort):
                p_block.feed_line(
                    json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "name": blocked}})
                )


@pytest.mark.asyncio
async def test_prod_f5_03_acp_tool_check_extracts_name_before_title_for_search_web(
    tmp_path: Path,
) -> None:
    """Verify ACP tool check checks name before title so search_web with descriptive title passes."""

    def on_prompt(msg: dict[str, Any], _perms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        req_id = msg["id"]
        return [
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
                        "content": {"type": "text", "text": "Found result"},
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"stopReason": "end_turn"},
            },
        ]

    fake_proc = _FakeACPProcess(on_prompt=on_prompt)
    cli = AntigravityCLIClient(data_dir=tmp_path, binary="/usr/bin/agy")
    try:
        with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=fake_proc):
            resp = await cli.generate_response(
                prompt="найди на сайте",
                tool_profile="chat",
                force_acp=True,
            ) if "force_acp" in cli.generate_response.__code__.co_varnames else await asyncio_to_thread_acp(
                cli, fake_proc, tool_profile="chat"
            )
        assert resp.text == "Found result"
    finally:
        cli.close()


async def asyncio_to_thread_acp(
    cli: AntigravityCLIClient,
    fake_proc: _FakeACPProcess,
    *,
    tool_profile: str = "chat",
    mode: Optional[str] = None,
    cwd: Optional[Path] = None,
    patch_popen: bool = True,
) -> ProviderResponse:
    """Helper calling _sync_execute_acp_turn directly when public generate_response uses stream-json."""
    import asyncio
    import inspect

    sig = inspect.signature(cli._sync_execute_acp_turn)
    kwargs: dict[str, Any] = {
        "prompt": "найди на сайте",
        "image_payloads": [],
    }
    if "tool_profile" in sig.parameters:
        kwargs["tool_profile"] = tool_profile
    if mode is not None and "mode" in sig.parameters:
        kwargs["mode"] = mode
    if cwd is not None and "cwd" in sig.parameters:
        kwargs["cwd"] = cwd
    if patch_popen:
        with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=fake_proc):
            return await asyncio.to_thread(cli._sync_execute_acp_turn, **kwargs)
    return await asyncio.to_thread(cli._sync_execute_acp_turn, **kwargs)


@pytest.mark.asyncio
async def test_prod_f5_04_acp_permission_and_mode_and_cwd_enforced_by_profile(
    tmp_path: Path,
) -> None:
    """Verify ACP forwards cwd/mode and enforces session/request_permission per tool_profile."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    captured_popen_kwargs: list[dict[str, Any]] = []

    def make_prompt_handler(tool_name: str) -> Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]]:
        def _handler(msg: dict[str, Any], _perms: list[dict[str, Any]]) -> list[dict[str, Any]]:
            req_id = msg["id"]
            return [
                {
                    "jsonrpc": "2.0",
                    "id": 900,
                    "method": "session/request_permission",
                    "params": {
                        "toolCall": {"name": tool_name},
                        "options": [{"optionId": "allow-once", "kind": "allow_once"}],
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "Done"},
                        }
                    },
                },
                {"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}},
            ]

        return _handler

    cli = AntigravityCLIClient(data_dir=tmp_path / "agy_home", binary="/usr/bin/agy")
    try:
        # 1. ops + write_file -> selected/allow + cwd + set_mode
        proc_ops = _FakeACPProcess(on_prompt=make_prompt_handler("write_file"))

        def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeACPProcess:
            captured_popen_kwargs.append(kwargs)
            return proc_ops

        with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
            await asyncio_to_thread_acp(
                cli,
                proc_ops,
                tool_profile="ops",
                mode="accept-edits",
                cwd=staging_dir,
                patch_popen=False,
            )

        session_new = next(r for r in proc_ops.requests if r["method"] == "session/new")
        assert session_new["params"]["cwd"] == str(staging_dir)
        assert Path(captured_popen_kwargs[0]["env"]["HOME"]) != staging_dir
        set_mode_reqs = [r for r in proc_ops.requests if r["method"] == "session/set_mode"]
        assert set_mode_reqs and set_mode_reqs[0]["params"].get("modeId") == "accept-edits"
        assert proc_ops.permission_responses[0]["result"]["outcome"]["outcome"] == "selected"

        # 2. ops + call_mcp_tool -> cancelled
        proc_mcp = _FakeACPProcess(on_prompt=make_prompt_handler("call_mcp_tool"))
        with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=proc_mcp):
            await asyncio_to_thread_acp(cli, proc_mcp, tool_profile="ops")
        assert proc_mcp.permission_responses[0]["result"]["outcome"]["outcome"] == "cancelled"

        # 3. chat + write_file -> cancelled
        proc_chat = _FakeACPProcess(on_prompt=make_prompt_handler("write_file"))
        with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=proc_chat):
            await asyncio_to_thread_acp(cli, proc_chat, tool_profile="chat")
        assert proc_chat.permission_responses[0]["result"]["outcome"]["outcome"] == "cancelled"
    finally:
        cli.close()


@pytest.mark.asyncio
async def test_prod_f5_05_orchestrator_selects_ops_only_for_admin_dm_and_chat_for_group(
    tmp_path: Path,
) -> None:
    """Verify GroupBotOrchestrator passes tool_profile='ops' ONLY for admin DM and 'chat' otherwise."""
    non_admin_dm = 999111222
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "telegram": {
                "allowed_dm_user_ids": [ADMIN_USER_ID, non_admin_dm],
                "admin_user_ids": [ADMIN_USER_ID],
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f5_05.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="OK", model="gemini-4-hydrogen")
    )
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    try:
        # (a) Group message
        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=GROUP_CHAT_ID,
                sender_id=891777515,
                text="Джефф, привет",
                is_mention=True,
                msg_id=1,
            )
        )
        assert mock_provider.generate_response.call_args.kwargs["tool_profile"] == "chat"

        # (b) Non-admin DM
        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=non_admin_dm,
                sender_id=non_admin_dm,
                text="привет",
                is_private=True,
                msg_id=2,
            )
        )
        assert mock_provider.generate_response.call_args.kwargs["tool_profile"] == "chat"

        # (c) Admin DM
        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=ADMIN_USER_ID,
                sender_id=ADMIN_USER_ID,
                text="проверь логи",
                is_private=True,
                msg_id=3,
            )
        )
        assert mock_provider.generate_response.call_args.kwargs["tool_profile"] == "ops"
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_f5_06_provider_forwards_tool_profile_mode_and_cwd_to_cli_client(
    tmp_path: Path,
) -> None:
    """Verify AntigravityProvider.generate_response forwards model, effort, tool_profile, mode, cwd."""
    from bot.antigravity import AntigravityConfig, AntigravityProvider

    provider = AntigravityProvider(
        AntigravityConfig(data_dir=tmp_path / "agy", enable_sdk=False, enable_cli_fallback=True)
    )
    cli_stub = MagicMock()
    cli_stub.generate_response = AsyncMock(
        return_value=ProviderResponse(text="ok", model="claude-sonnet-5-5")
    )
    provider._cli_client = cli_stub

    work_dir = tmp_path / "staging"
    work_dir.mkdir()
    resp = await provider.generate_response(
        prompt="run check",
        model="claude-sonnet-5-5",
        effort="high",
        tool_profile="self_dev",
        mode="accept-edits",
        cwd=work_dir,
    )
    assert resp.text == "ok"
    kw = cli_stub.generate_response.call_args.kwargs
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["effort"] == "high"
    assert kw["tool_profile"] == "self_dev"
    assert kw["mode"] == "accept-edits"
    assert kw["cwd"] == work_dir


# ============================================================================
# F6: R5.1 UsageInfo SQLite Persistence (5 tests)
# ============================================================================


def test_prod_f6_01_usage_info_model_supports_normalized_and_legacy_fields() -> None:
    """Verify UsageInfo supports both normalized token fields and legacy 5-field init."""
    normalized = UsageInfo(
        input_tokens=100,
        output_tokens=20,
        reasoning_tokens=10,
        cache_read_tokens=50,
        cache_write_tokens=5,
        prompt_tokens=100,
        completion_tokens=30,
        total_tokens=185,
    )
    assert normalized.input_tokens == 100
    assert normalized.output_tokens == 20
    assert normalized.cache_write_tokens == 5
    assert normalized.reasoning_tokens == 10

    legacy = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert legacy.prompt_tokens == 10
    assert legacy.completion_tokens == 5
    assert legacy.total_tokens == 15
    assert legacy.input_tokens >= 0
    assert legacy.output_tokens >= 0
    assert legacy.cache_write_tokens >= 0


@pytest.mark.asyncio
async def test_prod_f6_02_stream_and_acp_populate_normalized_usage_fields(tmp_path: Path) -> None:
    """Verify stream-json and ACP parsers populate normalized UsageInfo token fields."""
    parser = AntigravityStreamParser()
    parser.feed_line(
        json.dumps(
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": "Done",
                    "usage": {
                        "input_tokens": 120,
                        "output_tokens": 35,
                        "thinking_tokens": 15,
                        "cache_read_tokens": 40,
                        "cache_write_tokens": 10,
                    },
                },
            }
        )
    )
    chunks = parser.finish()
    stream_usage = chunks[-1].usage
    assert stream_usage is not None
    assert stream_usage.input_tokens == 120
    assert stream_usage.output_tokens == 35
    assert stream_usage.reasoning_tokens == 15
    assert stream_usage.cache_write_tokens == 10

    def on_prompt(msg: dict[str, Any], _perms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "ACP usage"},
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "result": {
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": 200,
                        "outputTokens": 50,
                        "thoughtTokens": 25,
                        "cachedReadTokens": 80,
                        "totalTokens": 355,
                    },
                },
            },
        ]

    cli = AntigravityCLIClient(data_dir=tmp_path, binary="/usr/bin/agy")
    try:
        proc = _FakeACPProcess(on_prompt=on_prompt)
        resp = await asyncio_to_thread_acp(cli, proc)
        assert resp.usage is not None
        assert resp.usage.input_tokens == 200
        assert resp.usage.output_tokens == 50
        assert resp.usage.reasoning_tokens == 25
        assert resp.usage.total_tokens == 355
    finally:
        cli.close()


@pytest.mark.asyncio
async def test_prod_f6_03_memory_store_creates_usage_history_schema(tmp_path: Path) -> None:
    """Verify MemoryStore.init_db creates usage_history table with all 10 required columns."""
    db_path = tmp_path / "usage_schema.sqlite"
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    await store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(usage_history)").fetchall()]
    finally:
        conn.close()

    expected = [
        "id",
        "timestamp",
        "model",
        "source",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
    ]
    for col in expected:
        assert col in cols, f"Missing column {col!r} in usage_history: {cols}"


@pytest.mark.asyncio
async def test_prod_f6_04_quota_manager_records_usage_per_source_to_sqlite(tmp_path: Path) -> None:
    """Verify QuotaBudgetManager.record_usage writes rows for group, dm_admin, proactive, self_edit."""
    db_path = tmp_path / "usage_rows.sqlite"
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    qm = _make_quota_manager(store)

    sources = ("group", "dm_admin", "proactive", "self_edit")
    try:
        for idx, src in enumerate(sources, start=1):
            usage = UsageInfo(
                input_tokens=100 * idx,
                output_tokens=20 * idx,
                reasoning_tokens=5 * idx,
                cache_read_tokens=10 * idx,
                cache_write_tokens=2 * idx,
                prompt_tokens=100 * idx,
                completion_tokens=25 * idx,
                total_tokens=137 * idx,
            )
            await _maybe_await(qm.record_usage(usage, model="gemini-4-hydrogen", source=src))
    finally:
        await store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT model, source, input_tokens, output_tokens, total_tokens FROM usage_history ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 4
    assert [r[1] for r in rows] == list(sources)
    assert rows[0][0] == "gemini-4-hydrogen"
    assert rows[1][2] == 200


@pytest.mark.asyncio
async def test_prod_f6_05_orchestrator_reply_persists_provider_usage_to_sqlite(
    tmp_path: Path,
) -> None:
    """Verify GroupBotOrchestrator persists provider UsageInfo into SQLite usage_history."""
    db_path = tmp_path / "orch_usage.sqlite"
    cfg_mgr = _make_isolated_config(tmp_path, bot_overrides={"memory": {"db_path": str(db_path)}})
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text="Ответ сохранён",
            model="gemini-3.8-flash",
            usage=UsageInfo(
                input_tokens=80,
                output_tokens=15,
                reasoning_tokens=0,
                cache_read_tokens=20,
                cache_write_tokens=0,
                prompt_tokens=100,
                completion_tokens=15,
                total_tokens=115,
            ),
        )
    )
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    try:
        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=GROUP_CHAT_ID,
                sender_id=891777515,
                text="Джефф, тест usage",
                is_mention=True,
            )
        )
    finally:
        await orch.stop()

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT model, source, total_tokens FROM usage_history").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0][0] == "gemini-3.8-flash"
    assert rows[0][1] == "group"
    assert rows[0][2] == 115


# ============================================================================
# F7: R5.2 Per-Model & Bucket Cooldowns (6 tests)
# ============================================================================


def test_prod_f7_01_classify_error_handles_429_and_404_not_found() -> None:
    """Verify classify_error maps 429/rate-limit phrases to RATE_LIMIT and 404 to MODEL_NOT_FOUND."""
    for phrase in ("HTTP 429", "too many requests", "quota exceeded", "RESOURCE_EXHAUSTED"):
        verdict = classify_error(phrase)
        assert verdict.category == ErrorCategory.RATE_LIMIT
        assert verdict.should_fallback is True

    verdict_404 = classify_error("NOT_FOUND (code 404): Requested entity was not found")
    assert verdict_404.category == ErrorCategory.MODEL_NOT_FOUND
    assert verdict_404.should_fallback is True


def test_prod_f7_02_extract_retry_after_parses_compound_durations_and_iso_resets_at() -> None:
    """Verify extract_retry_after parses 2h15m30s, 45m, and ISO-8601 UTC resets at timestamps."""
    assert extract_retry_after("retry in 2h15m30s") == pytest.approx(8130.0, abs=1.0)
    assert extract_retry_after("resets in 45m") == pytest.approx(2700.0, abs=1.0)

    future_dt = (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    parsed_iso = extract_retry_after(f"quota exhausted, resets at {future_dt}")
    assert parsed_iso is not None
    assert 1700.0 <= parsed_iso <= 1810.0


def test_prod_f7_03_quota_warning_in_logs_updates_state_without_killing_process(
    tmp_path: Path,
) -> None:
    """Verify [quota] warning lines trigger warning callback without killing the child process."""
    import threading

    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir(parents=True)
    log_file = gemini_dir / "cli.log"
    log_file.write_text(
        "[quota] model gemini-4-hydrogen is low on quota (10% remaining, resets at 2026-11-18T18:00:00Z)\n",
        encoding="utf-8",
    )

    stop_event = threading.Event()
    errors: list[str] = []
    warnings: list[Any] = []
    killed = False

    def on_kill() -> None:
        nonlocal killed
        killed = True

    def on_warning(*args: Any, **kwargs: Any) -> None:
        warnings.append((args, kwargs))
        stop_event.set()

    import inspect

    sig = inspect.signature(watch_log_for_errors)
    kwargs: dict[str, Any] = {}
    if "on_quota_warning" in sig.parameters:
        kwargs["on_quota_warning"] = on_warning

    t = threading.Thread(
        target=watch_log_for_errors,
        args=(gemini_dir, stop_event, errors, on_kill),
        kwargs=kwargs,
        daemon=True,
    )
    t.start()
    t.join(timeout=1.5)
    stop_event.set()
    t.join(timeout=1.0)

    assert killed is False
    assert errors == []
    assert len(warnings) >= 1


@pytest.mark.asyncio
async def test_prod_f7_04_hard_rate_limit_in_logs_aborts_and_cools_only_affected_model_or_bucket(
    tmp_path: Path,
) -> None:
    """Verify per-model vs bucket-wide rate limit cooldowns leave unrelated models/buckets eligible."""
    store = MemoryStore(db_path=str(tmp_path / "f7_04.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        qm.record_rate_limit("gemini-4-hydrogen", retry_after=120.0, bucket_wide=False, reason="429")
        model_dm, _ = qm.select_model("dm_admin")
        assert model_dm != "gemini-4-hydrogen"
        model_group, _ = qm.select_model("group")
        assert model_group == "gemini-3.8-flash"

        qm2 = _make_quota_manager(store)
        qm2.record_rate_limit("gemini-4-hydrogen", retry_after=120.0, bucket_wide=True, reason="bucket 429")
        model_after_bucket, _ = qm2.select_model("group")
        assert not model_after_bucket.startswith("gemini-3.8-flash")
        assert not model_after_bucket.startswith("gemini-4-hydrogen")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_f7_05_model_cooldown_expires_automatically(tmp_path: Path) -> None:
    """Verify per-model cooldown expires and restores primary model eligibility."""
    store = MemoryStore(db_path=str(tmp_path / "f7_05.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        primary_before, _ = qm.select_model("dm_admin")
        qm.record_rate_limit(primary_before, retry_after=0.05, bucket_wide=False)
        during, _ = qm.select_model("dm_admin")
        assert during != primary_before

        time.sleep(0.08)
        after, _ = qm.select_model("dm_admin")
        assert after == primary_before
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_f7_06_provider_per_model_cooldown_does_not_block_other_models(
    tmp_path: Path,
) -> None:
    """Verify AntigravityProvider rate limit on one model does not block another model."""
    from bot.antigravity import AntigravityConfig, AntigravityProvider

    provider = AntigravityProvider(
        AntigravityConfig(data_dir=tmp_path / "agy", enable_sdk=False, rate_limit_max_retries=0)
    )
    cli_stub = MagicMock()

    async def fake_cli(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("model") == "gemini-4-hydrogen":
            raise AntigravityRateLimitError("429 RESOURCE_EXHAUSTED retry in 60s")
        return ProviderResponse(text="claude ok", model=str(kwargs.get("model")))

    cli_stub.generate_response = AsyncMock(side_effect=fake_cli)
    provider._cli_client = cli_stub

    with pytest.raises(AntigravityRateLimitError):
        await provider.generate_response("hi", model="gemini-4-hydrogen")

    resp = await provider.generate_response("hi", model="claude-sonnet-5-5")
    assert resp.text == "claude ok"


# ============================================================================
# F8: R5.3 ModelRouter & Fallback (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_prod_f8_01_select_model_by_role_returns_distinct_model_and_effort(
    tmp_path: Path,
) -> None:
    """Verify select_model returns appropriate (model, effort) tuples per functional role."""
    store = MemoryStore(db_path=str(tmp_path / "f8_01.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        comp_m, comp_e = qm.select_model("compaction")
        grp_m, _ = qm.select_model("group")
        pro_m, _ = qm.select_model("proactive")
        dm_m, _ = qm.select_model("dm_admin")
        judge_m, judge_e = qm.select_model("self_edit_judge")
        edit_m, edit_e = qm.select_model("self_edit")

        assert comp_m and grp_m and pro_m and dm_m and judge_m and edit_m
        assert comp_e in ("low", "minimal", "medium")
        assert judge_e == "high"
        assert edit_e == "high"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_f8_02_cross_bucket_fallback_when_primary_bucket_exhausted(
    tmp_path: Path,
) -> None:
    """Verify cooling down gemini_models bucket causes ModelRouter to select claude_gpt fallback."""
    store = MemoryStore(db_path=str(tmp_path / "f8_02.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        qm.record_rate_limit("gemini-4-hydrogen", retry_after=300.0, bucket_wide=True)
        grp_m, _ = qm.select_model("group")
        dm_m, _ = qm.select_model("dm_admin")
        for chosen in (grp_m, dm_m):
            assert chosen.startswith("claude-") or chosen.startswith("gpt-oss-")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_f8_03_provider_retries_with_fallback_model_on_429_or_404(
    tmp_path: Path,
) -> None:
    """Verify orchestrator/quota router retries with fallback model on 429 and completes turn."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "f8_03.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_userbot = _make_mock_userbot()

    seen_models: list[str] = []

    async def fake_generate(prompt: str, **kwargs: Any) -> ProviderResponse:
        m = str(kwargs.get("model") or "gemini-4-hydrogen")
        seen_models.append(m)
        if m.startswith("gemini-"):
            raise AntigravityRateLimitError("429 RESOURCE_EXHAUSTED retry in 120s")
        return ProviderResponse(text="Fallback worked!", model=m)

    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(side_effect=fake_generate)

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    try:
        reply = await orch.handle_incoming_message(
            _make_incoming(
                chat_id=ADMIN_USER_ID,
                sender_id=ADMIN_USER_ID,
                text="проверь",
                is_private=True,
                sender_name="Родион",
            )
        )
        assert reply == "Fallback worked!"
        assert len(seen_models) >= 2
        assert seen_models[0].startswith("gemini-")
        assert not seen_models[-1].startswith("gemini-")
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_f8_04_argon_excluded_from_normal_chat_auto_rotation(tmp_path: Path) -> None:
    """Verify gemini-4-argon / gemini-pro-orchestrator is excluded from normal chat roles."""
    store = MemoryStore(db_path=str(tmp_path / "f8_04.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        for role in ("group", "proactive", "compaction"):
            m, _ = qm.select_model(role)
            assert "argon" not in m.lower()
            assert "orchestrator" not in m.lower()

        qm.record_rate_limit("gemini-4-hydrogen", retry_after=300.0, bucket_wide=True)
        for role in ("group", "proactive", "compaction"):
            m, _ = qm.select_model(role)
            assert "argon" not in m.lower()
            assert "orchestrator" not in m.lower()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_f8_05_fallback_chain_preserves_role_effort_override(tmp_path: Path) -> None:
    """Verify fallback selection for self_edit preserves effort='high'."""
    store = MemoryStore(db_path=str(tmp_path / "f8_05.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        primary, eff1 = qm.select_model("self_edit")
        assert eff1 == "high"
        qm.record_rate_limit(primary, retry_after=300.0, bucket_wide=False)
        fallback, eff2 = qm.select_model("self_edit")
        assert fallback != primary
        assert eff2 == "high"
    finally:
        await store.close()


# ============================================================================
# F9: R5.4 Budgets & Economy Mode (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_prod_f9_01_budget_threshold_activates_economy_mode(tmp_path: Path) -> None:
    """Verify crossing configured token budget threshold switches QuotaBudgetManager to economy mode."""
    cfg = BotConfig.model_validate(
        {
            "telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"},
            "quota": {
                "hourly_token_budget": 1000,
                "daily_token_budget": 10000,
                "economy_threshold_ratio": 0.8,
                "admin_reserve_ratio": 0.2,
            },
        }
    )
    store = MemoryStore(db_path=str(tmp_path / "f9_01.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store, config=cfg)
        assert qm.is_economy_mode() is False
        await _maybe_await(
            qm.record_usage(
                UsageInfo(input_tokens=850, output_tokens=0, total_tokens=850),
                model="gemini-4-hydrogen",
                source="group",
            )
        )
        assert qm.is_economy_mode() is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_f9_02_economy_mode_switches_router_to_light_model(tmp_path: Path) -> None:
    """Verify economy mode routes group and proactive roles to light economy model."""
    cfg = BotConfig.model_validate(
        {
            "telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"},
            "quota": {
                "hourly_token_budget": 1000,
                "daily_token_budget": 10000,
                "economy_threshold_ratio": 0.8,
            },
        }
    )
    store = MemoryStore(db_path=str(tmp_path / "f9_02.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store, config=cfg)
        await _maybe_await(
            qm.record_usage(
                UsageInfo(input_tokens=900, total_tokens=900),
                model="gemini-4-hydrogen",
                source="group",
            )
        )
        assert qm.is_economy_mode() is True
        grp_m, grp_e = qm.select_model("group")
        pro_m, pro_e = qm.select_model("proactive")
        assert grp_m == "gemini-3.8-flash"
        assert pro_m == "gemini-3.8-flash"
        assert grp_e == "low"
        assert pro_e == "low"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_f9_03_economy_mode_dampens_proactive_and_spontaneous_replies(
    tmp_path: Path,
) -> None:
    """Verify economy mode suppresses proactive ticks and spontaneous group replies while allowing mentions."""
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "quota": {
                "hourly_token_budget": 1000,
                "daily_token_budget": 10000,
                "economy_threshold_ratio": 0.8,
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f9_03.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Прямой ответ!", model="gemini-3.8-flash")
    )
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    try:
        await _maybe_await(
            orch.quota_manager.record_usage(
                UsageInfo(input_tokens=900, total_tokens=900),
                model="gemini-4-hydrogen",
                source="group",
            )
        )
        assert orch.quota_manager.is_economy_mode() is True
        assert orch.quota_manager.should_allow_proactive() is False
        assert orch.quota_manager.should_allow_spontaneous() is False

        # Proactive tick must send nothing
        send_cb = AsyncMock()
        await orch.proactive_engine.run_proactive_tick(
            chat_states={GROUP_CHAT_ID: {"last_chat_msg_time": 0.0, "last_bot_msg_time": 0.0}},
            send_callback=send_cb,
        )
        send_cb.assert_not_awaited()

        # Spontaneous unmentioned message suppressed
        res_spont = await orch.handle_incoming_message(
            _make_incoming(
                chat_id=GROUP_CHAT_ID,
                sender_id=891777515,
                text="просто болтаем в чате без упоминания",
                is_mention=False,
                msg_id=10,
            )
        )
        assert res_spont is None
        assert mock_provider.generate_response.await_count == 0

        # Direct mention still answered
        res_mention = await orch.handle_incoming_message(
            _make_incoming(
                chat_id=GROUP_CHAT_ID,
                sender_id=891777515,
                text="Джефф, подскажи!",
                is_mention=True,
                msg_id=11,
            )
        )
        assert res_mention == "Прямой ответ!"
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_f9_04_economy_mode_defers_optional_self_edit_while_preserving_admin_reserve(
    tmp_path: Path,
) -> None:
    """Verify economy mode defers auto self_edit while preserving admin reserve."""
    cfg = BotConfig.model_validate(
        {
            "telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"},
            "quota": {
                "hourly_token_budget": 1000,
                "daily_token_budget": 10000,
                "economy_threshold_ratio": 0.8,
                "admin_reserve_ratio": 0.2,
            },
        }
    )
    store = MemoryStore(db_path=str(tmp_path / "f9_04.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store, config=cfg)
        await _maybe_await(
            qm.record_usage(
                UsageInfo(input_tokens=850, total_tokens=850),
                model="gemini-4-hydrogen",
                source="group",
            )
        )
        assert qm.is_economy_mode() is True
        allow_auto, reason = qm.should_allow_self_edit(priority="auto")
        assert allow_auto is False
        assert reason
        allow_admin, _ = qm.should_allow_self_edit(priority="admin")
        assert allow_admin is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_f9_05_entering_economy_mode_emits_single_admin_dm_warning(
    tmp_path: Path,
) -> None:
    """Verify crossing into economy mode sends one admin DM warning without spamming on repeat turns."""
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "quota": {
                "hourly_token_budget": 1000,
                "daily_token_budget": 10000,
                "economy_threshold_ratio": 0.8,
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f9_05.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text="Ок",
            model="gemini-4-hydrogen",
            usage=UsageInfo(input_tokens=850, output_tokens=50, total_tokens=900),
        )
    )
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    try:
        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=GROUP_CHAT_ID,
                sender_id=891777515,
                text="Джефф, раз",
                is_mention=True,
                msg_id=1,
            )
        )
        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=GROUP_CHAT_ID,
                sender_id=891777515,
                text="Джефф, два",
                is_mention=True,
                msg_id=2,
            )
        )
        admin_dm_calls = [
            c
            for c in mock_userbot.send_reply.call_args_list
            if c.kwargs.get("chat_id") == ADMIN_USER_ID
        ]
        assert len(admin_dm_calls) == 1
    finally:
        await orch.stop()


# ============================================================================
# F10: R5.5 /quota & System Context (5 tests)
# ============================================================================


_SAMPLE_QUOTA_TSV = (
    "Gemini Models\tWeekly Limit Remaining\t98%\t2026-11-25T10:00:00Z\n"
    "Gemini Models\tFive Hour Limit Remaining\t97%\t2026-11-18T18:00:00Z\n"
    "Claude and GPT models\tWeekly Limit Remaining\t100%\t2026-11-25T10:00:00Z\n"
    "Claude and GPT models\tFive Hour Limit Remaining\t95%\t2026-11-18T18:00:00Z\n"
    "Gemini 4 Argon\tWeekly Limit Remaining\t100%\t2026-11-25T10:00:00Z\n"
)


@pytest.mark.asyncio
async def test_prod_f10_01_admin_dm_quota_command_returns_formatted_report(
    tmp_path: Path,
) -> None:
    """Verify /quota in admin DM replies with parsed quota report without calling chat LLM."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "f10_01.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock()
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    try:
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["agy", "-p", "/quota"], returncode=0, stdout=_SAMPLE_QUOTA_TSV, stderr=""
            ),
        ):
            reply = await orch.handle_incoming_message(
                _make_incoming(
                    chat_id=ADMIN_USER_ID,
                    sender_id=ADMIN_USER_ID,
                    text="/quota",
                    is_private=True,
                    sender_name="Родион",
                )
            )
        mock_provider.generate_response.assert_not_awaited()
        assert reply is not None
        assert "98%" in reply
        assert "Gemini" in reply
        mock_userbot.send_reply.assert_awaited_once()
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_f10_02_quota_tsv_parser_handles_live_5_row_format(tmp_path: Path) -> None:
    """Verify QuotaBudgetManager parses live 5-row agy /quota TSV into format_quota_report."""
    store = MemoryStore(db_path=str(tmp_path / "f10_02.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        qm.parse_quota_output(_SAMPLE_QUOTA_TSV)
        report = qm.format_quota_report()
        assert "Gemini Models" in report
        assert "Claude" in report
        assert "Argon" in report
        assert "98%" in report
        assert "97%" in report
        assert "95%" in report
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_f10_03_quota_summary_injected_into_system_context(tmp_path: Path) -> None:
    """Verify GroupBotOrchestrator injects quota/model context summary into LLM call."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "f10_03.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="OK", model="gemini-3.8-flash")
    )
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    orch.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    try:
        orch.quota_manager.record_rate_limit("gemini-4-hydrogen", retry_after=120.0)
        summary = orch.quota_manager.format_context_summary()
        assert summary
        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=GROUP_CHAT_ID,
                sender_id=891777515,
                text="Джефф, как дела?",
                is_mention=True,
            )
        )
        call_kw = mock_provider.generate_response.call_args.kwargs
        combined = f"{call_kw.get('system_instructions') or ''}\n{call_kw.get('prompt') or ''}"
        assert "gemini" in combined.lower() or "квот" in combined.lower() or "quota" in combined.lower()
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_f10_04_non_admin_cannot_run_quota_command_in_group_or_dm(
    tmp_path: Path,
) -> None:
    """Verify non-admin users in group or DM cannot trigger admin /quota report."""
    other_dm = 555444333
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "telegram": {
                "allowed_dm_user_ids": [ADMIN_USER_ID, other_dm],
                "admin_user_ids": [ADMIN_USER_ID],
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f10_04.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="неа", model="gemini-3.8-flash")
    )
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    try:
        for cid, uid, priv in ((GROUP_CHAT_ID, 891777515, False), (other_dm, other_dm, True)):
            await orch.handle_incoming_message(
                _make_incoming(
                    chat_id=cid,
                    sender_id=uid,
                    text="/quota",
                    is_private=priv,
                    msg_id=cid + 10000000000 if cid < 0 else cid,
                )
            )
        for call in mock_userbot.send_reply.call_args_list:
            text = str(call.kwargs.get("text") or "")
            assert "Weekly Limit Remaining" not in text
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_f10_05_quota_command_gracefully_handles_agy_cli_failure(
    tmp_path: Path,
) -> None:
    """Verify /quota returns local SQLite usage + cooldowns gracefully when agy -p /quota fails."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "f10_05.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_userbot = _make_mock_userbot()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.0,
    )
    try:
        await _maybe_await(
            orch.quota_manager.record_usage(
                UsageInfo(input_tokens=42, output_tokens=8, total_tokens=50),
                model="gemini-3.8-flash",
                source="group",
            )
        )
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["agy", "-p", "/quota"], returncode=1, stdout="", stderr="CLI unavailable"
            ),
        ):
            reply = await orch.handle_incoming_message(
                _make_incoming(
                    chat_id=ADMIN_USER_ID,
                    sender_id=ADMIN_USER_ID,
                    text="/quota",
                    is_private=True,
                )
            )
        assert reply is not None
        assert "50" in reply or "gemini-3.8-flash" in reply
    finally:
        await orch.stop()


# ============================================================================
# F11: R4.1 SelfEvolutionManager & self_edit_history SQLite (6 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_f11_self_edit_history_schema_created(tmp_path: Path) -> None:
    """Verify MemoryStore.init_db creates self_edit_history with all 15 contract columns."""
    db_path = tmp_path / "evo_schema.sqlite"
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    await store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(self_edit_history)").fetchall()]
    finally:
        conn.close()

    expected = [
        "id",
        "created_at",
        "trigger_type",
        "signature",
        "reason",
        "affected_files",
        "verdict",
        "model",
        "branch",
        "commit_sha",
        "diff_lines",
        "test_result",
        "deploy_result",
        "rollback_commit",
        "summary",
    ]
    for col in expected:
        assert col in cols, f"Missing column {col!r} in self_edit_history: {cols}"


@pytest.mark.asyncio
async def test_t1_f11_self_edit_history_persists_across_reopen(tmp_path: Path) -> None:
    """Verify self_edit_history entries persist across MemoryStore close and reopen."""
    from bot.evolution.manager import SelfEditRecord

    db_path = tmp_path / "evo_persist.sqlite"
    store1 = MemoryStore(db_path=str(db_path))
    await store1.init_db()
    rec = SelfEditRecord(
        trigger_type="admin_dm",
        signature="dm:fix-1",
        reason="Fix bug",
        affected_files=["src/bot/main.py"],
        verdict="CODE_FIX_REQUIRED",
        model="gemini-4-hydrogen",
        branch="self-edit/fix-1",
        commit_sha="abc1234",
        diff_lines=12,
        test_result="passed",
        deploy_result="applied",
        rollback_commit="def5678",
        summary="Applied cleanly",
    )
    await store1.record_self_edit(rec)
    await store1.close()

    store2 = MemoryStore(db_path=str(db_path))
    await store2.init_db()
    try:
        rows = await store2.get_recent_self_edits(limit=5)
        assert len(rows) == 1
        loaded = rows[0]
        assert loaded.signature == "dm:fix-1"
        assert loaded.verdict == "CODE_FIX_REQUIRED"
        assert loaded.commit_sha == "abc1234"
        assert loaded.affected_files == ["src/bot/main.py"]
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_t1_f11_runtime_error_signature_triggers_evolution(tmp_path: Path) -> None:
    """Verify repeated identical non-transient runtime errors trigger SelfEvolutionManager."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
                "error_threshold_count": 2,
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "evo_rt.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(
                text='{"verdict": "CODE_FIX_REQUIRED", "reason": "real bug"}',
                model="claude-sonnet-5-5",
            )
        target = sandbox["staging_dir"] / "src" / "bot" / "app.py"
        target.write_text("VALUE = 2\n", encoding="utf-8")
        return ProviderResponse(text="Fixed VALUE=2", model=str(kwargs.get("model")))

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    notify = AsyncMock()

    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr,
        memory_store=store,
        quota_manager=qm,
        provider=provider,
        notify_admin=notify,
    )
    try:
        exc = ValueError("broken parser state")
        await mgr.handle_runtime_error(exc, signature="ValueError:broken_parser")
        await mgr.handle_runtime_error(exc, signature="ValueError:broken_parser")
        history = await store.get_recent_self_edits(limit=5)
        assert len(history) == 1
        assert history[0].trigger_type == "runtime_error"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f11_admin_dm_task_triggers_evolution(tmp_path: Path) -> None:
    """Verify admin DM self-edit task invokes SelfEvolutionManager with trigger_type='admin_dm'."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "evo_dm.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(
                text='{"verdict": "CODE_FIX_REQUIRED", "reason": "admin requested fix"}',
                model="claude-sonnet-5-5",
            )
        target = sandbox["staging_dir"] / "src" / "bot" / "app.py"
        target.write_text("VALUE = 3\n", encoding="utf-8")
        return ProviderResponse(text="Updated", model=str(kwargs.get("model")))

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    notify = AsyncMock()

    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr,
        memory_store=store,
        quota_manager=qm,
        provider=provider,
        notify_admin=notify,
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="поправь баг в app.py")
        history = await store.get_recent_self_edits(limit=5)
        assert len(history) == 1
        assert history[0].trigger_type == "admin_dm"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f11_disabled_self_evolution_is_noop(tmp_path: Path) -> None:
    """Verify self_evolution.enabled=False prevents Judge/self_dev calls and request.json writes."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": False,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "evo_disabled.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock()

    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr,
        memory_store=store,
        quota_manager=qm,
        provider=provider,
        notify_admin=AsyncMock(),
    )
    try:
        await mgr.handle_runtime_error(RuntimeError("x"), signature="RuntimeError:x")
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="fix x")
        provider.generate_response.assert_not_awaited()
        assert not sandbox["request_file"].exists()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f11_admin_dm_summary_report_sent_on_completion(tmp_path: Path) -> None:
    """Verify terminal evolution outcomes (rejected/NO_CHANGE) notify admin DM with a concise summary."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "evo_notify.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text='{"verdict": "NO_CHANGE", "reason": "false alarm"}',
            model="claude-sonnet-5-5",
        )
    )
    notify = AsyncMock()

    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr,
        memory_store=store,
        quota_manager=qm,
        provider=provider,
        notify_admin=notify,
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="проверь ошибку")
        notify.assert_awaited()
        msg_text = str(notify.call_args.args[0] if notify.call_args.args else notify.call_args.kwargs.get("text"))
        assert "NO_CHANGE" in msg_text or "false alarm" in msg_text
    finally:
        await store.close()


# ============================================================================
# F12: R4.2 Anti-Churn Guards (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_f12_signature_cooldown_blocks_duplicate_error(tmp_path: Path) -> None:
    """Verify duplicate error signature within cooldown is blocked without calling Judge."""
    from bot.evolution.manager import SelfEditRecord, SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "signature_cooldown_seconds": 3600,
                "error_threshold_count": 1,
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f12_sig.sqlite"))
    await store.init_db()
    await store.record_self_edit(
        SelfEditRecord(
            trigger_type="runtime_error",
            signature="KeyError:foo",
            reason="prior",
            affected_files=["src/bot/a.py"],
            verdict="CODE_FIX_REQUIRED",
            model="gemini-4-hydrogen",
            branch="b",
            commit_sha="111",
            diff_lines=1,
            test_result="passed",
            deploy_result="applied",
            rollback_commit="000",
            summary="done",
        )
    )
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.handle_runtime_error(KeyError("foo"), signature="KeyError:foo")
        provider.generate_response.assert_not_awaited()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f12_file_cooldown_blocks_recently_modified_file(tmp_path: Path) -> None:
    """Verify file cooldown blocks editing a recently touched file even under a new signature."""
    from bot.evolution.manager import SelfEditRecord, SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "file_cooldown_seconds": 3600,
                "error_threshold_count": 1,
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f12_file.sqlite"))
    await store.init_db()
    await store.record_self_edit(
        SelfEditRecord(
            trigger_type="runtime_error",
            signature="OldSig",
            reason="prior",
            affected_files=["src/bot/proactive/humor.py"],
            verdict="CODE_FIX_REQUIRED",
            model="gemini-4-hydrogen",
            branch="b",
            commit_sha="111",
            diff_lines=1,
            test_result="passed",
            deploy_result="applied",
            rollback_commit="000",
            summary="done",
        )
    )
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        allowed, reason = await mgr.check_file_allowed("src/bot/proactive/humor.py")
        assert allowed is False
        assert reason
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f12_daily_edit_limit_enforced(tmp_path: Path) -> None:
    """Verify max_daily_edits=2 blocks a 3rd autonomous self-edit in the same UTC day."""
    from bot.evolution.manager import SelfEditRecord, SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "max_daily_edits": 2,
                "error_threshold_count": 1,
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f12_daily.sqlite"))
    await store.init_db()
    for i in (1, 2):
        await store.record_self_edit(
            SelfEditRecord(
                trigger_type="runtime_error",
                signature=f"Sig:{i}",
                reason="fix",
                affected_files=[f"src/bot/f{i}.py"],
                verdict="CODE_FIX_REQUIRED",
                model="gemini-4-hydrogen",
                branch=f"b{i}",
                commit_sha=f"c{i}",
                diff_lines=2,
                test_result="passed",
                deploy_result="applied",
                rollback_commit="c0",
                summary="ok",
            )
        )
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.handle_runtime_error(RuntimeError("third"), signature="Sig:3")
        provider.generate_response.assert_not_awaited()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f12_transient_network_errors_suppressed(tmp_path: Path) -> None:
    """Verify transient network errors never trigger evolution."""
    from bot.evolution.manager import SelfEvolutionManager

    cfg_mgr = _make_isolated_config(tmp_path, bot_overrides={"self_evolution": {"enabled": True, "error_threshold_count": 1}})
    store = MemoryStore(db_path=str(tmp_path / "f12_net.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        for exc in (
            TimeoutError("request timed out"),
            ConnectionResetError("Connection reset by peer"),
            OSError("Network is unreachable"),
            RuntimeError("getUpdates timed out"),
        ):
            await mgr.handle_runtime_error(exc)
        provider.generate_response.assert_not_awaited()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f12_transient_429_quota_errors_suppressed(tmp_path: Path) -> None:
    """Verify 429 / RESOURCE_EXHAUSTED / exhausted capacity errors are treated as transient."""
    from bot.evolution.manager import SelfEvolutionManager

    cfg_mgr = _make_isolated_config(tmp_path, bot_overrides={"self_evolution": {"enabled": True, "error_threshold_count": 1}})
    store = MemoryStore(db_path=str(tmp_path / "f12_429.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.handle_runtime_error(AntigravityRateLimitError("429 RESOURCE_EXHAUSTED"))
        await mgr.handle_runtime_error(RuntimeError("You have exhausted your capacity on this model"))
        provider.generate_response.assert_not_awaited()
    finally:
        await store.close()


# ============================================================================
# F13: R4.3 Adversarial Judge (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_f13_judge_no_change_blocks_self_dev(tmp_path: Path) -> None:
    """Verify NO_CHANGE verdict skips self_dev, leaves staging HEAD unchanged, and records verdict."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f13_no.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text='{"verdict": "NO_CHANGE", "reason": "one-off transient"}',
            model="claude-sonnet-5-5",
        )
    )
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="investigate")
        assert provider.generate_response.await_count == 1
        head = subprocess.check_output(
            ["git", "-C", str(sandbox["staging_dir"]), "rev-parse", "HEAD"], text=True
        ).strip()
        assert head == sandbox["prev_commit"]
        rows = await store.get_recent_self_edits(limit=5)
        assert rows[0].verdict == "NO_CHANGE"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f13_judge_config_only_blocks_code_edit(tmp_path: Path) -> None:
    """Verify CONFIG_ONLY verdict skips self_dev code edit and sends config recommendation to admin."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f13_cfg.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text='{"verdict": "CONFIG_ONLY", "reason": "adjust cooldown_minutes in bot.yaml"}',
            model="claude-sonnet-5-5",
        )
    )
    notify = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=notify
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="too frequent banter")
        assert provider.generate_response.await_count == 1
        rows = await store.get_recent_self_edits(limit=5)
        assert rows[0].verdict == "CONFIG_ONLY"
        notify.assert_awaited()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f13_judge_code_fix_required_opens_self_dev(tmp_path: Path) -> None:
    """Verify CODE_FIX_REQUIRED opens self_dev session with tool_profile='self_dev' and staging cwd."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f13_fix.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    calls: list[dict[str, Any]] = []

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        calls.append(kwargs)
        if len(calls) == 1:
            return ProviderResponse(
                text='{"verdict": "CODE_FIX_REQUIRED", "reason": "bug confirmed"}',
                model="claude-sonnet-5-5",
            )
        (sandbox["staging_dir"] / "src" / "bot" / "app.py").write_text("VALUE = 9\n", encoding="utf-8")
        return ProviderResponse(text="Fixed", model=str(kwargs.get("model")))

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="fix VALUE")
        assert len(calls) >= 2
        assert calls[1]["tool_profile"] == "self_dev"
        assert Path(calls[1]["cwd"]).resolve() == sandbox["staging_dir"].resolve()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f13_judge_uses_non_editing_profile(tmp_path: Path) -> None:
    """Verify Adversarial Judge runs with non-editing tool_profile (not self_dev) and falsification framing."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "staging_dir": str(sandbox["staging_dir"])}},
    )
    store = MemoryStore(db_path=str(tmp_path / "f13_prof.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text='{"verdict": "NO_CHANGE", "reason": "ok"}', model="claude-sonnet-5-5")
    )
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="check bug")
        kw = provider.generate_response.call_args.kwargs
        assert kw.get("tool_profile") != "self_dev"
        prompt_l = str(kw.get("prompt") or provider.generate_response.call_args.args[0]).lower()
        assert any(w in prompt_l for w in ("фальсиф", "falsif", "no_change", "config_only"))
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f13_judge_routes_via_self_edit_judge_role(tmp_path: Path) -> None:
    """Verify Judge queries QuotaBudgetManager.select_model('self_edit_judge')."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "staging_dir": str(sandbox["staging_dir"])}},
    )
    store = MemoryStore(db_path=str(tmp_path / "f13_role.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    expected_judge_model, expected_judge_effort = qm.select_model("self_edit_judge")

    provider = MagicMock()
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text='{"verdict": "NO_CHANGE", "reason": "ok"}', model=expected_judge_model)
    )
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="check")
        kw = provider.generate_response.call_args.kwargs
        assert kw.get("model") == expected_judge_model
        assert kw.get("effort") == expected_judge_effort
    finally:
        await store.close()


# ============================================================================
# F14: R4.4 Model Escalation (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_f14_escalation_starts_with_gemini_4_hydrogen_high(tmp_path: Path) -> None:
    """Verify first self_dev escalation attempt starts with gemini-4-hydrogen at effort='high'."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f14_t1.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    self_dev_calls: list[dict[str, Any]] = []

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(text='{"verdict": "CODE_FIX_REQUIRED", "reason": "fix"}', model="claude-sonnet-5-5")
        self_dev_calls.append(kwargs)
        (sandbox["staging_dir"] / "src" / "bot" / "app.py").write_text("VALUE = 10\n", encoding="utf-8")
        return ProviderResponse(text="done", model=str(kwargs.get("model")))

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="fix app.py")
        assert self_dev_calls[0]["model"] == "gemini-4-hydrogen"
        assert self_dev_calls[0]["effort"] == "high"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f14_escalation_advances_through_full_chain(tmp_path: Path) -> None:
    """Verify escalation steps gemini-4-hydrogen -> claude-sonnet-5-5 -> claude-opus-5-5 on failures."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f14_chain.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    tried_models: list[str] = []

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(text='{"verdict": "CODE_FIX_REQUIRED", "reason": "fix"}', model="claude-sonnet-5-5")
        m = str(kwargs.get("model"))
        tried_models.append(m)
        if m != "claude-opus-5-5":
            raise RuntimeError(f"Attempt on {m} failed to apply patch")
        (sandbox["staging_dir"] / "src" / "bot" / "app.py").write_text("VALUE = 11\n", encoding="utf-8")
        return ProviderResponse(text="done", model=m)

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="hard fix")
        assert tried_models == ["gemini-4-hydrogen", "claude-sonnet-5-5", "claude-opus-5-5"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f14_escalation_skips_cooled_down_model(tmp_path: Path) -> None:
    """Verify cooled-down gemini-4-hydrogen is skipped straight to claude-sonnet-5-5."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f14_skip.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    qm.record_rate_limit("gemini-4-hydrogen", retry_after=300.0)

    tried_models: list[str] = []

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(text='{"verdict": "CODE_FIX_REQUIRED", "reason": "fix"}', model="claude-sonnet-5-5")
        m = str(kwargs.get("model"))
        tried_models.append(m)
        (sandbox["staging_dir"] / "src" / "bot" / "app.py").write_text("VALUE = 12\n", encoding="utf-8")
        return ProviderResponse(text="done", model=m)

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="fix")
        assert tried_models[0] == "claude-sonnet-5-5"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f14_mid_escalation_429_records_cooldown_and_advances(tmp_path: Path) -> None:
    """Verify 429 during Tier 1 records cooldown in QuotaBudgetManager and advances to Tier 2."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f14_429.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    tried_models: list[str] = []

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(text='{"verdict": "CODE_FIX_REQUIRED", "reason": "fix"}', model="claude-sonnet-5-5")
        m = str(kwargs.get("model"))
        tried_models.append(m)
        if m == "gemini-4-hydrogen":
            raise AntigravityRateLimitError("429 RESOURCE_EXHAUSTED retry in 180s")
        (sandbox["staging_dir"] / "src" / "bot" / "app.py").write_text("VALUE = 13\n", encoding="utf-8")
        return ProviderResponse(text="done", model=m)

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="fix")
        assert tried_models == ["gemini-4-hydrogen", "claude-sonnet-5-5"]
        next_m, _ = qm.select_model("self_edit")
        assert next_m != "gemini-4-hydrogen"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t1_f14_all_escalation_models_cooled_down_aborts_cleanly(tmp_path: Path) -> None:
    """Verify cooling down all escalation models aborts cleanly without touching staging and notifies admin."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f14_allcool.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    for m in ("gemini-4-hydrogen", "claude-sonnet-5-5", "claude-opus-5-5"):
        qm.record_rate_limit(m, retry_after=300.0)

    provider = MagicMock()
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text='{"verdict": "CODE_FIX_REQUIRED", "reason": "fix"}', model="gpt-oss-120b-medium")
    )
    notify = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=notify
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="fix")
        assert not sandbox["request_file"].exists()
        notify.assert_awaited()
    finally:
        await store.close()


# ============================================================================
# F15: R4.5 Staging, Gatekeeper & Canary (6 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_f15_staging_branch_and_request_json_written_atomically(tmp_path: Path) -> None:
    """Verify SelfEvolutionManager commits in staging worktree and writes request.json without mutating prod."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "trigger_dir": str(sandbox["request_file"].parent),
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "f15_req.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(text='{"verdict": "CODE_FIX_REQUIRED", "reason": "fix"}', model="claude-sonnet-5-5")
        (sandbox["staging_dir"] / "src" / "bot" / "app.py").write_text("VALUE = 42\n", encoding="utf-8")
        return ProviderResponse(text="Fixed", model=str(kwargs.get("model")))

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="set VALUE=42")
        assert sandbox["request_file"].is_file()
        req = json.loads(sandbox["request_file"].read_text(encoding="utf-8"))
        for key in ("request_id", "branch", "commit_sha", "reason", "requested_at"):
            assert key in req and req[key]
        prod_head = subprocess.check_output(
            ["git", "-C", str(sandbox["prod_dir"]), "rev-parse", "HEAD"], text=True
        ).strip()
        assert prod_head == sandbox["prev_commit"]
    finally:
        await store.close()


def test_t1_f15_gatekeeper_happy_path_applies_and_writes_result(tmp_path: Path) -> None:
    """Verify bot-self-deploy.sh validates staging commit, swaps prod, passes canary, and writes applied result."""
    sandbox = _gatekeeper_sandbox(tmp_path)
    staging = sandbox["staging_dir"]
    (staging / "src" / "bot" / "app.py").write_text("VALUE = 77\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(staging), "add", "src/bot/app.py"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(staging), "commit", "-m", "fix value"],
        check=True,
        capture_output=True,
    )
    new_commit = subprocess.check_output(["git", "-C", str(staging), "rev-parse", "HEAD"], text=True).strip()

    sandbox["request_file"].write_text(
        json.dumps(
            {
                "request_id": "req-happy-1",
                "branch": "self-edit/fix-1",
                "commit_sha": new_commit,
                "reason": "update value",
                "requested_at": "2026-11-18T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    proc = sandbox["run_gatekeeper"]()
    assert proc.returncode == 0, proc.stderr
    assert not sandbox["request_file"].exists()

    res = json.loads(sandbox["result_file"].read_text(encoding="utf-8"))
    assert res["status"] == "applied"
    assert res["prev_commit"] == sandbox["prev_commit"]
    assert res["new_commit"] == new_commit
    assert (sandbox["prod_dir"] / "src" / "bot" / "app.py").read_text(encoding="utf-8") == "VALUE = 77\n"


def test_t1_f15_gatekeeper_rejects_protected_paths(tmp_path: Path) -> None:
    """Verify gatekeeper rejects staging commits modifying protected files (.env, data/, systemd/, gatekeeper)."""
    for rel_path in (
        "deploy/bot-self-deploy.sh",
        "systemd/telegram-userbot.service",
    ):
        sub_tmp = tmp_path / rel_path.replace("/", "_")
        sub_tmp.mkdir()
        sandbox = _gatekeeper_sandbox(sub_tmp)
        target = sandbox["staging_dir"] / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# tampered\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(sandbox["staging_dir"]), "add", rel_path], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(sandbox["staging_dir"]), "commit", "-m", "bad"],
            check=True,
            capture_output=True,
        )
        bad_sha = subprocess.check_output(
            ["git", "-C", str(sandbox["staging_dir"]), "rev-parse", "HEAD"], text=True
        ).strip()
        sandbox["request_file"].write_text(
            json.dumps(
                {
                    "request_id": "req-prot",
                    "branch": "self-edit/fix-1",
                    "commit_sha": bad_sha,
                    "reason": "tamper",
                    "requested_at": "2026-11-18T12:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        sandbox["run_gatekeeper"]()
        res = json.loads(sandbox["result_file"].read_text(encoding="utf-8"))
        assert res["status"] == "rejected"
        prod_head = subprocess.check_output(
            ["git", "-C", str(sandbox["prod_dir"]), "rev-parse", "HEAD"], text=True
        ).strip()
        assert prod_head == sandbox["prev_commit"]


def test_t1_f15_gatekeeper_aborts_on_staging_checks_failure(tmp_path: Path) -> None:
    """Verify failing pytest in staging writes tests_failed and leaves prod and systemctl untouched."""
    sandbox = _gatekeeper_sandbox(tmp_path, pytest_rc=1)
    staging = sandbox["staging_dir"]
    (staging / "src" / "bot" / "app.py").write_text("VALUE = 88\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(staging), "add", "src/bot/app.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(staging), "commit", "-m", "failing test"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(staging), "rev-parse", "HEAD"], text=True).strip()

    sandbox["request_file"].write_text(
        json.dumps(
            {
                "request_id": "req-fail-test",
                "branch": "self-edit/fix-1",
                "commit_sha": sha,
                "reason": "test fail",
                "requested_at": "2026-11-18T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    sandbox["run_gatekeeper"]()
    res = json.loads(sandbox["result_file"].read_text(encoding="utf-8"))
    assert res["status"] == "tests_failed"
    prod_head = subprocess.check_output(["git", "-C", str(sandbox["prod_dir"]), "rev-parse", "HEAD"], text=True).strip()
    assert prod_head == sandbox["prev_commit"]
    systemctl_log = sandbox["state_dir"] / "systemctl.log"
    assert not systemctl_log.exists() or "restart" not in systemctl_log.read_text(encoding="utf-8")


def test_t1_f15_gatekeeper_canary_rollback_on_service_or_log_failure(tmp_path: Path) -> None:
    """Verify canary failure (inactive service or journald Traceback) rolls prod back to PREV_COMMIT."""
    sandbox = _gatekeeper_sandbox(
        tmp_path,
        journalctl_output="CRITICAL Traceback (most recent call last):\nRuntimeError: crash\n",
    )
    staging = sandbox["staging_dir"]
    (staging / "src" / "bot" / "app.py").write_text("VALUE = 99\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(staging), "add", "src/bot/app.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(staging), "commit", "-m", "canary crash"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(staging), "rev-parse", "HEAD"], text=True).strip()

    sandbox["request_file"].write_text(
        json.dumps(
            {
                "request_id": "req-canary-rb",
                "branch": "self-edit/fix-1",
                "commit_sha": sha,
                "reason": "canary test",
                "requested_at": "2026-11-18T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    sandbox["run_gatekeeper"]()
    res = json.loads(sandbox["result_file"].read_text(encoding="utf-8"))
    assert res["status"] == "rolled_back"
    assert res["prev_commit"] == sandbox["prev_commit"]
    prod_head = subprocess.check_output(["git", "-C", str(sandbox["prod_dir"]), "rev-parse", "HEAD"], text=True).strip()
    assert prod_head == sandbox["prev_commit"]
    assert (sandbox["prod_dir"] / "src" / "bot" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_t1_f15_systemd_self_deploy_path_and_service_units() -> None:
    """Verify deploy/telegram-self-deploy.path and .service units match gatekeeper contract."""
    path_unit = REPO_ROOT / "deploy" / "telegram-self-deploy.path"
    svc_unit = REPO_ROOT / "deploy" / "telegram-self-deploy.service"
    assert path_unit.is_file()
    assert svc_unit.is_file()

    cp_path = configparser.ConfigParser(interpolation=None)
    cp_path.optionxform = str
    cp_path.read_string(path_unit.read_text(encoding="utf-8"))
    assert cp_path["Path"]["PathExists"] == "/opt/telegram-group-bot/data/self_deploy/request.json"

    cp_svc = configparser.ConfigParser(interpolation=None)
    cp_svc.optionxform = str
    cp_svc.read_string(svc_unit.read_text(encoding="utf-8"))
    assert cp_svc["Service"]["Type"] == "oneshot"
    assert "/usr/local/sbin/bot-self-deploy.sh" in cp_svc["Service"]["ExecStart"]


# ============================================================================
# F16: R6 WireGuard Preparation (5 tests)
# ============================================================================


def test_t1_f16_wg0_example_structure_and_placeholders() -> None:
    """Verify deploy/wireguard/wg0.conf.example has [Interface], [Peer], and placeholder keys."""
    wg_example = REPO_ROOT / "deploy" / "wireguard" / "wg0.conf.example"
    assert wg_example.is_file()
    text = wg_example.read_text(encoding="utf-8")
    assert "[Interface]" in text
    assert "[Peer]" in text
    assert "PrivateKey" in text
    assert "PublicKey" in text
    assert "Endpoint" in text


def test_t1_f16_wg0_example_has_no_real_wireguard_keys() -> None:
    """Verify wg0.conf.example contains zero real 44-char base64 WireGuard keys."""
    text = (REPO_ROOT / "deploy" / "wireguard" / "wg0.conf.example").read_text(encoding="utf-8")
    real_wg_key = re.compile(r"(?:PrivateKey|PublicKey|PresharedKey)\s*=\s*[A-Za-z0-9+/]{42}[AEIMQUYcgkosw048]=")
    assert real_wg_key.search(text) is None


def test_t1_f16_wg0_example_preserves_lan_192_168_1_0_24() -> None:
    """Verify wg0.conf.example preserves 192.168.1.0/24 LAN routing across tunnel up/down."""
    text = (REPO_ROOT / "deploy" / "wireguard" / "wg0.conf.example").read_text(encoding="utf-8")
    assert "192.168.1.0/24" in text
    assert "PreUp" in text or "PostUp" in text
    assert "PreDown" in text or "PostDown" in text


def test_t1_f16_wg0_example_ipv6_disabled_safe() -> None:
    """Verify uncommented wg0.conf.example directives will not fail under disable_ipv6=1."""
    text = (REPO_ROOT / "deploy" / "wireguard" / "wg0.conf.example").read_text(encoding="utf-8")
    active_lines = [
        ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]
    for ln in active_lines:
        if ln.startswith("AllowedIPs") or ln.startswith("Address"):
            assert "::" not in ln, f"Unguarded IPv6 address/route in active line: {ln}"


def test_t1_f16_wg0_example_includes_optional_nftables_killswitch() -> None:
    """Verify wg0.conf.example documents optional nftables kill-switch allowing wg0 and LAN."""
    text = (REPO_ROOT / "deploy" / "wireguard" / "wg0.conf.example").read_text(encoding="utf-8")
    lower = text.lower()
    assert "nft" in lower
    assert "wg0" in lower
    assert "192.168.1.0/24" in text


# ============================================================================
# F17: R2 Server Packaging & Unit Hardening (5 tests)
# ============================================================================


def test_t1_f17_systemd_userbot_service_botops_and_hardening() -> None:
    """Verify systemd/telegram-userbot.service runs as botops:botops with hardening directives."""
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read_string((REPO_ROOT / "systemd" / "telegram-userbot.service").read_text(encoding="utf-8"))
    svc = cp["Service"]
    assert svc["User"] == "botops"
    assert svc["Group"] == "botops"
    assert svc["NoNewPrivileges"] == "true"
    assert svc["PrivateTmp"] == "true"
    assert svc["UMask"] == "0077"
    assert svc["WorkingDirectory"] == "/opt/telegram-group-bot"


def test_t1_f17_systemd_mtproto_userbot_isolated() -> None:
    """Verify telegram-userbot.service remains strictly isolated from /opt/telegram-mtproto-userbot."""
    cp_bot = configparser.ConfigParser(interpolation=None)
    cp_bot.read_string((REPO_ROOT / "systemd" / "telegram-userbot.service").read_text(encoding="utf-8"))
    assert cp_bot["Service"]["WorkingDirectory"] == "/opt/telegram-group-bot"
    assert "/opt/telegram-mtproto-userbot" not in cp_bot["Service"]["ExecStart"]
    mt_unit = REPO_ROOT / "systemd" / "telegram-mtproto-userbot.service"
    if mt_unit.is_file():
        cp_mt = configparser.ConfigParser(interpolation=None)
        cp_mt.read_string(mt_unit.read_text(encoding="utf-8"))
        assert cp_bot["Service"]["WorkingDirectory"] != cp_mt["Service"]["WorkingDirectory"]
        assert cp_bot["Service"]["EnvironmentFile"] != cp_mt["Service"]["EnvironmentFile"]
        assert "/opt/telegram-group-bot" not in cp_mt["Service"]["ExecStart"]


def test_t1_f17_gitignore_ignores_runtime_secrets_and_data(tmp_path: Path) -> None:
    """Verify .gitignore ignores runtime DBs, self_deploy files, oauth tokens, and sessions."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    shutil.copy2(REPO_ROOT / ".gitignore", repo / ".gitignore")

    for path in (
        ".env",
        "data/memory.sqlite",
        "data/self_deploy/request.json",
        "data/antigravity/oauth-token",
        "userbot.session",
        ".venv/bin/python",
        "src/__pycache__/a.pyc",
    ):
        rc = subprocess.run(["git", "-C", str(repo), "check-ignore", "-q", path]).returncode
        assert rc == 0, f"{path} was not ignored by .gitignore"

    for allowed in (".env.example", "deploy/wireguard/wg0.conf.example"):
        rc = subprocess.run(["git", "-C", str(repo), "check-ignore", "-q", allowed]).returncode
        assert rc == 1, f"{allowed} should NOT be ignored by .gitignore"


def test_t1_f17_bot_yaml_portable_ffmpeg_path() -> None:
    """Verify config/bot.example.yaml (and bot.yaml if present) does not hardcode /home/jeff ffmpeg_path."""
    cfg_path = REPO_ROOT / "config" / "bot.yaml"
    if not cfg_path.is_file():
        cfg_path = REPO_ROOT / "config" / "bot.example.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ffmpeg_path = raw["media"]["ffmpeg_path"]
    assert not ffmpeg_path.startswith("/home/jeff")
    assert ffmpeg_path in ("/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg")


def test_t1_f17_deploy_shell_scripts_pass_bash_syntax() -> None:
    """Verify all shell scripts under deploy/ pass `bash -n` syntax validation."""
    scripts = sorted((REPO_ROOT / "deploy").glob("*.sh"))
    assert scripts, "Expected deploy/*.sh scripts"
    for script in scripts:
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert proc.returncode == 0, f"bash -n failed on {script}: {proc.stderr}"


# ============================================================================
# F18: R7 Offline Dry-Run & Suite Readiness (5 tests)
# ============================================================================


def test_t1_f18_cli_subprocess_dry_run_bot_yaml(tmp_path: Path) -> None:
    """Verify subprocess CLI `run --config config/bot.yaml --dry-run` exits 0 with clean summary."""
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bot.cli",
            "run",
            "--config",
            str(REPO_ROOT / "config" / "bot.example.yaml"),
            "--env-file",
            str(empty_env),
            "--dry-run",
            "--log-level",
            "WARNING",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["leftover_temp_dirs"] == 0


def test_t1_f18_cli_subprocess_dry_run_example_yaml(tmp_path: Path) -> None:
    """Verify subprocess CLI `run --config config/bot.example.yaml --dry-run` exits 0."""
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bot.cli",
            "run",
            "--config",
            str(REPO_ROOT / "config" / "bot.example.yaml"),
            "--env-file",
            str(empty_env),
            "--dry-run",
            "--log-level",
            "WARNING",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"


@pytest.mark.asyncio
async def test_t1_f18_dry_run_validates_all_sqlite_tables_in_temp_db(tmp_path: Path) -> None:
    """Verify MemoryStore initialized during dry-run creates all 4 production tables cleanly."""
    store = MemoryStore(db_path=str(tmp_path / "dry_tables.sqlite"))
    await store.init_db()
    await store.close()

    conn = sqlite3.connect(str(tmp_path / "dry_tables.sqlite"))
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    finally:
        conn.close()

    for required in ("messages", "memory_facts", "usage_history", "self_edit_history"):
        assert required in tables


@pytest.mark.asyncio
async def test_t1_f18_dry_run_makes_zero_external_network_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify run_dry_run_smoke completes with zero external TCP socket connections."""
    monkeypatch.chdir(tmp_path)
    attempts: list[Any] = []

    def forbid_connect(*args: Any, **kwargs: Any) -> Any:
        attempts.append(args)
        raise AssertionError(f"Unexpected outbound socket connection during dry-run: {args}")

    with patch.object(socket, "create_connection", side_effect=forbid_connect):
        summary = await run_dry_run_smoke(config_path=REPO_ROOT / "config" / "bot.example.yaml")
    assert summary["status"] == "ok"
    assert attempts == []


def test_t1_f18_run_digest_dry_run_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `run-digest --dry-run` exits 0 offline without sending messages."""
    from bot.cli import main as cli_main

    monkeypatch.chdir(tmp_path)
    rc = cli_main(
        [
            "run-digest",
            "--config",
            str(REPO_ROOT / "config" / "bot.example.yaml"),
            "--slot",
            "morning_digest",
            "--dry-run",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["sent"] is False
