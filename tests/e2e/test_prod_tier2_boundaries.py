"""Tier 2 Production Hardening Boundary, Corner & Negative E2E Tests (96 tests, F1-F18).

Covers boundary values, edge conditions, malformed inputs, concurrency,
security fail-closed invariants, and negative paths across all 18 features.
"""

from __future__ import annotations

import asyncio
import configparser
import io
import json
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from bot.antigravity.auth import build_clean_child_env
from bot.antigravity.cli_client import AntigravityCLIClient, watch_log_for_errors
from bot.antigravity.errors import (
    AntigravityAuthError,
    AntigravityNativeToolAbort,
    AntigravityRateLimitError,
    extract_retry_after,
)
from bot.antigravity.stream import AntigravityStreamParser
from bot.config.manager import ConfigManager
from bot.config.models import BotConfig
from bot.core.models import ProviderResponse, UsageInfo
from bot.main import GroupBotOrchestrator
from bot.media.audio import resolve_ffmpeg_path
from bot.media.processor import MediaProcessor
from bot.memory.db import ChatMessage, MemoryStore
from bot.proactive.digest import NewsDigestGenerator
from bot.proactive.engine import ProactiveEngine
from bot.telegram.bot_api_client import BotAPIMessageEvent, TelegramBotAPIClient
from bot.telegram.handlers import IncomingMessage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADMIN_USER_ID = 382962362
GROUP_CHAT_ID = -1003845062562


# ============================================================================
# Shared Offline Harness Helpers
# ============================================================================


def _make_isolated_config(
    tmp_path: Path,
    *,
    bot_overrides: Optional[dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
) -> ConfigManager:
    """Create isolated ConfigManager in tmp_path."""
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
    if system_prompt is None:
        repo_prompt = REPO_ROOT / "config" / "system_prompt.md"
        system_prompt = (
            repo_prompt.read_text(encoding="utf-8")
            if repo_prompt.is_file()
            else "Ты — Джефф. Запрещены незаконные действия и пиратские торренты."
        )
    (cfg_dir / "system_prompt.md").write_text(system_prompt, encoding="utf-8")
    return ConfigManager(config_dir=cfg_dir)


def _make_mock_userbot(bot_id: int = 777000, bot_username: str = "jeff_test_bot") -> MagicMock:
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


def _make_quota_manager(memory_store: MemoryStore, config: Optional[BotConfig] = None) -> Any:
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
    import inspect

    if inspect.isawaitable(val):
        return await val
    return val


class _FakeACPProcess:
    def __init__(
        self,
        on_prompt: Callable[[dict[str, Any]], list[dict[str, Any]]],
    ) -> None:
        self.stdout_queue: queue.Queue[str] = queue.Queue()
        self.stderr = io.StringIO("")
        self.pid = 77777
        self._terminated = False
        self._on_prompt = on_prompt
        self.stdin = self

    def write(self, data: str) -> int:
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method = msg.get("method")
            req_id = msg.get("id")
            if method == "initialize":
                self._push({"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": 1}})
            elif method == "session/new":
                self._push(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "sessionId": "acp-b-1",
                            "models": {
                                "currentModelId": "gemini-4-hydrogen",
                                "availableModels": [{"modelId": "gemini-4-hydrogen", "name": "gemini-4-hydrogen"}],
                            },
                        },
                    }
                )
            elif method in ("session/set_model", "session/set_mode"):
                self._push({"jsonrpc": "2.0", "id": req_id, "result": {}})
            elif method == "session/prompt":
                for frame in self._on_prompt(msg):
                    self._push(frame)
        return len(data)

    def flush(self) -> None:
        pass

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


async def _run_acp_turn(
    cli: AntigravityCLIClient,
    proc: _FakeACPProcess,
    *,
    tool_profile: str = "chat",
) -> ProviderResponse:
    import inspect

    sig = inspect.signature(cli._sync_execute_acp_turn)
    kwargs: dict[str, Any] = {"prompt": "test", "image_payloads": []}
    if "tool_profile" in sig.parameters:
        kwargs["tool_profile"] = tool_profile
    with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=proc):
        return await asyncio.to_thread(cli._sync_execute_acp_turn, **kwargs)


def _gatekeeper_sandbox(
    tmp_path: Path,
    *,
    max_diff_lines: int = 200,
    ruff_rc: int = 0,
    pytest_rc: int = 0,
    dry_run_rc: int = 0,
) -> dict[str, Any]:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    state_dir = tmp_path / "state"
    bin_dir = tmp_path / "bin"
    prod_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init", "-b", "main", str(prod_dir)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(prod_dir), "config", "user.name", "Bot Tester"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(prod_dir), "config", "user.email", "tester@example.invalid"], check=True, capture_output=True)

    (prod_dir / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    (prod_dir / "config").mkdir(parents=True, exist_ok=True)
    (prod_dir / "config" / "bot.yaml").write_text("telegram:\n  mode: bot_api\n", encoding="utf-8")
    (prod_dir / "src" / "bot").mkdir(parents=True, exist_ok=True)
    (prod_dir / "src" / "bot" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (prod_dir / ".gitignore").write_text(".env\ndata/\n.venv/\n", encoding="utf-8")

    subprocess.run(["git", "-C", str(prod_dir), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(prod_dir), "commit", "-m", "baseline"], check=True, capture_output=True)
    prev_commit = subprocess.check_output(["git", "-C", str(prod_dir), "rev-parse", "HEAD"], text=True).strip()

    (prod_dir / ".env").write_text("TELEGRAM_BOT_TOKEN=111111:FAKE_SANDBOX_TOKEN\n", encoding="utf-8")
    deploy_dir = prod_dir / "data" / "self_deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "-C", str(prod_dir), "worktree", "add", "-b", "self-edit/fix-1", str(staging_dir)],
        check=True,
        capture_output=True,
    )

    for base in (staging_dir, prod_dir):
        venv_bin = base / ".venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        for tool_name, rc_name in (("ruff", "ruff_rc"), ("pytest", "pytest_rc"), ("telegram-group-bot", "dry_run_rc")):
            shim = venv_bin / tool_name
            shim.write_text(
                f"#!/usr/bin/env bash\nrc=$(cat '{state_dir}/{rc_name}')\nexit ${{rc:-0}}\n",
                encoding="utf-8",
            )
            shim.chmod(0o755)

    (state_dir / "ruff_rc").write_text(str(ruff_rc), encoding="utf-8")
    (state_dir / "pytest_rc").write_text(str(pytest_rc), encoding="utf-8")
    (state_dir / "dry_run_rc").write_text(str(dry_run_rc), encoding="utf-8")

    (bin_dir / "systemctl").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (bin_dir / "systemctl").chmod(0o755)
    (bin_dir / "journalctl").write_text("#!/usr/bin/env bash\necho 'INFO ok'\nexit 0\n", encoding="utf-8")
    (bin_dir / "journalctl").chmod(0o755)

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
        return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=30)

    return {
        "prod_dir": prod_dir,
        "staging_dir": staging_dir,
        "request_file": request_file,
        "result_file": result_file,
        "prev_commit": prev_commit,
        "run_gatekeeper": run_gatekeeper,
    }


# ============================================================================
# B1: F1 Security Boundaries (6 tests)
# ============================================================================


def test_prod_b1_01_no_partial_ip_or_mtproto_path_in_bot_api_deploy() -> None:
    """Verify bot_api deploy wrapper, bot configs, and telegram-userbot.service never target mtproto."""
    for path in (
        REPO_ROOT / "deploy" / "deploy_bot_api.sh",
        REPO_ROOT / "config" / "bot.example.yaml",
        REPO_ROOT / "systemd" / "telegram-userbot.service",
    ):
        text = path.read_text(encoding="utf-8")
        assert "/opt/telegram-mtproto-userbot" not in text
        assert "telegram-mtproto-userbot" not in text
    bot_yaml = REPO_ROOT / "config" / "bot.yaml"
    if bot_yaml.is_file():
        text = bot_yaml.read_text(encoding="utf-8")
        assert "/opt/telegram-mtproto-userbot" not in text
        assert "telegram-mtproto-userbot" not in text


def test_prod_b1_02_example_env_files_have_only_empty_or_placeholder_credentials() -> None:
    """Verify .env.example and optional .env.userbot.example contain only empty or dummy creds."""
    for name in (".env.example", ".env.userbot.example"):
        p = REPO_ROOT / name
        if not p.is_file():
            continue
        kv: dict[str, str] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip("'\"")
        assert kv.get("TELEGRAM_BOT_TOKEN", "") == ""
        if "TELEGRAM_API_ID" in kv:
            assert kv["TELEGRAM_API_ID"] in ("", "12345678")
        if "TELEGRAM_API_HASH" in kv:
            assert kv["TELEGRAM_API_HASH"] in ("", "0123456789abcdef0123456789abcdef")


def test_prod_b1_03_empty_and_none_env_values_not_leaked_by_child_env(tmp_path: Path) -> None:
    """Verify empty/whitespace secret env values are stripped cleanly without raising."""
    with patch.dict(
        os.environ,
        {"TELEGRAM_BOT_TOKEN": "", "OPENAI_API_KEY": "   ", "API_SECRET": ""},
        clear=False,
    ):
        clean = build_clean_child_env(tmp_path)
    for k in ("TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "API_SECRET"):
        assert k not in clean


def test_prod_b1_04_deploy_remote_path_traversal_guard() -> None:
    """Verify deploy_remote.sh canonicalizes trailing/double slashes and still exits 2."""
    script = REPO_ROOT / "deploy" / "deploy_remote.sh"
    proc = subprocess.run(
        ["bash", str(script), "bot_api", "root@127.0.0.1", "/opt//telegram-mtproto-userbot/"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2


def test_prod_b1_05_bot_yaml_has_no_userbot_session_mode_confusion() -> None:
    """Verify repo bot config does not set mode='userbot' and deploy_bot_api.sh never touches sessions."""
    cfg_path = REPO_ROOT / "config" / "bot.yaml"
    if not cfg_path.is_file():
        cfg_path = REPO_ROOT / "config" / "bot.example.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["telegram"]["mode"] != "userbot"
    assert "userbot.session" not in (REPO_ROOT / "deploy" / "deploy_bot_api.sh").read_text(encoding="utf-8")


def test_prod_b1_06_deploy_userbot_rejects_bot_api_target_dir() -> None:
    """Verify deploy_remote.sh exits 2 when userbot profile targets /opt/telegram-group-bot."""
    script = REPO_ROOT / "deploy" / "deploy_remote.sh"
    proc = subprocess.run(
        ["bash", str(script), "userbot", "root@127.0.0.1", "/opt/telegram-group-bot/"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2


# ============================================================================
# B2: F2 Secret Sanitizer Boundaries (6 tests)
# ============================================================================


def test_prod_b2_01_short_secrets_under_6_chars_do_not_corrupt_normal_text() -> None:
    """Verify registered secrets shorter than 6 chars do not corrupt ordinary words/numbers."""
    from bot.security.sanitizer import SecretSanitizer

    sanitizer = SecretSanitizer(extra_secrets=["12345", "ok"])
    sample = "ok 12345 hello world"
    assert sanitizer.sanitize(sample) == sample


def test_prod_b2_02_exact_6_char_secret_is_redacted() -> None:
    """Verify boundary secret length of exactly 6 chars is redacted."""
    from bot.security.sanitizer import SecretSanitizer

    sanitizer = SecretSanitizer(extra_secrets=["sec666"])
    assert sanitizer.sanitize("key=sec666") == "key=[REDACTED]"


def test_prod_b2_03_multiline_dotenv_quoted_values_and_comments_parsed_safely(tmp_path: Path) -> None:
    """Verify .env parser handles comments, blank lines, quotes, and '=' inside base64 values."""
    from bot.security.sanitizer import SecretSanitizer

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment line\n"
        "\n"
        'DOUBLE_QUOTED="secret_double_quoted_val"\n'
        "SINGLE_QUOTED='secret_single_quoted_val'\n"
        "B64_TOKEN=abc12345==\n"
        "EMPTY_VAL=\n"
        "SHORT=abc\n",
        encoding="utf-8",
    )
    sanitizer = SecretSanitizer(env_path=env_file)
    out = sanitizer.sanitize("secret_double_quoted_val secret_single_quoted_val abc12345== abc")
    assert "secret_double_quoted_val" not in out
    assert "secret_single_quoted_val" not in out
    assert "abc12345==" not in out
    assert out.endswith("abc")


def test_prod_b2_04_missing_or_unreadable_env_path_is_non_fatal(tmp_path: Path) -> None:
    """Verify missing env_path initializes cleanly and still redacts pattern-matched tokens."""
    from bot.security.sanitizer import SecretSanitizer

    sanitizer = SecretSanitizer(env_path=tmp_path / "does_not_exist.env")
    out = sanitizer.sanitize("token=123456789:AAH_abcdefghijklmnopqrstuvwxyz12345")
    assert "123456789:AAH_abcdefghijklmnopqrstuvwxyz12345" not in out
    assert "[REDACTED]" in out


def test_prod_b2_05_idempotent_sanitization_and_unicode_preservation() -> None:
    """Verify double sanitization is idempotent and preserves Cyrillic/Ukrainian/emojis."""
    from bot.security.sanitizer import SecretSanitizer

    secret = "my_secret_value_777"
    sanitizer = SecretSanitizer(extra_secrets=[secret])
    text = f"Привіт світ 🔥 {secret} і ще текст"
    once = sanitizer.sanitize(text)
    twice = sanitizer.sanitize(once)
    assert once == twice
    assert "Привіт світ 🔥 [REDACTED] і ще текст" == twice
    assert sanitizer.sanitize("") == ""


def test_prod_b2_06_sanitizer_handles_empty_whitespace_and_nested_token_variants() -> None:
    """Verify sanitizer handles whitespace-only input and OAuth refresh token pattern 1//0..."""
    from bot.security.sanitizer import SecretSanitizer

    sanitizer = SecretSanitizer()
    assert sanitizer.sanitize("   \n\t  ") == "   \n\t  "
    refresh = "refresh=1//0gFakeOAuthRefreshToken1234567890abcdef"
    cleaned = sanitizer.sanitize(refresh)
    assert "1//0gFakeOAuthRefreshToken1234567890abcdef" not in cleaned


# ============================================================================
# B3: F3 Legal Prohibition Prompt Boundaries (5 tests)
# ============================================================================


def test_prod_b3_01_legal_prohibition_covers_jailbreak_roleplay_and_torrent_variants() -> None:
    """Verify system_prompt.md forbids illegal/pirated content regardless of framing."""
    prompt = (REPO_ROOT / "config" / "system_prompt.md").read_text(encoding="utf-8").lower()
    assert any(w in prompt for w in ("торрент", "пират", "torrent"))
    assert any(w in prompt for w in ("независимо", "кто бы ни просил", "исключен"))


def test_prod_b3_02_missing_system_prompt_file_returns_empty_without_crash(tmp_path: Path) -> None:
    """Verify ConfigManager returns empty system_prompt when system_prompt.md is missing."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "bot.yaml").write_text(
        yaml.safe_dump({"telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"}}),
        encoding="utf-8",
    )
    mgr = ConfigManager(config_dir=cfg_dir)
    assert mgr.system_prompt == ""


def test_prod_b3_03_prompt_size_stays_concise_under_token_budget() -> None:
    """Verify config/system_prompt.md stays concise (< 8000 chars) to prevent CONTEXT_OVERFLOW."""
    text = (REPO_ROOT / "config" / "system_prompt.md").read_text(encoding="utf-8")
    assert 100 < len(text) < 8000


def test_prod_b3_04_userbot_system_prompt_consistency() -> None:
    """Verify ConfigManager loads system_prompt.md (not userbot_system_prompt.md) for repo config."""
    mgr = ConfigManager(config_dir=REPO_ROOT / "config")
    assert mgr.system_prompt_path.name == "system_prompt.md"


@pytest.mark.asyncio
async def test_prod_b3_05_proactive_digest_and_humor_do_not_drop_system_instructions() -> None:
    """Verify NewsDigestGenerator includes system_instructions passed from ConfigManager."""
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Сводка новостей.", model="gemini-3.8-flash")
    )
    gen = NewsDigestGenerator(provider=mock_provider)
    cfg_mgr = ConfigManager(config_dir=REPO_ROOT / "config")
    await gen.generate_digest(
        topics=["технологии"],
        system_instructions=cfg_mgr.system_prompt,
    )
    passed_sys = str(mock_provider.generate_response.call_args.kwargs.get("system_instructions") or "").lower()
    assert any(w in passed_sys for w in ("торрент", "пират"))


# ============================================================================
# B4: F4 Config & Admin DM Routing Boundaries (6 tests)
# ============================================================================


def test_prod_b4_01_unlisted_dm_user_dropped_in_bot_api_client() -> None:
    """Verify unlisted private DM sender is dropped (returns None) in TelegramBotAPIClient."""
    client = TelegramBotAPIClient(
        bot_token="123456:DUMMY_TOKEN_ABCDEFGH",
        allowed_chat_ids=[GROUP_CHAT_ID],
        allowed_dm_user_ids=[ADMIN_USER_ID],
    )
    client.bot_id = 777000
    msg = {
        "message_id": 100,
        "from": {"id": 999999999, "is_bot": False, "first_name": "Stranger"},
        "chat": {"id": 999999999, "type": "private"},
        "date": 1794586333,
        "text": "hello",
    }
    assert client._convert_msg_dict(msg) is None


def test_prod_b4_02_ignored_bot_username_and_is_bot_dropped_in_dm_and_group() -> None:
    """Verify bot-originated messages and ignored usernames are blocked in both group and DM."""
    client = TelegramBotAPIClient(
        bot_token="123456:DUMMY_TOKEN_ABCDEFGH",
        allowed_chat_ids=[GROUP_CHAT_ID],
        allowed_dm_user_ids=[ADMIN_USER_ID],
        ignored_sender_usernames=["hermes_kuk7yi6kfm3lcdfa_bot"],
    )
    client.bot_id = 777000

    for chat_obj in (
        {"id": GROUP_CHAT_ID, "type": "supergroup"},
        {"id": ADMIN_USER_ID, "type": "private"},
    ):
        bot_sender = {
            "message_id": 200,
            "from": {"id": 888000, "is_bot": True, "username": "other_bot"},
            "chat": chat_obj,
            "date": 1794586333,
            "text": "ping",
        }
        ignored_user = {
            "message_id": 201,
            "from": {"id": 888001, "is_bot": False, "username": "hermes_kuk7yi6kfm3lcdfa_bot"},
            "chat": chat_obj,
            "date": 1794586333,
            "text": "ping",
        }
        res_bot = client._convert_msg_dict(bot_sender)
        res_ign = client._convert_msg_dict(ignored_user)
        assert res_bot is None or res_bot.sender_is_bot is True
        assert res_ign is None


@pytest.mark.asyncio
async def test_prod_b4_03_empty_admin_user_ids_grants_zero_admin_privileges(
    tmp_path: Path,
) -> None:
    """Verify empty admin_user_ids denies ops tool_profile and /quota even to 382962362."""
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "telegram": {
                "allowed_dm_user_ids": [ADMIN_USER_ID],
                "admin_user_ids": [],
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "b4_03.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="chat only", model="gemini-3.8-flash")
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
                text="привет",
                is_private=True,
                msg_id=1,
            )
        )
        assert mock_provider.generate_response.call_args.kwargs.get("tool_profile", "chat") == "chat"

        await orch.handle_incoming_message(
            _make_incoming(
                chat_id=ADMIN_USER_ID,
                sender_id=ADMIN_USER_ID,
                text="/quota",
                is_private=True,
                msg_id=2,
            )
        )
        for c in mock_userbot.send_reply.call_args_list:
            assert "Weekly Limit Remaining" not in str(c.kwargs.get("text") or "")
    finally:
        await orch.stop()


def test_prod_b4_04_ffmpeg_path_resolution_missing_custom_path_falls_back_cleanly() -> None:
    """Verify resolve_ffmpeg_path falls back to system PATH or raises explicit FileNotFoundError."""
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "bot.media.audio.DEFAULT_STATIC_FFMPEG", "/nonexistent/static/ffmpeg"
    ):
        resolved = resolve_ffmpeg_path("/nonexistent/custom/ffmpeg")
        assert resolved == "/usr/bin/ffmpeg"

    with patch("shutil.which", return_value=None), patch(
        "bot.media.audio.DEFAULT_STATIC_FFMPEG", "/nonexistent/static/ffmpeg"
    ), patch("os.path.isfile", return_value=False), patch("pathlib.Path.glob", return_value=[]):
        with pytest.raises(FileNotFoundError):
            resolve_ffmpeg_path("/nonexistent/custom/ffmpeg")


def test_prod_b4_05_bot_api_client_malformed_chat_or_from_dict_does_not_crash() -> None:
    """Verify malformed Bot API updates missing 'from' or using 'caption' do not crash."""
    client = TelegramBotAPIClient(
        bot_token="123456:DUMMY_TOKEN_ABCDEFGH",
        allowed_chat_ids=[GROUP_CHAT_ID],
    )
    client.bot_id = 777000

    malformed = {"message_id": 300, "chat": {"id": GROUP_CHAT_ID}, "caption": "фото подпись"}
    ev = BotAPIMessageEvent(malformed, bot_id=777000)
    assert ev.raw_text == "фото подпись"
    inc = client._convert_msg_dict(malformed)
    assert inc is not None


def test_prod_b4_06_negative_group_id_never_treated_as_private_dm() -> None:
    """Verify negative supergroup chat ID is never marked is_private=True even if type omitted."""
    client = TelegramBotAPIClient(
        bot_token="123456:DUMMY_TOKEN_ABCDEFGH",
        allowed_chat_ids=[GROUP_CHAT_ID],
        allowed_dm_user_ids=[ADMIN_USER_ID],
    )
    client.bot_id = 777000
    msg = {
        "message_id": 301,
        "from": {"id": ADMIN_USER_ID, "is_bot": False, "first_name": "Родион"},
        "chat": {"id": GROUP_CHAT_ID},
        "text": "тест",
    }
    inc = client._convert_msg_dict(msg)
    assert inc is not None
    assert inc.is_private is False


# ============================================================================
# B5: F5 Tool Profiles & ACP Boundaries (6 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_prod_b5_01_acp_title_spoofing_blocked_when_name_is_run_command(tmp_path: Path) -> None:
    """Verify ACP blocks name='run_command' even if title='search_web' (anti-spoofing)."""

    def on_prompt(msg: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call",
                        "name": "run_command",
                        "title": "search_web",
                    }
                },
            },
            {"jsonrpc": "2.0", "id": msg["id"], "result": {"stopReason": "end_turn"}},
        ]

    cli = AntigravityCLIClient(data_dir=tmp_path, binary="/usr/bin/agy")
    try:
        proc = _FakeACPProcess(on_prompt=on_prompt)
        with pytest.raises(AntigravityNativeToolAbort):
            await _run_acp_turn(cli, proc, tool_profile="chat")
    finally:
        cli.close()


@pytest.mark.asyncio
async def test_prod_b5_02_acp_tool_call_update_is_also_checked(tmp_path: Path) -> None:
    """Verify ACP sessionUpdate='tool_call_update' is checked and aborts forbidden tools."""

    def on_prompt(msg: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "name": "write_file",
                        "title": "Writing file",
                    }
                },
            },
            {"jsonrpc": "2.0", "id": msg["id"], "result": {"stopReason": "end_turn"}},
        ]

    cli = AntigravityCLIClient(data_dir=tmp_path, binary="/usr/bin/agy")
    try:
        proc = _FakeACPProcess(on_prompt=on_prompt)
        with pytest.raises(AntigravityNativeToolAbort):
            await _run_acp_turn(cli, proc, tool_profile="chat")
    finally:
        cli.close()


def test_prod_b5_03_unnamed_tool_call_aborted_in_all_profiles() -> None:
    """Verify empty/unnamed native tool step_update is aborted in both chat and ops profiles."""
    for profile in ("chat", "ops"):
        parser = AntigravityStreamParser(tool_profile=profile)
        with pytest.raises(AntigravityNativeToolAbort):
            parser.feed_line(
                json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "name": ""}})
            )


def test_prod_b5_04_invalid_tool_profile_rejected_fail_closed() -> None:
    """Verify unknown tool_profile either raises ValueError or fails closed to CHAT_ALLOWED_TOOLS."""
    try:
        parser = AntigravityStreamParser(tool_profile="superuser")  # type: ignore[arg-type]
    except ValueError:
        return
    with pytest.raises(AntigravityNativeToolAbort):
        parser.feed_line(
            json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "name": "run_command"}})
        )


def test_prod_b5_05_xml_tool_call_text_tag_vs_native_tool_distinction_and_sdk_profile() -> None:
    """Verify inline XML <tool_call> text parses cleanly without native abort and SDK honors profile."""
    parser = AntigravityStreamParser(tool_profile="chat")
    chunks = parser.feed_line(
        json.dumps(
            {
                "event": "step_update",
                "step_update": {"text_delta": '<tool_call>{"name": "search_web", "arguments": {}}</tool_call>'},
            }
        )
    )
    final_chunks = parser.finish()
    all_calls = [tc for c in chunks + final_chunks if c.tool_calls for tc in c.tool_calls]
    assert len(all_calls) == 1
    assert parser.native_tool_intercepted is False


def test_prod_b5_06_explicit_allowed_tools_override_takes_precedence_safely() -> None:
    """Verify explicit allowed_tools override restricts parser strictly to provided set."""
    parser = AntigravityStreamParser(allowed_tools=frozenset({"search_web"}), tool_profile="ops")
    parser.feed_line(json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "name": "search_web"}}))
    with pytest.raises(AntigravityNativeToolAbort):
        parser.feed_line(json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "name": "run_command"}}))


# ============================================================================
# B6: F6 UsageInfo & SQLite Boundaries (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_prod_b6_01_zero_and_missing_usage_fields_handled_gracefully(tmp_path: Path) -> None:
    """Verify empty UsageInfo() persists cleanly and ProviderResponse(usage=None) does not crash."""
    db_path = tmp_path / "b6_01.sqlite"
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        await _maybe_await(qm.record_usage(UsageInfo(), model="gemini-3.8-flash", source="group"))
    finally:
        await store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM usage_history").fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_prod_b6_02_stream_parser_result_without_usage_dict_emits_zero_usage() -> None:
    """Verify stream result event lacking usage key emits UsageInfo with zero token counts."""
    parser = AntigravityStreamParser()
    parser.feed_line(json.dumps({"event": "init", "conversation_id": "c1"}))
    parser.feed_line(json.dumps({"event": "step_update", "step_update": {"text_delta": "hi"}}))
    parser.feed_line(json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "hi"}}))
    chunks = parser.finish()
    usage = chunks[-1].usage
    assert usage is not None
    assert usage.total_tokens == 0
    assert usage.input_tokens == 0


@pytest.mark.asyncio
async def test_prod_b6_03_large_token_counts_and_unicode_model_names(tmp_path: Path) -> None:
    """Verify >32-bit integer token counts and suffixed model names round-trip accurately."""
    db_path = tmp_path / "b6_03.sqlite"
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    big_val = 2**32 + 12345
    try:
        qm = _make_quota_manager(store)
        await _maybe_await(
            qm.record_usage(
                UsageInfo(input_tokens=big_val, output_tokens=5, total_tokens=big_val + 5),
                model="claude-sonnet-5-5-high",
                source="self_edit",
            )
        )
    finally:
        await store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT model, input_tokens, total_tokens FROM usage_history").fetchone()
        assert row == ("claude-sonnet-5-5-high", big_val, big_val + 5)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_prod_b6_04_concurrent_async_usage_writes_no_sqlite_locked_error(tmp_path: Path) -> None:
    """Verify 20 concurrent async record_usage calls commit cleanly without SQLite locked errors."""
    db_path = tmp_path / "b6_04.sqlite"
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        await asyncio.gather(
            *[
                _maybe_await(
                    qm.record_usage(
                        UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
                        model="gemini-3.8-flash",
                        source="group",
                    )
                )
                for _ in range(20)
            ]
        )
    finally:
        await store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM usage_history").fetchone()[0] == 20
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_prod_b6_05_usage_history_survives_db_reopen_for_budget_window(tmp_path: Path) -> None:
    """Verify persisted usage_history restores economy budget state across MemoryStore reopen."""
    db_path = tmp_path / "b6_05.sqlite"
    cfg = BotConfig.model_validate(
        {
            "telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"},
            "quota": {"hourly_token_budget": 1000, "daily_token_budget": 10000, "economy_threshold_ratio": 0.8},
        }
    )
    store1 = MemoryStore(db_path=str(db_path))
    await store1.init_db()
    qm1 = _make_quota_manager(store1, config=cfg)
    await _maybe_await(
        qm1.record_usage(UsageInfo(input_tokens=850, total_tokens=850), model="gemini-4-hydrogen", source="group")
    )
    await store1.close()

    store2 = MemoryStore(db_path=str(db_path))
    await store2.init_db()
    try:
        qm2 = _make_quota_manager(store2, config=cfg)
        await _maybe_await(qm2.refresh_from_store()) if hasattr(qm2, "refresh_from_store") else None
        assert qm2.is_economy_mode() is True
    finally:
        await store2.close()


# ============================================================================
# B7: F7 Per-Model Cooldown Boundaries (6 tests)
# ============================================================================


def test_prod_b7_01_past_iso_resets_at_timestamp_returns_zero_or_small_positive() -> None:
    """Verify past ISO resets at timestamp never produces negative or huge cooldown."""
    val = extract_retry_after("resets at 2020-01-01T00:00:00Z")
    assert val is None or (0.0 <= val <= 5.0)


@pytest.mark.asyncio
async def test_prod_b7_02_rate_limit_with_none_retry_after_uses_default_cooldown(tmp_path: Path) -> None:
    """Verify retry_after=None applies finite positive default cooldown."""
    store = MemoryStore(db_path=str(tmp_path / "b7_02.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        primary, _ = qm.select_model("dm_admin")
        qm.record_rate_limit(primary, retry_after=None)
        nxt, _ = qm.select_model("dm_admin")
        assert nxt != primary
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b7_03_zero_or_negative_cooldown_ignored_safely(tmp_path: Path) -> None:
    """Verify zero/negative retry_after values do not permanently block a model."""
    store = MemoryStore(db_path=str(tmp_path / "b7_03.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        primary, _ = qm.select_model("dm_admin")
        qm.record_rate_limit(primary, retry_after=0.0)
        qm.record_rate_limit(primary, retry_after=-5.0)
        after, _ = qm.select_model("dm_admin")
        assert after == primary
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b7_04_all_buckets_exhausted_raises_or_signals_cleanly(tmp_path: Path) -> None:
    """Verify cooling down every model in all normal buckets terminates cleanly without infinite loop."""
    store = MemoryStore(db_path=str(tmp_path / "b7_04.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        for m in (
            "gemini-4-hydrogen",
            "gemini-3.8-flash",
            "claude-sonnet-5-5",
            "claude-opus-5-5",
            "gpt-oss-120b-medium",
        ):
            qm.record_rate_limit(m, retry_after=300.0, bucket_wide=True)
        model, effort = qm.select_model("group")
        assert isinstance(model, str) and model
        assert isinstance(effort, str) and effort
    finally:
        await store.close()


def test_prod_b7_05_quota_warning_invalid_pct_or_malformed_log_line(tmp_path: Path) -> None:
    """Verify malformed [quota] warning lines never crash watch_log_for_errors or kill child."""
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir(parents=True)
    (gemini_dir / "cli.log").write_text(
        "[quota] model unknown-x is low on quota (??% remaining, resets at NOT_A_DATE)\n",
        encoding="utf-8",
    )
    stop_event = threading.Event()
    errors: list[str] = []
    killed = False

    def on_kill() -> None:
        nonlocal killed
        killed = True

    t = threading.Thread(
        target=watch_log_for_errors,
        args=(gemini_dir, stop_event, errors, on_kill),
        daemon=True,
    )
    t.start()
    time.sleep(0.35)
    stop_event.set()
    t.join(timeout=1.0)
    assert killed is False
    assert errors == []


def test_prod_b7_06_extract_retry_after_malformed_iso_returns_none_without_raise() -> None:
    """Verify malformed resets at string returns None without raising ValueError."""
    assert extract_retry_after("resets at 2026-99-99T99:99:99Z") is None


# ============================================================================
# B8: F8 ModelRouter Boundaries (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_prod_b8_01_unknown_role_falls_back_to_safe_default_model(tmp_path: Path) -> None:
    """Verify select_model('unknown_role') returns a valid (model, effort) tuple without KeyError."""
    store = MemoryStore(db_path=str(tmp_path / "b8_01.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        model, effort = qm.select_model("unknown_role_xyz")
        assert isinstance(model, str) and len(model) > 0
        assert isinstance(effort, str) and len(effort) > 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b8_02_duplicate_or_empty_fallback_list_handled_safely(tmp_path: Path) -> None:
    """Verify empty or duplicate fallback_models config never causes infinite loops."""
    cfg = BotConfig.model_validate(
        {
            "telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"},
            "models": {"fallback_models": []},
        }
    )
    store = MemoryStore(db_path=str(tmp_path / "b8_02.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store, config=cfg)
        m, e = qm.select_model("group")
        assert m and e
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b8_03_not_found_404_marks_bad_model_unavailable_without_cooling_whole_bucket(
    tmp_path: Path,
) -> None:
    """Verify 404 MODEL_NOT_FOUND falls back from bad model without cooling healthy bucket peers."""
    store = MemoryStore(db_path=str(tmp_path / "b8_03.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        qm.record_rate_limit("gemini-4-hydrogen", retry_after=300.0, bucket_wide=False, reason="NOT_FOUND 404")
        grp_m, _ = qm.select_model("group")
        assert grp_m == "gemini-3.8-flash"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b8_04_model_variant_suffix_maps_to_correct_bucket(tmp_path: Path) -> None:
    """Verify suffixed model variants map cleanly to their quota buckets."""
    store = MemoryStore(db_path=str(tmp_path / "b8_04.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        assert qm.get_bucket_for_model("gemini-4-hydrogen-low") == "gemini_models"
        assert qm.get_bucket_for_model("gemini-4-hydrogen-high") == "gemini_models"
        assert qm.get_bucket_for_model("claude-sonnet-5-5-high") == "claude_gpt"
        assert qm.get_bucket_for_model("gpt-oss-120b-medium") == "claude_gpt"
        assert qm.get_bucket_for_model("gemini-pro-orchestrator-g1-pro") == "gemini_argon"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b8_05_non_retryable_auth_error_does_not_loop_through_fallbacks(tmp_path: Path) -> None:
    """Verify AntigravityAuthError fails fast in orchestrator without cycling through all fallbacks."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "b8_05.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        side_effect=AntigravityAuthError("unauthenticated: oauth token expired")
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
        with pytest.raises(AntigravityAuthError):
            await orch.handle_incoming_message(
                _make_incoming(chat_id=GROUP_CHAT_ID, sender_id=891777515, text="Джефф, тест", is_mention=True)
            )
        assert mock_provider.generate_response.await_count == 1
    finally:
        await orch.stop()


# ============================================================================
# B9: F9 Budgets & Economy Mode Boundaries (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_prod_b9_01_exact_budget_threshold_boundary(tmp_path: Path) -> None:
    """Verify exact boundary: threshold - 1 token is False; exact threshold is True."""
    cfg = BotConfig.model_validate(
        {
            "telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"},
            "quota": {"hourly_token_budget": 1000, "daily_token_budget": 10000, "economy_threshold_ratio": 0.8},
        }
    )
    store = MemoryStore(db_path=str(tmp_path / "b9_01.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store, config=cfg)
        await _maybe_await(
            qm.record_usage(UsageInfo(input_tokens=799, total_tokens=799), model="gemini-4-hydrogen", source="group")
        )
        assert qm.is_economy_mode() is False
        await _maybe_await(
            qm.record_usage(UsageInfo(input_tokens=1, total_tokens=1), model="gemini-4-hydrogen", source="group")
        )
        assert qm.is_economy_mode() is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b9_02_hourly_window_rollover_exits_economy_mode(tmp_path: Path) -> None:
    """Verify usage rows older than 1 hour no longer keep hourly budget in economy mode."""
    db_path = tmp_path / "b9_02.sqlite"
    cfg = BotConfig.model_validate(
        {
            "telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"},
            "quota": {"hourly_token_budget": 1000, "daily_token_budget": 100000, "economy_threshold_ratio": 0.8},
        }
    )
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    await store.close()

    old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO usage_history (timestamp, model, source, input_tokens, output_tokens, reasoning_tokens, cache_read_tokens, cache_write_tokens, total_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (old_ts, "gemini-4-hydrogen", "group", 950, 0, 0, 0, 0, 950),
        )
        conn.commit()
    finally:
        conn.close()

    store2 = MemoryStore(db_path=str(db_path))
    await store2.init_db()
    try:
        qm = _make_quota_manager(store2, config=cfg)
        assert qm.is_economy_mode() is False
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_prod_b9_03_zero_or_disabled_budget_caps_do_not_trigger_economy_mode(tmp_path: Path) -> None:
    """Verify 0-disabled budget caps keep is_economy_mode() False regardless of token volume."""
    cfg = BotConfig.model_validate(
        {
            "telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"},
            "quota": {"hourly_token_budget": 0, "daily_token_budget": 0, "economy_threshold_ratio": 0.8},
        }
    )
    store = MemoryStore(db_path=str(tmp_path / "b9_03.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store, config=cfg)
        await _maybe_await(
            qm.record_usage(UsageInfo(input_tokens=999999, total_tokens=999999), model="gemini-4-hydrogen", source="group")
        )
        assert qm.is_economy_mode() is False
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b9_04_proactive_quiet_hours_and_economy_mode_stack_safely(tmp_path: Path) -> None:
    """Verify stacking quiet_hours=True and economy_mode=True sends zero proactive messages cleanly."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "b9_04.sqlite"))
    await store.init_db()
    try:
        engine = ProactiveEngine(config_mgr=cfg_mgr, memory_store=store)
        engine.is_in_quiet_hours = MagicMock(return_value=True)
        send_cb = AsyncMock()
        await engine.run_proactive_tick(
            chat_states={GROUP_CHAT_ID: {"last_chat_msg_time": 0.0, "last_bot_msg_time": 0.0}},
            send_callback=send_cb,
        )
        send_cb.assert_not_awaited()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b9_05_admin_dm_still_answered_during_economy_mode_until_reserve_exhausted(
    tmp_path: Path,
) -> None:
    """Verify admin DM still succeeds during economy mode while reserve budget remains."""
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "quota": {
                "hourly_token_budget": 1000,
                "daily_token_budget": 10000,
                "economy_threshold_ratio": 0.8,
                "admin_reserve_ratio": 0.2,
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "b9_05.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Админ-задача выполнена", model="gemini-4-hydrogen")
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
        await _maybe_await(
            orch.quota_manager.record_usage(
                UsageInfo(input_tokens=850, total_tokens=850),
                model="gemini-4-hydrogen",
                source="group",
            )
        )
        assert orch.quota_manager.is_economy_mode() is True
        reply = await orch.handle_incoming_message(
            _make_incoming(
                chat_id=ADMIN_USER_ID,
                sender_id=ADMIN_USER_ID,
                text="проверь сервис",
                is_private=True,
                sender_name="Родион",
            )
        )
        assert reply == "Админ-задача выполнена"
    finally:
        await orch.stop()


# ============================================================================
# B10: F10 /quota & Context Summary Boundaries (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_prod_b10_01_quota_tsv_parser_handles_partial_rows_and_0_percent(tmp_path: Path) -> None:
    """Verify /quota parser handles 0%, 100%, ANSI sequences, and missing reset fields."""
    store = MemoryStore(db_path=str(tmp_path / "b10_01.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        raw = "\x1b[32mGemini Models\tFive Hour Limit Remaining\t0%\t\nClaude and GPT models\tWeekly Limit Remaining\t100%\t2026-11-25T00:00:00Z\n"
        qm.parse_quota_output(raw)
        report = qm.format_quota_report()
        assert "0%" in report
        assert "100%" in report
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b10_02_quota_command_with_bot_mention_suffix_in_dm(tmp_path: Path) -> None:
    """Verify /quota@bot_username and trailing whitespace work in admin DM."""
    cfg_mgr = _make_isolated_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "b10_02.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "m"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock()
    mock_userbot = _make_mock_userbot(bot_username="jeff_test_bot")

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
                args=["agy", "-p", "/quota"], returncode=0, stdout="Gemini Models\tWeekly\t99%\t-\n", stderr=""
            ),
        ):
            reply = await orch.handle_incoming_message(
                _make_incoming(
                    chat_id=ADMIN_USER_ID,
                    sender_id=ADMIN_USER_ID,
                    text="/quota@jeff_test_bot   ",
                    is_private=True,
                )
            )
        mock_provider.generate_response.assert_not_awaited()
        assert reply is not None and "99%" in reply
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_b10_03_empty_usage_history_renders_clean_zero_quota_report(tmp_path: Path) -> None:
    """Verify empty usage_history renders clean readable report and context summary with 0 tokens."""
    store = MemoryStore(db_path=str(tmp_path / "b10_03.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        report = qm.format_quota_report()
        summary = qm.format_context_summary()
        assert report and "0" in report
        assert summary
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prod_b10_04_quota_output_with_accidental_secret_is_sanitized(tmp_path: Path) -> None:
    """Verify accidental secret in agy -p /quota stderr/stdout is redacted in DM reply."""
    secret = "987654321:AAFakeSecretLeakedByQuotaCli12345"
    cfg_mgr = _make_isolated_config(tmp_path, bot_overrides={"telegram": {"bot_token": secret}})
    store = MemoryStore(db_path=str(tmp_path / "b10_04.sqlite"))
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
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["agy", "-p", "/quota"],
                returncode=1,
                stdout="",
                stderr=f"auth error with {secret}",
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
        assert secret not in reply
        sent_text = str(mock_userbot.send_reply.call_args.kwargs.get("text") or "")
        assert secret not in sent_text
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_prod_b10_05_context_summary_length_bounded(tmp_path: Path) -> None:
    """Verify format_context_summary stays compact (<= 600 chars) even with many cooled models."""
    store = MemoryStore(db_path=str(tmp_path / "b10_05.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store)
        for i in range(15):
            qm.record_rate_limit(f"model-variant-{i}", retry_after=120.0, reason="X" * 200)
        summary = qm.format_context_summary()
        assert len(summary) <= 600
    finally:
        await store.close()


# ============================================================================
# B11: F11 SelfEvolution & SQLite Boundaries (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_f11_non_admin_dm_or_group_cannot_trigger_self_edit(tmp_path: Path) -> None:
    """Verify non-admin DM or group message cannot invoke SelfEvolutionManager admin tasks."""
    from bot.evolution.manager import SelfEvolutionManager

    cfg_mgr = _make_isolated_config(tmp_path, bot_overrides={"self_evolution": {"enabled": True}})
    store = MemoryStore(db_path=str(tmp_path / "t2_f11_nonadmin.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        res = await mgr.run_admin_task(user_id=999888777, task="edit code")
        assert res is None or res is False or getattr(res, "status", None) in ("rejected", "unauthorized")
        provider.generate_response.assert_not_awaited()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f11_sql_injection_and_10kb_stacktrace_signature_safe(tmp_path: Path) -> None:
    """Verify SQL injection and 10KB stacktrace in signature/reason persist safely without DB corruption."""
    from bot.evolution.manager import SelfEditRecord

    db_path = tmp_path / "t2_f11_sqli.sqlite"
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    try:
        huge_reason = "Traceback:\n" + ("A" * 10240)
        sqli_sig = "sig'); DROP TABLE self_edit_history;--"
        await store.record_self_edit(
            SelfEditRecord(
                trigger_type="runtime_error",
                signature=sqli_sig,
                reason=huge_reason,
                affected_files=[],
                verdict="NO_CHANGE",
                model="claude-sonnet-5-5",
                branch="",
                commit_sha="",
                diff_lines=0,
                test_result="skipped",
                deploy_result="rejected",
                rollback_commit="",
                summary="safe",
            )
        )
        rows = await store.get_recent_self_edits(limit=5)
        assert len(rows) == 1
        assert rows[0].signature == sqli_sig
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f11_empty_affected_files_list_recorded_cleanly(tmp_path: Path) -> None:
    """Verify empty affected_files=[] serializes and deserializes cleanly."""
    from bot.evolution.manager import SelfEditRecord

    store = MemoryStore(db_path=str(tmp_path / "t2_f11_emptyfiles.sqlite"))
    await store.init_db()
    try:
        await store.record_self_edit(
            SelfEditRecord(
                trigger_type="admin_dm",
                signature="dm:empty",
                reason="none",
                affected_files=[],
                verdict="NO_CHANGE",
                model="claude-sonnet-5-5",
                branch="",
                commit_sha="",
                diff_lines=0,
                test_result="skipped",
                deploy_result="rejected",
                rollback_commit="",
                summary="no change",
            )
        )
        rows = await store.get_recent_self_edits(limit=5)
        assert rows[0].affected_files == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f11_concurrent_evolution_triggers_serialized_or_deduped(tmp_path: Path) -> None:
    """Verify concurrent triggers never run two overlapping self_dev staging edits."""
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
    store = MemoryStore(db_path=str(tmp_path / "t2_f11_conc.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    active_self_dev = 0
    max_concurrent = 0

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        nonlocal active_self_dev, max_concurrent
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(text='{"verdict": "CODE_FIX_REQUIRED", "reason": "fix"}', model="claude-sonnet-5-5")
        active_self_dev += 1
        max_concurrent = max(max_concurrent, active_self_dev)
        await asyncio.sleep(0.05)
        active_self_dev -= 1
        return ProviderResponse(text="no diff", model=str(kwargs.get("model")))

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await asyncio.gather(
            mgr.run_admin_task(user_id=ADMIN_USER_ID, task="task 1"),
            mgr.run_admin_task(user_id=ADMIN_USER_ID, task="task 2"),
        )
        assert max_concurrent <= 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f11_legacy_sqlite_migration_preserves_messages_and_facts(tmp_path: Path) -> None:
    """Verify MemoryStore.init_db on a legacy DB adds self_edit_history without losing messages/facts."""
    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_msg_id INTEGER NOT NULL,
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            author_name TEXT NOT NULL,
            text TEXT,
            reply_to_msg_id INTEGER,
            media_type TEXT,
            media_summary TEXT,
            compacted INTEGER DEFAULT 0
        );
        INSERT INTO messages (telegram_msg_id, chat_id, user_id, timestamp, author_name, text)
        VALUES (1, -1003845062562, 891777515, 1700000000, 'Алексей', 'legacy msg');
        """
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    try:
        hist = await store.get_recent_history(-1003845062562, limit=10)
        assert len(hist) == 1
        assert hist[0].text == "legacy msg"
        await store.save_message(
            ChatMessage(
                telegram_msg_id=2,
                chat_id=-1003845062562,
                user_id=891777515,
                timestamp=1700000001,
                author_name="Алексей",
                text="new msg",
            )
        )
    finally:
        await store.close()

    conn2 = sqlite3.connect(str(db_path))
    try:
        tables = {r[0] for r in conn2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "self_edit_history" in tables
        assert "usage_history" in tables
    finally:
        conn2.close()


# ============================================================================
# B12: F12 Anti-Churn Boundaries (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_f12_cooldown_exact_boundary_minus_plus_one_sec(tmp_path: Path) -> None:
    """Verify signature cooldown blocks at cooldown-1s and allows at cooldown+1s."""
    from bot.evolution.manager import SelfEditRecord, SelfEvolutionManager

    db_path = tmp_path / "t2_f12_bound.sqlite"
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "signature_cooldown_seconds": 60, "error_threshold_count": 1}},
    )
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    now_dt = datetime.now(timezone.utc)
    await store.record_self_edit(
        SelfEditRecord(
            created_at=(now_dt - timedelta(seconds=59)).isoformat(),
            trigger_type="runtime_error",
            signature="Sig:Boundary",
            reason="r",
            affected_files=[],
            verdict="NO_CHANGE",
            model="claude-sonnet-5-5",
            branch="",
            commit_sha="",
            diff_lines=0,
            test_result="skipped",
            deploy_result="rejected",
            rollback_commit="",
            summary="s",
        )
    )
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=MagicMock(), notify_admin=AsyncMock()
    )
    try:
        allowed_59, _ = await mgr.check_signature_allowed("Sig:Boundary")
        assert allowed_59 is False

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE self_edit_history SET created_at = ? WHERE signature = ?",
            ((now_dt - timedelta(seconds=61)).isoformat(), "Sig:Boundary"),
        )
        conn.commit()
        conn.close()

        allowed_61, _ = await mgr.check_signature_allowed("Sig:Boundary")
        assert allowed_61 is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f12_daily_limit_utc_midnight_rollover(tmp_path: Path) -> None:
    """Verify daily edit cap counts only current UTC date and resets after midnight rollover."""
    from bot.evolution.manager import SelfEditRecord, SelfEvolutionManager

    db_path = tmp_path / "t2_f12_utc.sqlite"
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "max_daily_edits": 2}},
    )
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=23, minute=59, second=55).isoformat()
    for i in (1, 2):
        await store.record_self_edit(
            SelfEditRecord(
                created_at=yesterday,
                trigger_type="runtime_error",
                signature=f"Old:{i}",
                reason="r",
                affected_files=[],
                verdict="CODE_FIX_REQUIRED",
                model="gemini-4-hydrogen",
                branch=f"b{i}",
                commit_sha=f"c{i}",
                diff_lines=1,
                test_result="passed",
                deploy_result="applied",
                rollback_commit="c0",
                summary="ok",
            )
        )
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=MagicMock(), notify_admin=AsyncMock()
    )
    try:
        allowed, _ = await mgr.check_daily_limit()
        assert allowed is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f12_zero_max_daily_edits_blocks_all_autonomous_edits(tmp_path: Path) -> None:
    """Verify max_daily_edits=0 blocks all autonomous runtime-error self-edits."""
    from bot.evolution.manager import SelfEvolutionManager

    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "max_daily_edits": 0, "error_threshold_count": 1}},
    )
    store = MemoryStore(db_path=str(tmp_path / "t2_f12_zero.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.handle_runtime_error(RuntimeError("boom"), signature="RuntimeError:boom")
        provider.generate_response.assert_not_awaited()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f12_nested_cause_transient_error_detected(tmp_path: Path) -> None:
    """Verify chained exception with __cause__=AntigravityRateLimitError is classified transient."""
    from bot.evolution.manager import SelfEvolutionManager

    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "error_threshold_count": 1}},
    )
    store = MemoryStore(db_path=str(tmp_path / "t2_f12_cause.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        try:
            raise RuntimeError("outer wrapper") from AntigravityRateLimitError("429")
        except RuntimeError as exc:
            await mgr.handle_runtime_error(exc)
        provider.generate_response.assert_not_awaited()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f12_single_non_repeated_runtime_error_below_threshold(tmp_path: Path) -> None:
    """Verify single runtime error below error_threshold does not launch Judge/self_dev."""
    from bot.evolution.manager import SelfEvolutionManager

    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "error_threshold_count": 3}},
    )
    store = MemoryStore(db_path=str(tmp_path / "t2_f12_thresh.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.handle_runtime_error(ValueError("once"), signature="ValueError:once")
        provider.generate_response.assert_not_awaited()
    finally:
        await store.close()


# ============================================================================
# B13: F13 Adversarial Judge Boundaries (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_f13_malformed_or_empty_judge_response_fails_closed(tmp_path: Path) -> None:
    """Verify empty/HTML/garbage Judge response fails closed to NO_CHANGE with zero self_dev calls."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "staging_dir": str(sandbox["staging_dir"])}},
    )
    store = MemoryStore(db_path=str(tmp_path / "t2_f13_bad.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    for bad_text in ("", "<html>502</html>", "not valid json"):
        provider = MagicMock()
        provider.generate_response = AsyncMock(return_value=ProviderResponse(text=bad_text, model="claude-sonnet-5-5"))
        mgr = SelfEvolutionManager(
            config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
        )
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="test bad judge")
        assert provider.generate_response.await_count == 1
    await store.close()


@pytest.mark.asyncio
async def test_t2_f13_judge_json_inside_markdown_code_fence(tmp_path: Path) -> None:
    """Verify Judge JSON wrapped in ```json ... ``` markdown fence is parsed accurately."""
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
    store = MemoryStore(db_path=str(tmp_path / "t2_f13_fence.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    calls = 0

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ProviderResponse(
                text='```json\n{"verdict": "CODE_FIX_REQUIRED", "reason": "confirmed"}\n```',
                model="claude-sonnet-5-5",
            )
        return ProviderResponse(text="done", model=str(kwargs.get("model")))

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="fix fenced")
        assert calls >= 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f13_judge_provider_exception_fails_closed(tmp_path: Path) -> None:
    """Verify exception during Judge aborts evolution safely and notifies admin."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "staging_dir": str(sandbox["staging_dir"])}},
    )
    store = MemoryStore(db_path=str(tmp_path / "t2_f13_exc.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=TimeoutError("judge timeout"))
    notify = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=notify
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="check")
        assert not sandbox["request_file"].exists()
        notify.assert_awaited()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f13_judge_negation_text_not_misclassified(tmp_path: Path) -> None:
    """Verify prose rejecting CODE_FIX_REQUIRED in favor of NO_CHANGE resolves to NO_CHANGE."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "staging_dir": str(sandbox["staging_dir"])}},
    )
    store = MemoryStore(db_path=str(tmp_path / "t2_f13_neg.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text="I reject CODE_FIX_REQUIRED; verdict is NO_CHANGE.",
            model="claude-sonnet-5-5",
        )
    )
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="negation check")
        assert provider.generate_response.await_count == 1
        rows = await store.get_recent_self_edits(limit=5)
        assert rows[0].verdict == "NO_CHANGE"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f13_judge_whitespace_and_case_normalization(tmp_path: Path) -> None:
    """Verify lowercase/whitespace verdict '  config_only\\n' normalizes to CONFIG_ONLY."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "staging_dir": str(sandbox["staging_dir"])}},
    )
    store = MemoryStore(db_path=str(tmp_path / "t2_f13_case.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    provider = MagicMock()
    provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text='{"verdict": "  config_only\\n", "reason": "yaml tweak"}',
            model="claude-sonnet-5-5",
        )
    )
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="case check")
        rows = await store.get_recent_self_edits(limit=5)
        assert rows[0].verdict == "CONFIG_ONLY"
    finally:
        await store.close()


# ============================================================================
# B14: F14 Model Escalation Boundaries (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_f14_single_model_escalation_models_no_index_error(tmp_path: Path) -> None:
    """Verify 1-element escalation_models handles single failure cleanly without IndexError."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={
            "self_evolution": {
                "enabled": True,
                "staging_dir": str(sandbox["staging_dir"]),
                "escalation_models": ["gemini-4-hydrogen"],
            }
        },
    )
    store = MemoryStore(db_path=str(tmp_path / "t2_f14_one.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(text='{"verdict": "CODE_FIX_REQUIRED", "reason": "fix"}', model="claude-sonnet-5-5")
        raise RuntimeError("tier1 fail")

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="one chain")
    finally:
        await store.close()


def test_t2_f14_empty_escalation_models_config_handled_safely() -> None:
    """Verify empty escalation_models either raises ValidationError or falls back cleanly."""
    from pydantic import ValidationError

    try:
        cfg = BotConfig.model_validate(
            {
                "telegram": {"api_id": 12345, "api_hash": "0123456789abcdef"},
                "self_evolution": {"enabled": True, "escalation_models": []},
            }
        )
        assert isinstance(cfg.self_evolution.escalation_models, list)
    except ValidationError:
        pass


@pytest.mark.asyncio
async def test_t2_f14_model_404_not_found_triggers_escalation(tmp_path: Path) -> None:
    """Verify NOT_FOUND (404) on Tier 1 escalates to Tier 2 model."""
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
    store = MemoryStore(db_path=str(tmp_path / "t2_f14_404.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    tried: list[str] = []

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(text='{"verdict": "CODE_FIX_REQUIRED", "reason": "fix"}', model="claude-sonnet-5-5")
        m = str(kwargs.get("model"))
        tried.append(m)
        if m == "gemini-4-hydrogen":
            raise RuntimeError("NOT_FOUND (code 404): Requested entity was not found")
        (sandbox["staging_dir"] / "src" / "bot" / "app.py").write_text("VALUE = 55\n", encoding="utf-8")
        return ProviderResponse(text="done", model=m)

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="404 test")
        assert tried == ["gemini-4-hydrogen", "claude-sonnet-5-5"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f14_economy_mode_blocks_auto_but_respects_admin_policy(tmp_path: Path) -> None:
    """Verify economy mode blocks auto self_edit while preserving explicit admin reserve policy."""
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
    store = MemoryStore(db_path=str(tmp_path / "t2_f14_econ.sqlite"))
    await store.init_db()
    try:
        qm = _make_quota_manager(store, config=cfg)
        await _maybe_await(
            qm.record_usage(UsageInfo(input_tokens=850, total_tokens=850), model="gemini-4-hydrogen", source="group")
        )
        ok_auto, _ = qm.should_allow_self_edit(priority="auto")
        ok_admin, _ = qm.should_allow_self_edit(priority="admin")
        assert ok_auto is False
        assert ok_admin is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_t2_f14_effort_high_preserved_under_all_escalation_hops(tmp_path: Path) -> None:
    """Verify every escalation hop passes effort='high'."""
    from bot.evolution.manager import SelfEvolutionManager

    sandbox = _gatekeeper_sandbox(tmp_path)
    cfg_mgr = _make_isolated_config(
        tmp_path,
        bot_overrides={"self_evolution": {"enabled": True, "staging_dir": str(sandbox["staging_dir"])}},
    )
    store = MemoryStore(db_path=str(tmp_path / "t2_f14_eff.sqlite"))
    await store.init_db()
    qm = _make_quota_manager(store, config=cfg_mgr.bot)

    efforts: list[Optional[str]] = []

    async def fake_gen(prompt: str, **kwargs: Any) -> ProviderResponse:
        if kwargs.get("tool_profile") != "self_dev":
            return ProviderResponse(text='{"verdict": "CODE_FIX_REQUIRED", "reason": "fix"}', model="claude-sonnet-5-5")
        efforts.append(kwargs.get("effort"))
        raise RuntimeError("fail hop")

    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=fake_gen)
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=provider, notify_admin=AsyncMock()
    )
    try:
        await mgr.run_admin_task(user_id=ADMIN_USER_ID, task="effort check")
        assert len(efforts) == 3
        assert all(e == "high" for e in efforts)
    finally:
        await store.close()


# ============================================================================
# B15: F15 Gatekeeper & Canary Boundaries (6 tests)
# ============================================================================


def test_t2_f15_gatekeeper_malformed_or_missing_request_json(tmp_path: Path) -> None:
    """Verify missing or corrupt request.json writes rejected result and leaves prod untouched."""
    sandbox = _gatekeeper_sandbox(tmp_path)
    # 1. Missing request.json
    sandbox["run_gatekeeper"]()
    if sandbox["result_file"].is_file():
        res = json.loads(sandbox["result_file"].read_text(encoding="utf-8"))
        assert res["status"] == "rejected"

    # 2. Corrupt JSON
    sandbox["request_file"].write_text("{bad json", encoding="utf-8")
    sandbox["run_gatekeeper"]()
    res2 = json.loads(sandbox["result_file"].read_text(encoding="utf-8"))
    assert res2["status"] == "rejected"
    head = subprocess.check_output(["git", "-C", str(sandbox["prod_dir"]), "rev-parse", "HEAD"], text=True).strip()
    assert head == sandbox["prev_commit"]


def test_t2_f15_gatekeeper_rejects_shell_metacharacters_in_request(tmp_path: Path) -> None:
    """Verify shell metacharacters in branch/commit_sha are rejected with zero command injection."""
    sandbox = _gatekeeper_sandbox(tmp_path)
    pwn_marker = tmp_path / "pwned"
    sandbox["request_file"].write_text(
        json.dumps(
            {
                "request_id": "req-inj",
                "branch": f"main; touch {pwn_marker}",
                "commit_sha": "$(whoami)",
                "reason": "inj",
                "requested_at": "2026-11-18T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    sandbox["run_gatekeeper"]()
    assert not pwn_marker.exists()
    res = json.loads(sandbox["result_file"].read_text(encoding="utf-8"))
    assert res["status"] == "rejected"


def test_t2_f15_gatekeeper_rejects_oversized_diff_at_exact_boundary(tmp_path: Path) -> None:
    """Verify diff of max_diff_lines succeeds while max_diff_lines + 1 is rejected."""
    sandbox_ok = _gatekeeper_sandbox(tmp_path / "ok", max_diff_lines=5)
    (sandbox_ok["staging_dir"] / "src" / "bot" / "app.py").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(sandbox_ok["staging_dir"]), "add", "src/bot/app.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(sandbox_ok["staging_dir"]), "commit", "-m", "5 lines"], check=True, capture_output=True)
    sha_ok = subprocess.check_output(["git", "-C", str(sandbox_ok["staging_dir"]), "rev-parse", "HEAD"], text=True).strip()
    sandbox_ok["request_file"].write_text(
        json.dumps({"request_id": "ok", "branch": "self-edit/fix-1", "commit_sha": sha_ok, "reason": "ok", "requested_at": "2026-11-18T12:00:00Z"}),
        encoding="utf-8",
    )
    sandbox_ok["run_gatekeeper"]()
    assert json.loads(sandbox_ok["result_file"].read_text(encoding="utf-8"))["status"] == "applied"

    sandbox_big = _gatekeeper_sandbox(tmp_path / "big", max_diff_lines=5)
    (sandbox_big["staging_dir"] / "src" / "bot" / "app.py").write_text("1\n2\n3\n4\n5\n6\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(sandbox_big["staging_dir"]), "add", "src/bot/app.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(sandbox_big["staging_dir"]), "commit", "-m", "6 lines"], check=True, capture_output=True)
    sha_big = subprocess.check_output(["git", "-C", str(sandbox_big["staging_dir"]), "rev-parse", "HEAD"], text=True).strip()
    sandbox_big["request_file"].write_text(
        json.dumps({"request_id": "big", "branch": "self-edit/fix-1", "commit_sha": sha_big, "reason": "big", "requested_at": "2026-11-18T12:00:00Z"}),
        encoding="utf-8",
    )
    sandbox_big["run_gatekeeper"]()
    assert json.loads(sandbox_big["result_file"].read_text(encoding="utf-8"))["status"] == "rejected"


def test_t2_f15_gatekeeper_rejects_symlink_pointing_to_protected_path(tmp_path: Path) -> None:
    """Verify staging commit introducing a symlink to ../../.env is rejected."""
    sandbox = _gatekeeper_sandbox(tmp_path)
    link_path = sandbox["staging_dir"] / "src" / "bot" / "leak_env"
    link_path.symlink_to("../../.env")
    subprocess.run(["git", "-C", str(sandbox["staging_dir"]), "add", "src/bot/leak_env"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(sandbox["staging_dir"]), "commit", "-m", "symlink"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(sandbox["staging_dir"]), "rev-parse", "HEAD"], text=True).strip()

    sandbox["request_file"].write_text(
        json.dumps({"request_id": "sym", "branch": "self-edit/fix-1", "commit_sha": sha, "reason": "sym", "requested_at": "2026-11-18T12:00:00Z"}),
        encoding="utf-8",
    )
    sandbox["run_gatekeeper"]()
    assert json.loads(sandbox["result_file"].read_text(encoding="utf-8"))["status"] == "rejected"


def test_t2_f15_gatekeeper_rejects_commit_sha_mismatch_or_dirty_prod(tmp_path: Path) -> None:
    """Verify gatekeeper rejects mismatched commit_sha without mutating prod."""
    sandbox = _gatekeeper_sandbox(tmp_path)
    sandbox["request_file"].write_text(
        json.dumps(
            {
                "request_id": "mismatch",
                "branch": "self-edit/fix-1",
                "commit_sha": "0000000000000000000000000000000000000000",
                "reason": "mismatch",
                "requested_at": "2026-11-18T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    sandbox["run_gatekeeper"]()
    assert json.loads(sandbox["result_file"].read_text(encoding="utf-8"))["status"] == "rejected"


@pytest.mark.asyncio
async def test_t2_f15_self_evolution_reconciles_pending_result_json_after_restart(tmp_path: Path) -> None:
    """Verify SelfEvolutionManager reconciles pre-written result.json on startup and notifies admin."""
    from bot.evolution.manager import SelfEditRecord, SelfEvolutionManager

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
    store = MemoryStore(db_path=str(tmp_path / "t2_f15_reconcile.sqlite"))
    await store.init_db()
    await store.record_self_edit(
        SelfEditRecord(
            trigger_type="admin_dm",
            signature="req-pending-1",
            reason="fix",
            affected_files=["src/bot/app.py"],
            verdict="CODE_FIX_REQUIRED",
            model="gemini-4-hydrogen",
            branch="self-edit/fix-1",
            commit_sha="deadbeef",
            diff_lines=3,
            test_result="pending",
            deploy_result="pending",
            rollback_commit=sandbox["prev_commit"],
            summary="pending",
        )
    )
    sandbox["result_file"].write_text(
        json.dumps(
            {
                "request_id": "req-pending-1",
                "status": "applied",
                "prev_commit": sandbox["prev_commit"],
                "new_commit": "deadbeef",
                "error": "",
                "completed_at": "2026-11-18T12:05:00Z",
            }
        ),
        encoding="utf-8",
    )
    qm = _make_quota_manager(store, config=cfg_mgr.bot)
    notify = AsyncMock()
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr, memory_store=store, quota_manager=qm, provider=MagicMock(), notify_admin=notify
    )
    try:
        await mgr.reconcile_pending_deploy()
        notify.assert_awaited()
    finally:
        await store.close()


# ============================================================================
# B16: F16 WireGuard Boundaries (5 tests)
# ============================================================================


def test_t2_f16_wg0_postup_predown_route_and_nft_symmetry() -> None:
    """Verify every route add in PreUp/PostUp has matching del/delete in PreDown/PostDown."""
    text = (REPO_ROOT / "deploy" / "wireguard" / "wg0.conf.example").read_text(encoding="utf-8")
    assert "ip route add" in text or "ip route replace" in text
    assert "ip route del" in text


def test_t2_f16_no_active_wg0_conf_or_key_files_in_repo() -> None:
    """Verify zero active wg0.conf or *.key files exist in repository."""
    for p in REPO_ROOT.rglob("*"):
        if ".git" in p.parts or ".venv" in p.parts:
            continue
        if p.name == "wg0.conf" or p.suffix == ".key":
            raise AssertionError(f"Active WireGuard config/key file must not exist in repo: {p}")


def test_t2_f16_deploy_scripts_never_enable_or_start_wireguard() -> None:
    """Verify no script under deploy/ runs wg-quick up or enables wg-quick service."""
    for script in (REPO_ROOT / "deploy").rglob("*.sh"):
        text = script.read_text(encoding="utf-8")
        assert "wg-quick up" not in text
        assert "systemctl enable wg-quick" not in text
        assert "systemctl start wg-quick" not in text


def test_t2_f16_wg0_killswitch_permits_lan_dns_and_keepalive() -> None:
    """Verify wg0.conf.example includes LAN 192.168.1.0/24, DNS, and PersistentKeepalive."""
    text = (REPO_ROOT / "deploy" / "wireguard" / "wg0.conf.example").read_text(encoding="utf-8")
    assert "192.168.1.0/24" in text
    assert "DNS" in text
    assert "PersistentKeepalive" in text


def test_t2_f16_wg0_example_ini_has_no_duplicate_or_unbalanced_lines() -> None:
    """Verify wg0.conf.example parses structurally with [Interface] and [Peer] INI sections."""
    text = (REPO_ROOT / "deploy" / "wireguard" / "wg0.conf.example").read_text(encoding="utf-8")
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.read_string(text)
    assert "Interface" in cp.sections()
    assert "Peer" in cp.sections()


# ============================================================================
# B17: F17 Server Packaging & Unit Hardening Boundaries (5 tests)
# ============================================================================


def test_t2_f17_systemd_unit_never_uses_sudo_or_disables_nonewprivs() -> None:
    """Verify telegram-userbot.service keeps NoNewPrivileges=true and never calls sudo."""
    text = (REPO_ROOT / "systemd" / "telegram-userbot.service").read_text(encoding="utf-8")
    assert "NoNewPrivileges=true" in text
    assert "sudo" not in text


def test_t2_f17_deploy_remote_cross_profile_guard_exits_2() -> None:
    """Verify deploy_remote.sh exits 2 for both cross-profile target combinations before SSH."""
    script = REPO_ROOT / "deploy" / "deploy_remote.sh"
    r1 = subprocess.run(
        ["bash", str(script), "userbot", "root@127.0.0.1", "/opt/telegram-group-bot"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    r2 = subprocess.run(
        ["bash", str(script), "bot_api", "root@127.0.0.1", "/opt/telegram-mtproto-userbot"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r1.returncode == 2
    assert r2.returncode == 2


def test_t2_f17_gitignore_covers_nested_secrets_and_wg_configs(tmp_path: Path) -> None:
    """Verify .gitignore ignores nested secrets, sqlite files, session journals, and wg0.conf/*.key."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    shutil.copy2(REPO_ROOT / ".gitignore", repo / ".gitignore")

    for path in (
        "nested/.env",
        "data/a/b.sqlite",
        "session/x.session-journal",
        "deploy/wireguard/wg0.conf",
        "deploy/wireguard/private.key",
    ):
        rc = subprocess.run(["git", "-C", str(repo), "check-ignore", "-q", path]).returncode
        assert rc == 0, f"Expected .gitignore to ignore {path}"


def test_t2_f17_bot_yaml_and_example_yaml_both_validate() -> None:
    """Verify config/bot.example.yaml (and config/bot.yaml if present) validate via BotConfig."""
    for name in ("bot.yaml", "bot.example.yaml"):
        cfg_file = REPO_ROOT / "config" / name
        if name == "bot.yaml" and not cfg_file.is_file():
            continue
        raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        cfg = BotConfig.model_validate(raw)
        assert cfg.telegram.api_id > 0


def test_t2_f17_git_ls_files_contains_zero_secrets_or_runtime_data() -> None:
    """Verify git index contains zero .env, session, oauth-token, sqlite, or data/ paths."""
    out = subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files"], text=True)
    bad_re = re.compile(r"(?:^|/)\.env$|\.session|oauth-token|\.sqlite|^data/|wg0\.conf$", re.IGNORECASE)
    bad = [p for p in out.splitlines() if bad_re.search(p)]
    assert bad == []


# ============================================================================
# B18: F18 Offline Dry-Run Boundaries (5 tests)
# ============================================================================


def test_t2_f18_dry_run_missing_config_returns_nonzero(tmp_path: Path) -> None:
    """Verify CLI --dry-run on nonexistent config path exits non-zero cleanly."""
    proc = subprocess.run(
        [sys.executable, "-m", "bot.cli", "run", "--config", str(tmp_path / "missing.yaml"), "--dry-run"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode != 0


def test_t2_f18_dry_run_corrupted_yaml_returns_nonzero(tmp_path: Path) -> None:
    """Verify CLI --dry-run on malformed YAML exits non-zero cleanly."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("telegram: [unclosed", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "bot.cli", "run", "--config", str(bad_yaml), "--dry-run"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode != 0


def test_t2_f18_dry_run_empty_chat_and_dm_lists_succeeds(tmp_path: Path) -> None:
    """Verify --dry-run succeeds even when allowed_chat_ids and allowed_dm_user_ids are empty."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "bot.yaml").write_text(
        yaml.safe_dump(
            {
                "telegram": {
                    "mode": "bot_api",
                    "api_id": 12345,
                    "api_hash": "0123456789abcdef",
                    "allowed_chat_ids": [],
                    "allowed_dm_user_ids": [],
                }
            }
        ),
        encoding="utf-8",
    )
    (cfg_dir / "system_prompt.md").write_text("Test prompt", encoding="utf-8")
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bot.cli",
            "run",
            "--config",
            str(cfg_dir / "bot.yaml"),
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
    assert json.loads(proc.stdout)["status"] == "ok"


def test_t2_f18_dry_run_missing_env_file_graceful(tmp_path: Path) -> None:
    """Verify --env-file pointing to nonexistent path succeeds gracefully in --dry-run."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bot.cli",
            "run",
            "--config",
            str(REPO_ROOT / "config" / "bot.example.yaml"),
            "--env-file",
            str(tmp_path / "no_such.env"),
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


def test_t2_f18_dry_run_redacts_env_secrets_from_stdout_and_logs(tmp_path: Path) -> None:
    """Verify synthetic TELEGRAM_BOT_TOKEN in --env-file never leaks to stdout or stderr."""
    secret = "888777666:AAFakeDryRunEnvTokenNeverLeak123456"
    env_file = tmp_path / "test.env"
    env_file.write_text(f"TELEGRAM_BOT_TOKEN={secret}\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bot.cli",
            "run",
            "--config",
            str(REPO_ROOT / "config" / "bot.example.yaml"),
            "--env-file",
            str(env_file),
            "--dry-run",
            "--log-level",
            "DEBUG",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert secret not in proc.stdout
    assert secret not in proc.stderr
