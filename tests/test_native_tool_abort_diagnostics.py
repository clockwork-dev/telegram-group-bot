"""Focused unit tests for AntigravityNativeToolAbort diagnostics, secret safety, and noise-free logging."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from PIL import Image

from bot.antigravity.cli_client import AntigravityCLIClient
from bot.antigravity.errors import (
    AntigravityNativeToolAbort,
    ErrorCategory,
    classify_error,
)
from bot.antigravity.stream import AntigravityStreamParser
from bot.config.manager import ConfigManager
from bot.core.models import MediaAttachment
from bot.evolution.manager import SelfEvolutionManager
from bot.main import GroupBotOrchestrator
from bot.media.processor import MediaProcessor
from bot.memory.db import MemoryStore
from bot.quota.manager import QuotaBudgetManager
from bot.security.sanitizer import SecretSanitizer
from bot.telegram.handlers import IncomingMessage


def _make_config(tmp_path: Path, *, evo_enabled: bool = False) -> ConfigManager:
    bot_yaml = tmp_path / "bot.yaml"
    bot_yaml.write_text(
        yaml.dump(
            {
                "telegram": {
                    "mode": "bot_api",
                    "bot_token": "123456:AAFakeToken1234567890abcdefghij",
                    "api_id": 12345,
                    "api_hash": "0123456789abcdef0123456789abcdef",
                    "allowed_chat_ids": [-100111222333],
                    "allowed_dm_user_ids": [1001],
                    "admin_user_ids": [1001],
                },
                "behavior": {"bot_names": ["Jeff"], "typing_simulation": False},
                "media": {"temp_dir": str(tmp_path / "media_tmp")},
                "antigravity": {"ops_tools_default_mode": "off"},
                "self_evolution": {
                    "enabled": evo_enabled,
                    "staging_dir": str(tmp_path / "staging"),
                    "trigger_dir": str(tmp_path / "trigger"),
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "system_prompt.md").write_text("System prompt", encoding="utf-8")
    (tmp_path / "friends.yaml").write_text("friends: {}\n", encoding="utf-8")
    return ConfigManager(config_path=bot_yaml)


class _DummyUserbot:
    bot_id = 777000
    bot_username = "jeff_bot"
    client = None

    def __init__(self) -> None:
        self.sent_replies: list[dict[str, Any]] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def register_message_handler(self, handler: Any) -> None:
        self._handler = handler

    async def send_reply(self, **kwargs: Any) -> dict[str, Any]:
        self.sent_replies.append(kwargs)
        return {"id": len(self.sent_replies)}


class _ScriptedAcpPopen:
    def __init__(self, script_fn: Any) -> None:
        self._script_fn = script_fn
        self._responses: list[str] = []
        self._resp_idx = 0
        self.pid = 9001
        self.returncode = 0
        self._terminated = False
        outer = self

        class _Stdin:
            def write(self, data: str) -> int:
                for line in data.splitlines():
                    if line.strip():
                        outer._on_line(line)
                return len(data)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        class _Stdout:
            def readline(self) -> str:
                if outer._resp_idx < len(outer._responses):
                    res = outer._responses[outer._resp_idx]
                    outer._resp_idx += 1
                    return res
                return ""

        self.stdin = _Stdin()
        self.stdout = _Stdout()
        self.stderr = MagicMock()
        self.stderr.readline = lambda: ""

    def _on_line(self, raw: str) -> None:
        msg = json.loads(raw)
        for reply in self._script_fn(msg):
            self._responses.append(json.dumps(reply) + "\n")

    def poll(self) -> int | None:
        return self.returncode if self._terminated else None

    def terminate(self) -> None:
        self._terminated = True

    def kill(self) -> None:
        self._terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self._terminated = True
        return self.returncode


def test_stream_abort_diagnostics_all_paths_and_no_secret_leaks() -> None:
    secret_cmd = "cat /opt/telegram-group-bot/.env"
    secret_tok = "sk-proj-SUPERSECRETVALUE1234567890"
    secret_prompt = "TOP_SECRET_USER_PROMPT_TEXT"

    # 1. step_update with forbidden named tool and raw parameters
    p1 = AntigravityStreamParser(tool_profile="chat")
    with pytest.raises(AntigravityNativeToolAbort) as exc1:
        p1.feed_line(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "tool",
                        "tool_name": "run_command",
                        "tool_info": {
                            "name": "run_command",
                            "parameters": {"command": secret_cmd, "token": secret_tok},
                        },
                    },
                }
            )
        )
    msg1 = str(exc1.value)
    assert "Antigravity native tool execution intercepted and terminated" in msg1
    assert "profile=chat" in msg1
    assert "source=step_update" in msg1
    assert "step_type=tool" in msg1
    assert "tool=run_command" in msg1
    assert "keys=" not in msg1
    assert secret_cmd not in msg1
    assert secret_tok not in msg1
    assert "parameters" not in msg1
    assert exc1.value.verdict.category == ErrorCategory.NATIVE_TOOL_ABORT
    assert classify_error(msg1).category == ErrorCategory.NATIVE_TOOL_ABORT

    # 2. step_update with missing tool name -> includes safe top-level keys only
    p2 = AntigravityStreamParser(tool_profile="ops")
    with pytest.raises(AntigravityNativeToolAbort) as exc2:
        p2.feed_line(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "tool",
                        "state": "ACTIVE",
                        "tool_info": {"parameters": {"secret": secret_tok}},
                    },
                }
            )
        )
    msg2 = str(exc2.value)
    assert "profile=ops" in msg2
    assert "source=step_update" in msg2
    assert "step_type=tool" in msg2
    assert "keys=state,step_type,tool_info" in msg2
    assert secret_tok not in msg2
    assert "parameters" not in msg2

    # 3. step_update with subagent (extracts type_name only, never role/prompt)
    p3 = AntigravityStreamParser(tool_profile="image")
    with pytest.raises(AntigravityNativeToolAbort) as exc3:
        p3.feed_line(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "subagent",
                        "tool_name": "invoke_subagent",
                        "subagent_info": {
                            "subagents": [
                                {
                                    "type_name": "research",
                                    "role": "SECRET_ROLE",
                                    "prompt": secret_prompt,
                                }
                            ]
                        },
                    },
                }
            )
        )
    msg3 = str(exc3.value)
    assert "profile=image" in msg3
    assert "source=step_update" in msg3
    assert "step_type=subagent" in msg3
    assert "tool=invoke_subagent" in msg3
    assert "subagents=research" in msg3
    assert "SECRET_ROLE" not in msg3
    assert secret_prompt not in msg3

    # 4. assistant message-level interception (unnamed tool with content)
    p4 = AntigravityStreamParser(tool_profile="chat")
    with pytest.raises(AntigravityNativeToolAbort) as exc4:
        p4.feed_line(
            json.dumps(
                {
                    "event": "assistant",
                    "message": {
                        "step_type": "tool",
                        "content": [{"type": "text", "text": secret_prompt}],
                    },
                }
            )
        )
    msg4 = str(exc4.value)
    assert "source=assistant" in msg4
    assert "step_type=tool" in msg4
    assert "keys=content,step_type" in msg4
    assert secret_prompt not in msg4

    # 5. assistant.content block interception (tool_use & subagent block)
    p5 = AntigravityStreamParser(tool_profile="chat")
    with pytest.raises(AntigravityNativeToolAbort) as exc5:
        p5.feed_line(
            json.dumps(
                {
                    "event": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "bash",
                                "input": {"command": secret_cmd, "token": secret_tok},
                            }
                        ]
                    },
                }
            )
        )
    msg5 = str(exc5.value)
    assert "source=assistant.content" in msg5
    assert "step_type=tool_use" in msg5
    assert "tool=bash" in msg5
    assert secret_cmd not in msg5
    assert secret_tok not in msg5

    p6 = AntigravityStreamParser(tool_profile="chat")
    with pytest.raises(AntigravityNativeToolAbort) as exc6:
        p6.feed_line(
            json.dumps(
                {
                    "event": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "subagent",
                                "tool_name": "invoke_subagent",
                                "subagent_info": {"subagents": []},
                            }
                        ]
                    },
                }
            )
        )
    msg6 = str(exc6.value)
    assert "source=assistant.content" in msg6
    assert "step_type=subagent" in msg6
    assert "tool=invoke_subagent" in msg6
    assert "subagents=none" in msg6


def test_stream_abort_redacts_secrets_smuggled_into_tool_name_or_keys() -> None:
    smuggled_bot_token = "999888777:AAH_EXFILTRATION_TRAP_TOKEN_123456"
    smuggled_cmd_name = "bash -c 'cat .env'"
    smuggled_key = "Bearer sk-secret1234567890abcdef"

    p1 = AntigravityStreamParser(tool_profile="chat")
    with pytest.raises(AntigravityNativeToolAbort) as exc1:
        p1.feed_line(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {"step_type": "tool", "tool_name": smuggled_bot_token},
                }
            )
        )
    assert smuggled_bot_token not in str(exc1.value)
    assert "tool=<redacted>" in str(exc1.value)

    p2 = AntigravityStreamParser(tool_profile="chat")
    with pytest.raises(AntigravityNativeToolAbort) as exc2:
        p2.feed_line(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {"step_type": "tool", "tool_name": smuggled_cmd_name},
                }
            )
        )
    assert smuggled_cmd_name not in str(exc2.value)
    assert "tool=<redacted>" in str(exc2.value)

    p3 = AntigravityStreamParser(tool_profile="chat")
    with pytest.raises(AntigravityNativeToolAbort) as exc3:
        p3.feed_line(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {"step_type": "tool", smuggled_key: "val"},
                }
            )
        )
    assert smuggled_key not in str(exc3.value)
    assert "keys=<redacted>,step_type" in str(exc3.value)


@pytest.mark.asyncio
async def test_acp_abort_diagnostics_and_title_non_leak(tmp_path: Path) -> None:
    img = tmp_path / "pic.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(img, format="JPEG")
    cli = AntigravityCLIClient(data_dir=tmp_path / "agy_home", binary="/mock/bin/agy")

    try:
        # 1. Explicit forbidden tool name in ACP tool_call
        def script_named(msg: dict[str, Any]) -> list[dict[str, Any]]:
            m, mid = msg.get("method"), msg.get("id")
            if m == "initialize":
                return [{"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}}]
            if m == "session/new":
                return [{"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "s1"}}]
            if m == "session/prompt":
                return [
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "tool_call",
                                "toolCallId": "tc1",
                                "name": "run_command",
                                "title": "search_web",
                                "rawInput": {"command": "SECRET_COMMAND_ARG"},
                            }
                        },
                    }
                ]
            return []

        with patch(
            "bot.antigravity.cli_client.popen_isolated_session",
            return_value=_ScriptedAcpPopen(script_named),
        ):
            with pytest.raises(AntigravityNativeToolAbort) as exc_named:
                await cli.generate_response(
                    prompt="Check image",
                    media_attachments=[MediaAttachment(file_path=img, mime_type="image/jpeg")],
                    tool_profile="chat",
                )
        msg_named = str(exc_named.value)
        assert "Antigravity native tool execution intercepted and terminated" in msg_named
        assert "profile=chat" in msg_named
        assert "source=acp" in msg_named
        assert "step_type=tool_call" in msg_named
        assert "tool=run_command" in msg_named
        assert "SECRET_COMMAND_ARG" not in msg_named
        assert classify_error(msg_named).category == ErrorCategory.NATIVE_TOOL_ABORT

        # 2. Unnamed ACP tool_call_update with descriptive title containing secret -> title must NOT leak
        def script_title_only(msg: dict[str, Any]) -> list[dict[str, Any]]:
            m, mid = msg.get("method"), msg.get("id")
            if m == "initialize":
                return [{"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}}]
            if m == "session/new":
                return [{"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "s2"}}]
            if m == "session/prompt":
                return [
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "tool_call_update",
                                "toolCallId": "tc2",
                                "title": "Running bash with sk-secret9876543210",
                                "rawInput": {"secret": "RAW_SECRET"},
                            }
                        },
                    }
                ]
            return []

        with patch(
            "bot.antigravity.cli_client.popen_isolated_session",
            return_value=_ScriptedAcpPopen(script_title_only),
        ):
            with pytest.raises(AntigravityNativeToolAbort) as exc_title:
                await cli.generate_response(
                    prompt="Check image",
                    media_attachments=[MediaAttachment(file_path=img, mime_type="image/jpeg")],
                    tool_profile="chat",
                )
        msg_title = str(exc_title.value)
        assert "Antigravity native tool execution intercepted and terminated" in msg_title
        assert "source=acp" in msg_title
        assert "step_type=tool_call_update" in msg_title
        assert "keys=rawInput,sessionUpdate,title,toolCallId" in msg_title
        assert "sk-secret9876543210" not in msg_title
        assert "RAW_SECRET" not in msg_title
        assert classify_error(msg_title).category == ErrorCategory.NATIVE_TOOL_ABORT
    finally:
        cli.close()


@pytest.mark.asyncio
async def test_orchestrator_native_tool_abort_single_warning_no_traceback_and_sanitized(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg_mgr = _make_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "mem.sqlite"))
    await store.init_db()
    userbot = _DummyUserbot()
    leaked_secret = "999888777:AAFakeOrchestratorSecret123456"
    sanitizer = SecretSanitizer(extra_secrets=[leaked_secret], load_environ=False)

    def _make_abort_err(*, already_logged: bool = False) -> AntigravityNativeToolAbort:
        err = AntigravityNativeToolAbort.from_context(
            tool_profile="chat",
            source="step_update",
            step_type="tool",
            tool_name="view_file",
        )
        if already_logged:
            setattr(err, "_native_tool_abort_logged", True)
        return err

    def _raise_unlogged(*_a: Any, **_kw: Any) -> Any:
        raise _make_abort_err(already_logged=False)

    def _raise_logged(*_a: Any, **_kw: Any) -> Any:
        raise _make_abort_err(already_logged=True)

    provider = MagicMock()
    provider.is_rate_limited = MagicMock(return_value=False)
    provider.generate_response = AsyncMock(side_effect=_raise_unlogged)

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=MediaProcessor(temp_dir_base=str(tmp_path / "media")),
        provider=provider,
        userbot=userbot,
        sanitizer=sanitizer,
        debounce_seconds=0.01,
    )
    try:
        incoming = IncomingMessage(
            msg_id=10,
            chat_id=-100111222333,
            sender_id=1001,
            sender_name="Tester",
            text="Jeff, check this",
            is_mention=True,
        )
        with caplog.at_level(logging.WARNING, logger="bot.main"):
            await orch._on_incoming_message(incoming)
            await asyncio.sleep(0.08)

        abort_records = [
            rec for rec in caplog.records if "Native tool execution aborted" in rec.getMessage()
        ]
        assert len(abort_records) == 1
        rec = abort_records[0]
        assert rec.levelno == logging.WARNING
        assert rec.exc_info is None
        assert "tool=view_file" in rec.getMessage()
        assert "profile=chat" in rec.getMessage()
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)
        assert len(userbot.sent_replies) == 1
        assert userbot.sent_replies[0]["chat_id"] == -100111222333
        assert userbot.sent_replies[0]["reply_to_msg_id"] == 10
        assert "profile=chat" in userbot.sent_replies[0]["text"]
        assert leaked_secret not in userbot.sent_replies[0]["text"]

        # Also verify non-debounced handle_incoming_message returns None cleanly with 1 warning (when not pre-logged)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="bot.main"):
            res = await orch.handle_incoming_message(incoming)
        assert res is None
        direct_records = [
            rec for rec in caplog.records if "Native tool execution aborted" in rec.getMessage()
        ]
        assert len(direct_records) == 1
        assert direct_records[0].exc_info is None
        assert len(userbot.sent_replies) == 2

        # When stream.py already logged (_native_tool_abort_logged=True), main.py skips duplicate WARNING
        caplog.clear()
        provider.generate_response = AsyncMock(side_effect=_raise_logged)
        with caplog.at_level(logging.WARNING, logger="bot.main"):
            res_logged = await orch.handle_incoming_message(incoming)
        assert res_logged is None
        assert [rec for rec in caplog.records if rec.levelno >= logging.WARNING] == []
        assert len(userbot.sent_replies) == 3

        # Verify _debounced_flush fallback also catches unlogged AntigravityNativeToolAbort without traceback
        caplog.clear()
        with patch.object(
            orch,
            "_process_buffered_messages",
            new=AsyncMock(side_effect=_make_abort_err()),
        ):
            with caplog.at_level(logging.WARNING, logger="bot.main"):
                await orch._debounced_flush(-100111222333, 0.0)
        flush_records = [
            rec for rec in caplog.records if "Native tool execution aborted" in rec.getMessage()
        ]
        assert len(flush_records) == 1
        assert flush_records[0].exc_info is None
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_evolution_judge_native_tool_abort_logs_once_with_diagnostics(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg_mgr = _make_config(tmp_path, evo_enabled=True)
    store = MemoryStore(db_path=str(tmp_path / "evo.sqlite"))
    await store.init_db()
    qm = QuotaBudgetManager(memory_store=store, config=cfg_mgr.bot)

    abort_err = AntigravityNativeToolAbort.from_context(
        tool_profile="chat",
        source="step_update",
        step_type="tool",
        tool_name="grep_search",
    )
    provider = MagicMock()
    provider.generate_response = AsyncMock(side_effect=abort_err)
    reports: list[str] = []

    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr,
        memory_store=store,
        quota_manager=qm,
        provider=provider,
        notify_admin=lambda msg: reports.append(msg),
    )
    try:
        with caplog.at_level(logging.WARNING, logger="bot.evolution.manager"):
            rec = await mgr.run_admin_task(user_id=1001, task="inspect bug")
        assert rec is not None
        assert rec.verdict == "NO_CHANGE"
        assert "profile=chat" in rec.summary
        assert "tool=grep_search" in rec.summary
        assert len(reports) == 1
        assert "tool=grep_search" in reports[0]
        judge_warnings = [
            r for r in caplog.records if "Adversarial Judge failed" in r.getMessage()
        ]
        assert len(judge_warnings) == 1
        assert judge_warnings[0].exc_info is None
    finally:
        await store.close()
