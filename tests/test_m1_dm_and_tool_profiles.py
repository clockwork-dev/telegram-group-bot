"""Unit and integration tests for M1 Bot API DM routing, tool profiles, ACP fixes, and orchestrator gating."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from bot.antigravity import (
    ALWAYS_FORBIDDEN_TOOLS,
    CHAT_ALLOWED_TOOLS,
    OPS_ALLOWED_TOOLS,
    SELF_DEV_ALLOWED_TOOLS,
    AntigravityCLIClient,
    AntigravityConfig,
    AntigravityNativeToolAbort,
    AntigravityProvider,
    AntigravitySDKClient,
    AntigravityStreamParser,
)
from bot.config.manager import ConfigManager
from bot.core.models import MediaAttachment, ProviderResponse
from bot.main import GroupBotOrchestrator
from bot.memory.db import MemoryStore
from bot.proactive.digest import NewsDigestGenerator
from bot.proactive.humor import HumorGenerator
from bot.telegram.bot_api_client import BotAPIMessageEvent, TelegramBotAPIClient
from bot.telegram.handlers import IncomingMessage
from bot.telegram.heuristics import ReplyAction, ReplyHeuristics


def test_bot_api_dm_routing_and_anti_loop():
    client = TelegramBotAPIClient(
        bot_token="123456:AAFakeToken1234567890abcdefghij",
        allowed_chat_ids=[-1003845062562],
        allowed_dm_user_ids=[382962362],
        ignored_sender_usernames=["hermes_kuk7yi6kfm3lcdfa_bot"],
        bot_names=["Jeff"],
    )
    client.bot_id = 999000
    client.bot_username = "jeff_bot"

    # 1. Allowed private DM from admin
    dm_allowed = client._convert_msg_dict(
        {
            "message_id": 10,
            "date": 1700000000,
            "chat": {"id": 382962362, "type": "private"},
            "from": {"id": 382962362, "is_bot": False, "username": "rodion", "first_name": "Rodion"},
            "text": "Привет, проверь логи",
            "reply_to_message": {
                "message_id": 9,
                "from": {"id": 999000, "first_name": "Jeff"},
                "text": "Ок",
            },
        }
    )
    assert dm_allowed is not None
    assert dm_allowed.is_private is True
    assert dm_allowed.sender_is_bot is False
    assert dm_allowed.sender_username == "rodion"
    assert dm_allowed.reply_to_sender_id == 999000

    # 2. Private DM from stranger dropped
    dm_stranger = client._convert_msg_dict(
        {
            "message_id": 11,
            "chat": {"id": 777888999, "type": "private"},
            "from": {"id": 777888999, "is_bot": False, "username": "stranger"},
            "text": "Hello",
        }
    )
    assert dm_stranger is None

    # 3. Bot sender dropped even in allowed group
    bot_msg = client._convert_msg_dict(
        {
            "message_id": 12,
            "chat": {"id": -1003845062562, "type": "supergroup"},
            "from": {"id": 111222, "is_bot": True, "username": "other_bot"},
            "text": "Jeff hi",
        }
    )
    assert bot_msg is None

    # 4. Ignored username dropped even if is_bot=False
    ignored_msg = client._convert_msg_dict(
        {
            "message_id": 13,
            "chat": {"id": -1003845062562, "type": "supergroup"},
            "from": {
                "id": 8710978944,
                "is_bot": False,
                "username": "hermes_kuk7yi6kfm3lcdfa_bot",
            },
            "text": "Jeff hi",
        }
    )
    assert ignored_msg is None

    # 5. Allowed group message accepted with is_private=False
    grp_msg = client._convert_msg_dict(
        {
            "message_id": 14,
            "chat": {"id": -1003845062562, "type": "supergroup"},
            "from": {"id": 382962362, "is_bot": False, "username": "rodion"},
            "text": "Jeff привет",
        }
    )
    assert grp_msg is not None
    assert grp_msg.is_private is False

    # 6. Unapproved group message from admin dropped
    unapproved_grp = client._convert_msg_dict(
        {
            "message_id": 15,
            "chat": {"id": -1009999999999, "type": "supergroup"},
            "from": {"id": 382962362, "is_bot": False, "username": "rodion"},
            "text": "Jeff привет",
        }
    )
    assert unapproved_grp is None

    # Check BotAPIMessageEvent metadata directly
    ev = BotAPIMessageEvent(
        {
            "message_id": 1,
            "date": 123456,
            "chat": {"id": 382962362, "type": "private"},
            "from": {"id": 382962362, "is_bot": False, "username": "rodion"},
            "text": "hi",
        },
        bot_id=999000,
    )
    assert ev.chat_type == "private"
    assert ev.is_private is True
    assert ev.sender_is_bot is False
    assert ev.sender_username == "rodion"
    assert ev.date == 123456


def test_heuristics_negative_group_without_is_private_never_becomes_dm():
    heuristics = ReplyHeuristics()
    ev = SimpleNamespace(
        chat_id=-1009999999999,
        sender_id=382962362,
        out=False,
        raw_text="Просто сообщение без упоминания",
        text="Просто сообщение без упоминания",
        is_private=None,
        sender_is_bot=False,
        sender_username="rodion",
    )
    action = heuristics.should_reply(
        event=ev,
        bot_id=999,
        bot_names=["Jeff"],
        allowed_chat_ids=[-1003845062562],
        allowed_dm_user_ids=[382962362],
        spontaneous_enabled=False,
    )
    assert action == ReplyAction.IGNORE


@pytest.mark.asyncio
async def test_hot_reload_syncs_bot_api_access_lists_and_batch_isolation(tmp_path: Path):
    bot_yaml = tmp_path / "bot.yaml"
    bot_yaml.write_text(
        yaml.dump(
            {
                "telegram": {
                    "mode": "bot_api",
                    "bot_token": "123456:AAFakeToken1234567890abcdefghij",
                    "api_id": 12345,
                    "api_hash": "0123456789abcdef0123456789abcdef",
                    "allowed_chat_ids": [-1003845062562],
                    "allowed_dm_user_ids": [],
                    "admin_user_ids": [382962362],
                },
                "behavior": {"bot_names": ["Jeff"], "typing_simulation": False},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "system_prompt.md").write_text("Prompt", encoding="utf-8")
    cfg_mgr = ConfigManager(config_path=bot_yaml)
    store = MemoryStore(db_path=str(tmp_path / "m.sqlite"))
    await store.init_db()

    orch = GroupBotOrchestrator(config_mgr=cfg_mgr, memory_store=store, provider=MagicMock())
    try:
        assert isinstance(orch.userbot, TelegramBotAPIClient)
        assert 382962362 in orch.userbot._allowed_dm_user_ids_set

        # Group isolation in _is_batch_permitted: unapproved negative group from admin must be False
        unapproved_group_msg = IncomingMessage(
            msg_id=1,
            chat_id=-100999888777,
            sender_id=382962362,
            sender_name="Rodion",
            text="voice",
            reply_to_msg_id=None,
            is_reply=False,
            is_mention=False,
            raw_event=SimpleNamespace(out=False, sender_id=382962362, is_private=False),
            is_private=False,
        )
        assert orch._is_batch_permitted([unapproved_group_msg], cfg_mgr.bot) is False

        # Modify bot.yaml to add new DM user and ignored username -> reload_changed syncs userbot
        time.sleep(0.05)
        bot_yaml.write_text(
            yaml.dump(
                {
                    "telegram": {
                        "mode": "bot_api",
                        "bot_token": "123456:AAFakeToken1234567890abcdefghij",
                        "api_id": 12345,
                        "api_hash": "0123456789abcdef0123456789abcdef",
                        "allowed_chat_ids": [-1003845062562],
                        "allowed_dm_user_ids": [555666],
                        "admin_user_ids": [382962362],
                        "ignored_sender_usernames": ["hermes_kuk7yi6kfm3lcdfa_bot", "spam_bot"],
                    },
                    "behavior": {"bot_names": ["Jeff"], "typing_simulation": False},
                }
            ),
            encoding="utf-8",
        )
        assert cfg_mgr.reload_changed() is True
        assert 555666 in orch.userbot._allowed_dm_user_ids_set
        assert 382962362 in orch.userbot._allowed_dm_user_ids_set
        assert "spam_bot" in orch.userbot._ignored_sender_usernames_set
    finally:
        await orch.stop()


def test_stream_parser_tool_profiles():
    assert CHAT_ALLOWED_TOOLS == frozenset({"search_web", "read_url_content"})
    assert len(OPS_ALLOWED_TOOLS) == 23
    assert SELF_DEV_ALLOWED_TOOLS == OPS_ALLOWED_TOOLS

    # Chat profile blocks FS/shell/browser/MCP/subagent
    for blocked in (
        "run_command",
        "read_file",
        "write_file",
        "view_file",
        "write_to_file",
        "browser_subagent",
        "call_mcp_tool",
        "invoke_subagent",
        "ask_question",
    ):
        p = AntigravityStreamParser(tool_profile="chat")
        with pytest.raises(AntigravityNativeToolAbort):
            p.feed_line(
                json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "toolName": blocked}})
            )

    # ops and self_dev allow all OPS_ALLOWED_TOOLS while blocking ALWAYS_FORBIDDEN_TOOLS
    for profile in ("ops", "self_dev"):
        p_ok = AntigravityStreamParser(tool_profile=profile)
        for tool in OPS_ALLOWED_TOOLS:
            p_ok.feed_line(
                json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "toolName": tool}})
            )

        for forbidden in (
            "browser_subagent",
            "call_mcp_tool",
            "swarm",
            "delete_knowledge",
            "manage_inbox",
            "ask_question",
            "ask_permission",
            "ask_custom_permission",
        ):
            assert forbidden in ALWAYS_FORBIDDEN_TOOLS
            p_bad = AntigravityStreamParser(tool_profile=profile)
            with pytest.raises(AntigravityNativeToolAbort):
                p_bad.feed_line(
                    json.dumps(
                        {"event": "step_update", "step_update": {"step_type": "tool", "tool_name": forbidden}}
                    )
                )


class _ScriptedAcpPopen:
    """Mock ACP process responding to initialize/session/new/set_mode/prompt with scripted updates."""

    def __init__(self, script_fn) -> None:
        self._script_fn = script_fn
        self._responses: list[str] = []
        self._resp_idx = 0
        self.sent_requests: list[dict] = []
        self.sent_responses: list[dict] = []
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
        if "method" in msg:
            self.sent_requests.append(msg)
        else:
            self.sent_responses.append(msg)
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


@pytest.mark.asyncio
async def test_acp_name_before_title_tool_call_update_and_permissions(tmp_path: Path):
    img = tmp_path / "pic.jpg"
    from PIL import Image

    Image.new("RGB", (8, 8), (10, 20, 30)).save(img, format="JPEG")
    cli = AntigravityCLIClient(data_dir=tmp_path / "agy_home", binary="/mock/bin/agy")

    # 1. search_web with descriptive title succeeds across tool_call AND tool_call_update (by toolCallId)
    def script_search_ok(msg):
        m = msg.get("method")
        mid = msg.get("id")
        if m == "initialize":
            return [{"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}}]
        if m == "session/new":
            return [
                {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "result": {
                        "sessionId": "s1",
                        "modes": {"currentModeId": "default"},
                    },
                }
            ]
        if m == "session/prompt":
            return [
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": "tc_web_1",
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
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": "tc_web_1",
                            "title": '"example.com" on Web (done)',
                            "status": "completed",
                        }
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "Found it."},
                        }
                    },
                },
                {"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}},
            ]
        return []

    proc_ok = _ScriptedAcpPopen(script_search_ok)
    with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=proc_ok):
        resp = await cli.generate_response(
            prompt="Check example.com",
            media_attachments=[MediaAttachment(file_path=img, mime_type="image/jpeg")],
            tool_profile="chat",
        )
    assert resp.text == "Found it."

    # 2. Title spoofing (name="run_command", title="search_web") aborts in chat profile
    def script_spoof(msg):
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
                            "sessionUpdate": "tool_call",
                            "toolCallId": "tc_spoof",
                            "name": "run_command",
                            "title": "search_web",
                        }
                    },
                }
            ]
        return []

    with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=_ScriptedAcpPopen(script_spoof)):
        with pytest.raises(AntigravityNativeToolAbort):
            await cli.generate_response(
                prompt="Run cmd",
                media_attachments=[MediaAttachment(file_path=img, mime_type="image/jpeg")],
                tool_profile="chat",
            )

    # 3. tool_call_update mutating to name="run_command" aborts in chat profile
    def script_mutate_update(msg):
        m, mid = msg.get("method"), msg.get("id")
        if m == "initialize":
            return [{"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}}]
        if m == "session/new":
            return [{"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "s3"}}]
        if m == "session/prompt":
            return [
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": "tc_mut",
                            "name": "search_web",
                        }
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": "tc_mut",
                            "name": "run_command",
                        }
                    },
                },
            ]
        return []

    with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=_ScriptedAcpPopen(script_mutate_update)):
        with pytest.raises(AntigravityNativeToolAbort):
            await cli.generate_response(
                prompt="Mutate",
                media_attachments=[MediaAttachment(file_path=img, mime_type="image/jpeg")],
                tool_profile="chat",
            )

    # 4. session/request_permission: always cancelled in chat; auto-approved for edit_file in ops; cancelled for call_mcp_tool in ops
    def script_permissions(msg):
        m, mid = msg.get("method"), msg.get("id")
        if m == "initialize":
            return [{"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}}]
        if m == "session/new":
            return [
                {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "result": {"sessionId": "s4", "modes": {"currentModeId": "default"}},
                }
            ]
        if m == "session/set_mode":
            return [{"jsonrpc": "2.0", "id": mid, "result": {}}]
        if m == "session/prompt":
            return [
                {
                    "jsonrpc": "2.0",
                    "id": "perm_1",
                    "method": "session/request_permission",
                    "params": {
                        "toolCall": {"name": "search_web"},
                        "options": [{"optionId": "allow_once", "kind": "allow_once"}],
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": "perm_2",
                    "method": "session/request_permission",
                    "params": {
                        "toolCall": {"name": "edit_file"},
                        "options": [{"optionId": "allow_once", "kind": "allow_once"}],
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": "perm_3",
                    "method": "session/request_permission",
                    "params": {
                        "toolCall": {"name": "call_mcp_tool"},
                        "options": [{"optionId": "allow_once", "kind": "allow_once"}],
                    },
                },
                {"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}},
            ]
        return []

    # In chat profile: search_web permission cancelled
    proc_chat_perm = _ScriptedAcpPopen(script_permissions)
    with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=proc_chat_perm):
        await cli.generate_response(
            prompt="Perm test chat",
            media_attachments=[MediaAttachment(file_path=img, mime_type="image/jpeg")],
            tool_profile="chat",
        )
    assert proc_chat_perm.sent_responses[0]["result"]["outcome"] == {"outcome": "cancelled"}

    # In ops profile with mode & cwd: search_web + edit_file approved, call_mcp_tool cancelled, HOME isolated, --add-dir & set_mode sent
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    captured_popen = []

    proc_ops = _ScriptedAcpPopen(script_permissions)

    def fake_popen_ops_ret(cmd, *, cwd, env):
        captured_popen.append((list(cmd), str(cwd), dict(env)))
        return proc_ops

    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen_ops_ret):
        await cli.generate_response(
            prompt="Perm test ops",
            media_attachments=[MediaAttachment(file_path=img, mime_type="image/jpeg")],
            tool_profile="ops",
            mode="accept-edits",
            cwd=staging_dir,
        )

    cmd_used, cwd_used, env_used = captured_popen[-1]
    assert "--add-dir" not in cmd_used
    assert cwd_used == str(staging_dir.resolve())
    assert env_used["HOME"] != str(staging_dir.resolve())
    assert Path(env_used["HOME"]) == (tmp_path / "agy_home").resolve()
    set_mode_reqs = [r for r in proc_ops.sent_requests if r.get("method") == "session/set_mode"]
    assert len(set_mode_reqs) == 1 and set_mode_reqs[0]["params"]["modeId"] == "accept-edits"
    new_reqs = [r for r in proc_ops.sent_requests if r.get("method") == "session/new"]
    assert new_reqs[0]["params"]["cwd"] == str(staging_dir.resolve())
    assert proc_ops.sent_responses[0]["result"]["outcome"] == {"outcome": "selected", "optionId": "allow_once"}
    assert proc_ops.sent_responses[1]["result"]["outcome"] == {"outcome": "selected", "optionId": "allow_once"}
    assert proc_ops.sent_responses[2]["result"]["outcome"] == {"outcome": "cancelled"}
    cli.close()


@pytest.mark.asyncio
async def test_sdk_client_tool_profile_builtin_tools_and_workspaces(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123456")
    sdk = AntigravitySDKClient(data_dir=tmp_path)
    captured_configs = []

    mock_resp = MagicMock()
    mock_resp.text = AsyncMock(return_value="ok")
    mock_resp.stop_reason = "stop"
    mock_resp.usage_metadata = None

    async def empty_thoughts():
        if False:
            yield ""

    mock_resp.thoughts = empty_thoughts()
    mock_agent = MagicMock()
    mock_agent.chat = AsyncMock(return_value=mock_resp)
    mock_agent.conversation_id = "c1"
    mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
    mock_agent.__aexit__ = AsyncMock(return_value=None)

    def capture_agent(cfg):
        captured_configs.append(cfg)
        return mock_agent

    work_dir = tmp_path / "repo"
    work_dir.mkdir()
    with patch("bot.antigravity.sdk_client.Agent", side_effect=capture_agent):
        await sdk.generate_response(prompt="hi", tool_profile="ops", cwd=work_dir)
        await sdk.generate_response(prompt="hi", tool_profile="chat")

    ops_cfg, chat_cfg = captured_configs
    ops_tools = {t.value if hasattr(t, "value") else str(t) for t in ops_cfg.capabilities.enabled_tools}
    chat_tools = {t.value if hasattr(t, "value") else str(t) for t in chat_cfg.capabilities.enabled_tools}
    assert "run_command" in ops_tools and "edit_file" in ops_tools
    assert ops_cfg.workspaces == [str(work_dir.resolve())]
    assert chat_tools == {"search_web", "read_url_content"}
    assert chat_cfg.capabilities.enable_subagents is False


@pytest.mark.asyncio
async def test_orchestrator_and_proactive_tool_profile_routing(tmp_path: Path):
    staging = tmp_path / "ops_ws"
    staging.mkdir()
    bot_yaml = tmp_path / "bot.yaml"
    bot_yaml.write_text(
        yaml.dump(
            {
                "telegram": {
                    "api_id": 123456,
                    "api_hash": "0123456789abcdef0123456789abcdef",
                    "allowed_chat_ids": [-1003845062562],
                    "allowed_dm_user_ids": [382962362, 999888],
                    "admin_user_ids": [382962362],
                },
                "behavior": {"bot_names": ["Jeff"], "typing_simulation": False},
                "antigravity": {
                    "ops_mode": "accept-edits",
                    "ops_cwd": str(staging),
                    "ops_tools_default_mode": "admin",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "system_prompt.md").write_text("Prompt", encoding="utf-8")
    cfg_mgr = ConfigManager(config_path=bot_yaml)
    store = MemoryStore(db_path=str(tmp_path / "mem.sqlite"))
    await store.init_db()

    calls: list[dict] = []

    class _SpyProvider:
        async def generate_response(
            self,
            prompt,
            system_instructions=None,
            media_attachments=None,
            conversation_id=None,
            model=None,
            effort=None,
            tool_profile="chat",
            mode=None,
            cwd=None,
        ):
            calls.append({"tool_profile": tool_profile, "mode": mode, "cwd": cwd})
            return ProviderResponse(text="ok", model="test")

        def close(self):
            pass

    class _MockUserbot:
        bot_id = 777
        bot_username = "jeff_bot"
        client = MagicMock()

        def register_message_handler(self, h):
            pass

        async def send_reply(self, chat_id, text, reply_to_msg_id=None, simulate_typing=True):
            return {"result": {"message_id": 1}}

        async def send_typing_action(self, chat_id, action="typing"):
            pass

        async def send_reaction(self, chat_id, msg_id, emoji):
            return True

        async def stop(self):
            pass

    spy = _SpyProvider()
    orch = GroupBotOrchestrator(config_mgr=cfg_mgr, memory_store=store, provider=spy, userbot=_MockUserbot())
    try:
        # 1. Admin private DM -> ops
        admin_dm = IncomingMessage(
            msg_id=1,
            chat_id=382962362,
            sender_id=382962362,
            sender_name="Rodion",
            text="Check disk",
            reply_to_msg_id=None,
            is_reply=False,
            is_mention=False,
            raw_event=SimpleNamespace(id=1, chat_id=382962362, sender_id=382962362, out=False, is_private=True, sender_is_bot=False, raw_text="Check disk"),
            is_private=True,
        )
        await orch.handle_incoming_message(admin_dm)
        assert calls[-1] == {"tool_profile": "ops", "mode": "accept-edits", "cwd": str(staging)}

        # 2. Admin in group -> chat (NOT ops!)
        admin_grp = IncomingMessage(
            msg_id=2,
            chat_id=-1003845062562,
            sender_id=382962362,
            sender_name="Rodion",
            text="Jeff привет",
            reply_to_msg_id=None,
            is_reply=False,
            is_mention=True,
            raw_event=SimpleNamespace(id=2, chat_id=-1003845062562, sender_id=382962362, out=False, is_private=False, sender_is_bot=False, raw_text="Jeff привет"),
            is_private=False,
        )
        await orch.handle_incoming_message(admin_grp)
        assert calls[-1]["tool_profile"] == "chat"

        # 3. Non-admin allowed DM -> chat (NOT ops!)
        other_dm = IncomingMessage(
            msg_id=3,
            chat_id=999888,
            sender_id=999888,
            sender_name="Friend",
            text="Привет",
            reply_to_msg_id=None,
            is_reply=False,
            is_mention=False,
            raw_event=SimpleNamespace(id=3, chat_id=999888, sender_id=999888, out=False, is_private=True, sender_is_bot=False, raw_text="Привет"),
            is_private=True,
        )
        await orch.handle_incoming_message(other_dm)
        assert calls[-1]["tool_profile"] == "chat"

        # 4. Compaction, digest, humor -> all pass tool_profile="chat"
        await orch._summarize_transcript_via_provider("sample transcript")
        assert calls[-1]["tool_profile"] == "chat"

        digest_gen = NewsDigestGenerator(provider=spy)
        await digest_gen.generate_digest(slot_name="morning_digest", slot_label="morning", timezone="Europe/Kyiv")
        assert calls[-1]["tool_profile"] == "chat"

        humor_gen = HumorGenerator(provider=spy)
        await humor_gen.generate_banter()
        assert calls[-1]["tool_profile"] == "chat"

        # 5. AntigravityProvider propagates model/effort/tool_profile/mode/cwd to CLI
        provider = AntigravityProvider(AntigravityConfig(data_dir=tmp_path / "agy", enable_sdk=False))
        provider._cli_client = MagicMock()
        provider._cli_client.generate_response = AsyncMock(return_value=ProviderResponse(text="done"))
        await provider.generate_response(
            prompt="hi",
            model="claude-sonnet-5-5",
            effort="high",
            tool_profile="self_dev",
            mode="auto",
            cwd=staging,
        )
        kw = provider._cli_client.generate_response.call_args.kwargs
        assert kw["model"] == "claude-sonnet-5-5"
        assert kw["effort"] == "high"
        assert kw["tool_profile"] == "self_dev"
        assert kw["mode"] == "auto"
        assert kw["cwd"] == staging
    finally:
        await orch.stop()
