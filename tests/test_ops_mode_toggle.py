"""Tests for toggleable extended ops tool mode, subagent/task allowlist, and /ops commands."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from bot.antigravity import (
    ALWAYS_FORBIDDEN_TOOLS,
    CHAT_ALLOWED_TOOLS,
    OPS_ALLOWED_TOOLS,
    SELF_DEV_ALLOWED_TOOLS,
    AntigravityNativeToolAbort,
    AntigravityStreamParser,
)
from bot.config.manager import ConfigManager
from bot.core.models import ProviderResponse
from bot.evolution.manager import SelfEvolutionManager
from bot.main import GroupBotOrchestrator
from bot.memory.db import MemoryStore, SelfEditRecord
from bot.telegram.handlers import IncomingMessage

ADMIN_ID = 382962362
FRIEND_ID = 555666777
GROUP_ID = -1003845062562


class _SpyProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []

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
        self.calls.append(
            {
                "prompt": prompt,
                "tool_profile": tool_profile,
                "mode": mode,
                "cwd": cwd,
                "model": model,
            }
        )
        return ProviderResponse(text="ok", model=model or "test")

    def close(self) -> None:
        pass


class _MockUserbot:
    def __init__(self) -> None:
        self.bot_id = 777000
        self.bot_username = "jeff_bot"
        self.client = MagicMock()
        self.replies: list[dict] = []

    def register_message_handler(self, handler) -> None:
        pass

    async def send_reply(self, chat_id, text, reply_to_msg_id=None, simulate_typing=True):
        self.replies.append({"chat_id": chat_id, "text": text, "reply_to_msg_id": reply_to_msg_id})
        return {"result": {"message_id": len(self.replies) + 100}}

    async def send_typing_action(self, chat_id, action="typing"):
        pass

    async def send_reaction(self, chat_id, msg_id, emoji):
        return True

    async def stop(self) -> None:
        pass


def _make_config(
    tmp_path: Path,
    *,
    ops_default_mode: str = "on",
    admin_ids: list[int] | None = None,
    ops_cwd: str | None = None,
) -> ConfigManager:
    bot_yaml = tmp_path / "bot.yaml"
    bot_yaml.write_text(
        yaml.dump(
            {
                "telegram": {
                    "api_id": 123456,
                    "api_hash": "0123456789abcdef0123456789abcdef",
                    "allowed_chat_ids": [GROUP_ID],
                    "allowed_dm_user_ids": [FRIEND_ID],
                    "admin_user_ids": [ADMIN_ID] if admin_ids is None else admin_ids,
                },
                "behavior": {"bot_names": ["Jeff"], "typing_simulation": False},
                "antigravity": {
                    "ops_mode": "auto",
                    "ops_cwd": ops_cwd,
                    "ops_tools_default_mode": ops_default_mode,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "system_prompt.md").write_text("System prompt", encoding="utf-8")
    return ConfigManager(config_path=bot_yaml)


def _msg(
    *,
    msg_id: int,
    chat_id: int,
    sender_id: int,
    text: str,
    is_private: bool = False,
    is_mention: bool = False,
) -> IncomingMessage:
    return IncomingMessage(
        msg_id=msg_id,
        chat_id=chat_id,
        sender_id=sender_id,
        sender_name="Admin" if sender_id == ADMIN_ID else "Friend",
        text=text,
        reply_to_msg_id=None,
        is_reply=False,
        is_mention=is_mention,
        raw_event=SimpleNamespace(
            id=msg_id,
            chat_id=chat_id,
            sender_id=sender_id,
            out=False,
            is_private=is_private,
            sender_is_bot=False,
            raw_text=text,
        ),
        is_private=is_private,
    )


def test_extended_ops_tool_allowlist_and_subagent_gating() -> None:
    extended_subagent_tools = {
        "invoke_subagent",
        "define_subagent",
        "manage_subagents",
        "manage_task",
        "send_message",
        "run_workflow",
        "schedule",
    }
    assert CHAT_ALLOWED_TOOLS == frozenset({"search_web", "read_url_content"})
    assert len(OPS_ALLOWED_TOOLS) == 23
    assert SELF_DEV_ALLOWED_TOOLS == OPS_ALLOWED_TOOLS
    assert extended_subagent_tools.issubset(OPS_ALLOWED_TOOLS)
    assert extended_subagent_tools.isdisjoint(ALWAYS_FORBIDDEN_TOOLS)

    # ops & self_dev permit all OPS_ALLOWED_TOOLS including subagent steps
    for profile in ("ops", "self_dev"):
        parser = AntigravityStreamParser(tool_profile=profile)
        for tool in OPS_ALLOWED_TOOLS:
            parser.feed_line(
                json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "toolName": tool}})
            )
        parser.feed_line(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "subagent",
                        "tool_name": "invoke_subagent",
                        "subagent_info": {"subagents": [{"type_name": "research"}]},
                    },
                }
            )
        )
        # Still blocks browser, MCP, swarm, interactive asks
        for forbidden in ("browser_subagent", "call_mcp_tool", "swarm", "ask_question"):
            assert forbidden in ALWAYS_FORBIDDEN_TOOLS
            p_bad = AntigravityStreamParser(tool_profile=profile)
            with pytest.raises(AntigravityNativeToolAbort):
                p_bad.feed_line(
                    json.dumps(
                        {"event": "step_update", "step_update": {"step_type": "tool", "tool_name": forbidden}}
                    )
                )

    # chat profile STILL blocks subagents, workflows, shell, and file tools
    for blocked in ("run_command", "edit_file", "invoke_subagent", "run_workflow", "schedule"):
        p_chat = AntigravityStreamParser(tool_profile="chat")
        with pytest.raises(AntigravityNativeToolAbort):
            p_chat.feed_line(
                json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "tool_name": blocked}})
            )

    p_chat_sub = AntigravityStreamParser(tool_profile="chat")
    with pytest.raises(AntigravityNativeToolAbort):
        p_chat_sub.feed_line(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "subagent",
                        "tool_name": "invoke_subagent",
                        "subagent_info": {"subagents": [{"type_name": "research"}]},
                    },
                }
            )
        )

    # image profile STILL allows ONLY image-generator subagent
    p_img = AntigravityStreamParser(tool_profile="image")
    with pytest.raises(AntigravityNativeToolAbort):
        p_img.feed_line(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "subagent",
                        "tool_name": "invoke_subagent",
                        "subagent_info": {"subagents": [{"type_name": "research"}]},
                    },
                }
            )
        )


@pytest.mark.asyncio
async def test_ops_mode_default_on_toggle_off_admin_and_sqlite_persistence(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg_mgr = _make_config(tmp_path, ops_default_mode="on", ops_cwd=str(workspace))
    db_path = str(tmp_path / "mem.sqlite")
    store = MemoryStore(db_path=db_path)
    await store.init_db()

    spy = _SpyProvider()
    ub = _MockUserbot()
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=spy,
        userbot=ub,
        debounce_seconds=0.0,
    )
    try:
        # 1. Default mode is ON -> group non-admin gets ops profile + ops prompt guidance
        await orch.handle_incoming_message(
            _msg(msg_id=1, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="Jeff проверь файлы", is_mention=True)
        )
        assert spy.calls[-1]["tool_profile"] == "ops"
        assert spy.calls[-1]["mode"] == "auto"
        assert spy.calls[-1]["cwd"] == str(workspace)
        assert "РАСШИРЕННЫЙ РЕЖИМ ИНСТРУМЕНТОВ" in spy.calls[-1]["prompt"]

        # 2. Non-admin allowed DM: plain greeting stays on cheap chat profile, action query gets ops
        await orch.handle_incoming_message(
            _msg(msg_id=2, chat_id=FRIEND_ID, sender_id=FRIEND_ID, text="Привет", is_private=True)
        )
        assert spy.calls[-1]["tool_profile"] == "chat"

        await orch.handle_incoming_message(
            _msg(msg_id=200, chat_id=FRIEND_ID, sender_id=FRIEND_ID, text="Покажи статус сервера", is_private=True)
        )
        assert spy.calls[-1]["tool_profile"] == "ops"

        # 3. Non-admin CANNOT toggle /ops off
        await orch.handle_incoming_message(
            _msg(msg_id=3, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="/ops off")
        )
        assert "только администратор" in ub.replies[-1]["text"]
        assert orch._ops_tools_mode == "on"

        # 4. Command addressed to another bot is ignored
        replies_before = len(ub.replies)
        await orch.handle_incoming_message(
            _msg(msg_id=4, chat_id=GROUP_ID, sender_id=ADMIN_ID, text="/ops@other_bot off")
        )
        assert len(ub.replies) == replies_before
        assert orch._ops_tools_mode == "on"

        # 5. Admin toggles /ops off directly in group (without bot name mention)
        await orch.handle_incoming_message(
            _msg(msg_id=5, chat_id=GROUP_ID, sender_id=ADMIN_ID, text="/ops@jeff_bot off")
        )
        assert orch._ops_tools_mode == "off"
        assert "отключён" in ub.replies[-1]["text"]
        assert await store.get_setting("ops_tools_mode") == "off"

        # 6. After /ops off -> group AND admin DM use chat-only safe profile
        await orch.handle_incoming_message(
            _msg(msg_id=6, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="Jeff запусти команду", is_mention=True)
        )
        assert spy.calls[-1]["tool_profile"] == "chat"
        assert "РАСШИРЕННЫЙ РЕЖИМ ИНСТРУМЕНТОВ" not in spy.calls[-1]["prompt"]

        await orch.handle_incoming_message(
            _msg(msg_id=7, chat_id=ADMIN_ID, sender_id=ADMIN_ID, text="Проверь диск", is_private=True)
        )
        assert spy.calls[-1]["tool_profile"] == "chat"

        # 7. Admin switches to /ops admin -> only admin private DM gets ops
        await orch.handle_incoming_message(
            _msg(msg_id=8, chat_id=GROUP_ID, sender_id=ADMIN_ID, text="/tools admin")
        )
        assert orch._ops_tools_mode == "admin"

        await orch.handle_incoming_message(
            _msg(msg_id=9, chat_id=GROUP_ID, sender_id=ADMIN_ID, text="Jeff тест", is_mention=True)
        )
        assert spy.calls[-1]["tool_profile"] == "chat"

        await orch.handle_incoming_message(
            _msg(msg_id=10, chat_id=ADMIN_ID, sender_id=ADMIN_ID, text="Проверь логи", is_private=True)
        )
        assert spy.calls[-1]["tool_profile"] == "ops"

        # 8. Admin turns safemode on (/safemode on == /ops off)
        await orch.handle_incoming_message(
            _msg(msg_id=11, chat_id=ADMIN_ID, sender_id=ADMIN_ID, text="/safemode on", is_private=True)
        )
        assert orch._ops_tools_mode == "off"
    finally:
        await orch.stop()

    # 9. Restart orchestrator against same SQLite DB -> persisted "off" survives restart!
    store2 = MemoryStore(db_path=db_path)
    await store2.init_db()
    spy2 = _SpyProvider()
    ub2 = _MockUserbot()
    orch2 = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store2,
        provider=spy2,
        userbot=ub2,
        debounce_seconds=0.0,
    )
    try:
        await orch2.handle_incoming_message(
            _msg(msg_id=12, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="Jeff привет", is_mention=True)
        )
        assert orch2._ops_tools_mode == "off"
        assert spy2.calls[-1]["tool_profile"] == "chat"

        # Re-enable with /ops on: plain banter stays on chat, action query escalates to ops
        await orch2.handle_incoming_message(
            _msg(msg_id=13, chat_id=GROUP_ID, sender_id=ADMIN_ID, text="/ops on")
        )
        assert orch2._ops_tools_mode == "on"
        await orch2.handle_incoming_message(
            _msg(msg_id=14, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="Jeff привет", is_mention=True)
        )
        assert spy2.calls[-1]["tool_profile"] == "chat"
        await orch2.handle_incoming_message(
            _msg(msg_id=15, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="Jeff проверь файлы", is_mention=True)
        )
        assert spy2.calls[-1]["tool_profile"] == "ops"
    finally:
        await orch2.stop()


@pytest.mark.asyncio
async def test_evolve_and_quota_commands_in_group_respect_ops_mode(tmp_path: Path) -> None:
    cfg_mgr = _make_config(tmp_path, ops_default_mode="on")
    store = MemoryStore(db_path=str(tmp_path / "evo.sqlite"))
    await store.init_db()

    spy = _SpyProvider()
    ub = _MockUserbot()
    evo_mock = MagicMock()
    evo_mock.run_admin_task = AsyncMock(
        return_value=SelfEditRecord(
            trigger_type="ops_chat",
            signature="ops:123",
            reason="add feature",
            verdict="NO_CHANGE",
            deploy_result="skipped",
            summary="No code change needed",
        )
    )
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=spy,
        userbot=ub,
        evolution_manager=evo_mock,
        debounce_seconds=0.0,
    )
    try:
        # 1. With mode ON, group friend can trigger /evolve without mentioning bot name
        await orch.handle_incoming_message(
            _msg(msg_id=1, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="/evolve проверь баг")
        )
        evo_mock.run_admin_task.assert_awaited_once_with(
            user_id=FRIEND_ID,
            task="проверь баг",
            allow_non_admin=True,
        )
        assert "Self-Edit" in ub.replies[-1]["text"]

        # 2. Admin disables ops mode -> /evolve is blocked
        await orch.handle_incoming_message(
            _msg(msg_id=2, chat_id=GROUP_ID, sender_id=ADMIN_ID, text="/ops off")
        )
        evo_mock.run_admin_task.reset_mock()
        await orch.handle_incoming_message(
            _msg(msg_id=3, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="/evolve сломай всё")
        )
        evo_mock.run_admin_task.assert_not_awaited()
        assert "отключён" in ub.replies[-1]["text"]

        # 3. Admin can still check /quota even when ops mode is OFF
        await orch.handle_incoming_message(
            _msg(msg_id=4, chat_id=ADMIN_ID, sender_id=ADMIN_ID, text="/quota", is_private=True)
        )
        assert ub.replies[-1]["chat_id"] == ADMIN_ID
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_evolution_manager_allow_non_admin_flag(tmp_path: Path) -> None:
    cfg_mgr = _make_config(tmp_path, ops_default_mode="on")
    store = MemoryStore(db_path=str(tmp_path / "evo_mgr.sqlite"))
    await store.init_db()
    quota = MagicMock()
    quota.should_allow_self_edit = MagicMock(return_value=(True, "ok"))
    mgr = SelfEvolutionManager(
        config_mgr=cfg_mgr,
        memory_store=store,
        quota_manager=quota,
        provider=MagicMock(),
    )
    mgr._execute_evolution_pipeline = AsyncMock(
        return_value=SelfEditRecord(
            trigger_type="ops_chat",
            signature="ops:abc",
            reason="fix",
            verdict="NO_CHANGE",
            deploy_result="skipped",
            summary="checked",
        )
    )

    # Default allow_non_admin=False rejects non-admin
    rejected = await mgr.run_admin_task(user_id=FRIEND_ID, task="fix")
    assert rejected is None
    mgr._execute_evolution_pipeline.assert_not_awaited()

    # allow_non_admin=True permits execution from ops-enabled chat
    accepted = await mgr.run_admin_task(user_id=FRIEND_ID, task="fix", allow_non_admin=True)
    assert accepted is not None
    assert mgr._execute_evolution_pipeline.await_args.kwargs["trigger_type"] == "ops_chat"
    await store.close()
