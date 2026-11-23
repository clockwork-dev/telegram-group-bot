"""Focused tests for ops prompt sync, media/admin-DM ops routing, and AntigravityNativeToolAbort user replies."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from bot.antigravity import AntigravityNativeToolAbort
from bot.config.manager import ConfigManager
from bot.core.models import ProviderResponse
from bot.main import GroupBotOrchestrator
from bot.media.processor import MediaProcessor
from bot.memory.db import MemoryStore
from bot.security.sanitizer import SecretSanitizer
from bot.telegram.handlers import IncomingMessage
from bot.telegram.heuristics import ReplyAction

REPO_ROOT = Path(__file__).resolve().parents[1]
ADMIN_ID = 382962362
FRIEND_ID = 555666777
GROUP_ID = -1003845062562


class _SpyProvider:
    def __init__(self, *, side_effect: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.side_effect = side_effect

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
                "system_instructions": system_instructions,
                "tool_profile": tool_profile,
                "mode": mode,
                "cwd": cwd,
            }
        )
        if self.side_effect is not None:
            raise self.side_effect
        return ProviderResponse(text="ok", model=model or "test")

    def close(self) -> None:
        pass


class _DummyUserbot:
    def __init__(self, *, send_error: Exception | None = None) -> None:
        self.bot_id = 777000
        self.bot_username = "jeff_bot"
        self.client = MagicMock()
        self.replies: list[dict] = []
        self.send_error = send_error

    def register_message_handler(self, handler) -> None:
        pass

    async def send_reply(self, chat_id, text, reply_to_msg_id=None, simulate_typing=True):
        if self.send_error is not None:
            raise self.send_error
        self.replies.append(
            {
                "chat_id": chat_id,
                "text": text,
                "reply_to_msg_id": reply_to_msg_id,
                "simulate_typing": simulate_typing,
            }
        )
        return {"result": {"message_id": len(self.replies) + 100}}

    async def send_typing_action(self, chat_id, action="typing"):
        pass

    async def send_reaction(self, chat_id, msg_id, emoji):
        return True

    async def stop(self) -> None:
        pass


def _make_orchestrator_deps(
    tmp_path: Path,
    *,
    ops_default_mode: str = "on",
    system_prompt_text: str | None = None,
) -> tuple[ConfigManager, MemoryStore, MediaProcessor]:
    workspace = tmp_path / "ops_ws"
    workspace.mkdir(exist_ok=True)
    bot_yaml = tmp_path / "bot.yaml"
    bot_yaml.write_text(
        yaml.dump(
            {
                "telegram": {
                    "api_id": 123456,
                    "api_hash": "0123456789abcdef0123456789abcdef",
                    "allowed_chat_ids": [GROUP_ID],
                    "allowed_dm_user_ids": [FRIEND_ID],
                    "admin_user_ids": [ADMIN_ID],
                },
                "behavior": {"bot_names": ["Jeff"], "typing_simulation": False},
                "media": {"temp_dir": str(tmp_path / "media_tmp")},
                "antigravity": {
                    "data_dir": str(tmp_path / "agy_data"),
                    "ops_mode": "auto",
                    "ops_cwd": str(workspace),
                    "ops_tools_default_mode": ops_default_mode,
                },
            }
        ),
        encoding="utf-8",
    )
    if system_prompt_text is None:
        system_prompt_text = (REPO_ROOT / "config" / "system_prompt.md").read_text(encoding="utf-8")
    (tmp_path / "system_prompt.md").write_text(system_prompt_text, encoding="utf-8")
    cfg_mgr = ConfigManager(config_path=bot_yaml)
    store = MemoryStore(db_path=str(tmp_path / "mem.sqlite"))
    media_proc = MediaProcessor(temp_dir_base=str(tmp_path / "media_tmp"))
    return cfg_mgr, store, media_proc


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


@pytest.mark.asyncio
async def test_system_prompt_and_context_do_not_advertise_ops_when_disabled(tmp_path: Path) -> None:
    cfg_mgr, store, media_proc = _make_orchestrator_deps(tmp_path)
    await store.init_db()
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=_SpyProvider(),
        userbot=_DummyUserbot(),
        debounce_seconds=0.0,
    )
    try:
        sys_prompt = (REPO_ROOT / "config" / "system_prompt.md").read_text(encoding="utf-8")
        sys_lower = sys_prompt.lower()
        assert "по умолчанию), у тебя есть доступ" not in sys_lower
        assert "выполнение команд" not in sys_lower
        assert "чтение/запись/редактирование файлов" not in sys_lower
        assert "субагенты" not in sys_lower

        incoming = _msg(msg_id=1, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="Jeff привет", is_mention=True)
        chat_ctx = orch.build_context_prompt(
            incoming=incoming,
            recent_history=[],
            facts=[],
            friend_profile=None,
            ops_tools_enabled=False,
        )
        combined_chat = f"{sys_prompt}\n\n{chat_ctx}".lower()
        assert "расширенный режим инструментов" not in combined_chat
        assert "выполнение команд" not in combined_chat
        assert "субагенты" not in combined_chat

        ops_ctx = orch.build_context_prompt(
            incoming=incoming,
            recent_history=[],
            facts=[],
            friend_profile=None,
            ops_tools_enabled=True,
        )
        assert "РАСШИРЕННЫЙ РЕЖИМ ИНСТРУМЕНТОВ (ops, без root)" in ops_ctx
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_needs_ops_tools_media_text_and_admin_dm_ops_retention(tmp_path: Path) -> None:
    assert GroupBotOrchestrator._needs_ops_tools("просто болтаем") is False
    assert GroupBotOrchestrator._needs_ops_tools(
        "слушай сюда",
        media_text="проверь статус сервиса и логи",
    ) is True

    cfg_mgr, store, media_proc = _make_orchestrator_deps(tmp_path, ops_default_mode="on")
    await store.init_db()
    spy = _SpyProvider()
    ub = _DummyUserbot()
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=spy,
        userbot=ub,
        debounce_seconds=0.0,
    )
    try:
        # 1. Admin private DM retains ops even for short conversational follow-ups
        admin_dm = _msg(
            msg_id=10,
            chat_id=ADMIN_ID,
            sender_id=ADMIN_ID,
            text="что там?",
            is_private=True,
        )
        await orch._complete_reply_turn(
            chat_id=ADMIN_ID,
            bot_cfg=cfg_mgr.bot,
            state={"last_chat_msg_time": 0.0, "last_bot_msg_time": 0.0},
            now=100.0,
            decision=ReplyAction.MUST_REPLY,
            primary=admin_dm,
            messages=[admin_dm],
            combined_text="что там?",
            media_target=None,
            is_reply_media=False,
            media_summary=None,
            media_text=None,
            raw_media_text=None,
            attachments=[],
            is_voice_target=False,
        )
        assert spy.calls[-1]["tool_profile"] == "ops"
        assert "РАСШИРЕННЫЙ РЕЖИМ ИНСТРУМЕНТОВ" in spy.calls[-1]["prompt"]

        # 2. Group banter stays on cheap chat profile
        grp_banter = _msg(
            msg_id=11,
            chat_id=GROUP_ID,
            sender_id=FRIEND_ID,
            text="Jeff привет",
            is_mention=True,
        )
        await orch._complete_reply_turn(
            chat_id=GROUP_ID,
            bot_cfg=cfg_mgr.bot,
            state={"last_chat_msg_time": 0.0, "last_bot_msg_time": 0.0},
            now=101.0,
            decision=ReplyAction.MUST_REPLY,
            primary=grp_banter,
            messages=[grp_banter],
            combined_text="Jeff привет",
            media_target=None,
            is_reply_media=False,
            media_summary=None,
            media_text=None,
            raw_media_text=None,
            attachments=[],
            is_voice_target=False,
        )
        assert spy.calls[-1]["tool_profile"] == "chat"
        assert "РАСШИРЕННЫЙ РЕЖИМ ИНСТРУМЕНТОВ" not in spy.calls[-1]["prompt"]

        # 3. Group voice/attachment STT media_text triggers ops when mode is on
        grp_voice = _msg(
            msg_id=12,
            chat_id=GROUP_ID,
            sender_id=FRIEND_ID,
            text="Jeff",
            is_mention=True,
        )
        await orch._complete_reply_turn(
            chat_id=GROUP_ID,
            bot_cfg=cfg_mgr.bot,
            state={"last_chat_msg_time": 0.0, "last_bot_msg_time": 0.0},
            now=102.0,
            decision=ReplyAction.MUST_REPLY,
            primary=grp_voice,
            messages=[grp_voice],
            combined_text="Jeff",
            media_target=grp_voice,
            is_reply_media=False,
            media_summary="голосовое сообщение",
            media_text=None,
            raw_media_text="проверь логи на сервере",
            attachments=[],
            is_voice_target=True,
        )
        assert spy.calls[-1]["tool_profile"] == "ops"

        # 4. /ops off strictly enforces chat even for admin DM
        orch._ops_tools_mode = "off"
        await orch._complete_reply_turn(
            chat_id=ADMIN_ID,
            bot_cfg=cfg_mgr.bot,
            state={"last_chat_msg_time": 0.0, "last_bot_msg_time": 0.0},
            now=103.0,
            decision=ReplyAction.MUST_REPLY,
            primary=admin_dm,
            messages=[admin_dm],
            combined_text="что там?",
            media_target=None,
            is_reply_media=False,
            media_summary=None,
            media_text=None,
            raw_media_text=None,
            attachments=[],
            is_voice_target=False,
        )
        assert spy.calls[-1]["tool_profile"] == "chat"
        assert "РАСШИРЕННЫЙ РЕЖИМ ИНСТРУМЕНТОВ" not in spy.calls[-1]["prompt"]
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_native_tool_abort_notifies_direct_and_admin_dm_without_duplicate_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg_mgr, store, media_proc = _make_orchestrator_deps(tmp_path, ops_default_mode="on")
    await store.init_db()
    secret_token = "999888777:AAFakeBotTokenForSanitizerCheck1234"
    sanitizer = SecretSanitizer(extra_secrets=[secret_token], load_environ=False)

    # Case 1: stream.py already logged -> _native_tool_abort_logged=True -> NO duplicate WARNING in main.py
    logged_abort = AntigravityNativeToolAbort.from_context(
        tool_profile="chat",
        source="step_update",
        step_type="tool",
        tool_name="read_file",
    )
    setattr(logged_abort, "_native_tool_abort_logged", True)

    spy = _SpyProvider(side_effect=logged_abort)
    ub = _DummyUserbot()
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=spy,
        userbot=ub,
        sanitizer=sanitizer,
        debounce_seconds=0.0,
    )
    try:
        # 1a. Group MUST_REPLY -> receives short sanitized notice, zero duplicate WARNING
        direct_grp = _msg(
            msg_id=20,
            chat_id=GROUP_ID,
            sender_id=FRIEND_ID,
            text="Jeff привет",
            is_mention=True,
        )
        with caplog.at_level(logging.WARNING, logger="bot.main"):
            res = await orch._complete_reply_turn(
                chat_id=GROUP_ID,
                bot_cfg=cfg_mgr.bot,
                state={"last_chat_msg_time": 0.0, "last_bot_msg_time": 0.0},
                now=200.0,
                decision=ReplyAction.MUST_REPLY,
                primary=direct_grp,
                messages=[direct_grp],
                combined_text="Jeff привет",
                media_target=None,
                is_reply_media=False,
                media_summary=None,
                media_text=None,
                raw_media_text=None,
                attachments=[],
                is_voice_target=False,
            )
        assert res is None
        assert len(ub.replies) == 1
        assert ub.replies[0]["chat_id"] == GROUP_ID
        assert ub.replies[0]["reply_to_msg_id"] == 20
        assert "profile=chat" in ub.replies[0]["text"]
        assert "защита инструментов" in ub.replies[0]["text"]
        assert secret_token not in ub.replies[0]["text"]
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

        # 1b. Group SPONTANEOUS_REPLY -> silent (no chat spam)
        spont_grp = _msg(
            msg_id=21,
            chat_id=GROUP_ID,
            sender_id=FRIEND_ID,
            text="ну такое",
            is_mention=False,
        )
        res_spont = await orch._complete_reply_turn(
            chat_id=GROUP_ID,
            bot_cfg=cfg_mgr.bot,
            state={"last_chat_msg_time": 0.0, "last_bot_msg_time": 0.0},
            now=201.0,
            decision=ReplyAction.SPONTANEOUS_REPLY,
            primary=spont_grp,
            messages=[spont_grp],
            combined_text="ну такое",
            media_target=None,
            is_reply_media=False,
            media_summary=None,
            media_text=None,
            raw_media_text=None,
            attachments=[],
            is_voice_target=False,
        )
        assert res_spont is None
        assert len(ub.replies) == 1

        # 1c. Unlogged abort (e.g., ACP or mock) in admin DM -> 1 clean WARNING + admin DM notice
        caplog.clear()
        unlogged_abort = AntigravityNativeToolAbort.from_context(
            tool_profile="ops",
            source="acp",
            step_type="tool_call",
            tool_name="call_mcp_tool",
        )
        spy.side_effect = unlogged_abort
        admin_dm = _msg(
            msg_id=22,
            chat_id=ADMIN_ID,
            sender_id=ADMIN_ID,
            text="проверь",
            is_private=True,
        )
        with caplog.at_level(logging.WARNING, logger="bot.main"):
            res_dm = await orch._complete_reply_turn(
                chat_id=ADMIN_ID,
                bot_cfg=cfg_mgr.bot,
                state={"last_chat_msg_time": 0.0, "last_bot_msg_time": 0.0},
                now=202.0,
                decision=ReplyAction.MUST_REPLY,
                primary=admin_dm,
                messages=[admin_dm],
                combined_text="проверь",
                media_target=None,
                is_reply_media=False,
                media_summary=None,
                media_text=None,
                raw_media_text=None,
                attachments=[],
                is_voice_target=False,
            )
        assert res_dm is None
        assert len(ub.replies) == 2
        assert ub.replies[1]["chat_id"] == ADMIN_ID
        assert "profile=ops" in ub.replies[1]["text"]
        warn_recs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn_recs) == 1
        assert warn_recs[0].exc_info is None

        # 1d. _debounced_flush fallback for admin DM sends notice & skips duplicate warning if already logged
        caplog.clear()
        with patch.object(orch, "_process_buffered_messages", new=AsyncMock(side_effect=logged_abort)):
            with caplog.at_level(logging.WARNING, logger="bot.main"):
                await orch._debounced_flush(ADMIN_ID, 0.0)
        assert len(ub.replies) == 3
        assert ub.replies[2]["chat_id"] == ADMIN_ID
        assert "profile=chat" in ub.replies[2]["text"]
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

        # 1e. Notification failure does not crash orchestrator
        ub.send_error = RuntimeError(f"Telegram failed with {secret_token}")
        await orch._complete_reply_turn(
            chat_id=ADMIN_ID,
            bot_cfg=cfg_mgr.bot,
            state={"last_chat_msg_time": 0.0, "last_bot_msg_time": 0.0},
            now=203.0,
            decision=ReplyAction.MUST_REPLY,
            primary=admin_dm,
            messages=[admin_dm],
            combined_text="проверь",
            media_target=None,
            is_reply_media=False,
            media_summary=None,
            media_text=None,
            raw_media_text=None,
            attachments=[],
            is_voice_target=False,
        )
    finally:
        await orch.stop()
