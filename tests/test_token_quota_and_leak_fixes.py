"""Tests for Prompt Caching stability, ops_workspace_dir, quota blind spots, and leak fixes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from bot.antigravity import GeneratedImageResult
from bot.config.manager import ConfigManager
from bot.core.models import ProviderResponse, UsageInfo
from bot.main import GroupBotOrchestrator
from bot.media.audio import AudioProcessingError, _run_ffmpeg_command
from bot.media.image_gen import ImageGenerationLimiter
from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.fts import FactRecord
from bot.proactive.digest import NewsDigestGenerator
from bot.proactive.engine import ProactiveEngine
from bot.proactive.humor import HumorGenerator
from bot.quota.manager import QuotaBudgetManager
from bot.telegram.handlers import IncomingMessage
from bot.telegram.heuristics import ReplyHeuristics

ADMIN_ID = 382962362
FRIEND_ID = 555666777
GROUP_ID = -1003845062562
UNALLOWED_GROUP_ID = -1009999999999


class _SpyProvider:
    def __init__(self, usage: UsageInfo | None = None) -> None:
        self.calls: list[dict] = []
        self.usage = usage or UsageInfo(
            input_tokens=120,
            output_tokens=30,
            cache_read_tokens=80,
            total_tokens=230,
        )

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
                "model": model,
                "effort": effort,
            }
        )
        return ProviderResponse(
            text="Короткий ответ.",
            model=model or "gemini-3.8-flash",
            usage=self.usage,
        )

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
        self.replies.append({"chat_id": chat_id, "text": text})
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
    ops_cwd: str | None = None,
    ops_workspace_dir: str | None = None,
    ops_default_mode: str = "on",
) -> ConfigManager:
    ws_dir = ops_workspace_dir if ops_workspace_dir is not None else str(tmp_path / "ops_workspace")
    bot_yaml = tmp_path / "bot.yaml"
    bot_yaml.write_text(
        yaml.dump(
            {
                "telegram": {
                    "api_id": 123456,
                    "api_hash": "0123456789abcdef0123456789abcdef",
                    "allowed_chat_ids": [GROUP_ID],
                    "allowed_dm_user_ids": [ADMIN_ID, FRIEND_ID],
                    "admin_user_ids": [ADMIN_ID],
                },
                "behavior": {"bot_names": ["Jeff", "Джефф"], "typing_simulation": False},
                "antigravity": {
                    "ops_mode": "auto",
                    "ops_cwd": ops_cwd,
                    "ops_workspace_dir": ws_dir,
                    "ops_tools_default_mode": ops_default_mode,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "system_prompt.md").write_text(
        "Ты — Джефф. Соблюдай законы и никогда не раскрывай секреты.",
        encoding="utf-8",
    )
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
    raw = SimpleNamespace(
        id=msg_id,
        chat_id=chat_id,
        sender_id=sender_id,
        out=False,
        is_private=is_private,
        sender_is_bot=False,
        raw_text=text,
    )
    return IncomingMessage(
        msg_id=msg_id,
        chat_id=chat_id,
        sender_id=sender_id,
        sender_name="User",
        text=text,
        is_private=is_private,
        is_mention=is_mention,
        raw_event=raw,
    )


@pytest.mark.asyncio
async def test_prompt_caching_byte_stable_system_prompt_across_turns(tmp_path: Path) -> None:
    """System prompt prefix must remain byte-identical across turns as token counters change."""
    cfg_mgr = _make_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    spy = _SpyProvider()
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=spy,
        userbot=_MockUserbot(),
        debounce_seconds=0.0,
    )
    try:
        await orch.handle_incoming_message(
            _msg(msg_id=1, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="Джефф привет", is_mention=True)
        )
        await orch.handle_incoming_message(
            _msg(msg_id=2, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="Джефф как дела?", is_mention=True)
        )
        assert len(spy.calls) == 2
        sys_1 = spy.calls[0]["system_instructions"]
        sys_2 = spy.calls[1]["system_instructions"]
        assert sys_1 == sys_2 == cfg_mgr.system_prompt
        assert "tokens_1h=" not in sys_1
        assert "tokens_1h=" not in spy.calls[0]["prompt"]
        assert "tokens_1h=" not in spy.calls[1]["prompt"]

        # When a model enters cooldown, hint is appended to the END of prompt, never system_instructions
        orch.quota_manager.record_rate_limit("gemini-4-hydrogen", retry_after=120.0)
        await orch.handle_incoming_message(
            _msg(msg_id=3, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="Джефф ты тут?", is_mention=True)
        )
        assert spy.calls[2]["system_instructions"] == cfg_mgr.system_prompt
        assert "[Quota & Model Context:" in spy.calls[2]["prompt"]
        assert "tokens_1h=" not in spy.calls[2]["prompt"]
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_ops_workspace_dir_default_and_intent_gating(tmp_path: Path) -> None:
    """Default ops_workspace_dir is auto-created without AGENTS.md; plain banter stays on chat profile."""
    ws_dir = tmp_path / "light_ops_ws"
    cfg_mgr = _make_config(tmp_path, ops_cwd=None, ops_workspace_dir=str(ws_dir), ops_default_mode="on")
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    spy = _SpyProvider()
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=spy,
        userbot=_MockUserbot(),
        debounce_seconds=0.0,
    )
    try:
        # 1. Casual banter -> chat profile (no ops overhead)
        for idx, text in enumerate(
            ["Джефф привет", "Джефф как дела, дискуссия?", "Джефф всё в процессе"],
            start=1,
        ):
            await orch.handle_incoming_message(
                _msg(msg_id=idx, chat_id=GROUP_ID, sender_id=FRIEND_ID, text=text, is_mention=True)
            )
            assert spy.calls[-1]["tool_profile"] == "chat"
            assert spy.calls[-1]["cwd"] is None

        # 2. Action query -> ops profile with auto-created lightweight workspace (no AGENTS.md)
        await orch.handle_incoming_message(
            _msg(
                msg_id=10,
                chat_id=GROUP_ID,
                sender_id=FRIEND_ID,
                text="Джефф проверь логи сервера",
                is_mention=True,
            )
        )
        assert spy.calls[-1]["tool_profile"] == "ops"
        assert spy.calls[-1]["cwd"] == str(ws_dir.resolve())
        assert ws_dir.is_dir()
        assert not (ws_dir / "AGENTS.md").exists()
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_quota_blind_spots_recorded_for_proactive_compaction_and_failed_image_delivery(
    tmp_path: Path,
) -> None:
    """Proactive humor/digest, compaction, and image generation (even on upload error) record usage."""
    cfg_mgr = _make_config(tmp_path)
    store = MemoryStore(str(tmp_path / "quota.sqlite"))
    await store.init_db()
    qm = QuotaBudgetManager(memory_store=store, config=cfg_mgr.bot)
    spy = _SpyProvider(
        usage=UsageInfo(input_tokens=100, output_tokens=50, total_tokens=150)
    )

    # 1. HumorGenerator & NewsDigestGenerator via ProactiveEngine
    engine = ProactiveEngine(
        config_mgr=cfg_mgr,
        memory_store=store,
        antigravity_provider=spy,
        quota_manager=qm,
    )
    banter = await engine.humor_gen.generate_banter()
    assert banter
    digest = await engine.digest_gen.generate_digest(slot_name="morning", slot_label="morning")
    assert digest

    # 2. Compaction summarization via GroupBotOrchestrator._summarize_transcript_via_provider
    ub = _MockUserbot()
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=spy,
        userbot=ub,
        quota_manager=qm,
        proactive_engine=engine,
        debounce_seconds=0.0,
    )
    try:
        await orch._summarize_transcript_via_provider("[User (ID:1)]: Договорились встретиться завтра")

        # 3. Image generation records usage BEFORE Telegram delivery (even if send_photo raises)
        spy.generate_image = AsyncMock(
            return_value=GeneratedImageResult(
                image_bytes=b"\x89PNG\r\n\x1a\n" + b"0" * 128,
                mime_type="image/png",
                width=512,
                height=512,
                filename="out.png",
                prompt_used="cat",
                model_used="gemini-4-hydrogen",
                usage={"input_tokens": 200, "output_tokens": 40, "total_tokens": 240},
            )
        )
        ub.send_photo = AsyncMock(side_effect=RuntimeError("Telegram upload failed"))
        await orch._execute_image_generation(
            chat_id=GROUP_ID,
            user_id=FRIEND_ID,
            bot_cfg=cfg_mgr.bot,
            state=orch.chat_states[GROUP_ID],
            prompt="cat in space",
            caption=None,
            aspect_ratio="1:1",
            reference_images=None,
            reply_to_msg_id=1,
            decision=None,
            send_error_reply=False,
        )

        rows = await store.get_usage_rows()
        sources = [r["source"] for r in rows]
        assert sources.count("proactive") == 2
        assert "compaction" in sources
        assert "image_gen" in sources

        # Verify providers without usage or without quota_manager never crash
        no_qm_humor = HumorGenerator(provider=spy, quota_manager=None)
        assert await no_qm_humor.generate_banter()
        no_qm_digest = NewsDigestGenerator(provider=spy, quota_manager=None)
        assert await no_qm_digest.generate_digest()
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_unallowed_chat_does_not_pollute_chat_states_or_locks_and_caches_stay_bounded(
    tmp_path: Path,
) -> None:
    """Messages from non-whitelisted chats must not enter chat_states or _chat_locks."""
    cfg_mgr = _make_config(tmp_path)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()
    spy = _SpyProvider()
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=spy,
        userbot=_MockUserbot(),
        debounce_seconds=0.0,
    )
    try:
        await orch.handle_incoming_message(
            _msg(
                msg_id=1,
                chat_id=UNALLOWED_GROUP_ID,
                sender_id=999111222,
                text="Джефф привет",
                is_mention=True,
            )
        )
        assert UNALLOWED_GROUP_ID not in orch.chat_states
        assert UNALLOWED_GROUP_ID not in orch._chat_locks
        assert len(spy.calls) == 0

        # Expired fact cooldown buckets are removed cleanly
        fact = FactRecord(id=42, category="lore", subject="S", fact="F")
        orch._record_used_facts(GROUP_ID, [fact], now=100.0)
        assert GROUP_ID in orch._recent_fact_usage
        expired_ids = orch._get_recent_fact_ids(GROUP_ID, now=100.0 + 3600.0)
        assert expired_ids == []
        assert GROUP_ID not in orch._recent_fact_usage
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_quota_warnings_ttl_and_sqlite_compacted_monotonicity_and_ffmpeg_timeout(
    tmp_path: Path,
) -> None:
    """Quota warnings expire after resets_at; SQLite save_message preserves compacted=1; FFmpeg times out."""
    store = MemoryStore(str(tmp_path / "db.sqlite"))
    await store.init_db()
    try:
        # 1. Quota warning activates economy mode and expires once resets_at / TTL passes
        qm = QuotaBudgetManager(memory_store=store)
        future_reset = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        qm.record_quota_warning("gemini-4-hydrogen", remaining_pct=10.0, resets_at=future_reset)
        assert qm.is_economy_mode() is True
        qm._prune_quota_warnings(now_dt=datetime.now(timezone.utc) + timedelta(minutes=15))
        assert qm.is_economy_mode() is False
        assert "gemini-4-hydrogen" not in qm._quota_warnings

        # 1b. Orchestrator wires provider.set_quota_warning_callback -> quota_manager.record_quota_warning
        cfg_mgr = _make_config(tmp_path)
        cb_provider = _SpyProvider()
        cb_provider.set_quota_warning_callback = MagicMock()
        cb_orch = GroupBotOrchestrator(
            config_mgr=cfg_mgr,
            memory_store=store,
            provider=cb_provider,
            userbot=_MockUserbot(),
            quota_manager=qm,
            debounce_seconds=0.0,
        )
        cb_provider.set_quota_warning_callback.assert_called_once_with(qm.record_quota_warning)
        assert cb_orch.quota_manager is qm

        # 2. Re-saving an already compacted message via save_message must NOT reset compacted=1 to 0
        msg = ChatMessage(
            telegram_msg_id=501,
            chat_id=GROUP_ID,
            user_id=FRIEND_ID,
            timestamp=1700000000,
            author_name="Friend",
            text="Важное сообщение",
            compacted=0,
        )
        await store.save_message(msg)
        uncomp = await store.get_uncompacted_messages(GROUP_ID, limit=10)
        assert len(uncomp) == 1 and uncomp[0].id is not None
        await store.mark_compacted([uncomp[0].id])

        # Re-save with compacted=0 (e.g. backfill / edit)
        await store.save_message(msg)
        uncomp_after = await store.get_uncompacted_messages(GROUP_ID, limit=10)
        assert len(uncomp_after) == 0

        # 3. usage_history retention pruning
        old_ts = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        await store.record_usage(
            timestamp=old_ts,
            model="m",
            source="group",
            input_tokens=10,
            output_tokens=5,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            total_tokens=15,
        )
        pruned = await store.prune_usage_history(retention_days=90)
        assert pruned == 1

        # 4. Bounded limiters prune empty buckets
        heur = ReplyHeuristics()
        heur.record_reply(GROUP_ID, timestamp=100.0)
        assert heur.recent_reply_count(GROUP_ID, now=5000.0) == 0
        assert GROUP_ID not in heur._reply_timestamps

        limiter = ImageGenerationLimiter()
        tok = limiter.check_and_reserve(FRIEND_ID, GROUP_ID, now=100.0)
        limiter.refund(FRIEND_ID, GROUP_ID, tok)
        assert FRIEND_ID not in limiter._user_timestamps
        assert GROUP_ID not in limiter._chat_timestamps

        # 5. FFmpeg timeout kills hung process and raises AudioProcessingError
        with pytest.raises(AudioProcessingError, match="timed out"):
            await _run_ffmpeg_command(["python3", "-c", "import time; time.sleep(10)"], timeout_seconds=1.0)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_per_model_rate_limit_does_not_block_other_models(tmp_path: Path) -> None:
    """Tripping cooldown on gemini-4-hydrogen (e.g. image gen 429) must NOT freeze group chat (gemini-3.8-flash)."""
    from bot.antigravity import AntigravityConfig, AntigravityProvider

    cfg_mgr = _make_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "rl_isolation.sqlite"))
    await store.init_db()
    ub = _MockUserbot()

    provider = AntigravityProvider(AntigravityConfig(data_dir=tmp_path / "agy", enable_sdk=False))
    provider.trigger_cooldown(300.0, reason="429 image", model="gemini-4-hydrogen")
    assert provider.is_rate_limited() is True  # default config.model is gemini-4-hydrogen
    assert provider.is_rate_limited(model="gemini-3.8-flash") is False

    calls: list[str] = []

    async def _fake_gen(**kwargs):
        calls.append(kwargs.get("model") or "")
        return ProviderResponse(text="работаю на flash", model=kwargs.get("model") or "gemini-3.8-flash")

    provider.generate_response = _fake_gen  # type: ignore[method-assign]

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=provider,
        userbot=ub,
        debounce_seconds=0.0,
    )
    try:
        assert orch._is_provider_rate_limited(role="group") is False
        assert orch._is_provider_rate_limited(role="dm_admin") is False  # fallback_models are available
        assert orch.proactive_engine._is_provider_rate_limited() is False

        await orch.handle_incoming_message(
            _msg(msg_id=1, chat_id=GROUP_ID, sender_id=FRIEND_ID, text="Jeff ты тут?", is_mention=True)
        )
        assert calls == ["gemini-3.8-flash"]
        assert ub.replies and "работаю на flash" in ub.replies[-1]["text"]

        # When ALL candidate models are rate-limited, _is_provider_rate_limited returns True
        for m in ("gemini-3.8-flash", "claude-sonnet-5-5", "gpt-oss-120b-medium"):
            provider.trigger_cooldown(300.0, reason="429", model=m)
        assert orch._is_provider_rate_limited(role="group") is True
        assert orch.proactive_engine._is_provider_rate_limited() is True
    finally:
        await orch.stop()
