"""Tests for token-leak optimizations, sibling-bot ID isolation, compaction empty-JSON handling, and STT idle unload."""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import patch

import pytest

from bot.config.models import BotConfig
from bot.main import GroupBotOrchestrator
from bot.media.image_gen import parse_imagine_command
from bot.media.stt import STTTranscriber
from bot.memory.compaction import (
    CompactionEngine,
    canonicalize_subject,
    is_explicit_empty_fact_response,
)
from bot.memory.db import ChatMessage, MemoryStore
from bot.telegram.bot_api_client import TelegramBotAPIClient
from bot.telegram.heuristics import ReplyAction, ReplyHeuristics


@dataclass
class _DummyEv:
    chat_id: int = -1003845062562
    sender_id: int = 9218923960
    out: bool = False
    is_reply: bool = False
    reply_to_sender_id: Optional[int] = None
    raw_text: Optional[str] = "Рептилоиды снова облучают роутер"
    entities: Optional[List[Any]] = None
    is_private: bool = False
    sender_is_bot: bool = False
    sender_username: Optional[str] = None


def test_ignored_sender_ids_blocks_mtproto_sibling_in_bot_api_and_heuristics() -> None:
    client = TelegramBotAPIClient(
        bot_token="123456:AAFakeTokenForUnitTestOnly1234567890",
        allowed_chat_ids=[-1003845062562],
        ignored_sender_ids=[9218923960],
    )
    client.bot_id = 8710978944

    # Message from sibling MTProto userbot (is_bot=False, username=None, id=9218923960)
    msg_dict = {
        "message_id": 501,
        "date": 1700000000,
        "chat": {"id": -1003845062562, "type": "supergroup"},
        "from": {"id": 9218923960, "is_bot": False, "first_name": "Потужнiч"},
        "text": "піструнькиня, приём",
    }
    assert client._convert_msg_dict(msg_dict) is None

    # Hot-reload sync_access_lists updates ignored_sender_ids
    client.sync_access_lists(ignored_sender_ids=[111222333])
    assert client.ignored_sender_ids == [111222333]
    assert client._convert_msg_dict(msg_dict) is not None

    heuristics = ReplyHeuristics()
    ev = _DummyEv(sender_id=9218923960, raw_text="піструнькиня привет")
    assert (
        heuristics.should_reply(
            event=ev,
            bot_id=8710978944,
            bot_names=["піструнькиня"],
            allowed_chat_ids=[-1003845062562],
            ignored_sender_ids=[9218923960],
        )
        == ReplyAction.IGNORE
    )


def test_slash_commands_addressed_to_other_bot_when_username_is_none() -> None:
    res = parse_imagine_command("/imagine@hermes_kuk7yi6kfm3lcdfa_bot кибер-кот", bot_username=None)
    assert res.is_command is False
    assert res.addressed_to_other_bot is True

    split_res = GroupBotOrchestrator._split_slash_command(
        "/quota@hermes_kuk7yi6kfm3lcdfa_bot",
        bot_username=None,
    )
    assert split_res == ("/quota", "", True)


def test_active_window_probability_override_and_skip_attempt_cooldown() -> None:
    heuristics = ReplyHeuristics(default_max_per_hour=5, default_active_window_cooldown_seconds=25.0)
    ev = _DummyEv(sender_id=382962362, raw_text="Содержательная реплика в активном окне")

    # Custom active_window_probability=0.20 overrides the legacy 0.45 boost
    with patch("random.random", return_value=0.30):
        act = heuristics.should_reply(
            event=ev,
            bot_id=8710978944,
            bot_names=["піструнькиня"],
            allowed_chat_ids=[-1003845062562],
            spontaneous_enabled=True,
            time_since_last_bot_msg=35.0,
            probability=0.08,
            active_conversation_window_seconds=60.0,
            active_window_probability=0.20,
            now=1000.0,
        )
        assert act == ReplyAction.IGNORE

    # When a spontaneous attempt fires (even if LLM later returns [SKIP] so time_since_last_bot_msg stays 37s),
    # _last_spontaneous_attempt blocks another immediate LLM call 2 seconds later.
    with patch("random.random", return_value=0.05):
        first = heuristics.should_reply(
            event=ev,
            bot_id=8710978944,
            bot_names=["піструнькиня"],
            allowed_chat_ids=[-1003845062562],
            spontaneous_enabled=True,
            time_since_last_bot_msg=35.0,
            probability=0.08,
            active_conversation_window_seconds=60.0,
            active_window_probability=0.20,
            now=1000.0,
        )
        assert first == ReplyAction.SPONTANEOUS_REPLY

        second_too_soon = heuristics.should_reply(
            event=ev,
            bot_id=8710978944,
            bot_names=["піструнькиня"],
            allowed_chat_ids=[-1003845062562],
            spontaneous_enabled=True,
            time_since_last_bot_msg=37.0,
            probability=0.08,
            active_conversation_window_seconds=60.0,
            active_window_probability=0.20,
            now=1002.0,
        )
        assert second_too_soon == ReplyAction.IGNORE


@pytest.mark.asyncio
async def test_compaction_explicit_empty_json_skips_noisy_fallback_and_ignores_potuzhnych() -> None:
    assert is_explicit_empty_fact_response([]) is True
    assert is_explicit_empty_fact_response("[]") is True
    assert is_explicit_empty_fact_response("```json\n[]\n```") is True
    assert is_explicit_empty_fact_response('{"facts": []}') is True
    assert is_explicit_empty_fact_response("") is False
    assert is_explicit_empty_fact_response(None) is False

    # Sibling bot names rejected in canonicalize_subject
    assert canonicalize_subject("Потужнич")[0] is None
    assert canonicalize_subject("Потужнiч")[0] is None
    assert canonicalize_subject("potuzhnych")[0] is None

    store = MemoryStore(db_path=":memory:")
    await store.init_db()
    chat_id = -1003845062562
    for i in range(8):
        await store.save_message(
            ChatMessage(
                telegram_msg_id=100 + i,
                chat_id=chat_id,
                user_id=382962362,
                timestamp=1700000000 + i,
                author_name="Rodion",
                text=f"Обычное рабочее обсуждение без долгосрочных фактов номер {i}",
            )
        )

    engine = CompactionEngine(
        memory_store=store,
        summarize_fn=lambda _t: "```json\n[]\n```",
        threshold=8,
        batch_size=8,
        min_substantive_messages=6,
    )
    compacted = await engine.run_compaction_for_chat(chat_id)
    assert compacted == 8

    # Explicit [] from LLM must NOT pollute memory_facts via deterministic fallback
    conn = store._get_connection()
    count = conn.execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0]
    assert count == 0
    await store.close()


@pytest.mark.asyncio
async def test_stt_transcriber_auto_unloads_idle_model(tmp_path: Path) -> None:
    wav_file = tmp_path / "sample.wav"
    with wave.open(str(wav_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 8000)

    load_count = 0

    class _FakeWhisper:
        def transcribe(self, audio: Any, **kwargs: Any):
            return [SimpleNamespace(text="проверка голоса")], SimpleNamespace(
                language="ru", language_probability=0.99
            )

    def _factory(*args: Any, **kwargs: Any) -> _FakeWhisper:
        nonlocal load_count
        load_count += 1
        return _FakeWhisper()

    stt = STTTranscriber(
        enabled=True,
        idle_unload_seconds=0.08,
        model_factory=_factory,
    )
    try:
        res = await stt.transcribe(wav_file)
        assert res.text == "проверка голоса"
        assert load_count == 1
        assert stt._model is not None

        # Wait slightly past idle_unload_seconds -> model released automatically
        deadline = time.monotonic() + 1.0
        while stt._model is not None and time.monotonic() < deadline:
            time.sleep(0.03)
        assert stt._model is None
    finally:
        stt.close()


def test_config_models_accept_new_optimization_fields() -> None:
    cfg = BotConfig.model_validate(
        {
            "telegram": {
                "api_id": 123,
                "api_hash": "abc",
                "ignored_sender_ids": [9218923960],
            },
            "behavior": {
                "spontaneous_replies": {
                    "probability": 0.08,
                    "active_window_probability": 0.22,
                    "max_per_hour": 2,
                }
            },
            "memory": {
                "compaction_threshold": 220,
                "compaction_batch_size": 140,
                "compaction_min_substantive": 12,
            },
            "voice": {"stt": {"idle_unload_seconds": 300.0}},
        }
    )
    assert cfg.telegram.ignored_sender_ids == [9218923960]
    assert cfg.behavior.spontaneous_replies.active_window_probability == 0.22
    assert cfg.memory.compaction_min_substantive == 12
    assert cfg.voice.stt.idle_unload_seconds == 300.0
