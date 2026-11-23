"""Offline E2E tests for native image generation wiring in GroupBotOrchestrator."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
import yaml
from PIL import Image

from bot.antigravity import (
    ImagePolicyRefusalError,
    ImageTimeoutError,
)
from bot.config.manager import ConfigManager
from bot.core.models import GeneratedImageResult, MediaAttachment, ProviderResponse
from bot.main import GroupBotOrchestrator
from bot.media.processor import MediaProcessor
from bot.memory.db import MemoryStore
from bot.telegram.handlers import IncomingMessage

GROUP_CHAT_ID = -1003845062562
USER_ID = 111222333
ADMIN_USER_ID = 382962362


def _make_jpeg_bytes(width: int = 64, height: int = 64, color: tuple[int, int, int] = (30, 140, 220)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="JPEG")
    return buf.getvalue()


class _FakeBotClient:
    """In-memory Telegram Bot API stand-in recording photos, docs, replies, and chat actions."""

    def __init__(self, *, bot_id: int = 999000, bot_username: str = "jeff_test_bot") -> None:
        self.bot_id = bot_id
        self.bot_username = bot_username
        self.sent_replies: List[Dict[str, Any]] = []
        self.sent_photos: List[Dict[str, Any]] = []
        self.sent_documents: List[Dict[str, Any]] = []
        self.chat_actions: List[Dict[str, Any]] = []
        self.reactions: List[Dict[str, Any]] = []
        self.client = self  # downloader adapter
        self._msg_seq = 5000

    def register_message_handler(self, handler: Any) -> None:
        self._handler = handler

    def sync_access_lists(self, **kwargs: Any) -> None:
        self._synced = kwargs

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def download_media(self, message: Any, file: str) -> str:
        src_path = getattr(message, "file_path", None)
        if src_path:
            Path(file).write_bytes(Path(src_path).read_bytes())
            return file
        Path(file).write_bytes(_make_jpeg_bytes(32, 32))
        return file

    async def send_typing_action(self, chat_id: int, action: str = "typing") -> None:
        self.chat_actions.append({"chat_id": chat_id, "action": action})

    async def send_reply(
        self,
        chat_id: int,
        text: str,
        reply_to_msg_id: Optional[int] = None,
        simulate_typing: bool = True,
    ) -> Dict[str, Any]:
        self._msg_seq += 1
        rec = {
            "chat_id": chat_id,
            "text": text,
            "reply_to_msg_id": reply_to_msg_id,
            "message_id": self._msg_seq,
        }
        self.sent_replies.append(rec)
        return {"ok": True, "result": {"message_id": self._msg_seq}}

    async def send_photo(
        self,
        chat_id: int,
        photo: str | Path | bytes,
        *,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
        timeout: float = 90.0,
    ) -> int:
        self._msg_seq += 1
        self.sent_photos.append(
            {
                "chat_id": chat_id,
                "photo": photo,
                "caption": caption,
                "filename": filename,
                "mime_type": mime_type,
                "reply_to_message_id": reply_to_message_id,
                "message_id": self._msg_seq,
            }
        )
        return self._msg_seq

    async def send_document(
        self,
        chat_id: int,
        file_path: str | Path | bytes,
        caption: Optional[str] = None,
        reply_to_msg_id: Optional[int] = None,
        simulate_typing: bool = True,
        filename: Optional[str] = None,
        timeout: float = 120.0,
        *,
        mime_type: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._msg_seq += 1
        self.sent_documents.append(
            {
                "chat_id": chat_id,
                "file_path": file_path,
                "caption": caption,
                "filename": filename,
                "mime_type": mime_type,
                "reply_to_message_id": reply_to_message_id or reply_to_msg_id,
                "message_id": self._msg_seq,
            }
        )
        return {"ok": True, "result": {"message_id": self._msg_seq}}

    async def send_reaction(self, chat_id: int, msg_id: int, emoji: str) -> bool:
        self.reactions.append({"chat_id": chat_id, "msg_id": msg_id, "emoji": emoji})
        return True


class _FakeProvider:
    """Controllable provider double recording chat and image generation calls."""

    def __init__(
        self,
        *,
        chat_reply: str = "Ок!",
        image_result: Optional[GeneratedImageResult] = None,
        image_error: Optional[Exception] = None,
    ) -> None:
        self.chat_reply = chat_reply
        self.image_result = image_result or GeneratedImageResult(
            image_bytes=_make_jpeg_bytes(512, 512),
            mime_type="image/jpeg",
            filename="cat_space.jpg",
            width=512,
            height=512,
            prompt_used="cat in space",
            model_used="gemini-4-hydrogen",
            usage={
                "input_tokens": 120,
                "output_tokens": 45,
                "total_tokens": 165,
            },
        )
        self.image_error = image_error
        self.chat_calls: List[Dict[str, Any]] = []
        self.image_calls: List[Dict[str, Any]] = []

    def is_rate_limited(self, model: Optional[str] = None) -> bool:
        return False

    async def generate_response(self, **kwargs: Any) -> ProviderResponse:
        self.chat_calls.append(kwargs)
        return ProviderResponse(text=self.chat_reply, model="gemini-3.8-flash")

    async def generate_image(
        self,
        prompt: str,
        *,
        aspect_ratio: Optional[str] = None,
        reference_images: Optional[List[MediaAttachment]] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 120.0,
        output_dir: str | Path = "data/tmp/images",
        max_image_size_mb: float = 20.0,
        cleanup_artifacts: bool = True,
    ) -> GeneratedImageResult:
        self.image_calls.append(
            {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "reference_images": reference_images,
                "model": model,
                "timeout_seconds": timeout_seconds,
                "output_dir": output_dir,
                "max_image_size_mb": max_image_size_mb,
                "cleanup_artifacts": cleanup_artifacts,
            }
        )
        if self.image_error is not None:
            raise self.image_error
        return self.image_result

    def close(self) -> None:
        pass


def _make_config_manager(
    tmp_path: Path,
    *,
    image_overrides: Optional[Dict[str, Any]] = None,
    spontaneous_enabled: bool = False,
) -> ConfigManager:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    img_cfg: Dict[str, Any] = {
        "enabled": True,
        "allow_natural_language_trigger": True,
        "timeout_seconds": 60.0,
        "max_prompt_chars": 1000,
        "max_per_user_per_hour": 5,
        "user_cooldown_seconds": 30.0,
        "max_per_chat_per_hour": 15,
        "admin_bypass_limits": True,
        "output_dir": str(tmp_path / "img_out"),
        "max_image_size_mb": 20.0,
        "send_as_document_threshold_mb": 9.5,
        "model": None,
        "cleanup_after_send": True,
    }
    if image_overrides:
        img_cfg.update(image_overrides)

    bot_yaml = {
        "telegram": {
            "mode": "bot_api",
            "bot_token": "123456:TEST_OFFLINE_DUMMY_TOKEN_000000000",
            "api_id": 12345,
            "api_hash": "0123456789abcdef0123456789abcdef",
            "allowed_chat_ids": [GROUP_CHAT_ID],
            "allowed_dm_user_ids": [ADMIN_USER_ID],
            "admin_user_ids": [ADMIN_USER_ID],
            "ignored_sender_usernames": ["hermes_kuk7yi6kfm3lcdfa_bot"],
        },
        "behavior": {
            "bot_names": ["Джефф", "jeff"],
            "typing_simulation": False,
            "reactions_enabled": True,
            "spontaneous_replies": {
                "enabled": spontaneous_enabled,
                "min_interval_seconds": 3600,
                "probability": 0.0,
                "max_per_hour": 0,
                "active_conversation_window_seconds": 0,
            },
        },
        "proactive": {
            "enabled": False,
            "quiet_hours": {
                "enabled": True,
                "start": "00:00",
                "end": "23:59",
                "timezone": "UTC",
            },
        },
        "media": {
            "max_doc_size_mb": 10,
            "temp_dir": str(tmp_path / "media_tmp"),
        },
        "memory": {
            "db_path": str(tmp_path / "memory.sqlite"),
        },
        "voice": {
            "tts": {"enabled": False},
            "stt": {"enabled": False},
        },
        "image_generation": img_cfg,
    }
    (cfg_dir / "bot.yaml").write_text(yaml.safe_dump(bot_yaml, sort_keys=False), encoding="utf-8")
    (cfg_dir / "system_prompt.md").write_text("Ты — Джефф, участница чата.", encoding="utf-8")
    return ConfigManager(config_path=cfg_dir / "bot.yaml")


async def _build_orchestrator(
    tmp_path: Path,
    *,
    provider: Optional[_FakeProvider] = None,
    image_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[GroupBotOrchestrator, _FakeBotClient, _FakeProvider, MemoryStore, ConfigManager]:
    cfg_mgr = _make_config_manager(tmp_path, image_overrides=image_overrides)
    store = MemoryStore(db_path=str(tmp_path / "memory.sqlite"))
    await store.init_db()
    media_proc = MediaProcessor(
        max_doc_size_mb=10,
        temp_dir_base=str(tmp_path / "media_tmp"),
    )
    bot_client = _FakeBotClient()
    prov = provider or _FakeProvider()
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        media_processor=media_proc,
        provider=prov,
        userbot=bot_client,
        debounce_seconds=0.0,
    )
    # Force quiet hours ON to prove explicit /imagine bypasses quiet-hours / spontaneous gating
    orch.proactive_engine.is_in_quiet_hours = lambda: True  # type: ignore[method-assign]
    return orch, bot_client, prov, store, cfg_mgr


def _make_group_msg(
    text: str,
    *,
    msg_id: int = 101,
    sender_id: int = USER_ID,
    chat_id: int = GROUP_CHAT_ID,
    is_private: bool = False,
    is_mention: bool = False,
    reply_to_text: Optional[str] = None,
    reply_to_media_event: Any = None,
    raw_event: Any = None,
    media: Any = None,
) -> IncomingMessage:
    if raw_event is None:
        class _Raw:
            pass

        ev = _Raw()
        ev.id = msg_id
        ev.chat_id = chat_id
        ev.sender_id = sender_id
        ev.out = False
        ev.is_private = is_private
        ev.is_reply = bool(reply_to_text or reply_to_media_event)
        ev.is_reply_to_bot = False
        ev.raw_text = text
        ev.text = text
        ev.reply_to_text = reply_to_text
        ev.reply_to_media_event = reply_to_media_event
        ev.photo = bool(media)
        ev.media = bool(media)
        raw_event = ev

    return IncomingMessage(
        msg_id=msg_id,
        chat_id=chat_id,
        sender_id=sender_id,
        sender_name="Алексей",
        text=text,
        is_reply=bool(reply_to_text or reply_to_media_event),
        is_mention=is_mention,
        raw_event=raw_event,
        media=media,
        reply_to_text=reply_to_text,
        reply_to_media_event=reply_to_media_event,
        is_private=is_private,
    )


@pytest.mark.asyncio
async def test_imagine_and_img_in_group_bypass_heuristics_and_log_memory(tmp_path: Path) -> None:
    """`/imagine` and `/img` work in groups during quiet hours, send upload-photo & photo, and log to SQLite."""
    orch, client, prov, store, _ = await _build_orchestrator(
        tmp_path,
        image_overrides={"user_cooldown_seconds": 0.0},
    )
    try:
        # 1. Bare /imagine in group (no @mention, spontaneous disabled, quiet hours ON)
        msg1 = _make_group_msg("/imagine кот в скафандре на Марсе --ar 16:9", msg_id=201)
        res1 = await orch.handle_incoming_message(msg1)

        assert res1 is not None
        assert len(prov.image_calls) == 1
        assert prov.image_calls[0]["prompt"] == "кот в скафандре на Марсе"
        assert prov.image_calls[0]["aspect_ratio"] == "16:9"
        assert any(a["action"] == "upload-photo" for a in client.chat_actions)
        assert len(client.sent_photos) == 1
        assert client.sent_photos[0]["reply_to_message_id"] == 201
        assert client.sent_photos[0]["mime_type"] == "image/jpeg"
        assert client.sent_photos[0]["filename"] == "cat_space.jpg"
        assert len(client.sent_documents) == 0

        # Verify SQLite memory saved text summary with media_type="photo" (no binary)
        history = await store.get_recent_history(GROUP_CHAT_ID, limit=10)
        bot_photo_msgs = [m for m in history if m.media_type == "photo"]
        assert len(bot_photo_msgs) == 1
        assert "Сгенерированное изображение:" in (bot_photo_msgs[0].media_summary or "")
        assert "cat in space" in (bot_photo_msgs[0].media_summary or "")

        # Verify token usage recorded with source="image_gen"
        usage_rows = await store.get_usage_rows()
        assert any(r.get("source") == "image_gen" and int(r.get("total_tokens", 0)) == 165 for r in usage_rows)

        # 2. `/img` alias targeted at our bot username
        msg2 = _make_group_msg("/img@jeff_test_bot неоновый киберпанк город", msg_id=202)
        res2 = await orch.handle_incoming_message(msg2)
        assert res2 is not None
        assert len(prov.image_calls) == 2
        assert prov.image_calls[1]["prompt"] == "неоновый киберпанк город"
        assert len(client.sent_photos) == 2
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_imagine_edge_cases_other_bot_bare_and_disabled(tmp_path: Path) -> None:
    """`/imagine@otherbot` is ignored; bare `/imagine` gives usage hint; disabled returns notice."""
    orch, client, prov, _, cfg_mgr = await _build_orchestrator(tmp_path)
    try:
        # 1. Addressed to another bot -> completely ignored
        res_other = await orch.handle_incoming_message(
            _make_group_msg("/imagine@some_other_bot a dog", msg_id=301)
        )
        assert res_other is None
        assert len(prov.image_calls) == 0
        assert len(client.sent_replies) == 0
        assert len(client.sent_photos) == 0

        # 2. Bare /imagine without prompt or reply -> friendly usage hint
        res_bare = await orch.handle_incoming_message(_make_group_msg("/imagine", msg_id=302))
        assert res_bare is not None
        assert "/imagine" in res_bare
        assert len(prov.image_calls) == 0
        assert len(client.sent_replies) == 1
        assert client.sent_replies[0]["reply_to_msg_id"] == 302

        # 3. Disabled image_generation -> disabled notice
        assert cfg_mgr.bot is not None
        cfg_mgr.bot.image_generation.enabled = False
        res_dis = await orch.handle_incoming_message(
            _make_group_msg("/imagine sunset over ocean", msg_id=303)
        )
        assert res_dis is not None
        assert "отключена" in res_dis.lower()
        assert len(prov.image_calls) == 0
        assert len(client.sent_replies) == 2
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_imagine_reply_to_text_and_reply_to_photo_and_photo_caption(tmp_path: Path) -> None:
    """Replying to text merges/uses reply text; replying to photo or photo caption passes reference_images."""
    orch, client, prov, _, _ = await _build_orchestrator(
        tmp_path,
        image_overrides={"user_cooldown_seconds": 0.0, "max_prompt_chars": 80},
    )
    try:
        # 1. Bare /imagine replying to a text message -> uses reply_to_text as prompt
        msg_reply_text = _make_group_msg(
            "/imagine",
            msg_id=401,
            reply_to_text="рыжий кот пьёт кофе на крыше Парижа",
        )
        await orch.handle_incoming_message(msg_reply_text)
        assert len(prov.image_calls) == 1
        assert prov.image_calls[0]["prompt"] == "рыжий кот пьёт кофе на крыше Парижа"
        assert prov.image_calls[0]["reference_images"] is None

        # 2. /imagine with prompt replying to a text message -> merges reply context & respects max_prompt_chars
        msg_merge = _make_group_msg(
            "/imagine сделай в стиле акварели",
            msg_id=402,
            reply_to_text="очень длинный контекст " * 10,
        )
        await orch.handle_incoming_message(msg_merge)
        assert len(prov.image_calls) == 2
        assert "сделай в стиле акварели" in prov.image_calls[1]["prompt"]
        assert len(prov.image_calls[1]["prompt"]) <= 80

        # 3. /imagine replying to a photo -> downloads & passes reference_images
        ref_jpg = tmp_path / "ref_reply.jpg"
        ref_jpg.write_bytes(_make_jpeg_bytes(48, 48, (200, 50, 50)))

        class _PhotoReplyEvent:
            photo = True
            media = True
            file_path = str(ref_jpg)
            file_name = "ref_reply.jpg"
            mime_type = "image/jpeg"

        msg_reply_photo = _make_group_msg(
            "/imagine добавь солнечные очки",
            msg_id=403,
            reply_to_media_event=_PhotoReplyEvent(),
        )
        await orch.handle_incoming_message(msg_reply_photo)
        assert len(prov.image_calls) == 3
        refs = prov.image_calls[2]["reference_images"]
        assert refs is not None and len(refs) == 1
        assert refs[0].mime_type.startswith("image/")
        assert refs[0].get_bytes().startswith(b"\xff\xd8\xff")

        # 4. Photo sent directly with `/imagine` caption -> passes reference_images
        cap_jpg = tmp_path / "cap_photo.jpg"
        cap_jpg.write_bytes(_make_jpeg_bytes(40, 40, (20, 200, 80)))

        class _CaptionPhotoEvent:
            id = 404
            chat_id = GROUP_CHAT_ID
            sender_id = USER_ID
            out = False
            is_private = False
            is_reply = False
            is_reply_to_bot = False
            raw_text = "/imagine перерисуй в стиле пиксель-арт"
            text = raw_text
            caption = raw_text
            photo = True
            media = True
            file_path = str(cap_jpg)
            file_name = "cap_photo.jpg"
            mime_type = "image/jpeg"

        msg_cap_photo = _make_group_msg(
            "/imagine перерисуй в стиле пиксель-арт",
            msg_id=404,
            raw_event=_CaptionPhotoEvent(),
            media=True,
        )
        await orch.handle_incoming_message(msg_cap_photo)
        assert len(prov.image_calls) == 4
        cap_refs = prov.image_calls[3]["reference_images"]
        assert cap_refs is not None and len(cap_refs) == 1
        assert cap_refs[0].get_bytes().startswith(b"\xff\xd8\xff")
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_natural_language_img_tag_trigger_and_malformed_strip(tmp_path: Path) -> None:
    """Natural-language [IMG:{...}] triggers image gen, strips tags cleanly, and never leaks malformed tags."""
    prov = _FakeProvider(
        chat_reply=(
            "Щас нарисую, держи! "
            '[IMG:{"prompt":"fluffy space cat floating near Saturn","caption":"Вот твой космокот","aspect_ratio":"16:9"}]'
        )
    )
    orch, client, prov, store, cfg_mgr = await _build_orchestrator(
        tmp_path,
        provider=prov,
        image_overrides={"user_cooldown_seconds": 0.0},
    )
    try:
        # 1. Verify prompt guidance injects [IMG:...] instructions on drawing intent and omits on plain chat
        draw_msg = _make_group_msg("Джефф, нарисуй кота в космосе", msg_id=501, is_mention=True)
        prompt_with_intent = orch.build_context_prompt(draw_msg, [], [], None)
        assert "[IMG:" in prompt_with_intent

        plain_msg = _make_group_msg("Джефф, как дела сегодня?", msg_id=500, is_mention=True)
        prompt_plain = orch.build_context_prompt(plain_msg, [], [], None)
        assert "[IMG:" not in prompt_plain

        # 2. Execute natural-language drawing turn
        res = await orch.handle_incoming_message(draw_msg)
        assert res is not None
        assert len(client.sent_replies) == 1
        assert client.sent_replies[0]["text"] == "Щас нарисую, держи!"
        assert "[IMG:" not in client.sent_replies[0]["text"]

        assert len(prov.image_calls) == 1
        assert prov.image_calls[0]["prompt"] == "fluffy space cat floating near Saturn"
        assert prov.image_calls[0]["aspect_ratio"] == "16:9"
        assert len(client.sent_photos) == 1
        assert client.sent_photos[0]["caption"] == "Вот твой космокот"

        # 3. Malformed [IMG:{bad json tag is stripped cleanly without leaking or calling generate_image
        prov.chat_reply = 'Вот ответ без тега [IMG:{"prompt": broken'
        res_bad = await orch.handle_incoming_message(
            _make_group_msg("Джефф, нарисуй что-нибудь ещё", msg_id=502, is_mention=True)
        )
        assert res_bad == "Вот ответ без тега"
        assert "[IMG:" not in client.sent_replies[-1]["text"]
        assert len(prov.image_calls) == 1  # unchanged

        # 4. Natural language trigger disabled -> tag still stripped, generate_image NOT called
        assert cfg_mgr.bot is not None
        cfg_mgr.bot.image_generation.allow_natural_language_trigger = False
        prov.chat_reply = 'Держи текст [IMG:{"prompt":"a castle","caption":"Замок"}]'
        res_nl_off = await orch.handle_incoming_message(
            _make_group_msg("Джефф, нарисуй замок", msg_id=503, is_mention=True)
        )
        assert res_nl_off == "Держи текст"
        assert "[IMG:" not in client.sent_replies[-1]["text"]
        assert len(prov.image_calls) == 1
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_oversized_or_huge_dimension_image_falls_back_to_send_document(tmp_path: Path) -> None:
    """Images exceeding send_as_document_threshold_mb or w+h > 10000 are sent via send_document."""
    huge_dim_result = GeneratedImageResult(
        image_bytes=_make_jpeg_bytes(64, 64),
        mime_type="image/jpeg",
        filename="panorama.jpg",
        width=8000,
        height=3000,  # sum = 11000 > 10000
        prompt_used="ultra wide panorama",
        model_used="gemini-4-hydrogen",
    )
    prov = _FakeProvider(image_result=huge_dim_result)
    orch, client, prov, store, _ = await _build_orchestrator(
        tmp_path,
        provider=prov,
        image_overrides={"user_cooldown_seconds": 0.0, "send_as_document_threshold_mb": 9.5},
    )
    try:
        # 1. Dimension sum > 10000 -> send_document
        await orch.handle_incoming_message(_make_group_msg("/imagine панорама гор", msg_id=601))
        assert len(client.sent_photos) == 0
        assert len(client.sent_documents) == 1
        assert client.sent_documents[0]["filename"] == "panorama.jpg"
        assert client.sent_documents[0]["reply_to_message_id"] == 601

        history = await store.get_recent_history(GROUP_CHAT_ID, limit=10)
        assert any(m.media_type == "document" and "Сгенерированное изображение:" in (m.media_summary or "") for m in history)

        # 2. Byte size > threshold (use low threshold_mb=0.001 so test stays tiny & fast)
        orch.config_mgr.bot.image_generation.send_as_document_threshold_mb = 0.001  # type: ignore[union-attr]
        prov.image_result = GeneratedImageResult(
            image_bytes=b"\xff\xd8\xff" + (b"\x00" * 2048),
            mime_type="image/jpeg",
            filename="heavy.jpg",
            width=512,
            height=512,
            prompt_used="heavy art",
        )
        await orch.handle_incoming_message(_make_group_msg("/imagine тяжёлый арт", msg_id=602))
        assert len(client.sent_photos) == 0
        assert len(client.sent_documents) == 2
        assert client.sent_documents[1]["filename"] == "heavy.jpg"
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_rate_limit_timeout_refund_policy_refusal_and_sanitizer(tmp_path: Path) -> None:
    """Rate limit blocks without calling provider; timeout refunds; policy refusal stays sanitized."""
    orch, client, prov, store, _ = await _build_orchestrator(
        tmp_path,
        image_overrides={
            "max_per_user_per_hour": 1,
            "user_cooldown_seconds": 60.0,
            "admin_bypass_limits": True,
        },
    )
    orch.sanitizer.register_secret("SUPER_SECRET_BOT_KEY_98765")
    try:
        # 1. Timeout refunds the limiter slot so immediate retry succeeds!
        prov.image_error = ImageTimeoutError("timed out at /secret/path/SUPER_SECRET_BOT_KEY_98765")
        res_timeout = await orch.handle_incoming_message(
            _make_group_msg("/imagine дракон в облаках", msg_id=701)
        )
        assert res_timeout is not None
        assert "время ожидания" in res_timeout.lower()
        assert "SUPER_SECRET_BOT_KEY_98765" not in client.sent_replies[-1]["text"]
        assert "/secret/path" not in client.sent_replies[-1]["text"]

        # Immediate retry by SAME user succeeds because timeout refunded reservation
        prov.image_error = None
        prov.image_result = GeneratedImageResult(
            image_bytes=_make_jpeg_bytes(64, 64),
            mime_type="image/jpeg",
            filename="dragon.jpg",
            width=64,
            height=64,
            prompt_used="dragon with SUPER_SECRET_BOT_KEY_98765",
            caption_suggestion="Дракон SUPER_SECRET_BOT_KEY_98765",
        )
        res_ok = await orch.handle_incoming_message(
            _make_group_msg("/imagine дракон в облаках", msg_id=702)
        )
        assert res_ok is not None
        assert len(client.sent_photos) == 1
        assert "SUPER_SECRET_BOT_KEY_98765" not in (client.sent_photos[0]["caption"] or "")
        assert "[REDACTED]" in (client.sent_photos[0]["caption"] or "")

        # Verify secret also redacted in SQLite media_summary
        hist = await store.get_recent_history(GROUP_CHAT_ID, limit=10)
        for m in hist:
            assert "SUPER_SECRET_BOT_KEY_98765" not in (m.media_summary or "")

        # 2. Second request by same non-admin user hits rate limit -> provider NOT called
        calls_before = len(prov.image_calls)
        res_rl = await orch.handle_incoming_message(
            _make_group_msg("/imagine ещё один дракон", msg_id=703)
        )
        assert res_rl is not None
        assert "подожди" in res_rl.lower() or "лимит" in res_rl.lower()
        assert len(prov.image_calls) == calls_before

        # 3. Admin bypasses rate limit; PolicyRefusal sends polite Russian refusal without leaking internals
        prov.image_error = ImagePolicyRefusalError("blocked internal detail /var/lib/secret")
        res_refusal = await orch.handle_incoming_message(
            _make_group_msg(
                "/imagine запрещённый сюжет",
                msg_id=704,
                sender_id=ADMIN_USER_ID,
            )
        )
        assert res_refusal is not None
        assert "ограничение контента" in res_refusal.lower()
        assert "/var/lib/secret" not in client.sent_replies[-1]["text"]
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_hot_reload_updates_image_limiter_config(tmp_path: Path) -> None:
    """ConfigManager hot-reload propagates updated ImageGenerationConfig to orchestrator._image_limiter."""
    orch, client, prov, _, cfg_mgr = await _build_orchestrator(
        tmp_path,
        image_overrides={
            "max_per_user_per_hour": 1,
            "user_cooldown_seconds": 0.0,
        },
    )
    try:
        # 1st image consumes the 1/hour quota
        await orch.handle_incoming_message(_make_group_msg("/imagine первый кот", msg_id=801))
        assert len(prov.image_calls) == 1

        # 2nd image hits hourly cap
        res_blocked = await orch.handle_incoming_message(
            _make_group_msg("/imagine второй кот", msg_id=802)
        )
        assert res_blocked is not None
        assert "лимит" in res_blocked.lower()
        assert len(prov.image_calls) == 1

        # Mutate bot.yaml on disk to raise max_per_user_per_hour -> 5 and hot-reload
        raw_cfg = yaml.safe_load(cfg_mgr.bot_path.read_text(encoding="utf-8"))
        raw_cfg["image_generation"]["max_per_user_per_hour"] = 5
        cfg_mgr.bot_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")
        # Bump mtime deterministically so reload_changed() always detects the update
        import os

        cur_stat = cfg_mgr.bot_path.stat()
        os.utime(cfg_mgr.bot_path, (cur_stat.st_atime + 2, cur_stat.st_mtime + 2))

        assert cfg_mgr.reload_changed() is True

        # Now 2nd image succeeds immediately under updated limiter config
        await orch.handle_incoming_message(_make_group_msg("/imagine второй кот после reload", msg_id=803))
        assert len(prov.image_calls) == 2
        assert len(client.sent_photos) == 2
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_upload_photo_keepalive_task_cancelled_cleanly_on_error(tmp_path: Path) -> None:
    """Verify background upload-photo keepalive task is always cancelled even if provider raises."""
    orch, client, prov, _, _ = await _build_orchestrator(tmp_path)
    prov.image_error = RuntimeError("unexpected boom")
    try:
        orch._keep_chat_action_alive = AsyncMock(side_effect=orch._keep_chat_action_alive)  # type: ignore[method-assign]
        res = await orch.handle_incoming_message(_make_group_msg("/imagine космос", msg_id=901))
        assert res is not None
        assert "не получилось" in res.lower()
        assert any(a["action"] == "upload-photo" for a in client.chat_actions)
        assert orch._keep_chat_action_alive.call_count == 1
    finally:
        await orch.stop()
