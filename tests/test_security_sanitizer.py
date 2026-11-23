"""Unit and integration tests for SecretSanitizer, SanitizingLogFilter, and outbound redaction."""

from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from bot.config.manager import ConfigManager
from bot.core.models import ProviderResponse
from bot.main import GroupBotOrchestrator
from bot.memory.db import MemoryStore
from bot.security.sanitizer import (
    REDACTED,
    SanitizingLogFilter,
    SecretSanitizer,
    install_log_sanitizer,
)
from bot.telegram.bot_api_client import TelegramBotAPIClient
from bot.telegram.handlers import IncomingMessage


def test_sanitizer_env_loading_and_non_secret_exclusions(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "TELEGRAM_API_ID=987654321",
                "TELEGRAM_SESSION_PATH=/opt/data/session/userbot.session",
                "ANTIGRAVITY_DATA_DIR=/opt/data/antigravity",
                "ANTIGRAVITY_BINARY=/usr/local/bin/agy",
                "TELEGRAM_BOT_TOKEN=123456789:AAFakeBotTokenForTest1234567890abc",
                "TELEGRAM_API_HASH=deadbeefcafebabe1234567890abcdef",
                "TELEGRAM_PHONE=+79991112233",
                "ANTIGRAVITY_PROXY=http://proxyuser:SuperProxyPass99@10.0.0.2:8080",
                "SHORT_VAL=abcde",
                "DUMMY_HASH=0123456789abcdef0123456789abcdef",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    sanitizer = SecretSanitizer(env_path=env_file, load_environ=False)
    # load_env_file must NOT mutate os.environ
    assert "TELEGRAM_BOT_TOKEN" not in os.environ

    # Secrets and proxy credentials redacted
    sample = (
        "tok=123456789:AAFakeBotTokenForTest1234567890abc "
        "hash=deadbeefcafebabe1234567890abcdef "
        "phone=+79991112233 "
        "proxy=http://proxyuser:SuperProxyPass99@10.0.0.2:8080"
    )
    cleaned = sanitizer.sanitize(sample)
    assert "AAFakeBotTokenForTest" not in cleaned
    assert "deadbeefcafebabe1234567890abcdef" not in cleaned
    assert "+79991112233" not in cleaned
    assert "SuperProxyPass99" not in cleaned
    assert "10.0.0.2:8080" in cleaned  # host/port preserved

    # Non-secret env values, short values (<6 chars), and dummy placeholders must NOT be redacted
    safe_text = (
        "id=987654321 sess=/opt/data/session/userbot.session "
        "agy=/usr/local/bin/agy short=abcde dummy=0123456789abcdef0123456789abcdef"
    )
    assert sanitizer.sanitize(safe_text) == safe_text


def test_sanitizer_patterns_and_git_sha_preservation():
    sanitizer = SecretSanitizer(extra_secrets=["My$ecret/Value+1"], load_environ=False)

    # URL-encoded registered secret
    encoded = urllib.parse.quote("My$ecret/Value+1", safe="")
    assert sanitizer.sanitize(f"raw=My$ecret/Value+1 enc={encoded}") == f"raw={REDACTED} enc={REDACTED}"

    # Telegram bot token inside Bot API URL (preceded directly by `bot`)
    url_msg = "POST https://api.telegram.org/bot8877665544:AAHTestSecretTokenValue123456789/getUpdates failed"
    redacted_url = sanitizer.sanitize(url_msg)
    assert "8877665544:AAHTestSecretTokenValue123456789" not in redacted_url
    assert f"https://api.telegram.org/bot{REDACTED}/getUpdates" in redacted_url

    # Bearer, Google OAuth access token (ya29.), Google OAuth refresh token (1//0...), API keys, AWS, JWT
    tokens = (
        "Authorization: Bearer abc.def.ghi12345 "
        "oauth=ya29.a0AfH6SMC1234567890abcdef "
        "refresh=1//0gTestRefreshTokenValue1234567890ABCDEF "
        "gkey=AIzaSyDummyKeyValue1234567890abcdef "
        "aqkey=AQ.Ab8RN6DummyGoogleKeyValue1234567890abcdef "
        "openai=sk-proj-TestKey1234567890abcdef "
        "gh=ghp_1234567890abcdefghij1234567890 "
        "aws=AKIAIOSFODNN7EXAMPLE "
        "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    cleaned_tokens = sanitizer.sanitize(tokens)
    for leaked in (
        "abc.def.ghi12345",
        "ya29.a0AfH6SMC1234567890abcdef",
        "1//0gTestRefreshTokenValue1234567890ABCDEF",
        "AIzaSyDummyKeyValue1234567890abcdef",
        "AQ.Ab8RN6DummyGoogleKeyValue1234567890abcdef",
        "sk-proj-TestKey1234567890abcdef",
        "ghp_1234567890abcdefghij1234567890",
        "AKIAIOSFODNN7EXAMPLE",
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ):
        assert leaked not in cleaned_tokens

    # WireGuard PrivateKey & PresharedKey (context-scoped)
    wg_cfg = (
        "[Interface]\n"
        "PrivateKey = cGFzc3dvcmRfMzJfYnl0ZXNfd2lyZWd1YXJkX2tleQ==\n"
        "[Peer]\n"
        "PresharedKey = YW5vdGhlcl9wcmVzaGFyZWRfa2V5XzMyX2J5dGVzX2s=\n"
    )
    cleaned_wg = sanitizer.sanitize(wg_cfg)
    assert "cGFzc3dvcmRfMzJfYnl0ZXNfd2lyZWd1YXJkX2tleQ==" not in cleaned_wg
    assert "YW5vdGhlcl9wcmVzaGFyZWRfa2V5XzMyX2J5dGVzX2s=" not in cleaned_wg
    assert f"PrivateKey = {REDACTED}" in cleaned_wg
    assert f"PresharedKey = {REDACTED}" in cleaned_wg

    # PEM Private Key
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAtest\n-----END RSA PRIVATE KEY-----"
    assert sanitizer.sanitize(pem) == REDACTED

    # Labeled api_hash redacted, bare 40-char git commit SHA preserved!
    commit_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
    labeled_and_sha = (
        f"TELEGRAM_API_HASH=0f1e2d3c4b5a69788796a5b4c3d2e1f0 "
        f"Deployed commit {commit_sha} cleanly"
    )
    cleaned_hash = sanitizer.sanitize(labeled_and_sha)
    assert "0f1e2d3c4b5a69788796a5b4c3d2e1f0" not in cleaned_hash
    assert f"TELEGRAM_API_HASH={REDACTED}" in cleaned_hash
    assert commit_sha in cleaned_hash


def test_logging_filter_redacts_args_exc_info_and_child_loggers():
    secret = "super_secret_runtime_token_xyz"
    sanitizer = SecretSanitizer(extra_secrets=[secret], load_environ=False)

    root_logger = logging.getLogger()
    records_out: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records_out.append(self.format(record))

    handler = _ListHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root_logger.addHandler(handler)
    old_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)

    try:
        # Install idempotently twice
        install_log_sanitizer(sanitizer)
        install_log_sanitizer(sanitizer)
        assert sum(isinstance(f, SanitizingLogFilter) for f in handler.filters) == 1

        child = logging.getLogger("bot.telegram.bot_api_client")
        httpx_logger = logging.getLogger("httpx")

        # 1. Positional args containing secret URL
        child.warning(
            "Request failed for %s",
            f"https://api.telegram.org/bot9988776655:AAFtestBotToken1234567890abcdef/getMe?x={secret}",
        )
        # 2. Exception traceback containing secret
        try:
            raise RuntimeError(f"Connection failed using 1//0testRefreshToken123456789012345 and {secret}")
        except RuntimeError:
            httpx_logger.exception("HTTP call crashed")

        joined = "\n".join(records_out)
        assert "AAFtestBotToken1234567890abcdef" not in joined
        assert "1//0testRefreshToken123456789012345" not in joined
        assert secret not in joined
        assert REDACTED in joined
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)


@pytest.mark.asyncio
async def test_bot_api_client_redacts_outbound_and_error_traces(tmp_path: Path):
    token = "123456789:AAFakeTokenSecretForBotApiTest123456"
    sanitizer = SecretSanitizer(extra_secrets=["MyExtraSecret123"], load_environ=False)
    client = TelegramBotAPIClient(
        bot_token=token,
        allowed_chat_ids=[-100123],
        typing_min_seconds=0.0,
        typing_max_seconds=0.0,
        sanitizer=sanitizer,
    )

    captured_payloads: list[dict] = []

    async def fake_call_api(method: str, payload=None, timeout: float = 35.0):
        captured_payloads.append({"method": method, "payload": payload})
        return {"ok": True, "result": {"message_id": 1}}

    with patch.object(client, "call_api", side_effect=fake_call_api):
        await client.send_reply(
            chat_id=-100123,
            text=f"Here is secret MyExtraSecret123 and token {token}",
            simulate_typing=False,
        )
    sent_text = captured_payloads[-1]["payload"]["text"]
    assert "MyExtraSecret123" not in sent_text
    assert token not in sent_text
    assert REDACTED in sent_text

    # Error raising in _sync_execute_request must redact token in HTTP/network error
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError(f"Failed connecting to https://api.telegram.org/bot{token}/getMe"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            client._sync_execute_request("getMe", None, {})
    assert token not in str(exc_info.value)
    assert REDACTED in str(exc_info.value)


@pytest.mark.asyncio
async def test_orchestrator_sanitizes_outbound_reply_and_proactive(tmp_path: Path):
    bot_yaml = tmp_path / "bot.yaml"
    bot_yaml.write_text(
        yaml.dump(
            {
                "telegram": {
                    "api_id": 123456,
                    "api_hash": "custom_real_hash_value_999999999",
                    "allowed_chat_ids": [-100555],
                    "allowed_dm_user_ids": [382962362],
                    "admin_user_ids": [382962362],
                },
                "behavior": {"bot_names": ["Jeff"], "typing_simulation": False},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "system_prompt.md").write_text("Test prompt", encoding="utf-8")
    cfg_mgr = ConfigManager(config_path=bot_yaml)
    store = MemoryStore(db_path=str(tmp_path / "mem.sqlite"))
    await store.init_db()

    sent_replies: list[str] = []

    class _MockUserbot:
        bot_id = 777
        bot_username = "jeff_bot"
        client = MagicMock()

        def register_message_handler(self, h):
            pass

        async def send_reply(self, chat_id, text, reply_to_msg_id=None, simulate_typing=True):
            sent_replies.append(text)
            return {"result": {"message_id": len(sent_replies)}}

        async def send_typing_action(self, chat_id, action="typing"):
            pass

        async def send_reaction(self, chat_id, msg_id, emoji):
            return True

        async def stop(self):
            pass

    class _LeakingProvider:
        async def generate_response(self, prompt, system_instructions=None, media_attachments=None, conversation_id=None, **kwargs):
            return ProviderResponse(
                text="Oops leaked custom_real_hash_value_999999999 and 987654321:AAFakeBotTokenForLeakingTest12345",
                model="test",
            )

        def close(self):
            pass

    sanitizer = SecretSanitizer(load_environ=False)
    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=_LeakingProvider(),
        userbot=_MockUserbot(),
        sanitizer=sanitizer,
    )
    try:
        ev = MagicMock(
            id=1,
            chat_id=-100555,
            sender_id=42,
            out=False,
            is_private=False,
            sender_is_bot=False,
            sender_username="friend",
            raw_text="Jeff hello",
            text="Jeff hello",
            photo=False,
            voice=False,
            video=False,
            video_note=False,
            sticker=False,
            gif=False,
            animation=False,
            document=False,
            media=False,
        )
        inc = IncomingMessage(
            msg_id=1,
            chat_id=-100555,
            sender_id=42,
            sender_name="Friend",
            text="Jeff hello",
            reply_to_msg_id=None,
            is_reply=False,
            is_mention=True,
            raw_event=ev,
            is_private=False,
        )
        await orch.handle_incoming_message(inc)
        await orch._send_proactive(-100555, "Proactive custom_real_hash_value_999999999")
        await orch.send_sanitized_message(382962362, "DM sk-ant-secretkey1234567890abcdef")

        assert len(sent_replies) == 3
        for reply in sent_replies:
            assert "custom_real_hash_value_999999999" not in reply
            assert "AAFakeBotTokenForLeakingTest12345" not in reply
            assert "sk-ant-secretkey1234567890abcdef" not in reply
            assert REDACTED in reply
    finally:
        await orch.stop()


def test_install_log_sanitizer_covers_tts_and_http_loggers():
    gemini_secret = "AIzaSyLogSanitizerTestKey1234567890abcdef"
    sanitizer = SecretSanitizer(extra_secrets=[gemini_secret], load_environ=False)

    root_logger = logging.getLogger()
    records_out: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records_out.append(self.format(record))

    handler = _ListHandler()
    handler.setFormatter(logging.Formatter("%(name)s:%(message)s"))
    root_logger.addHandler(handler)
    old_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)

    target_loggers = ("httpx2", "google_genai", "google", "aiohttp", "edge_tts")
    saved_states = {}
    try:
        install_log_sanitizer(sanitizer)
        for name in target_loggers:
            sub = logging.getLogger(name)
            saved_states[name] = (sub.level, sub.propagate)
            sub.setLevel(logging.DEBUG)
            sub.propagate = True
            assert any(isinstance(f, SanitizingLogFilter) for f in sub.filters)
            sub.info("Request failed key=%s aq=AQ.Ab8RN6SecretToken1234567890", gemini_secret)

        joined = "\n".join(records_out)
        assert gemini_secret not in joined
        assert "AQ.Ab8RN6SecretToken1234567890" not in joined
        assert joined.count(REDACTED) >= len(target_loggers) * 2
    finally:
        for name, (lvl, prop) in saved_states.items():
            sub = logging.getLogger(name)
            sub.setLevel(lvl)
            sub.propagate = prop
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)
