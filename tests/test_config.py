"""Unit tests for configuration models and TTS voice/style defaults."""

from __future__ import annotations

from pathlib import Path

import yaml

from bot.config.models import BotConfig, TTSConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_GEMINI_STYLE_PROMPT = (
    "Sharp-tongued, confident young woman roasting friends in a group chat; "
    "bold, dry deadpan sarcasm and biting humor with crisp, punchy "
    "Russian/Ukrainian delivery—never sweet, breathy, flirtatious, or announcer-like."
)


def test_tts_config_defaults_use_kore_and_sharp_deadpan_style():
    cfg = TTSConfig()

    assert cfg.provider == "auto"
    assert cfg.fallback_to_edge is True
    assert cfg.gemini_model == "gemini-2.5-flash-preview-tts"
    assert cfg.gemini_ru_voice == "Kore"
    assert cfg.gemini_uk_voice == "Kore"
    assert cfg.gemini_style_prompt == EXPECTED_GEMINI_STYLE_PROMPT

    lower_prompt = cfg.gemini_style_prompt.lower()
    assert "deadpan" in lower_prompt
    assert "sarcasm" in lower_prompt
    assert "never sweet, breathy, flirtatious" in lower_prompt
    for unwanted in ("warm", "playful", "subtle"):
        assert unwanted not in lower_prompt


def test_bot_example_yaml_tts_config_matches_defaults():
    example_path = REPO_ROOT / "config" / "bot.example.yaml"
    raw = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    cfg = BotConfig.model_validate(raw)

    assert cfg.voice.tts.gemini_ru_voice == "Kore"
    assert cfg.voice.tts.gemini_uk_voice == "Kore"
    assert cfg.voice.tts.gemini_style_prompt == EXPECTED_GEMINI_STYLE_PROMPT


def test_bot_yaml_tts_config_if_present():
    bot_yaml_path = REPO_ROOT / "config" / "bot.yaml"
    if not bot_yaml_path.exists():
        return

    raw = yaml.safe_load(bot_yaml_path.read_text(encoding="utf-8"))
    tts_cfg = TTSConfig.model_validate(raw["voice"]["tts"])
    assert tts_cfg.gemini_ru_voice == "Kore"
    assert tts_cfg.gemini_uk_voice == "Kore"
    assert tts_cfg.gemini_style_prompt == EXPECTED_GEMINI_STYLE_PROMPT
