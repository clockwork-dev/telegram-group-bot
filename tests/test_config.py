"""Unit tests for configuration models and TTS voice/style defaults."""

from __future__ import annotations

from pathlib import Path

import yaml

from bot.config.models import BotConfig, TTSConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_GEMINI_STYLE_PROMPT = (
    "Lively, quick-witted young woman chatting naturally with close friends; "
    "authentic everyday intonation, natural breaths and conversational timing, "
    "cheeky teasing humor and crisp Russian/Ukrainian—never robotic, monotone, "
    "ASMR-whispering, or flirtatious."
)


def test_tts_config_defaults_use_aoede_and_lively_human_style():
    cfg = TTSConfig()

    assert cfg.provider == "auto"
    assert cfg.fallback_to_edge is True
    assert cfg.gemini_model == "gemini-2.5-flash-preview-tts"
    assert cfg.gemini_ru_voice == "Aoede"
    assert cfg.gemini_uk_voice == "Aoede"
    assert cfg.gemini_style_prompt == EXPECTED_GEMINI_STYLE_PROMPT

    lower_prompt = cfg.gemini_style_prompt.lower()
    assert "lively" in lower_prompt
    assert "natural breaths" in lower_prompt
    assert "cheeky teasing humor" in lower_prompt
    assert "never robotic, monotone, asmr-whispering, or flirtatious" in lower_prompt
    assert "deadpan" not in lower_prompt


def test_bot_example_yaml_tts_config_matches_defaults():
    example_path = REPO_ROOT / "config" / "bot.example.yaml"
    raw = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    cfg = BotConfig.model_validate(raw)

    assert cfg.voice.tts.gemini_ru_voice == "Aoede"
    assert cfg.voice.tts.gemini_uk_voice == "Aoede"
    assert cfg.voice.tts.gemini_style_prompt == EXPECTED_GEMINI_STYLE_PROMPT


def test_bot_yaml_tts_config_if_present():
    bot_yaml_path = REPO_ROOT / "config" / "bot.yaml"
    if not bot_yaml_path.exists():
        return

    raw = yaml.safe_load(bot_yaml_path.read_text(encoding="utf-8"))
    tts_cfg = TTSConfig.model_validate(raw["voice"]["tts"])
    assert tts_cfg.gemini_ru_voice == "Aoede"
    assert tts_cfg.gemini_uk_voice == "Aoede"
    assert tts_cfg.gemini_style_prompt == EXPECTED_GEMINI_STYLE_PROMPT
