"""Unit tests for configuration models and TTS voice/style defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bot.config.models import (
    BotConfig,
    MemoryConfig,
    MemoryEmbeddingsConfig,
    MemoryReflectionConfig,
    TTSConfig,
)

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


def test_memory_embeddings_and_reflection_config_defaults_and_validation():
    emb = MemoryEmbeddingsConfig()
    assert emb.enabled is True
    assert emb.provider == "gemini"
    assert emb.gemini_model == "gemini-embedding-001"
    assert emb.gemini_api_key_env == "GEMINI_API_KEY"
    assert emb.gemini_api_key is None
    assert emb.output_dimensionality == 768
    assert emb.timeout_seconds == 3.0
    assert emb.min_similarity == 0.52
    assert emb.query_cache_size == 128
    assert emb.backfill_batch_size == 20
    assert emb.cooldown_seconds == 120.0

    # repr=False hides raw API key
    with_key = MemoryEmbeddingsConfig(gemini_api_key="AIzaSyDummySecret1234567890", provider="GEMINI")  # type: ignore[arg-type]
    assert with_key.provider == "gemini"
    assert "AIzaSyDummySecret1234567890" not in repr(with_key)

    with pytest.raises(ValidationError):
        MemoryEmbeddingsConfig(output_dimensionality=32)

    refl = MemoryReflectionConfig()
    assert refl.enabled is True
    assert refl.interval_hours == 24.0
    assert refl.min_active_facts_per_subject == 15
    assert refl.max_subjects_per_cycle == 3
    assert refl.target_summary_facts == 4
    assert refl.subject_cooldown_hours == 72.0
    assert refl.agreement_default_ttl_days == 14
    assert refl.event_default_ttl_days == 30

    mem = MemoryConfig()
    assert mem.embeddings.gemini_model == "gemini-embedding-001"
    assert mem.reflection.min_active_facts_per_subject == 15


def test_yaml_memory_embeddings_and_reflection_sections():
    example_path = REPO_ROOT / "config" / "bot.example.yaml"
    raw_example = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    cfg_example = BotConfig.model_validate(raw_example)
    assert cfg_example.memory.embeddings.enabled is True
    assert cfg_example.memory.embeddings.provider == "gemini"
    assert cfg_example.memory.embeddings.gemini_model == "gemini-embedding-001"
    assert cfg_example.memory.embeddings.gemini_api_key_env == "GEMINI_API_KEY"
    assert cfg_example.memory.embeddings.output_dimensionality == 768
    assert cfg_example.memory.embeddings.min_similarity == 0.52
    assert cfg_example.memory.reflection.enabled is True
    assert cfg_example.memory.reflection.min_active_facts_per_subject == 15
    assert cfg_example.memory.reflection.agreement_default_ttl_days == 14
    assert cfg_example.memory.reflection.event_default_ttl_days == 30

    bot_yaml_path = REPO_ROOT / "config" / "bot.yaml"
    if not bot_yaml_path.exists():
        return

    raw_bot = yaml.safe_load(bot_yaml_path.read_text(encoding="utf-8"))
    mem_cfg = MemoryConfig.model_validate(raw_bot["memory"])
    assert mem_cfg.embeddings.enabled is True
    assert mem_cfg.embeddings.gemini_model == "gemini-embedding-001"
    assert mem_cfg.reflection.enabled is True
    assert mem_cfg.reflection.event_default_ttl_days == 30

