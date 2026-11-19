"""Unit tests for M1 config models, bot.yaml/example validation, FFmpeg fallback, and legal prompts."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import bot.antigravity as agy_pkg
import bot.config as cfg_pkg
from bot.config.manager import ConfigManager
from bot.config.models import (
    AntigravityProviderConfig,
    BotConfig,
    ModelRoleConfig,
    ModelRouterConfig,
    QuotaConfig,
    SelfEvolutionConfig,
    TelegramConfig,
)
from bot.main import build_antigravity_config
from bot.media.audio import resolve_ffmpeg_path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_config_models_defaults_and_backward_compatibility():
    # TelegramConfig.admin_user_ids default & None coercion
    tg = TelegramConfig(api_id=123, api_hash="abc", admin_user_ids=None)  # type: ignore[arg-type]
    assert tg.admin_user_ids == []

    tg_admin = TelegramConfig(api_id=123, api_hash="abc", admin_user_ids=[382962362])
    assert tg_admin.admin_user_ids == [382962362]

    # Legacy minimal YAML without antigravity/models/quota/self_evolution
    legacy = BotConfig.model_validate({"telegram": {"api_id": 1, "api_hash": "h"}})
    assert isinstance(legacy.antigravity, AntigravityProviderConfig)
    assert isinstance(legacy.models, ModelRouterConfig)
    assert isinstance(legacy.quota, QuotaConfig)
    assert isinstance(legacy.self_evolution, SelfEvolutionConfig)
    assert legacy.models.group.model == "gemini-3.8-flash"
    assert legacy.models.self_edit.model == "claude-sonnet-5-5"
    assert any(
        m.model == "gpt-oss-120b-medium" and m.effort is None
        for m in legacy.models.fallback_models
    )

    # Re-exports in bot.config must NOT shadow bot.antigravity.AntigravityConfig
    assert hasattr(cfg_pkg, "AntigravityProviderConfig")
    assert hasattr(cfg_pkg, "ModelRouterConfig")
    assert hasattr(cfg_pkg, "QuotaConfig")
    assert hasattr(cfg_pkg, "SelfEvolutionConfig")
    assert not hasattr(cfg_pkg, "AntigravityConfig")
    assert agy_pkg.AntigravityConfig is not AntigravityProviderConfig

    runtime_agy = build_antigravity_config(legacy)
    assert isinstance(runtime_agy, agy_pkg.AntigravityConfig)
    assert runtime_agy.model == legacy.antigravity.model


def test_bot_yaml_and_example_yaml_valid_and_secret_free():
    bot_yaml_path = REPO_ROOT / "config" / "bot.yaml"
    example_yaml_path = REPO_ROOT / "config" / "bot.example.yaml"

    if bot_yaml_path.exists():
        mgr = ConfigManager(config_path=bot_yaml_path)
        assert mgr.bot is not None
        assert 382962362 in mgr.bot.telegram.allowed_dm_user_ids
        assert 382962362 in mgr.bot.telegram.admin_user_ids
        assert mgr.bot.media.ffmpeg_path == "/usr/local/bin/ffmpeg"
        assert mgr.bot.antigravity.model == "gemini-4-hydrogen"
        assert isinstance(mgr.bot.models.group, ModelRoleConfig)
        assert mgr.bot.quota.enabled is True
        assert mgr.bot.self_evolution.max_daily_edits >= 1
        assert mgr.bot.telegram.bot_token is None

    raw_example = yaml.safe_load(example_yaml_path.read_text(encoding="utf-8"))
    example_cfg = BotConfig.model_validate(raw_example)
    assert example_cfg.telegram.mode == "bot_api"
    assert example_cfg.telegram.allowed_dm_user_ids == []
    assert example_cfg.telegram.admin_user_ids == []
    assert "hermes_kuk7yi6kfm3lcdfa_bot" in example_cfg.telegram.ignored_sender_usernames
    assert example_cfg.media.ffmpeg_path == "/usr/local/bin/ffmpeg"
    assert example_cfg.telegram.bot_token is None
    assert "382962362" not in example_yaml_path.read_text(encoding="utf-8")


def test_resolve_ffmpeg_path_fallback_order(tmp_path: Path):
    fake_custom = tmp_path / "custom_ffmpeg"
    fake_custom.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(fake_custom, 0o755)

    # 1. Valid custom_path wins
    assert resolve_ffmpeg_path(str(fake_custom)) == str(fake_custom.resolve())

    # 2. Missing custom_path falls back to shutil.which("ffmpeg")
    fake_which = str(tmp_path / "which_ffmpeg")
    with patch("shutil.which", return_value=fake_which):
        assert resolve_ffmpeg_path("/nonexistent/ffmpeg") == fake_which

    # 3. Missing custom_path and no which -> checks /usr/local/bin/ffmpeg before DEFAULT_STATIC_FFMPEG
    with (
        patch("shutil.which", return_value=None),
        patch("os.path.isfile", side_effect=lambda p: str(p) == "/usr/local/bin/ffmpeg"),
        patch("os.access", side_effect=lambda p, m: str(p) == "/usr/local/bin/ffmpeg"),
    ):
        assert resolve_ffmpeg_path("/nonexistent/ffmpeg") == "/usr/local/bin/ffmpeg"

    # 4. Nothing exists -> FileNotFoundError
    with patch("os.path.isfile", return_value=False), patch("shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="FFmpeg binary not found"):
            resolve_ffmpeg_path("/nonexistent/ffmpeg")


def test_system_prompts_enforce_unconditional_legal_prohibition():
    sys_prompt = (REPO_ROOT / "config" / "system_prompt.md").read_text(encoding="utf-8").lower()
    contents = [sys_prompt]
    userbot_prompt_path = REPO_ROOT / "config" / "userbot_system_prompt.md"
    if userbot_prompt_path.is_file():
        contents.append(userbot_prompt_path.read_text(encoding="utf-8").lower())

    for content in contents:
        assert "торрент" in content
        assert "незакон" in content or "нелегаль" in content
        assert "независимо от того, кто просит" in content
        assert "администратор" in content or "владел" in content
