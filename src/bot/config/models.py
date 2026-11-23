"""Pydantic v2 Configuration Models for Bot, Telegram, Memory, Friends & Proactive."""

from typing import Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator, model_validator


class TelegramConfig(BaseModel):
    """Configuration for Telegram client (supports both Classic Bot API and MTProto Userbot)."""

    mode: str = "auto"  # "auto", "bot_api", or "userbot"
    bot_token: Optional[str] = None
    api_id: int
    api_hash: str
    phone: Optional[str] = None
    session_path: str = "data/session/userbot.session"
    allowed_chat_ids: List[int] = Field(default_factory=list)
    allowed_dm_user_ids: List[int] = Field(default_factory=list)
    admin_user_ids: List[int] = Field(default_factory=list)
    ignored_sender_usernames: List[str] = Field(
        default_factory=lambda: ["hermes_kuk7yi6kfm3lcdfa_bot"]
    )

    @field_validator(
        "allowed_chat_ids",
        "allowed_dm_user_ids",
        "admin_user_ids",
        "ignored_sender_usernames",
        mode="before",
    )
    @classmethod
    def _coerce_none_to_list(cls, v: Any) -> Any:
        if v is None:
            return []
        return v


class SpontaneousRepliesConfig(BaseModel):
    """Heuristic settings for spontaneous in-chat conversation joining."""

    enabled: bool = True
    min_interval_seconds: int = 300
    probability: float = 0.20
    max_per_hour: int = 3
    active_conversation_window_seconds: int = Field(default=90, ge=0)


class BehaviorConfig(BaseModel):
    """In-chat conversational persona and behavioral timings."""

    bot_names: List[str] = Field(default_factory=lambda: ["Джефф", "бот"])
    typing_simulation: bool = True
    typing_speed_cpm: int = 300
    typing_min_seconds: float = 1.5
    typing_max_seconds: float = 8.0
    reactions_enabled: bool = True
    spontaneous_replies: SpontaneousRepliesConfig = Field(default_factory=SpontaneousRepliesConfig)


class QuietHoursConfig(BaseModel):
    """Quiet hours window during which proactive actions are silenced."""

    enabled: bool = True
    start: str = "23:00"
    end: str = "08:00"
    timezone: str = "Europe/Moscow"


class CronDigestJobConfig(BaseModel):
    """Scheduled cron job configuration for proactive news digests."""

    id: str
    enabled: bool = True
    cron: str
    timezone: str = "Europe/Kyiv"
    slot_label: str = "morning"
    prompt_hint: str = ""
    topics: List[str] = Field(default_factory=list)
    respect_quiet_hours: bool = True
    chat_ids: List[int] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("CronDigestJobConfig.id must not be empty")
        return cleaned

    @field_validator("cron")
    @classmethod
    def _validate_cron(cls, value: str) -> str:
        # Lazy import avoids circular dependency via bot.proactive.__init__ -> engine -> manager.
        from bot.proactive.cron import CronExpression

        parsed = CronExpression.parse(value)
        return parsed.expression

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Timezone must not be empty")
        try:
            ZoneInfo(cleaned)
        except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid timezone: {value!r}") from exc
        return cleaned


class ProactiveConfig(BaseModel):
    """Proactive message scheduling, triggers, and content limits."""

    enabled: bool = True
    quiet_hours: QuietHoursConfig = Field(default_factory=QuietHoursConfig)
    silence_trigger_hours: float = 6.0
    scheduled_interval_minutes: int = 180
    jitter_minutes: int = 45
    cooldown_minutes: int = 90
    humor_style: str = "friendly_sarcasm"
    forbidden_topics: List[str] = Field(default_factory=list)
    cron_digests: List[CronDigestJobConfig] = Field(default_factory=list)


class MediaConfig(BaseModel):
    """Media processing thresholds and tool binaries."""

    max_doc_size_mb: int = 10
    temp_dir: str = "data/tmp"
    ffmpeg_path: Optional[str] = None


class TTSConfig(BaseModel):
    """Text-to-Speech (Edge TTS) synthesis and voice-reply configuration."""

    enabled: bool = True
    ru_voice: str = "ru-RU-SvetlanaNeural"
    uk_voice: str = "uk-UA-PolinaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    max_chars: int = Field(default=600, ge=50, le=4000)
    timeout_sec: float = Field(default=25.0, gt=0.0)
    reply_voice_to_voice: bool = True


class STTConfig(BaseModel):
    """Speech-to-Text (local transcription) configuration for incoming voice notes."""

    enabled: bool = True
    model_size: str = "base"
    compute_type: str = "int8"
    cpu_threads: int = Field(default=2, ge=1, le=16)
    max_duration_sec: int = Field(default=120, ge=5, le=600)
    timeout_sec: float = Field(default=45.0, gt=0.0)
    allowed_languages: List[str] = Field(default_factory=lambda: ["ru", "uk"])
    model_dir: Optional[str] = "data/models/whisper"


class VoiceConfig(BaseModel):
    """Combined TTS and STT configuration."""

    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)


class ImageGenerationConfig(BaseModel):
    """Native Antigravity image generation limits, delivery thresholds, and cleanup settings."""

    enabled: bool = True
    allow_natural_language_trigger: bool = True
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=600.0)
    max_prompt_chars: int = Field(default=1000, ge=20, le=4000)
    max_per_user_per_hour: int = Field(default=5, ge=1, le=100)
    user_cooldown_seconds: float = Field(default=30.0, ge=0.0, le=3600.0)
    max_per_chat_per_hour: int = Field(default=15, ge=1, le=200)
    admin_bypass_limits: bool = True
    output_dir: str = "data/tmp/images"
    max_image_size_mb: float = Field(default=20.0, gt=0.0, le=45.0)
    send_as_document_threshold_mb: float = Field(default=9.5, gt=0.0, le=10.0)
    model: Optional[str] = None
    cleanup_after_send: bool = True


class MemoryBackupConfig(BaseModel):
    """Automated SQLite memory backup settings."""

    enabled: bool = True
    backup_dir: str = "data/backups"
    interval_hours: float = Field(default=24.0, gt=0.0)
    retain_count: int = Field(default=14, ge=1)
    compress: bool = True


class MemoryConfig(BaseModel):
    """Memory database settings and compaction thresholds."""

    db_path: str = "data/memory.sqlite"
    short_term_limit: int = 30
    compaction_threshold: int = 60
    compaction_batch_size: int = 30
    backup: MemoryBackupConfig = Field(default_factory=MemoryBackupConfig)


class AntigravityProviderConfig(BaseModel):
    """Antigravity runtime settings (distinct from bot.antigravity.AntigravityConfig)."""

    data_dir: str = "data/antigravity"
    model: str = "gemini-4-hydrogen"
    effort: Optional[str] = "low"
    enable_sdk: bool = True
    enable_cli_fallback: bool = True
    max_retries: int = Field(default=3, ge=0)
    rate_limit_max_retries: int = Field(default=0, ge=0)
    rate_limit_cooldown_seconds: float = Field(default=300.0, ge=0.0)
    backoff_base: float = Field(default=1.0, ge=0.0)
    backoff_max: float = Field(default=30.0, gt=0.0)
    ops_mode: str = "auto"
    self_dev_mode: str = "auto"
    ops_cwd: Optional[str] = None
    # Lightweight default workspace for ops turns (no AGENTS.md / repo context).
    # `ops_cwd`, when set, remains an explicit override.
    ops_workspace_dir: Optional[str] = "data/ops_workspace"
    ops_tools_default_mode: Literal["on", "admin", "off"] = "on"

    @field_validator("ops_tools_default_mode", mode="before")
    @classmethod
    def _normalize_ops_tools_default_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class ModelRoleConfig(BaseModel):
    """Model and reasoning effort assignment for a specific bot role."""

    model: str
    effort: Optional[str] = "low"


class ModelRouterConfig(BaseModel):
    """Role-based model selection and cross-bucket fallback chain."""

    group: ModelRoleConfig = Field(
        default_factory=lambda: ModelRoleConfig(model="gemini-3.8-flash", effort="low")
    )
    proactive: ModelRoleConfig = Field(
        default_factory=lambda: ModelRoleConfig(model="gemini-3.8-flash", effort="low")
    )
    compaction: ModelRoleConfig = Field(
        default_factory=lambda: ModelRoleConfig(model="gemini-3.8-flash", effort="low")
    )
    dm_admin: ModelRoleConfig = Field(
        default_factory=lambda: ModelRoleConfig(model="gemini-4-hydrogen", effort="medium")
    )
    self_edit_judge: ModelRoleConfig = Field(
        default_factory=lambda: ModelRoleConfig(model="gemini-4-hydrogen", effort="high")
    )
    self_edit: ModelRoleConfig = Field(
        default_factory=lambda: ModelRoleConfig(model="claude-sonnet-5-5", effort="high")
    )
    fallback_models: List[ModelRoleConfig] = Field(
        default_factory=lambda: [
            ModelRoleConfig(model="gemini-3.8-flash", effort="low"),
            ModelRoleConfig(model="gemini-4-hydrogen", effort="low"),
            ModelRoleConfig(model="claude-sonnet-5-5", effort="medium"),
            ModelRoleConfig(model="gpt-oss-120b-medium", effort=None),
        ]
    )


class QuotaConfig(BaseModel):
    """Token budget limits, admin reserve, economy mode thresholds, and quota buckets."""

    enabled: bool = True
    hourly_token_budget: int = Field(default=300_000, ge=0)
    daily_token_budget: int = Field(default=2_500_000, ge=0)
    admin_reserve_ratio: float = Field(default=0.20, ge=0.0, le=0.9)
    economy_threshold_ratio: float = Field(default=0.75, gt=0.0, le=1.0)
    default_cooldown_seconds: float = Field(default=300.0, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_pct_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            updated = dict(data)
            if "economy_threshold_pct" in updated and "economy_threshold_ratio" not in updated:
                updated["economy_threshold_ratio"] = updated.pop("economy_threshold_pct")
            if "admin_reserve_pct" in updated and "admin_reserve_ratio" not in updated:
                updated["admin_reserve_ratio"] = updated.pop("admin_reserve_pct")
            return updated
        return data
    buckets: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "gemini_models": ["gemini-3.8-flash", "gemini-4-hydrogen"],
            "claude_gpt": ["claude-sonnet-5-5", "claude-opus-5-5", "gpt-oss-120b"],
            "gemini_argon": ["gemini-4-argon", "gemini-pro-orchestrator"],
        }
    )


class SelfEvolutionConfig(BaseModel):
    """Autonomous self-editing, adversarial judge, anti-churn, and staging gatekeeper settings."""

    enabled: bool = True
    staging_dir: str = "/opt/telegram-group-bot-staging"
    trigger_dir: str = "data/self_deploy"
    request_file: Optional[str] = None
    result_file: Optional[str] = None
    max_daily_edits: int = Field(default=2, ge=0)
    signature_cooldown_seconds: int = Field(default=21600, ge=0)
    file_cooldown_seconds: int = Field(default=14400, ge=0)
    error_threshold_count: int = Field(default=3, ge=1)
    error_window_seconds: int = Field(default=1800, ge=60)
    max_diff_lines: int = Field(default=400, ge=10)
    canary_wait_seconds: int = Field(default=18, ge=5, le=120)
    protected_paths: List[str] = Field(
        default_factory=lambda: [
            ".env",
            "data/",
            "deploy/bot-self-deploy.sh",
            "systemd/",
        ]
    )
    escalation_models: List[str] = Field(
        default_factory=lambda: [
            "gemini-4-hydrogen",
            "claude-sonnet-5-5",
            "claude-opus-5-5",
        ]
    )
    self_edit_effort: str = "high"
    judge_effort: str = "high"


class BotConfig(BaseModel):
    """Root configuration model combining all subsystem configs."""

    telegram: TelegramConfig
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    image_generation: ImageGenerationConfig = Field(default_factory=ImageGenerationConfig)
    antigravity: AntigravityProviderConfig = Field(default_factory=AntigravityProviderConfig)
    models: ModelRouterConfig = Field(default_factory=ModelRouterConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
    self_evolution: SelfEvolutionConfig = Field(default_factory=SelfEvolutionConfig)


class FriendProfile(BaseModel):
    """Rich conversational profile of a known friend or group participant."""

    name: str
    aliases: List[str] = Field(default_factory=list)
    bio: str = ""
    role: str = ""
    communication_style: str = ""
    banter_topics: List[str] = Field(default_factory=list)
    sensitive_topics: List[str] = Field(default_factory=list)


class FriendsConfig(BaseModel):
    """Friends directory mapping telegram_user_id to FriendProfile."""

    friends: Dict[int, FriendProfile] = Field(default_factory=dict)
