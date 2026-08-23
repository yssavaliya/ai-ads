from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _coerce_empty(value: Any, field_name: str, default: Any, cls: type) -> Any:
    """GitHub Actions passes unset secrets as '' — fall back to the field default."""
    if isinstance(value, str) and value.strip() == "":
        return default
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_path: Path = Field(default=Path("agent.db"), description="SQLite database path")

    # LLM Providers (using OpenCode CLI + API fallbacks)
    opencode_command: str = Field(default="opencode", description="OpenCode CLI command")
    opencode_timeout_seconds: int = Field(default=120, description="OpenCode CLI timeout")

    # Groq API (primary)
    groq_api_key: str | None = Field(default=None, description="Groq API key")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model")

    # Gemini API (fallback)
    gemini_api_key: str | None = Field(default=None, description="Gemini API key")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model")

    # Apify (social scraping)
    apify_token: str | None = Field(default=None, description="Apify API token")

    # Email (SMTP Gmail)
    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP host")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_user: str | None = Field(default=None, description="SMTP username")
    smtp_password: str | None = Field(default=None, description="SMTP app password")
    smtp_from: str | None = Field(default=None, description="Sender email")
    smtp_tls: bool = Field(default=True, description="Use TLS")

    # GitHub (for feedback issues)
    github_token: str | None = Field(default=None, description="GitHub token for issue creation")
    github_repo: str | None = Field(default=None, description="GitHub repo (owner/repo)")

    # Scheduling
    daily_cron: str = Field(default="29 18 * * *", description="Daily run cron (UTC) - 11:59 PM IST")
    weekly_cron: str = Field(default="0 2 * * 0", description="Weekly run cron (UTC) - Sunday 8 AM IST")
    feedback_cron: str = Field(default="0 * * * *", description="Feedback processing cron (hourly)")

    # Virality
    virality_threshold: int = Field(default=75, description="Minimum virality score (0-100)")
    max_regeneration_attempts: int = Field(default=5, description="Max prompt regeneration attempts")

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    prompts_dir: Path = Field(default=Path("prompts"), description="Prompt templates directory")
    templates_dir: Path = Field(default=Path("templates"), description="Email templates directory")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")

    # Timezone
    default_timezone: str = Field(default="Asia/Kolkata", description="Default timezone")

    @field_validator(
        "smtp_port",
        "opencode_timeout_seconds",
        "virality_threshold",
        "max_regeneration_attempts",
        mode="before",
    )
    @classmethod
    def coerce_empty_int(cls, value: Any, info: ValidationInfo) -> Any:
        return _coerce_empty(value, info.field_name, cls.model_fields[info.field_name].default, cls)

    @field_validator("smtp_tls", mode="before")
    @classmethod
    def coerce_empty_bool(cls, value: Any, info: ValidationInfo) -> Any:
        return _coerce_empty(value, info.field_name, cls.model_fields[info.field_name].default, cls)


settings = Settings()


def get_businesses_path() -> Path:
    return settings.data_dir / "businesses.json"


def get_weights_path() -> Path:
    return settings.data_dir / "weights.json"