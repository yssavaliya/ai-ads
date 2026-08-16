from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    groq_model: str = Field(default="llama-3.1-70b-versatile", description="Groq model")
    
    # Gemini API (fallback)
    gemini_api_key: str | None = Field(default=None, description="Gemini API key")
    gemini_model: str = Field(default="gemini-1.5-flash", description="Gemini model")
    
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
    
    # Embeddings (local, no API)
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local embedding model"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    
    # Timezone
    default_timezone: str = Field(default="Asia/Kolkata", description="Default timezone")


settings = Settings()


def get_businesses_path() -> Path:
    return settings.data_dir / "businesses.json"


def get_weights_path() -> Path:
    return settings.data_dir / "weights.json"