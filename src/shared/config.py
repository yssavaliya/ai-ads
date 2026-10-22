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

    # LLM Providers (free tiers: Groq -> Gemini)
    groq_api_key: str | None = Field(default=None, description="Groq API key")
    groq_model: str = Field(default="openai/gpt-oss-120b", description="Groq model")

    # Gemini API (fallback)
    gemini_api_key: str | None = Field(default=None, description="Gemini API key")
    gemini_model: str = Field(default="gemini-3.6-flash", description="Gemini model")

    # Hard cap on generated tokens per call. Without it long JSON answers (the prompt
    # forge returns ~8k chars) were cut off mid-string by provider defaults.
    llm_max_tokens: int = Field(default=8192, description="max_tokens sent with every LLM call")

    # Extra models tried (same key) when the primary model is rate limited / retired.
    # Groq and Gemini enforce limits per model, so a sibling model has its own quota.
    groq_fallback_models: str = Field(default="llama-3.3-70b-versatile,llama-3.1-8b-instant")
    gemini_fallback_models: str = Field(default="gemini-2.5-flash,gemini-2.0-flash")

    # More free providers, tried in LLM_PROVIDER_ORDER when the ones above fail.
    llm_provider_order: str = Field(
        default="groq,gemini,github,cerebras,openrouter,mistral,huggingface,pollinations"
    )
    github_models_enabled: bool = Field(default=False, description="Opt-in: GitHub Models via GITHUB_TOKEN (Actions: models: read)")
    github_models_model: str = Field(default="openai/gpt-4o-mini")
    cerebras_api_key: str | None = Field(default=None, description="Cerebras free tier key")
    cerebras_model: str = Field(default="llama-3.3-70b")
    openrouter_api_key: str | None = Field(default=None, description="OpenRouter key (free ':free' models)")
    openrouter_models: str = Field(
        default="meta-llama/llama-3.3-70b-instruct:free,deepseek/deepseek-chat-v3-0324:free,google/gemma-3-27b-it:free"
    )
    mistral_api_key: str | None = Field(default=None, description="Mistral free 'experiment' tier key")
    mistral_model: str = Field(default="mistral-small-latest")
    hf_llm_model: str = Field(default="meta-llama/Llama-3.3-70B-Instruct", description="Used with HF_TOKEN")
    llm_keyless_fallback: bool = Field(default=True, description="Allow the keyless Pollinations text endpoint as a last resort")
    pollinations_model: str = Field(default="openai")

    # Image providers (free tiers), tried in IMAGE_PROVIDER_ORDER; a local branded card
    # is rendered when all of them fail so a post never ships without a visual.
    image_provider_order: str = Field(default="pollinations,gemini,huggingface,together,cloudflare")
    image_card_fallback: bool = Field(default=True, description="Render a local branded card when every image provider fails")
    gemini_image_model: str = Field(default="gemini-2.5-flash-image")
    together_api_key: str | None = Field(default=None, description="Together AI key (free FLUX.1-schnell)")
    together_image_model: str = Field(default="black-forest-labs/FLUX.1-schnell-Free")
    cloudflare_account_id: str | None = Field(default=None, description="Cloudflare Workers AI account id")
    cloudflare_api_token: str | None = Field(default=None, description="Cloudflare Workers AI token")

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
    # Optional comma-separated GitHub logins allowed to submit feedback issues.
    # Unset = anyone who can open an issue (fine for private repos).
    feedback_allowed_users: str | None = Field(default=None, description="Allowed feedback issue authors")
    github_token: str | None = Field(default=None, description="GitHub token for issue creation")
    github_repo: str | None = Field(default=None, description="GitHub repo (owner/repo)")

    # Lead discovery wall-clock budget: stop scraping new candidates after this
    # many seconds so the job finishes (and persists state) inside its timeout.
    lead_time_budget_seconds: int = Field(default=2100, description="Max seconds spent scraping lead candidates")

    # Delivery: how finished content reaches people. "site" (default) publishes a static
    # website; "email" is the legacy SMTP e-mail; "both" does both.
    delivery_mode: str = Field(default="site", description="site | email | both")
    site_dir: Path = Field(default=Path("site"), description="Checkout of the published gh-pages branch")
    site_retention_days: int = Field(default=45, description="Days of prompts kept on the site")
    site_secret: str | None = Field(default=None, description="If set, page URLs carry a keyed hash and the index lists nothing")
    site_base_url: str | None = Field(default=None, description="Public URL override (default: <owner>.github.io/<repo>)")

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

    # Hugging Face (optional free-tier image fallback: FLUX.1-schnell)
    hf_token: str | None = Field(default=None, description="Hugging Face API token")

    # Dry run: write rendered emails/assets to logs/preview/ instead of sending
    dry_run: bool = Field(default=False, description="Preview mode — no SMTP, files written to logs/preview/")

    @field_validator("*", mode="before")
    @classmethod
    def coerce_empty(cls, value: Any, info: ValidationInfo) -> Any:
        """Unset GitHub secrets arrive as '' — treat them as "use the default"
        for every field (ints, bools AND strings such as GROQ_MODEL/SMTP_HOST)."""
        if info.field_name is None:
            return value
        field = cls.model_fields[info.field_name]
        return _coerce_empty(value, info.field_name, field.get_default(call_default_factory=True), cls)


settings = Settings()


def get_businesses_path() -> Path:
    return settings.data_dir / "businesses.json"


def get_weights_path() -> Path:
    return settings.data_dir / "weights.json"
