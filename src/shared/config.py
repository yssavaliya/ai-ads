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

    # Optional provider credentials (Groq is the default text provider, Cloudflare the default image provider).
    groq_api_key: str | None = Field(default=None, description="Groq API key")
    groq_model: str = Field(default="openai/gpt-oss-120b", description="Groq model")

    # Gemini API (fallback)
    gemini_api_key: str | None = Field(default=None, description="Gemini API key")
    gemini_model: str = Field(default="gemini-3.6-flash", description="Gemini model")

    # Hard cap on generated tokens per call. Without it long JSON answers (the prompt
    # forge returns ~8k chars) were cut off mid-string by provider defaults.
    llm_max_tokens: int = Field(default=8192, description="max_tokens sent with every LLM call")

    # Free tiers meter tokens per minute as prompt + requested max_tokens, so a fixed 8192 reservation made
    # every Groq call "too large" (413). The client sizes max_tokens per call to fit these budgets.
    # Format: "key=value,key=value".
    llm_tpm_limits: str = Field(default="groq=8000", description="provider=tokens-per-minute budget per request")
    # gpt-oss models spend completion tokens on hidden reasoning; at the default effort a capped request can come
    # back with EMPTY content (all tokens used thinking). "low" keeps the answer inside the budget. Empty = off.
    groq_reasoning_effort: str = Field(default="low", description="reasoning_effort for Groq openai/gpt-oss-* models")
    llm_model_output_caps: str = Field(
        default="qwen/qwen3.8-27b=1000", description="model=max output tokens (providers' output-per-minute limits)"
    )

    # Extra models tried (same key) when the primary model is rate limited / retired.
    # Groq and Gemini enforce limits per model, so a sibling model has its own quota.
    groq_fallback_models: str = Field(default="openai/gpt-oss-20b,qwen/qwen3.8-27b")
    gemini_fallback_models: str = Field(default="gemini-3.5-flash,gemini-3.1-flash-lite,gemini-flash-latest")

    # Providers tried in LLM_PROVIDER_ORDER; the scheduled workflows set it explicitly (groq).
    llm_provider_order: str = Field(default="groq")
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
    hf_llm_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct", description="Apache-2.0 model used with HF_TOKEN")
    llm_keyless_fallback: bool = Field(default=False, description="Opt-in: let the keyless public Pollinations text endpoint see prompts as a last resort")
    pollinations_model: str = Field(default="openai")

    # Image providers tried in IMAGE_PROVIDER_ORDER; a local branded card
    # is rendered when all of them fail so a post never ships without a visual.
    image_provider_order: str = Field(default="cloudflare")
    image_card_fallback: bool = Field(default=True, description="Render a local branded card when every image provider fails")
    gemini_image_model: str = Field(default="gemini-3.1-flash-image,gemini-2.5-flash-image", description="Comma-separated, tried in order")
    together_api_key: str | None = Field(default=None, description="Together AI key (free FLUX.1-schnell)")
    together_image_model: str = Field(default="black-forest-labs/FLUX.1-schnell-Free")
    cloudflare_account_id: str | None = Field(default=None, description="Cloudflare Workers AI account id")
    cloudflare_api_token: str | None = Field(default=None, description="Cloudflare Workers AI token")
    cloudflare_model: str = Field(default="@cf/meta/llama-3.3-70b-instruct-fp8-fast", description="Workers AI text model")

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
    # Comma-separated GitHub logins allowed to submit feedback issues. Unset = feedback is ignored (fails closed).
    feedback_allowed_users: str | None = Field(
        default=None, description="GitHub logins whose feedback issues count. REQUIRED: feedback fails closed without it"
    )
    feedback_signing_key: str | None = Field(
        default=None, description="HMAC key for feedback tokens. REQUIRED: without it no feedback links are issued or accepted"
    )
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

    # Buffer (Social publishing)
    buffer_access_token: str | None = Field(default=None, description="Buffer API access token")
    buffer_profile_ids: str | None = Field(default=None, description="Comma-separated Buffer profile IDs (or single ID)")
    buffer_draft_mode: bool = Field(default=True, description="Whether to create Buffer posts as drafts for review")
    buffer_now: bool = Field(default=False, description="Whether to post immediately to Buffer channels rather than queueing")

    # Concurrency caps (1 = the old strictly sequential behaviour). Independent network work overlaps;
    # the LLM cap is small because free tiers rate-limit tokens per minute, not requests.
    llm_concurrency: int = Field(default=2, ge=1, le=16, description="Max simultaneous LLM calls")
    image_concurrency: int = Field(default=3, ge=1, le=16, description="Max simultaneous image-provider calls")
    fetch_concurrency: int = Field(default=2, ge=1, le=16, description="Max simultaneous scraper runs")


    # Virality
    virality_threshold: int = Field(default=75, description="Minimum virality score (0-100)")
    below_threshold_policy: str = Field(default="flag", description="flag | block: what to do with prompts under the quality bar")
    lead_strategy_count: int = Field(default=3, ge=0, le=10, description="Top leads that get a recommended campaign per run")
    experiments_enabled: bool = Field(default=True, description="Run one designed experiment per week (hook, post block, format)")
    competitor_search_enabled: bool = Field(default=True, description="Find competitors by Instagram search (real accounts), not only LLM memory")
    competitor_probe_limit: int = Field(default=10, ge=5, le=20, description="Max competitor candidates scraped per run")
    campaign_strategy_enabled: bool = Field(
        default=True, description="Run the campaign-strategy stage (objective, angles, rubric) before creative generation"
    )
    max_regeneration_attempts: int = Field(default=5, description="Max prompt regeneration attempts")

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    templates_dir: Path = Field(default=Path("templates"), description="Jinja templates directory")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")

    # Timezone
    default_timezone: str = Field(default="Asia/Kolkata", description="Default timezone")

    # Hugging Face (optional free-tier image fallback: FLUX.1-schnell)
    hf_token: str | None = Field(default=None, description="Hugging Face API token")

    # Dry run: write rendered emails/assets to logs/preview/ instead of sending
    dry_run: bool = Field(default=False, description="Preview mode — no SMTP, files written to logs/preview/")

    @field_validator("delivery_mode")
    @classmethod
    def _valid_delivery_mode(cls, value: str) -> str:
        mode = (value or "site").strip().lower()
        if mode not in ("site", "email", "both"):
            raise ValueError(f"DELIVERY_MODE must be site, email or both (got {value!r})")
        return mode

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
