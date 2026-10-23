"""Free-tier LLM providers (all speak the OpenAI chat-completions protocol) plus a
small health tracker that lets the client skip providers/models that are rate
limited, out of quota, unauthorised or retired instead of waiting on them.

Every provider is optional: only the ones with a key (or keyless ones) take part,
in the order given by LLM_PROVIDER_ORDER.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.shared.config import settings

logger = logging.getLogger(__name__)

DEFAULT_ORDER = "groq,gemini,github,cerebras,openrouter,mistral,huggingface,pollinations"


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key: str
    models: tuple[str, ...]
    headers: dict[str, str] = field(default_factory=dict)


def csv(*values: str | None) -> list[str]:
    """Merge comma-separated lists, dropping blanks and duplicates, keeping order."""
    out: list[str] = []
    for value in values:
        for item in (value or "").split(","):
            item = item.strip()
            if item and item not in out:
                out.append(item)
    return out


def _referer() -> str:
    repo = settings.github_repo
    return f"https://github.com/{repo}" if repo else "https://github.com"


def _catalog() -> dict[str, Callable[[], Provider | None]]:
    s = settings

    def groq() -> Provider | None:
        if not s.groq_api_key:
            return None
        return Provider("groq", "https://api.groq.com/openai/v1", s.groq_api_key,
                        tuple(csv(s.groq_model, s.groq_fallback_models)))

    def gemini() -> Provider | None:
        if not s.gemini_api_key:
            return None
        return Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
                        s.gemini_api_key, tuple(csv(s.gemini_model, s.gemini_fallback_models)))

    def github() -> Provider | None:
        # Free in GitHub Actions with the built-in token (needs `models: read`).
        if not (s.github_token and s.github_models_enabled):
            return None
        return Provider("github", "https://models.github.ai/inference", s.github_token,
                        tuple(csv(s.github_models_model)),
                        {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})

    def cerebras() -> Provider | None:
        if not s.cerebras_api_key:
            return None
        return Provider("cerebras", "https://api.cerebras.ai/v1", s.cerebras_api_key,
                        tuple(csv(s.cerebras_model)))

    def openrouter() -> Provider | None:
        if not s.openrouter_api_key:
            return None
        return Provider("openrouter", "https://openrouter.ai/api/v1", s.openrouter_api_key,
                        tuple(csv(s.openrouter_models)),
                        {"HTTP-Referer": _referer(), "X-Title": "ai-ads"})

    def mistral() -> Provider | None:
        if not s.mistral_api_key:
            return None
        return Provider("mistral", "https://api.mistral.ai/v1", s.mistral_api_key,
                        tuple(csv(s.mistral_model)))

    def huggingface() -> Provider | None:
        if not s.hf_token:
            return None
        return Provider("huggingface", "https://router.huggingface.co/v1", s.hf_token,
                        tuple(csv(s.hf_llm_model)))

    def pollinations() -> Provider | None:
        # Keyless last resort: prompts go to a public endpoint, so it can be switched off.
        if not s.llm_keyless_fallback:
            return None
        return Provider("pollinations", "https://text.pollinations.ai/openai", "keyless",
                        tuple(csv(s.pollinations_model)))

    return {
        "groq": groq, "gemini": gemini, "github": github, "cerebras": cerebras,
        "openrouter": openrouter, "mistral": mistral, "huggingface": huggingface,
        "pollinations": pollinations,
    }


def provider_order() -> list[str]:
    return csv(settings.llm_provider_order) or csv(DEFAULT_ORDER)


def get_provider(name: str) -> Provider | None:
    factory = _catalog().get(name)
    return factory() if factory else None


def configured_providers() -> list[Provider]:
    found = [p for n in provider_order() if (p := get_provider(n)) is not None]
    return found


# ---------------------------------------------------------------------------
# health / circuit breaker
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Failure:
    kind: str                 # auth | model | rate_limit | transient | bad_request
    cooldown: float           # seconds; math.inf = for the rest of this process
    scope: str                # "provider" | "model" | "none"


def classify(exc: BaseException) -> Failure:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    msg = str(exc).lower()

    if status in (401, 403):
        return Failure("auth", math.inf, "provider")
    if "free limit" in msg or "free_limit" in msg or ("retry after" in msg and status == 402):
        match = re.search(r"retry after\s*(\d+(?:\.\d+)?)", msg)
        wait = float(match.group(1)) if match else 30.0
        return Failure("rate_limit", min(wait, 60.0), "model")
    if status == 402 or "no remaining credits" in msg or "insufficient_quota" in msg or "payment required" in msg:
        return Failure("quota", 3600.0, "provider")  # prepaid credits gone: don't retry within the hour
    if msg.startswith("non-json response"):
        return Failure("bad_response", 600.0, "provider")  # endpoint answered with something that is not an API reply
    if status in (404, 410) or "model_not_found" in msg or "does not exist" in msg or "not found" in msg:
        return Failure("model", math.inf, "model")
    if status == 429 or any(k in msg for k in ("rate limit", "rate_limit", "quota", "resource_exhausted", "too many requests")):
        retry_after = 0.0
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                retry_after = float(headers.get("retry-after", 0) or 0)
            except (TypeError, ValueError):
                retry_after = 0.0
        daily = any(k in msg for k in ("per day", "daily", "tpd", "rpd", "quota"))
        cooldown = retry_after if retry_after > 0 else (900.0 if daily else 60.0)
        return Failure("rate_limit", min(cooldown, 3600.0), "model")
    if status == 400:
        return Failure("bad_request", 0.0, "none")
    if status is not None and status >= 500:
        # "model overloaded" is per model: sibling models of the same provider may be fine
        return Failure("transient", 30.0, "model")
    return Failure("transient", 30.0, "provider")  # connection errors / timeouts: the provider itself is unreachable


class Health:
    """Remembers which (provider, model) pairs are unusable and until when."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._until: dict[tuple[str, str | None], float] = {}
        self._lock = threading.Lock()  # shared by every worker thread

    def reset(self) -> None:
        with self._lock:
            self._until.clear()

    def available(self, provider: str, model: str | None = None) -> bool:
        now = self._clock()
        with self._lock:
            return all(self._until.get(key, 0.0) <= now for key in ((provider, None), (provider, model)))

    def mark(self, provider: str, model: str | None, failure: Failure) -> None:
        if failure.scope == "none" or failure.cooldown <= 0:
            return
        key = (provider, None) if failure.scope == "provider" else (provider, model)
        until = math.inf if math.isinf(failure.cooldown) else self._clock() + failure.cooldown
        with self._lock:
            self._until[key] = max(self._until.get(key, 0.0), until)

    def seconds_until_ready(self, candidates: list[tuple[str, str]]) -> float:
        """Shortest wait before ANY candidate becomes usable (inf if none ever will)."""
        now = self._clock()
        best = math.inf
        with self._lock:
            for provider, model in candidates:
                wait = max(self._until.get((provider, None), 0.0), self._until.get((provider, model), 0.0)) - now
                best = min(best, max(wait, 0.0))
        return best


HEALTH = Health()


def describe_setup() -> str:
    """Human-readable summary of what is configured (never prints secrets)."""
    parts = []
    for name in provider_order():
        p = get_provider(name)
        parts.append(f"{name}: {', '.join(p.models)}" if p else f"{name}: not configured")
    return "; ".join(parts)


def log_setup(log: logging.Logger | Any = logger) -> None:
    log.info("LLM provider chain: %s", describe_setup())
