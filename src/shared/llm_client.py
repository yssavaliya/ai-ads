from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from src.shared.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    stdout: str
    stderr: str
    exit_code: int
    elapsed_seconds: float


# Backwards-compatible alias (older code/tests referred to OpenCodeResult).
OpenCodeResult = LLMResult


class AllProvidersFailedError(RuntimeError):
    """Raised when every configured LLM provider failed for a call."""


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def extract_json_text(text: str) -> str:
    text = strip_markdown_fences(text)
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass
    return text


class LLMClient:
    """LLM interface — free tiers only: Groq API → Gemini API → Pollinations
    (keyless last resort). No local models are ever run on CPU."""

    def __init__(self, timeout_seconds: int | None = None, command: str | None = None) -> None:
        self.timeout_seconds = timeout_seconds or 60

    def run(
        self,
        prompt: str,
        label: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> LLMResult:
        """Run a prompt through the provider chain. Returns first success;
        raises AllProvidersFailedError when every provider fails."""
        result = self._try_groq(prompt, label, temperature)
        if result is not None:
            return result

        logger.info("[%s] Groq unavailable; trying Gemini", label or "llm")
        result = self._try_gemini(prompt, label, temperature)
        if result is not None:
            return result

        logger.info("[%s] Gemini unavailable; trying Pollinations (keyless)", label or "llm")
        result = self._try_pollinations(prompt, label, temperature)
        if result is not None:
            return result

        raise AllProvidersFailedError(
            f"All LLM providers failed for {label}. "
            "Set GROQ_API_KEY (free at console.groq.com) as a repo secret."
        )

    def _try_groq(self, prompt: str, label: str | None, temperature: float) -> LLMResult | None:
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY unset; skipping Groq")
            return None
        # Current free-tier models: https://console.groq.com/docs/models
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=self.timeout_seconds,
                max_retries=2,
            )
            start = time.monotonic()
            response = self._chat(client, settings.groq_model, prompt, temperature, 2)
            content = (response.choices[0].message.content or "").strip()
            logger.info(
                "::notice::[%s] Groq answered via %s in %.1fs (%d chars)",
                label or "llm", settings.groq_model, time.monotonic() - start, len(content),
            )
            return LLMResult(stdout=strip_markdown_fences(content), stderr="", exit_code=0, elapsed_seconds=time.monotonic() - start)
        except Exception as exc:
            logger.error("Groq failed for %s: %s", label or "llm", exc)
            return None

    def _try_gemini(self, prompt: str, label: str | None, temperature: float) -> LLMResult | None:
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY unset; skipping Gemini")
            return None
        # Current free-tier models: https://ai.google.dev/gemini-api/docs/models
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=self.timeout_seconds,
                max_retries=2,
            )
            start = time.monotonic()
            response = self._chat(client, settings.gemini_model, prompt, temperature, 2)
            content = (response.choices[0].message.content or "").strip()
            logger.info(
                "::notice::[%s] Gemini answered via %s in %.1fs (%d chars)",
                label or "llm", settings.gemini_model, time.monotonic() - start, len(content),
            )
            return LLMResult(stdout=strip_markdown_fences(content), stderr="", exit_code=0, elapsed_seconds=time.monotonic() - start)
        except Exception as exc:
            logger.error("Gemini failed for %s: %s", label or "llm", exc)
            return None

    @staticmethod
    def _chat(client: Any, model: str, prompt: str, temperature: float, max_retries: int):
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001 - provider SDK raises many types
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(2 * attempt)
        raise last_exc  # type: ignore[misc]

    def _try_pollinations(self, prompt: str, label: str | None, temperature: float) -> LLMResult | None:
        """Keyless free fallback (text.pollinations.ai GET API) — anonymous tier
        is rate-limited and low-quality; loud logs make its use visible."""
        try:
            import httpx
            from urllib.parse import quote

            start = time.monotonic()
            with httpx.Client(timeout=120) as client:
                response = client.get(
                    f"https://text.pollinations.ai/{quote(prompt, safe='')}",
                    params={"model": "openai-fast"},
                )
                response.raise_for_status()
                content = response.text.strip()
            if not content or content.startswith('{"error"'):
                logger.error("Pollinations returned no usable content for %s", label or "llm")
                return None
            logger.warning(
                "::warning::[%s] used Pollinations fallback (keyless, lowest quality) in %.1fs "
                "— set GROQ_API_KEY for reliable output",
                label or "llm", time.monotonic() - start,
            )
            return LLMResult(
                stdout=strip_markdown_fences(content), stderr="", exit_code=0, elapsed_seconds=time.monotonic() - start
            )
        except Exception as exc:
            logger.error("Pollinations failed for %s: %s", label or "llm", exc)
            return None


def run_llm(prompt: str, label: str | None = None, temperature: float = 0.7) -> str:
    """Convenience wrapper returning just the text output."""
    return LLMClient().run(prompt, label=label, temperature=temperature).stdout
