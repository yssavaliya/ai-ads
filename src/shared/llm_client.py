from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.shared.config import settings

logger = logging.getLogger(__name__)


class OpenCodeError(RuntimeError):
    def __init__(self, message: str, exit_code: int, stdout: str, stderr: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class OpenCodeResult:
    stdout: str
    stderr: str
    exit_code: int
    elapsed_seconds: float


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
    return text


class LLMClient:
    """LLM interface: primary = OpenCode CLI (keyless/free gateway),
    fallbacks = Groq API (free tier), then Gemini API (free tier).
    No local models are ever run on CPU."""

    def __init__(
        self,
        timeout_seconds: int | None = None,
        command: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds or settings.opencode_timeout_seconds
        self.command = command or settings.opencode_command

    def run(
        self,
        prompt: str,
        label: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> OpenCodeResult:
        """Run a prompt through the LLM chain. Returns first successful result."""
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            start = time.monotonic()
            try:
                result = self._run_opencode(prompt, label)
                logger.info(
                    "::notice::[%s] opencode answered in %.1fs (%d chars)",
                    label or "llm",
                    result.elapsed_seconds,
                    len(result.stdout),
                )
                return result
            except OpenCodeError as exc:
                last_error = exc
                logger.warning(
                    "opencode attempt %s/%s failed (exit=%s): %s",
                    attempt, max_retries, exc.exit_code, (exc.stderr or exc.stdout or "")[:300],
                )
                if attempt < max_retries:
                    time.sleep(2 * attempt)

        logger.info("opencode unavailable after %s attempts; trying Groq", max_retries)
        result = self._try_groq(prompt, label, temperature)
        if result is not None:
            return result

        logger.info("Groq unavailable; trying Gemini")
        result = self._try_gemini(prompt, label, temperature)
        if result is not None:
            return result

        raise RuntimeError(f"All LLM providers failed for {label}: {last_error}") from last_error

    def _run_opencode(self, prompt: str, label: str | None) -> OpenCodeResult:
        start = time.monotonic()
        command_line = [self.command, "run", prompt]
        command_line = self._resolve_windows_command(command_line)
        popen_command = command_line if isinstance(command_line, str) else [str(part) for part in command_line]

        try:
            process = subprocess.Popen(
                popen_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
        except FileNotFoundError as exc:
            raise OpenCodeError(
                f"OpenCode binary not found: {self.command!r}", -1, "", str(exc)
            ) from exc

        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.is_set():
                heartbeat_stop.wait(30)
                if not heartbeat_stop.is_set():
                    logger.info(
                        "::notice::[OpenCode:%s] still running... %.0fs elapsed (timeout=%ds)",
                        label or "llm",
                        time.monotonic() - start,
                        self.timeout_seconds,
                    )

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)
            process.kill()
            stdout, stderr = process.communicate()
            raise OpenCodeError("OpenCode command timed out", -1, stdout or "", stderr or "")
        else:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)

        elapsed = time.monotonic() - start
        if process.returncode != 0:
            raise OpenCodeError(
                f"OpenCode command failed with exit code {process.returncode}",
                process.returncode,
                stdout or "",
                stderr or "",
            )
        return OpenCodeResult(stdout=stdout or "", stderr=stderr or "", exit_code=0, elapsed_seconds=elapsed)

    def _resolve_windows_command(self, command_line: list[str]) -> list[str] | str:
        if sys.platform != "win32":
            return command_line
        resolved = shutil.which(self.command)
        if resolved is None:
            return command_line
        if resolved.lower().endswith((".cmd", ".bat")):
            rest = subprocess.list2cmdline(command_line[1:]) if len(command_line) > 1 else ""
            return f'cmd.exe /d /c call "{resolved}"' + (f" {rest}" if rest else "")
        return [resolved, *command_line[1:]]

    def _try_groq(self, prompt: str, label: str | None, temperature: float) -> OpenCodeResult | None:
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY unset; skipping Groq fallback")
            return None
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=60,
                max_retries=2,
            )
            start = time.monotonic()
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = (response.choices[0].message.content or "").strip()
            logger.info(
                "::notice::[%s] Groq answered via %s in %.1fs (%d chars)",
                label or "llm", settings.groq_model, time.monotonic() - start, len(content),
            )
            return OpenCodeResult(stdout=strip_markdown_fences(content), stderr="", exit_code=0, elapsed_seconds=time.monotonic() - start)
        except Exception as exc:
            logger.error("Groq failed for %s: %s", label or "llm", exc)
            return None

    def _try_gemini(self, prompt: str, label: str | None, temperature: float) -> OpenCodeResult | None:
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY unset; skipping Gemini fallback")
            return None
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=60,
                max_retries=2,
            )
            start = time.monotonic()
            response = client.chat.completions.create(
                model=settings.gemini_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = (response.choices[0].message.content or "").strip()
            logger.info(
                "::notice::[%s] Gemini answered via %s in %.1fs (%d chars)",
                label or "llm", settings.gemini_model, time.monotonic() - start, len(content),
            )
            return OpenCodeResult(stdout=strip_markdown_fences(content), stderr="", exit_code=0, elapsed_seconds=time.monotonic() - start)
        except Exception as exc:
            logger.error("Gemini failed for %s: %s", label or "llm", exc)
            return None
