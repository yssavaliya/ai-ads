"""A minimal OpenAI-compatible `/chat/completions` client on plain httpx.

Every provider in the chain (Groq, Gemini, Cerebras, OpenRouter, Mistral, Hugging Face, ...) speaks
the same protocol, and we only ever make one kind of call. The official SDK pulled in ~6.7 s of
imports (measured, cold) and a large dependency tree on every workflow run for that single POST.

The client keeps the SDK's call shape (`client.chat.completions.create(...)`), so the retry /
circuit-breaker logic in llm_client is unchanged, and it reuses TCP/TLS connections per endpoint.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx

_clients: dict[tuple[str, float], httpx.Client] = {}
_lock = threading.Lock()


class ChatHTTPError(Exception):
    """An HTTP error reply. Carries `.status_code` and `.response` (headers, e.g. Retry-After) and the
    start of the body in the message: providers put the real reason there ("tokens per day", "model
    does not exist"), and llm_providers.classify() reads it to decide how long to back off."""

    def __init__(self, response: httpx.Response) -> None:
        self.status_code = response.status_code
        self.response = response
        try:
            body = response.text[:400].replace("\n", " ")
        except Exception:  # noqa: BLE001
            body = ""
        super().__init__(f"Error code: {response.status_code} - {body}")


@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message
    finish_reason: str | None = None


@dataclass
class _Response:
    choices: list[_Choice]


def _client_for(timeout: float) -> httpx.Client:
    """One pooled client per timeout value (connections are reused across calls and threads)."""
    key = ("pool", float(timeout))
    with _lock:
        client = _clients.get(key)
        if client is None:
            client = httpx.Client(
                timeout=httpx.Timeout(timeout, connect=min(timeout, 15.0)),
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            )
            _clients[key] = client
        return client


def reset_clients() -> None:
    """Close and forget pooled clients (tests, or after changing proxy/TLS settings)."""
    with _lock:
        for client in _clients.values():
            client.close()
        _clients.clear()


class HttpChatClient:
    def __init__(self, base_url: str, api_key: str, timeout: float, headers: dict[str, str] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", **(headers or {})}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int | None = None) -> Any:
        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        response = _client_for(self.timeout).post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers)
        if response.status_code >= 400:
            raise ChatHTTPError(response)
        if "json" not in response.headers.get("content-type", "").lower():
            return response.text  # not an API reply (some gateways answer a plain "OK"): the caller diagnoses it
        try:
            data = response.json()
            choice = data["choices"][0]
            return _Response([_Choice(_Message(choice["message"].get("content")), choice.get("finish_reason"))])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed chat completion reply: {response.text[:200]!r}") from exc
