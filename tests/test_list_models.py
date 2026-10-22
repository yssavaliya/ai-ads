"""scripts/list_models.py: the diagnostic that replaces guessing free-tier model names."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx

from src.shared.config import settings

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "list_models.py"


def _load():
    spec = importlib.util.spec_from_file_location("list_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lists_offered_models_marks_configured_ones_and_flags_missing(monkeypatch, capsys):
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "groq_model", "gpt-oss")
    monkeypatch.setattr(settings, "groq_fallback_models", "retired-model")
    monkeypatch.setattr(settings, "llm_provider_order", "groq")
    seen = {}

    def fake_get(url, headers, timeout):
        seen["url"], seen["auth"] = url, headers["Authorization"]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": [{"id": "gpt-oss"}, {"id": "qwen3"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    assert _load().main() == 0
    out = capsys.readouterr().out
    assert seen["url"] == "https://api.groq.com/openai/v1/models" and seen["auth"] == "Bearer k"
    assert " * gpt-oss" in out and "   qwen3" in out
    assert "configured but not offered to this key: retired-model" in out  # exactly the real 404 case


def test_gemini_ids_lose_their_models_prefix_and_openrouter_shows_only_free(monkeypatch, capsys):
    monkeypatch.setattr(settings, "gemini_api_key", "k")
    monkeypatch.setattr(settings, "openrouter_api_key", "o")
    monkeypatch.setattr(settings, "llm_provider_order", "gemini,openrouter")
    payloads = {
        "generativelanguage": {"data": [{"id": "models/gemini-x"}]},
        "openrouter": {"data": [{"id": "a/b:free"}, {"id": "paid/model"}]},
    }
    monkeypatch.setattr(httpx, "get", lambda url, headers, timeout: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: next(v for k, v in payloads.items() if k in url)))
    _load().main()
    out = capsys.readouterr().out
    assert "gemini-x" in out and "models/gemini-x" not in out
    assert "a/b:free" in out and "paid/model" not in out


def test_a_failing_provider_never_fails_the_diagnostic(monkeypatch, capsys):
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "llm_provider_order", "groq")

    def boom(url, headers, timeout):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", boom)
    assert _load().main() == 0
    assert "could not list models: ConnectError" in capsys.readouterr().out
