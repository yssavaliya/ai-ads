"""The health-check scripts run in CI (drift_check.yml). They crashed with a
TypeError for months while the workflow reported success."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from src.shared.config import settings
from src.shared.llm_client import LLMClient, LLMResult

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def two_providers(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "g")
    monkeypatch.setattr(settings, "cerebras_api_key", "c")
    monkeypatch.setattr(settings, "llm_provider_order", "groq,cerebras")


def _ok(name):
    return LLMResult(stdout="Bake the viral loaf", stderr="", exit_code=0, elapsed_seconds=0.1, provider=name, model="m")


def test_check_llm_probes_every_configured_provider(two_providers, capsys):
    check = _load("check_llm")
    # autospec enforces the real signature, so a wrong call raises TypeError
    with patch.object(LLMClient, "_try_provider", autospec=True, side_effect=lambda self, name, *a: _ok(name)) as probe:
        assert check.main() == 0
    assert [c.args[1] for c in probe.call_args_list] == ["groq", "cerebras"]
    assert "2/2 providers healthy" in capsys.readouterr().out


def test_check_llm_passes_when_one_fallback_works_but_flags_the_broken_one(two_providers, capsys):
    check = _load("check_llm")
    with patch.object(
        LLMClient, "_try_provider", autospec=True,
        side_effect=lambda self, name, *a: None if name == "groq" else _ok(name),
    ):
        assert check.main() == 0  # one healthy provider is enough...
    out = capsys.readouterr().out
    assert "providers failing: groq" in out and "1/2 providers healthy" in out  # ...but the failure is visible


def test_check_llm_fails_when_every_provider_fails(two_providers):
    check = _load("check_llm")
    with patch.object(LLMClient, "_try_provider", autospec=True, return_value=None):
        assert check.main() == 1


def test_check_llm_fails_when_nothing_is_configured(monkeypatch):
    check = _load("check_llm")
    monkeypatch.setattr(settings, "llm_keyless_fallback", False)
    assert check.main() == 1


@pytest.mark.parametrize("name", ["check_llm", "check_apify", "init_db", "onboard_business", "run_auto_learn"])
def test_every_script_imports_without_pythonpath(name):
    _load(name)  # raises ModuleNotFoundError if a script relies on PYTHONPATH
