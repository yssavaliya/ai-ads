"""The health-check scripts run in CI (drift_check.yml). They crashed with a
TypeError for months while the workflow reported success."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from src.shared.llm_client import LLMClient, LLMResult

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_llm_calls_providers_with_a_valid_signature():
    check = _load("check_llm")
    ok = LLMResult(stdout="Bake the viral loaf", stderr="", exit_code=0, elapsed_seconds=0.1)
    # autospec enforces the real signatures, so a wrong call raises TypeError
    with (
        patch.object(LLMClient, "_try_groq", autospec=True, return_value=ok),
        patch.object(LLMClient, "_try_gemini", autospec=True, return_value=None),
    ):
        assert check.main() == 0  # one healthy provider is enough


def test_check_llm_fails_when_every_provider_fails():
    check = _load("check_llm")
    with (
        patch.object(LLMClient, "_try_groq", autospec=True, return_value=None),
        patch.object(LLMClient, "_try_gemini", autospec=True, return_value=None),
    ):
        assert check.main() == 1


@pytest.mark.parametrize("name", ["check_llm", "check_apify", "init_db", "onboard_business", "run_auto_learn"])
def test_every_script_imports_without_pythonpath(name):
    _load(name)  # raises ModuleNotFoundError if a script relies on PYTHONPATH
