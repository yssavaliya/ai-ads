"""Test isolation: the suite must never touch the repo's real state.

Previously some tests wrote into the git-tracked data/weights.json (leaking
fixture businesses such as "regenbiz") and created agent.db in the project root.
"""
from __future__ import annotations

import os
import tempfile

# Must run before any `src` import: the SQLAlchemy engine is built at import time.
_SESSION_TMP = tempfile.mkdtemp(prefix="ai-ads-tests-")
os.environ["DATABASE_PATH"] = os.path.join(_SESSION_TMP, "agent.db")
os.environ["DRY_RUN"] = "true"  # no test can ever send a real email
os.environ["LLM_KEYLESS_FALLBACK"] = "false"  # no test may reach a public LLM endpoint
# Blank (not unset) so a developer's real keys / local .env can never leak into tests.
for _name in (
    "GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY",
    "HF_TOKEN", "GITHUB_TOKEN", "GITHUB_REPO", "APIFY_TOKEN", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM",
):
    os.environ[_name] = ""

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Per-test data/logs dirs. Tests that need their own still override these."""
    from src.shared.config import settings

    data_dir = tmp_path / "_data"
    logs_dir = tmp_path / "_logs"
    data_dir.mkdir()
    logs_dir.mkdir()
    from src.shared.llm_providers import HEALTH

    HEALTH.reset()  # provider cool-downs must not leak between tests
    monkeypatch.setattr(settings, "data_dir", data_dir)
    monkeypatch.setattr(settings, "logs_dir", logs_dir)
    monkeypatch.setattr(settings, "site_dir", tmp_path / "_site")
    monkeypatch.setattr(settings, "site_secret", None)
    monkeypatch.setattr(settings, "delivery_mode", "site")
    monkeypatch.setattr(settings, "templates_dir", ROOT / "templates")  # cwd-independent
    yield
