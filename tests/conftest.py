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

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Per-test data/logs dirs. Tests that need their own still override these."""
    from src.shared.config import settings

    data_dir = tmp_path / "_data"
    logs_dir = tmp_path / "_logs"
    data_dir.mkdir()
    logs_dir.mkdir()
    monkeypatch.setattr(settings, "data_dir", data_dir)
    monkeypatch.setattr(settings, "logs_dir", logs_dir)
    yield
