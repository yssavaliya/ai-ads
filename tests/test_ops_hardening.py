"""Posting cooldown, degraded-run resolution, partial-failure reporting."""
from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from src.delivery import github_feedback as gf
from src.scheduler import daily_job as dj

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _publish_module(tmp_path, entries):
    spec = importlib.util.spec_from_file_location("publish_buffer_ops", SCRIPTS / "publish_buffer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log = tmp_path / "published_log.json"
    log.write_text(json.dumps({"posted": entries}))
    module.PUBLISHED_LOG = log
    return module


def test_cooldown_counts_hours_since_the_newest_post(tmp_path):
    now = datetime(2026, 11, 14, 12, tzinfo=UTC)
    module = _publish_module(tmp_path, [
        {"id": "a", "at": (now - timedelta(hours=50)).isoformat()},
        {"id": "b", "at": (now - timedelta(hours=5)).isoformat()},
        {"id": "c", "at": "garbage"},
    ])
    assert round(module.hours_since_last_post(now), 1) == 5.0


def test_cooldown_is_none_without_history(tmp_path):
    module = _publish_module(tmp_path, [])
    assert module.hours_since_last_post() is None
    (tmp_path / "published_log.json").unlink()
    assert module.hours_since_last_post() is None


def test_queue_limit_default_is_conservative(tmp_path):
    module = _publish_module(tmp_path, [])
    assert module.DEFAULT_QUEUE_LIMIT <= 3 and module.DEFAULT_MIN_HOURS_BETWEEN_POSTS >= 20


class _Resp:
    def __init__(self, payload=None):
        self._payload = payload or []

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_resolve_degraded_runs_closes_only_degraded_issues(monkeypatch):
    calls = []
    issues = [
        {"number": 1, "title": "[pipeline] degraded run — 2026-08-23"},
        {"number": 2, "title": "unrelated bug"},
        {"number": 3, "title": "[pipeline] degraded run — 2026-11-07", "pull_request": {}},
    ]
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(issues))
    monkeypatch.setattr(httpx, "post", lambda url, **k: calls.append(("post", url)) or _Resp())
    monkeypatch.setattr(httpx, "patch", lambda url, **k: calls.append(("patch", url, k["json"])) or _Resp())
    closed = gf.GithubFeedback(token="t", repo="o/r").resolve_degraded_runs()
    assert closed == [1]
    assert ("patch", "https://api.github.com/repos/o/r/issues/1", {"state": "closed", "state_reason": "completed"}) in calls


def test_resolve_degraded_runs_is_a_noop_without_credentials():
    assert gf.GithubFeedback(token="", repo="").resolve_degraded_runs() == []


def test_resolve_degraded_runs_never_raises(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", boom)
    assert gf.GithubFeedback(token="t", repo="o/r").resolve_degraded_runs() == []


def _pipeline(monkeypatch, result):
    reports, resolved = [], []
    monkeypatch.setattr(dj, "load_businesses", lambda: [{"name": "Biz", "is_active": True}])
    pipeline = dj.DailyPipeline.__new__(dj.DailyPipeline)
    pipeline.run_for_business = lambda business: result  # type: ignore[method-assign]
    pipeline.feedback = type("F", (), {
        "report_degraded_run": lambda self, failures: reports.append(failures),
        "resolve_degraded_runs": lambda self: resolved.append(True),
    })()
    return pipeline, reports, resolved


def test_unscored_prompts_are_reported_as_a_degraded_scoring_stage(monkeypatch):
    pipeline, reports, resolved = _pipeline(monkeypatch, {"business": "Biz", "prompts": 3, "unscored": 2})
    pipeline.run_all()
    assert reports and reports[0][0]["stage"] == "scoring" and "2 of 3" in reports[0][0]["error"]
    assert not resolved


def test_clean_run_resolves_open_degraded_issues(monkeypatch):
    pipeline, reports, resolved = _pipeline(monkeypatch, {"business": "Biz", "prompts": 3, "unscored": 0})
    pipeline.run_all()
    assert not reports and resolved == [True]
