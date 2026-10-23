"""Audit H-06 (per-business quality bar), H-07 (timezones), H-08 (config propagation)."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.generation.virality_gate import ViralityGate, apply_threshold_policy
from src.learning.auto_learner import AutoLearner
from src.scheduler.daily_job import DailyPipeline
from src.shared.config import settings
from src.storage.models import Base, Business


# ===================================================================== H-07 timezones
@pytest.mark.parametrize("value,expected", [
    ("2026-10-01T14:00:00Z", datetime(2026, 10, 1, 14, 0)),
    ("2026-10-01T19:30:00+05:30", datetime(2026, 10, 1, 14, 0)),     # same instant, other offset
    ("2026-10-01T09:00:00-05:00", datetime(2026, 10, 1, 14, 0)),
    ("2026-10-01T14:00:00", datetime(2026, 10, 1, 14, 0)),           # naive = UTC (instaloader date_utc)
    (1790863200, datetime(2026, 10, 1, 14, 0)),                      # epoch seconds
    (datetime(2026, 10, 1, 19, 30, tzinfo=__import__("zoneinfo").ZoneInfo("Asia/Kolkata")), datetime(2026, 10, 1, 14, 0)),
])
def test_parse_date_converts_to_naive_utc(value, expected):
    assert AutoLearner._parse_date(value) == expected


@pytest.mark.parametrize("junk", [None, "", "not a date", True, 12, [], {}])
def test_parse_date_rejects_garbage(junk):
    assert AutoLearner._parse_date(junk) is None


def _posts(day, hour_utc, n=4):
    return {"posts": [{"posted_at": datetime(2026, 7 if day == "jul" else 1, 1 + i, hour_utc).isoformat() + "Z"} for i in range(n)]}


def test_best_post_hour_is_the_businesss_local_hour():
    pipe = DailyPipeline.__new__(DailyPipeline)
    assert "07:00 PM" in pipe._recommend_post_time({"timezone": "Asia/Kolkata"}, _posts("jul", 14))  # 14:00Z = 19:30 IST
    assert "02:00 PM" in pipe._recommend_post_time({"timezone": "UTC"}, _posts("jul", 14))


@pytest.mark.parametrize("month,expected", [("jul", "10:00 AM"), ("jan", "09:00 AM")])
def test_best_post_hour_respects_daylight_saving(month, expected):
    """14:00Z is 10:00 in New York in July (EDT) but 09:00 in January (EST)."""
    pipe = DailyPipeline.__new__(DailyPipeline)
    out = pipe._recommend_post_time({"timezone": "America/New_York"}, _posts(month, 14))
    assert expected in out and "America/New York" in out


def test_matching_window_compares_the_same_instant_not_wall_clock_digits():
    learner = AutoLearner()
    sent = datetime(2026, 10, 1, 20, 0)  # naive UTC
    prompts = [{"id": 1, "prompt_type": "educational", "prompt_text": "alpha beta gamma delta", "caption": "alpha beta gamma delta",
                "format": "", "sent_at": sent}]
    before = {"post_id": "a", "caption": "alpha beta gamma delta", "posted_at": "2026-10-02T01:00:00+05:30"}  # = 19:30Z, BEFORE the send
    after = {"post_id": "b", "caption": "alpha beta gamma delta", "posted_at": "2026-10-02T03:00:00+05:30"}   # = 21:30Z, after
    assert learner._match([before], prompts) == []
    assert [m[0] for m in learner._match([after], prompts)] == [1]


# ===================================================================== H-06 quality bar
class _Forge:
    def generate_variants(self, **kw):
        return [{"prompt_type": t, "prompt_text": f"text {t}", "format": "reel"} for t in kw["prompt_types"]]

    def refine_with_hints(self, best, *a):
        return best


def _predictor(score):
    class P:
        def score(self, *a, **k):
            return {"virality_score": score, "virality_label": "MEDIUM", "virality_probability": 0.5, "key_factors": [], "risks": [],
                    "reasoning": "", "improvement_hints": []}
    return P()


@pytest.mark.parametrize("bar,below", [(70, False), (72, False), (73, True), (90, True)])
def test_gate_judges_against_the_per_business_bar(bar, below):
    gate = ViralityGate(threshold=75, max_attempts=1, forge=_Forge(), predictor=_predictor(72))
    out = gate.run({}, {}, {}, ["educational"], threshold=bar)
    assert out and all(p["threshold"] == bar and p["below_threshold"] is below for p in out)


def test_default_bar_still_applies_without_an_override():
    out = ViralityGate(threshold=75, max_attempts=1, forge=_Forge(), predictor=_predictor(72)).run({}, {}, {}, ["educational"])
    assert out[0]["threshold"] == 75 and out[0]["below_threshold"] is True


def test_a_higher_bar_keeps_refining_where_a_lower_one_stops():
    forge, calls = _Forge(), {"n": 0}
    orig = forge.generate_variants

    def counting(**kw):
        calls["n"] += 1
        return orig(**kw)

    forge.generate_variants = counting  # type: ignore[method-assign]
    class P:
        def score(self, *a, **k):
            return {"virality_score": 72, "virality_label": "M", "virality_probability": 0.5, "key_factors": [], "risks": [], "reasoning": "",
                    "improvement_hints": ["h"]}
    ViralityGate(threshold=75, max_attempts=3, forge=forge, predictor=P()).run({}, {}, {}, ["educational"], threshold=70)
    low = calls["n"]
    calls["n"] = 0
    ViralityGate(threshold=75, max_attempts=3, forge=forge, predictor=P()).run({}, {}, {}, ["educational"], threshold=90)
    assert low == 1 and calls["n"] == 3  # passes at 70 immediately; at 90 it uses every attempt


def test_policy_flag_keeps_but_marks_and_block_drops(monkeypatch):
    prompts = [{"title": "ok", "virality_score": 88, "threshold": 80, "below_threshold": False},
               {"title": "weak", "virality_score": 61, "threshold": 80, "below_threshold": True}]
    flagged = apply_threshold_policy([dict(p) for p in prompts], "flag")
    assert [p["title"] for p in flagged] == ["ok", "weak"]
    assert "Below your quality bar (61/100, bar 80)" in flagged[1]["warnings"][0] and "warnings" not in flagged[0]
    assert [p["title"] for p in apply_threshold_policy([dict(p) for p in prompts], "block")] == ["ok"]
    monkeypatch.setattr(settings, "below_threshold_policy", "block")
    assert [p["title"] for p in apply_threshold_policy([dict(p) for p in prompts])] == ["ok"]
    assert settings.below_threshold_policy == "block"


# ===================================================================== H-08 config propagation
def _session(tmp_path: Path):
    eng = create_engine(f"sqlite:///{tmp_path / 'cfg.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    return sessionmaker(bind=eng, autoflush=False, autocommit=False)()


CONFIG = {"name": "Cfg Co", "website_url": "https://a.example", "instagram_handle": "cfgco", "linkedin_handle": "",
          "email": "a@example.com", "timezone": "Asia/Kolkata", "virality_threshold": 75, "prompt_types": ["educational"]}


def test_edited_config_reaches_the_existing_row(tmp_path, caplog):
    db = _session(tmp_path)
    pipe = DailyPipeline.__new__(DailyPipeline)
    biz = pipe._get_or_create_business(db, dict(CONFIG))
    bid = biz.id
    edited = {**CONFIG, "email": "new.owner@example.com", "timezone": "America/New_York", "virality_threshold": 60,
              "instagram_handle": "newhandle", "prompt_types": ["educational", "promotional"], "website_url": "https://b.example"}
    with caplog.at_level(logging.INFO):
        again = pipe._get_or_create_business(db, edited)
    assert again.id == bid  # same row, updated in place
    db.expire_all()
    row = db.get(Business, bid)
    assert (row.email, row.timezone, row.virality_threshold, row.instagram_handle, row.website_url) == (
        "new.owner@example.com", "America/New_York", 60, "newhandle", "https://b.example")
    assert row.prompt_types == ["educational", "promotional"]
    assert "updated" in caplog.text and "email" in caplog.text
    assert "new.owner@example.com" not in caplog.text  # field names are logged, never values (PII)


def test_unchanged_config_is_a_no_op(tmp_path, caplog):
    db = _session(tmp_path)
    pipe = DailyPipeline.__new__(DailyPipeline)
    pipe._get_or_create_business(db, dict(CONFIG))
    db.expire_all()
    before = db.query(Business).one().updated_at
    with caplog.at_level(logging.INFO):
        pipe._get_or_create_business(db, dict(CONFIG))
    db.expire_all()
    assert db.query(Business).one().updated_at == before and "config changed" not in caplog.text


def test_email_resolved_from_an_env_reference_propagates(tmp_path, monkeypatch):
    """businesses.json stores env:OWNER_EMAIL; load_businesses resolves it; the row follows the secret."""
    import json

    from src.shared.utils import load_businesses

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "businesses.json").write_text(json.dumps([{**CONFIG, "email": "env:OWNER_EMAIL"}]))
    db = _session(tmp_path)
    pipe = DailyPipeline.__new__(DailyPipeline)
    monkeypatch.setenv("OWNER_EMAIL", "first@example.com")
    pipe._get_or_create_business(db, load_businesses()[0])
    monkeypatch.setenv("OWNER_EMAIL", "second@example.com")
    pipe._get_or_create_business(db, load_businesses()[0])
    db.expire_all()
    assert db.query(Business).one().email == "second@example.com"
