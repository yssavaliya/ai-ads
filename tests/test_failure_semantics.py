"""Audit H-04 / H-05: a pipeline that produced nothing, or whose output reached nobody, must exit
non-zero (so Actions turns red and the degraded-run issue opens) and must not record side effects
that only make sense after delivery."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.delivery.github_feedback as gf
import src.scheduler.lead_job as lj
import src.scheduler.weekly_job as wj
from src.delivery.channels import DeliveryResult
from src.delivery.site_publisher import SitePublisher
from src.shared.config import settings
from src.shared.llm_client import AllProvidersFailedError, LLMClient, LLMResult
from src.storage.models import Base, Business, Prompt, PromptType


def _session(tmp_path: Path):
    eng = create_engine(f"sqlite:///{tmp_path / 'fs.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    return sessionmaker(bind=eng, autoflush=False, autocommit=False)


@pytest.fixture
def degraded_reports(monkeypatch):
    reports: list[list[dict]] = []
    monkeypatch.setattr(gf.GithubFeedback, "report_degraded_run", lambda self, failures: reports.append(failures))
    return reports


# ====================================================================== weekly (H-04)
@pytest.fixture
def weekly_env(tmp_path, monkeypatch):
    Session = _session(tmp_path)
    db = Session()
    db.add(Business(name="Weekly Co", email=""))
    db.commit()
    db.close()
    monkeypatch.setattr(wj, "SessionLocal", Session)
    monkeypatch.setattr(wj, "load_businesses", lambda: [{"name": "Weekly Co", "email": "", "is_active": True}])
    return Session


def _llm(monkeypatch, *, fail=False):
    def run(self, prompt, label=None, temperature=0.7, max_retries=3, exclude=None):
        if fail:
            raise AllProvidersFailedError("providers down")
        return LLMResult(json.dumps({"top_performing": [], "audience_insights": ["real analysis"]}), "", 0, 0.0, provider="p", model="m")

    monkeypatch.setattr(LLMClient, "run", run)


def test_weekly_success_exits_zero(weekly_env, degraded_reports, monkeypatch):
    _llm(monkeypatch)
    wj.main()  # no SystemExit
    assert degraded_reports == []


def test_weekly_llm_failure_delivers_a_placeholder_but_fails_the_run(weekly_env, degraded_reports, monkeypatch, tmp_path):
    _llm(monkeypatch, fail=True)
    with pytest.raises(SystemExit) as exit_info:
        wj.main()
    assert exit_info.value.code == 1
    assert degraded_reports and degraded_reports[0][0]["stage"] == "report" and degraded_reports[0][0]["business"] == "Weekly Co"
    # the placeholder is still delivered so the owner sees the problem on the site, without the internal marker
    manifest = json.loads((settings.site_dir / "data/weekly-co.json").read_text(encoding="utf-8"))
    assert "Report generation failed" in json.dumps(manifest["weekly"][0]["report"])
    assert "_degraded" not in json.dumps(manifest)


def test_weekly_delivery_failure_fails_the_run(weekly_env, degraded_reports, monkeypatch):
    _llm(monkeypatch)
    monkeypatch.setattr(SitePublisher, "send_weekly_report", lambda self, b, r: False)
    with pytest.raises(SystemExit) as exit_info:
        wj.main()
    assert exit_info.value.code == 1 and degraded_reports[0][0]["stage"] == "delivery"


def test_weekly_unknown_business_fails_the_run(weekly_env, degraded_reports, monkeypatch):
    _llm(monkeypatch)
    monkeypatch.setattr(wj, "load_businesses", lambda: [{"name": "Not In Db", "email": "", "is_active": True}])
    with pytest.raises(SystemExit):
        wj.main()
    assert degraded_reports[0][0]["stage"] == "pipeline"


def test_weekly_failure_classification_is_exhaustive():
    results = [{"business": "a", "delivered": True, "degraded": False}, {"business": "b", "error": "boom"},
               {"business": "c", "delivered": True, "degraded": True}, {"business": "d", "delivered": False, "degraded": False}]
    assert [(f["business"], f["stage"]) for f in wj.weekly_failures(results)] == [("b", "pipeline"), ("c", "report"), ("d", "delivery")]


# ====================================================================== lead (H-05)
def _lead_prompt(i=0):
    return {"title": f"Hook {i}", "prompt_text": f"Sample brief {i}", "prompt_type": "educational", "virality_score": 88,
            "virality_label": "HIGH", "audience_segment": "bakers", "caption": f"cap {i}"}


def _run_lead_pipeline(tmp_path, monkeypatch, *, delivered: bool):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    Session = _session(tmp_path)
    monkeypatch.setattr(lj, "SessionLocal", Session)
    pipeline = lj.LeadPipeline()
    monkeypatch.setattr(pipeline, "_merge_known_leads", lambda db, tl, cfg: tl)
    lead = {"name": "Vicky", "website_url": "https://vicky.example", "instagram_handle": "vicky", "linkedin_handle": "",
            "niche": "bakery", "location": "Ahmedabad", "found_via": "search"}

    async def discover(*a):
        return [dict(lead)]

    async def enrich(c):
        return c

    async def fetch(self, handle, limit=20):
        return {"handle": handle, "follower_count": 900, "posts": []}

    outcome = DeliveryResult({"site": delivered})
    with (
        patch.object(lj.LeadPipeline, "_learning_pass", return_value={}),
        patch.object(pipeline.finder, "discover", discover),
        patch.object(pipeline.finder, "enrich_with_socials", enrich),
        patch.object(pipeline.scorer, "score", lambda c, s, **k: {"score": 80, "followers": 900, "reasons": ["ok"], "eligible": True}),
        patch("src.fetch.social_scraper.SocialScraper.fetch_instagram", fetch),
        patch("src.intelligence.audience_mapper.AudienceMapper.map_audience", return_value={}),
        patch("src.intelligence.trend_radar.TrendRadar.scan", return_value={}),
        patch.object(pipeline.gate, "run", return_value=[_lead_prompt(0), _lead_prompt(1)]),
        patch.object(pipeline.delivery, "send_lead_digest", return_value=outcome),
        patch("src.generation.creative_forge.CreativeForge.render_prompt", side_effect=RuntimeError("no render in test")),
    ):
        result = pipeline.run()
    return result, Session


def test_failed_digest_leaves_prompts_unsent(tmp_path, monkeypatch):
    result, Session = _run_lead_pipeline(tmp_path, monkeypatch, delivered=False)
    assert result["delivered"] is False and result["prompts_generated"] == 2
    rows = Session().query(Prompt).all()
    assert len(rows) == 2 and all(r.sent_at is None for r in rows)  # not "sent": must not feed dedup/learning


def test_delivered_digest_stamps_each_prompt_exactly_once(tmp_path, monkeypatch):
    result, Session = _run_lead_pipeline(tmp_path, monkeypatch, delivered=True)
    assert result["delivered"] is True
    rows = Session().query(Prompt).all()
    assert len(rows) == 2 and all(isinstance(r.sent_at, datetime) for r in rows)
    first = {r.id: r.sent_at for r in rows}
    pipeline = lj.LeadPipeline.__new__(lj.LeadPipeline)
    pipeline._stamp_sent(list(first))  # a second stamp must not rewrite history
    assert {r.id: r.sent_at for r in Session().query(Prompt).all()} == first


def test_lead_main_exits_nonzero_when_nothing_was_delivered(monkeypatch, degraded_reports):
    monkeypatch.setattr(lj.LeadPipeline, "run", lambda self: {"top_leads": 2, "prompts_generated": 4, "delivered": False, "channels": {"site": False}})
    with pytest.raises(SystemExit) as exit_info:
        lj.main()
    assert exit_info.value.code == 1 and degraded_reports[0][0]["stage"] == "delivery"


def test_lead_main_succeeds_when_delivered(monkeypatch, degraded_reports):
    monkeypatch.setattr(lj.LeadPipeline, "run", lambda self: {"top_leads": 2, "prompts_generated": 4, "delivered": True, "channels": {"site": True}})
    lj.main()
    assert degraded_reports == []


def test_save_prompts_never_marks_anything_sent(tmp_path):
    Session = _session(tmp_path)
    db = Session()
    biz = Business(name="L", email="")
    db.add(biz)
    db.commit()
    pipeline = lj.LeadPipeline.__new__(lj.LeadPipeline)
    ids = pipeline._save_prompts(db, biz.id, [_lead_prompt(0)])
    db.expire_all()
    assert len(ids) == 1 and db.get(Prompt, ids[0]).sent_at is None and db.get(Prompt, ids[0]).prompt_type == PromptType.EDUCATIONAL
