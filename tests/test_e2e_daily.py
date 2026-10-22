"""End-to-end dry run of the daily pipeline with every network/LLM edge stubbed.

Exercises the real wiring: profile -> audience -> trends -> virality gate ->
dedupe -> posting brief -> asset render -> DB persistence -> templated email
(written to logs/preview in DRY_RUN) -> learning bookkeeping.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.shared.config import settings
from src.shared.llm_client import LLMClient, LLMResult
from src.storage.models import Base, Business, Prompt

ROOT = Path(__file__).resolve().parents[1]
TYPES = ["educational", "storytelling", "social_proof"]


def _fake_llm_run(self, prompt, label=None, temperature=0.7, max_retries=3):
    if label == "competitor-discovery":
        out = []
    elif label == "audience-mapper":
        out = {
            "personas": [{"name": "Home bakers", "best_posting_times": "Tue-Thu 7 PM IST"}],
            "top_pain_points": ["time"], "content_themes": [{"theme": "t", "strength": "high"}],
        }
    elif label == "trend-radar":
        out = {"trending_topics": [], "trending_hashtags": ["#bake"]}
    elif label == "prompt-forge":
        out = [
            {
                "prompt_type": TYPES[i % 3], "title": f"Idea {i}", "hook": f"Hook number {i}",
                "prompt_text": f"Detailed creative brief {i} about sourdough proofing tips {i * 7}",
                "hashtags": ["#bake", "#ahmedabad"], "platform_focus": "instagram",
                "format": "single image", "angle": "a", "audience_segment": "Home bakers",
                "reel_beats": [],
            }
            for i in range(5)
        ]
    elif label == "virality-predictor":
        out = {
            "virality_score": 90, "virality_label": "HIGH", "virality_probability": 0.9,
            "key_factors": ["hook"], "risks": [], "reasoning": "strong", "improvement_hints": ["more specifics"],
        }
    else:  # pragma: no cover - guards against an unexpected extra LLM call
        raise AssertionError(f"unexpected LLM label {label!r}")
    return LLMResult(stdout=json.dumps(out), stderr="", exit_code=0, elapsed_seconds=0.0)


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    from PIL import Image

    import src.fetch.social_scraper as social
    import src.generation.creative_forge as cf
    import src.scheduler.daily_job as dj

    monkeypatch.setattr(settings, "templates_dir", ROOT / "templates")
    monkeypatch.setattr(settings, "dry_run", True)
    monkeypatch.setattr(cf, "FONT_PATH", ROOT / "assets" / "fonts" / "Poppins-Bold.ttf")
    monkeypatch.setattr(LLMClient, "run", _fake_llm_run)
    monkeypatch.setattr(dj.random, "random", lambda: 1.0)  # no exploration variant
    monkeypatch.setattr(dj.WebsiteScraper, "scrape", lambda self, url: {
        "title": "E2E Bakery", "meta_description": "Fresh bread", "headings": ["Our bread"],
    })
    monkeypatch.setattr(social, "fetch_instagram_sync", lambda handle: {"follower_count": 1000, "posts": []})

    def fake_image(self, image_prompt, width, height, name):
        path = self.output_dir / f"{name}.png"
        Image.new("RGB", (width // 4, height // 4), (200, 120, 60)).save(path)
        return str(path)

    monkeypatch.setattr(cf.CreativeForge, "generate_image", fake_image)

    eng = create_engine(f"sqlite:///{tmp_path / 'e2e.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    Session = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    monkeypatch.setattr(dj, "SessionLocal", Session)
    return dj, Session


BUSINESS = {
    "name": "E2E Bakery", "website_url": "https://e2e.example", "instagram_handle": "e2ebakery",
    "email": "owner@e2e.example", "prompt_types": TYPES, "virality_threshold": 75,
    "timezone": "Asia/Kolkata",
}


def test_daily_pipeline_end_to_end(pipeline_env):
    dj, Session = pipeline_env
    result = dj.DailyPipeline().run_for_business(dict(BUSINESS))

    assert result == {"business": "E2E Bakery", "prompts": 3, "email_sent": True}

    db = Session()
    rows = db.query(Prompt).all()
    assert len(rows) == 3
    assert {r.prompt_type.value for r in rows} == set(TYPES)
    for r in rows:
        assert r.sent_at is not None  # stamped only because the email went out
        assert r.virality_score == 90
        assert r.caption and "#bake" in r.caption
        assert r.asset_type == "image" and Path(r.asset_path).exists()
    biz = db.query(Business).one()
    assert biz.profile_json["title"] == "E2E Bakery"  # scraped profile persisted
    assert biz.audience_personas_json["personas"][0]["name"] == "Home bakers"

    previews = list((settings.logs_dir / "preview").glob("*.html"))
    assert len(previews) == 1
    html = previews[0].read_text(encoding="utf-8")
    assert "E2E Bakery" in html and "Caption" in html and "data:image/png;base64," in html
    assert "cid:" not in html  # inline images were resolved for the preview


def test_second_run_does_not_resend_near_duplicates(pipeline_env):
    dj, Session = pipeline_env
    pipeline = dj.DailyPipeline()
    pipeline.run_for_business(dict(BUSINESS))
    second = pipeline.run_for_business(dict(BUSINESS))
    # Identical briefs were just sent: dedupe keeps only the single best one.
    assert second["prompts"] == 1
    assert Session().query(Prompt).count() == 4


def test_run_all_skips_demo_and_inactive_businesses(pipeline_env, monkeypatch):
    dj, _ = pipeline_env
    (settings.data_dir / "businesses.json").write_text(json.dumps([
        {**BUSINESS, "name": "Demo", "demo": True},
        {**BUSINESS, "name": "Off", "is_active": False},
    ]))
    assert dj.DailyPipeline().run_all() == []


def test_business_email_can_come_from_an_env_reference(monkeypatch):
    from src.shared.utils import load_businesses

    monkeypatch.setenv("CLIENT_EMAIL", "secret@client.io")
    (settings.data_dir / "businesses.json").write_text(json.dumps([{"name": "X", "email": "env:CLIENT_EMAIL"}]))
    assert load_businesses()[0]["email"] == "secret@client.io"
    monkeypatch.delenv("CLIENT_EMAIL")
    assert load_businesses()[0]["email"] == ""  # unset -> empty, with a logged warning



def test_zero_prompts_is_reported_degraded_and_fails_the_job(pipeline_env, monkeypatch):
    """The real 10-21 run produced 0 prompts (empty model names) and still went green."""
    import src.delivery.github_feedback as gf
    from src.shared.llm_client import AllProvidersFailedError

    dj, _ = pipeline_env

    def boom(self, prompt, label=None, temperature=0.7, max_retries=3):
        raise AllProvidersFailedError(f"down: {label}")

    monkeypatch.setattr(LLMClient, "run", boom)
    reports: list = []
    monkeypatch.setattr(gf.GithubFeedback, "report_degraded_run", lambda self, failures: reports.append(failures))
    (settings.data_dir / "businesses.json").write_text(json.dumps([BUSINESS]))

    results = dj.DailyPipeline().run_all()
    assert results[0]["prompts"] == 0
    assert reports and reports[0][0]["stage"] == "generation"

    with pytest.raises(SystemExit) as exc:
        dj.main()
    assert exc.value.code == 1
