"""Increment 2: measured competitor benchmarks, evidence-based posting window, per-lead campaigns."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from src.intelligence import competitor_analysis as ca
from src.intelligence import posting_time as pt

BASE = datetime(2026, 11, 1, 12, tzinfo=UTC)


def post(days_ago, likes, caption="Stop guessing. Save this. #ads #brand", comments=0, hour_utc=None, **kw):
    when = BASE - timedelta(days=days_ago)
    if hour_utc is not None:
        when = when.replace(hour=hour_utc)
    return {"caption": caption, "likes": likes, "comments": comments, "posted_at": when.isoformat(), **kw}


def competitor(name, posts, followers=1000):
    return {"name": name, "handle": name.lower(), "top_posts": posts, "profile": {"follower_count": followers}}


# ------------------------------------------------------------- competitor benchmarks
def test_analysis_needs_enough_posts():
    assert ca.analyze_competitor(competitor("A", [post(1, 5), post(2, 5)])) is None


def test_competitor_metrics_are_computed_from_posts():
    posts = [
        post(1, 50, "Brand story today. DM us to order #cake #bake", media_type="video"),
        post(8, 10, "New flavour #cake", product_type="clips"),
        post(15, 30, "Behind the scenes of our kitchen and the whole team hard at work #bake"),
        post(22, 20, "Order now! Link in bio #cake"),
    ]
    a = ca.analyze_competitor(competitor("Sweet", posts, followers=2000))
    assert a["posts_seen"] == 4 and a["video_share"] == 0.5
    assert a["posts_per_week"] == round(4 / (21 / 7), 1)
    assert a["median_interactions"] == 25.0 and a["median_engagement_rate"] == round(25 / 2000, 4)
    assert a["cta_share"] == 0.5 and a["hashtags"][0] == "#cake"
    assert a["best_openers"][0] == {"opener": "Brand story today. DM us to order #cake", "interactions": 50}


def test_benchmark_uses_only_verified_competitors_and_explains_in_numbers():
    ghost = {"name": "Ghost", "handle": "ghost", "top_posts": []}
    comps = [
        competitor("A", [post(1, 5, media_type="image"), post(8, 4), post(15, 6), post(22, 5)], 500),
        competitor("B", [post(2, 40), post(9, 30), post(16, 50), post(23, 35)], 800),
        ghost,
    ]
    b = ca.benchmark(comps)
    assert b["available"] and len(b["competitors"]) == 2
    text = " ".join(b["observations"])
    assert "Video/Reel share" in text and "call to action" in text and "times per week" in text
    assert "Strongest engagement relative to audience: B" in text
    assert ca.benchmark([ghost]) == {"available": False, "competitors": [], "observations": [],
                                     "source": "scraped public posts (verified competitors only)"}


def test_strategist_prompt_receives_measured_facts_or_an_explicit_absence():
    from src.intelligence.campaign_strategist import CampaignStrategist

    seen = []

    def fake(prompt, label=None, **kw):
        seen.append(prompt)
        return {}

    strategist = CampaignStrategist()
    with patch.object(strategist.llm, "run_json", fake):
        strategist.plan({"name": "B"}, {}, {}, benchmarks={"observations": ["Competitors post about 4.0 times per week (median)"]})
        strategist.plan({"name": "B"}, {}, {}, benchmarks={"available": False, "observations": []})
    assert "4.0 times per week" in seen[0] and "MEASURED" in seen[0]
    assert "No verified competitor data" in seen[1]


# ------------------------------------------------------------- posting window
def evening_wins():
    own = [post(d, 40 if d % 2 else 8, hour_utc=13) for d in range(1, 9)]       # 18:30 IST: strong
    own += [post(d + 20, 5, hour_utc=4) for d in range(1, 5)]                    # 09:30 IST: weak
    return own


def test_window_is_chosen_by_relative_performance_not_posting_frequency():
    own = evening_wins() + [post(d + 40, 4, hour_utc=4) for d in range(1, 9)]    # posts MORE often at 09:30, performs worse
    result = pt.recommend_window(own, None, "Asia/Kolkata")
    assert result["source"] == "scraped post performance" and result["block"] == "18:00-20:59"
    assert result["confidence"] in {"low", "medium"} and "relative engagement" in result["evidence"]


def test_competitors_count_half_and_are_normalised_per_account():
    own = [post(d, 10, hour_utc=13) for d in range(1, 5)]
    giant = [post(d, 5000 if d % 2 else 4000, hour_utc=13) for d in range(1, 5)]   # huge numbers must not dominate
    result = pt.recommend_window(own, [giant], "Asia/Kolkata")
    assert result["samples"] == 8 and result["block"] == "18:00-20:59"


def test_too_little_data_falls_back_with_an_honest_label():
    result = pt.recommend_window([post(1, 5, hour_utc=13)], None, "Asia/Kolkata", audience_hint_hour=20)
    assert result["hour"] == 20 and result["confidence"] == "none" and "not measured" in result["source"]
    habit = pt.recommend_window([post(d, 0, hour_utc=14) for d in range(1, 4)], None, "UTC")
    assert habit["hour"] == 14 and "habit" in habit["source"]
    assert pt.recommend_window(None, None)["hour"] == pt.DEFAULT_HOUR


def test_hidden_likes_are_missing_data_not_zero_performance():
    posts = [post(d, -1, hour_utc=13, likes_hidden=True) for d in range(1, 12)]
    assert pt.recommend_window(posts, None, "UTC")["confidence"] == "none"


def test_bad_timezone_name_does_not_crash():
    assert pt.recommend_window(evening_wins(), None, "Mars/Phobos")["source"] == "scraped post performance"


def test_daily_pipeline_keeps_the_evidence_for_the_hub():
    from src.scheduler.daily_job import DailyPipeline

    pipe = DailyPipeline.__new__(DailyPipeline)
    prompts = [{"title": "t", "hook": "h", "caption": "c", "hashtags": []}]
    pipe._attach_posting_brief({"name": "B", "timezone": "Asia/Kolkata"}, prompts, {"posts": evening_wins()}, {}, [])
    assert prompts[0]["post_time_evidence"]["source"] == "scraped post performance"
    assert "06:00 PM" in prompts[0]["recommended_post_time"] or "07:00 PM" in prompts[0]["recommended_post_time"]


# ------------------------------------------------------------- per-lead campaign
def test_top_leads_get_a_recommended_campaign_and_failures_never_block(monkeypatch):
    from src.scheduler.lead_job import LeadPipeline
    from src.shared.config import settings

    pipeline = LeadPipeline.__new__(LeadPipeline)
    seen = []

    def fake_plan(profile, audience, trends, *a, **k):
        seen.append(profile)
        return {
            "objective": "lead_generation", "persona": "Local shopper", "customer_problem": "No online orders",
            "emotional_trigger": "aspiration", "cta": "DM to order", "kpis": ["saves"], "test_hypothesis": "h",
            "rationale": "r", "chosen": {"name": "Show the bake", "hook_type": "before_after", "total": 80.0},
            "angles": [{"name": "Show the bake", "total": 80.0}, {"name": "Menu", "total": 70.0}],
        }

    pipeline.strategist = type("S", (), {"plan": staticmethod(fake_plan)})()
    monkeypatch.setattr(settings, "lead_strategy_count", 1)
    first, second = {"name": "Sweet Crumbs", "why_target": "scores 80/100"}, {"name": "Other"}
    pipeline._attach_campaign(first, 0)
    pipeline._attach_campaign(second, 1)
    assert first["campaign"]["angle"] == "Show the bake" and "campaign" not in second
    assert seen[0]["verified_facts"] == [] and seen[0]["why_selected"] == "scores 80/100"

    pipeline.strategist = type("S", (), {"plan": staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))})()
    third = {"name": "Third"}
    pipeline._attach_campaign(third, 0)  # must not raise
    assert "campaign" not in third


def test_digest_shows_the_recommended_campaign(tmp_path, monkeypatch):
    from src.delivery.site_publisher import SitePublisher
    from src.shared.config import settings

    monkeypatch.setattr(settings, "site_dir", tmp_path / "site")
    lead = {"name": "Sweet Crumbs", "opportunity_score": 82, "opportunity_reasons": ["ok"], "followers": 4000,
            "campaign": {"objective": "lead_generation", "persona": "Local shopper", "problem": "No online orders",
                         "angle": "Show the bake", "rationale": "scored 80/100", "cta": "DM to order",
                         "kpis": ["saves"], "alternatives": [{"name": "Menu", "score": 70.0}]}, "prompts": []}
    SitePublisher().send_lead_digest("owner@example.com", [lead])
    html = "\n".join(p.read_text(encoding="utf-8") for p in (tmp_path / "site").rglob("index.html"))
    assert "Recommended campaign" in html and "Show the bake" in html and "DM to order" in html
