import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.scheduler.lead_job import LeadPipeline
from src.shared.config import settings


def _lead_prompt() -> dict:
    return {
        "title": "Hook",
        "prompt_text": "Show the oven",
        "prompt_type": "educational",
        "virality_score": 88,
        "virality_label": "high",
        "audience_segment": "home bakers",
    }


def test_mark_seen_only_records_delivered_leads(tmp_path, monkeypatch):
    """Regression: raw candidates rejected by scoring must NOT enter the
    21-day cooldown cache — otherwise one empty run starves every later run.

    Isolation: mark_seen()/_recently_seen()/_save_leads() all resolve their
    file paths from settings.data_dir at call time, so redirecting that
    shared setting to tmp_path keeps this test from writing fabricated
    "Vicky Bakery" fixture data into the real, git-tracked data/leads.json
    and data/lead_seen.json — which is exactly what happened before this
    fix (verified by reproducing it)."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    pipeline = LeadPipeline()
    monkeypatch.setattr(pipeline, "_merge_known_leads", lambda db, tl, cfg: tl)

    eligible = {
        "name": "Vicky Bakery",
        "website_url": "https://vickybakery.com/",
        "instagram_handle": "vickybakery",
        "linkedin_handle": "",
        "niche": "bakery",
        "location": "Ahmedabad",
        "found_via": "search",
    }
    ineligible = {
        "name": "Junk Directory",
        "website_url": "https://homes.com/",
        "instagram_handle": "",
        "linkedin_handle": "",
        "niche": "home decor",
        "location": "Ahmedabad",
        "found_via": "search",
    }

    async def fake_discover(niches, locations, count, platform_focus):
        return [eligible, ineligible]

    async def fake_enrich(candidates):
        return candidates

    def fake_score(candidate, social_data, min_followers=500, max_followers=500000):
        if candidate.get("instagram_handle"):
            return {
                "score": 80,
                "followers": 900,
                "reasons": ["ok"],
                "eligible": True,
            }
        return {"score": 0, "followers": 0, "reasons": ["too small"], "eligible": False}

    async def fake_fetch(self, handle, limit=20):
        return {"handle": handle, "follower_count": 900, "posts": []}

    db = MagicMock()
    biz = MagicMock()
    biz.id = 1
    db.query.return_value.filter.return_value.first.return_value = None

    with (
        patch.object(LeadPipeline, "_learning_pass", return_value={"leads_checked": 0}),
        patch.object(pipeline.finder, "discover", fake_discover),
        patch.object(pipeline.finder, "enrich_with_socials", fake_enrich),
        patch.object(pipeline.scorer, "score", fake_score),
        patch("src.fetch.social_scraper.SocialScraper.fetch_instagram", fake_fetch),
        patch("src.intelligence.audience_mapper.AudienceMapper.map_audience", return_value={}),
        patch("src.intelligence.trend_radar.TrendRadar.scan", return_value={}),
        patch("src.learning.bayesian_updater.BayesianUpdater.to_forge_weights", return_value={}),
        patch.object(pipeline.gate, "run", return_value=[_lead_prompt()]),
        patch("src.scheduler.lead_job.SessionLocal", return_value=db),
        patch("src.delivery.email_sender.EmailSender.send_lead_digest", return_value=True),
        patch("src.generation.creative_forge.CreativeForge.render_prompt") as fake_render,
        patch("src.delivery.github_feedback.GithubFeedback.report_degraded_run"),
        patch.object(
            pipeline.finder, "mark_seen", wraps=pipeline.finder.mark_seen
        ) as fake_mark,
    ):
        fake_render.return_value = {"asset_path": "a.jpg", "asset_type": "image", "thumbnail": ""}
        result = pipeline.run()

    assert result["top_leads"] == 1
    assert fake_mark.call_count == 1
    marked = fake_mark.call_args[0][0]
    assert len(marked) == 1
    assert marked[0]["name"] == "Vicky Bakery"
    assert all(c["name"] != "Junk Directory" for c in marked)

    # Isolation check: the fixture data landed in tmp_path, not the real
    # git-tracked data/ directory.
    assert (tmp_path / "leads.json").exists()
    assert (tmp_path / "lead_seen.json").exists()
    assert not (Path("data") / "leads.json").samefile(tmp_path / "leads.json")


def test_too_big_candidates_go_to_long_cooldown_reject_cache(tmp_path, monkeypatch):
    """A structurally-ineligible ('too big') candidate should be cached in
    the 90-day rejection list, not just skipped — otherwise the same
    obviously-wrong result (a global brand, a real-estate portal) gets
    re-discovered and re-scored every single scheduled run forever."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    pipeline = LeadPipeline()
    monkeypatch.setattr(pipeline, "_merge_known_leads", lambda db, tl, cfg: tl)

    too_small = {
        "name": "Unverified Candidate",
        "website_url": "https://tinybakery.example/",
        "instagram_handle": "",
        "linkedin_handle": "",
        "niche": "bakery",
        "location": "Ahmedabad",
        "found_via": "search",
    }
    too_big = {
        "name": "Global Luxury Co",
        "website_url": "https://global-luxury.example/",
        "instagram_handle": "",
        "linkedin_handle": "",
        "niche": "jewellery store",
        "location": "Ahmedabad",
        "found_via": "search",
    }

    async def fake_discover(niches, locations, count, platform_focus):
        return [too_small, too_big]

    async def fake_enrich(candidates):
        return candidates

    def fake_score(candidate, social_data, min_followers=500, max_followers=500000):
        if candidate["name"] == "Global Luxury Co":
            return {"score": 0, "followers": 5_000_000, "reasons": ["too big: likely already has an agency"], "eligible": False}
        return {"score": 0, "followers": 0, "reasons": ["too small: under min follower threshold"], "eligible": False}

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    with (
        patch.object(LeadPipeline, "_learning_pass", return_value={"leads_checked": 0}),
        patch.object(pipeline.finder, "discover", fake_discover),
        patch.object(pipeline.finder, "enrich_with_socials", fake_enrich),
        patch.object(pipeline.scorer, "score", fake_score),
        patch("src.scheduler.lead_job.SessionLocal", return_value=db),
        patch("src.delivery.email_sender.EmailSender.send_lead_digest", return_value=True),
        patch("src.delivery.github_feedback.GithubFeedback.report_degraded_run"),
        patch.object(pipeline.finder, "mark_seen") as fake_mark_seen,
        patch.object(
            pipeline.finder, "mark_rejected", wraps=pipeline.finder.mark_rejected
        ) as fake_mark_rejected,
    ):
        result = pipeline.run()

    assert result["top_leads"] == 0
    fake_mark_seen.assert_called_once_with([])  # nothing eligible was delivered
    assert fake_mark_rejected.call_count == 1
    rejected = fake_mark_rejected.call_args[0][0]
    assert len(rejected) == 1
    assert rejected[0]["name"] == "Global Luxury Co"

    # A second discover() call should now filter it out via the reject cache.
    seen_recent = pipeline.finder._recently_seen() | pipeline.finder._recently_rejected()
    assert "global-luxury.example" in seen_recent
    assert "tinybakery.example" not in seen_recent  # "too small" stays retryable


def test_is_local_drops_overseas():
    cfg = {"locations": ["Ahmedabad"]}
    overseas = {"name": "Bakery City", "website_url": "bakerycity.com", "instagram_handle": "bakerycity"}
    assert LeadPipeline._is_local(overseas, cfg) is False


def test_is_local_keeps_indian_signals():
    cfg = {"locations": ["Ahmedabad"]}
    assert LeadPipeline._is_local({"name": "Ahmedabad Bakers", "website_url": "ahmedabadbakers.in"}, cfg) is True
    assert LeadPipeline._is_local({"name": "Gujarat Sweets", "website_url": "gujaratsweets.com"}, cfg) is True
    assert LeadPipeline._is_local({"name": "India Decor Co", "website_url": "indiadecor.co.in"}, cfg) is True


def test_is_local_drops_unconfirmed():
    cfg = {"locations": ["Ahmedabad"], "enforce_location": True}
    # No overseas marker, but also no positive local signal -> skip (quality over quantity)
    assert LeadPipeline._is_local({"name": "Generic Bakery", "website_url": "genericbakery.com"}, cfg) is False

