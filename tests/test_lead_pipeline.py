import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.scheduler.lead_job import LeadPipeline


def _lead_prompt() -> dict:
    return {
        "title": "Hook",
        "prompt_text": "Show the oven",
        "prompt_type": "educational",
        "virality_score": 88,
        "virality_label": "high",
        "audience_segment": "home bakers",
    }


def test_mark_seen_only_records_delivered_leads(tmp_path):
    """Regression: raw candidates rejected by scoring must NOT enter the
    21-day cooldown cache — otherwise one empty run starves every later run."""
    pipeline = LeadPipeline()

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
