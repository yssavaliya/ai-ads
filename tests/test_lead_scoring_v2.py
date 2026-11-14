"""Explainable multi-dimension lead scoring."""
from __future__ import annotations

from src.intelligence.lead_scorer import MAX_POINTS, LeadScorer


def post(likes, caption, reach=None):
    return {"likes": likes, "comments": 0, "shares": 0, "saves": 0, "reach": reach or likes * 100, "caption": caption}


SELLING = [
    post(20, "Fresh cakes today! DM to order, delivery across the city"),
    post(15, "Wedding cake pre-order open. Price on request, WhatsApp us"),
    post(18, "New menu. Shop the catalogue, link in bio"),
]
GENERIC = [post(20, "good morning"), post(15, "happy friday"), post(18, "weekend vibes")]


def lead(**kw):
    base = {"name": "Sweet Crumbs Bakery", "website_url": "https://sweet.example", "instagram_handle": "sweetcrumbs",
            "email": "hello@sweet.example", "niche": "bakery"}
    base.update(kw)
    return base


def test_point_maxima_sum_to_one_hundred():
    assert sum(MAX_POINTS.values()) == 100


def test_result_keeps_the_old_contract_and_adds_an_explanation():
    result = LeadScorer().score(lead(), {"follower_count": 5000, "posts": SELLING})
    assert {"score", "followers", "reasons", "eligible"} <= set(result)
    assert result["eligible"] is True and 0 <= result["score"] <= 100
    assert set(result["breakdown"]) == set(MAX_POINTS)
    assert sum(b["points"] for b in result["breakdown"].values()) == result["score"] or abs(
        sum(b["points"] for b in result["breakdown"].values()) - result["score"]) <= 1
    assert all(0 <= b["points"] <= b["max"] for b in result["breakdown"].values())
    assert "Sweet Crumbs Bakery" in result["why_target"] and "/100" in result["why_target"]
    assert result["decision"] in {"target", "maybe", "skip"}


def test_buying_intent_and_visual_products_raise_the_score():
    scorer = LeadScorer()
    social_selling = {"follower_count": 5000, "posts": SELLING}
    social_generic = {"follower_count": 5000, "posts": GENERIC}
    selling = scorer.score(lead(), social_selling)
    generic = scorer.score(lead(niche="consulting", name="Acme Advisory"), social_generic)
    assert selling["breakdown"]["buying_intent"]["points"] > generic["breakdown"]["buying_intent"]["points"]
    assert selling["breakdown"]["content_potential"]["points"] > generic["breakdown"]["content_potential"]["points"]
    assert selling["score"] > generic["score"]
    assert any("sells directly" in r for r in selling["reasons"])


def test_unreachable_lead_is_penalised_and_the_gap_is_reported():
    scorer = LeadScorer()
    social = {"follower_count": 5000, "posts": SELLING}
    reachable = scorer.score(lead(), social)
    hidden = scorer.score(lead(email=""), social)
    assert hidden["breakdown"]["reachability"]["points"] < reachable["breakdown"]["reachability"]["points"]
    assert "no contact email" in hidden["data_gaps"]


def test_missing_data_is_neutral_and_listed_never_invented():
    result = LeadScorer().score({"name": "Mystery"}, {"follower_count": 3000, "posts": []})
    assert result["eligible"] is True
    gaps = " ".join(result["data_gaps"])
    assert "engagement could not be measured" in gaps and "no recent posts" in gaps
    assert "local competition was not measured" in gaps and "buying intent could not be judged" in gaps
    assert result["breakdown"]["engagement_gap"]["note"] == "no engagement data yet"


def test_competition_uses_the_niche_count_when_known():
    scorer = LeadScorer()
    social = {"follower_count": 5000, "posts": SELLING}
    quiet = scorer.score(lead(niche_competitors=1), social)
    crowded = scorer.score(lead(niche_competitors=15), social)
    assert quiet["breakdown"]["competition"]["points"] > crowded["breakdown"]["competition"]["points"]
    assert "15 similar local businesses" in crowded["breakdown"]["competition"]["note"]


def test_ineligible_leads_are_skipped_with_a_reason():
    tiny = LeadScorer().score(lead(), {"follower_count": 40, "posts": []})
    huge = LeadScorer().score(lead(), {"follower_count": 9_000_000, "posts": []}, max_followers=500_000)
    for result, text in ((tiny, "too small"), (huge, "too big")):
        assert result["eligible"] is False and result["score"] == 0 and result["decision"] == "skip"
        assert text in result["why_target"]


def test_high_engagement_still_scores_below_low_engagement():
    scorer = LeadScorer()
    healthy = [post(500, "great cake DM to order", reach=8000), post(700, "shop now", reach=10000), post(600, "price list", reach=9000)]
    weak = [post(30, "great cake DM to order"), post(40, "shop now"), post(25, "price list")]
    assert scorer.score(lead(), {"follower_count": 10000, "posts": weak})["score"] > \
        scorer.score(lead(), {"follower_count": 10000, "posts": healthy})["score"]


def test_digest_page_shows_why_and_breakdown(tmp_path, monkeypatch):
    from src.delivery.site_publisher import SitePublisher
    from src.shared.config import settings

    monkeypatch.setattr(settings, "site_dir", tmp_path / "site")
    scored = LeadScorer().score(lead(), {"follower_count": 5000, "posts": SELLING})
    entry = {**lead(), "opportunity_score": scored["score"], "opportunity_reasons": scored["reasons"],
             "followers": 5000, "opportunity_breakdown": scored["breakdown"], "why_target": scored["why_target"],
             "opportunity_decision": scored["decision"], "data_gaps": scored["data_gaps"], "prompts": []}
    publisher = SitePublisher()
    publisher.send_lead_digest("owner@example.com", [entry])
    pages = [p.read_text(encoding="utf-8") for p in (tmp_path / "site").rglob("index.html") if "leads" in str(p) or "l-" in str(p)]
    html = "\n".join(pages)
    assert "Why target" in html and "Score breakdown" in html and "Sweet Crumbs Bakery" in html
