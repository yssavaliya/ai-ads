from src.intelligence.lead_scorer import LeadScorer


def make_post(likes, comments=0, shares=0, saves=0, reach=None, caption="some caption"):
    return {
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "saves": saves,
        "reach": reach or (likes * 100),
        "caption": caption,
    }


def test_eligible_sweet_spot():
    candidate = {"name": "Fit Studio", "website_url": "https://fitstudio.in", "instagram_handle": "fitstudio"}
    social = {
        "follower_count": 5000,
        "posts": [make_post(20, caption="post 1"), make_post(15, caption="post 2"), make_post(30, caption="post 3")],
    }
    result = LeadScorer().score(candidate, social)
    assert result["eligible"] is True
    assert result["score"] >= 50
    assert "low engagement rate" in " ".join(result["reasons"]).lower() or "below-average" in " ".join(result["reasons"])


def test_too_small_ineligible():
    candidate = {"name": "Tiny Biz"}
    social = {"follower_count": 50, "posts": []}
    result = LeadScorer().score(candidate, social)
    assert result["eligible"] is False
    assert result["score"] == 0


def test_too_big_ineligible():
    candidate = {"name": "Big Brand"}
    social = {"follower_count": 2_000_000, "posts": [make_post(5000)]}
    result = LeadScorer().score(candidate, social, max_followers=500_000)
    assert result["eligible"] is False


def test_high_engagement_scores_lower():
    candidate = {"name": "A", "website_url": "https://a.in", "instagram_handle": "a"}
    active = {
        "follower_count": 10000,
        "posts": [
            make_post(500, reach=8000, caption="p1"),
            make_post(700, reach=10000, caption="p2"),
            make_post(600, reach=9000, caption="p3"),
        ],
    }
    low_eng = {
        "follower_count": 10000,
        "posts": [
            make_post(30, caption="p1"),
            make_post(40, caption="p2"),
            make_post(25, caption="p3"),
        ],
    }
    scorer = LeadScorer()
    high = scorer.score(candidate, active)
    low = scorer.score(candidate, low_eng)
    assert low["score"] > high["score"]
