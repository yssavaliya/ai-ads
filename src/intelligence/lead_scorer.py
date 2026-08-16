from __future__ import annotations

import logging
import statistics
from typing import Any

from src.learning.auto_learner import engagement_rate

logger = logging.getLogger(__name__)


class LeadScorer:
    """Scores a candidate business's 'content opportunity': a business that is
    active on social media, has a real audience, but gets weak engagement —
    that's a business that needs help generating viral content."""

    def score(
        self,
        candidate: dict[str, Any],
        social_data: dict[str, Any],
        min_followers: int = 500,
        max_followers: int = 500_000,
    ) -> dict[str, Any]:
        score = 0
        reasons: list[str] = []

        followers = int(social_data.get("follower_count") or 0)
        posts = social_data.get("posts", []) or []

        if followers < min_followers:
            return {
                "score": 0,
                "followers": followers,
                "reasons": ["too small: under min follower threshold"],
                "eligible": False,
            }
        if followers > max_followers:
            return {
                "score": 0,
                "followers": followers,
                "reasons": ["too big: likely already has an agency"],
                "eligible": False,
            }

        score += 20
        reasons.append(f"{followers} followers (sweet spot)")

        active_posts = [p for p in posts if p.get("caption")]
        if len(active_posts) >= 3:
            score += 20
            reasons.append(f"active: {len(active_posts)} recent posts")
        else:
            score += 10
            reasons.append("limited recent activity")

        rates = [engagement_rate(p, followers) for p in posts if engagement_rate(p, followers) > 0]
        if rates:
            median_rate = statistics.median(rates)
            if median_rate < 0.03:
                score += 30
                reasons.append(f"low engagement rate ({median_rate:.2%}) — needs content help")
            elif median_rate < 0.06:
                score += 20
                reasons.append(f"below-average engagement ({median_rate:.2%})")
            else:
                score += 5
                reasons.append(f"healthy engagement ({median_rate:.2%})")
        else:
            score += 15
            reasons.append("no engagement data yet")

        if candidate.get("website_url"):
            score += 10
            reasons.append("has website (full profile available)")

        if candidate.get("instagram_handle") or candidate.get("linkedin_handle"):
            score += 10
            reasons.append("has social handle")

        return {
            "score": min(score, 100),
            "followers": followers,
            "reasons": reasons,
            "eligible": True,
        }