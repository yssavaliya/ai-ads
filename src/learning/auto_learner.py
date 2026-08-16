from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func

from src.learning.bayesian_updater import BayesianUpdater
from src.storage.models import Business, Engagement, Platform, Prompt, SessionLocal
from src.shared.utils import load_businesses

logger = logging.getLogger(__name__)

MIN_SIMILARITY = 0.15          # min caption/prompt token overlap to count as a match
POST_MATCH_WINDOW_DAYS = 5     # how many days after a prompt the user may have posted
MATCH_LOOKBACK_DAYS = 14       # how far back to look for sent prompts
BENCHMARK_LOOKBACK_DAYS = 30   # median engagement window


def tokenize(text: str) -> set[str]:
    return {
        w for w in re.split(r"[^a-z0-9#]+", (text or "").lower()) if len(w) > 2
    }


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def engagement_rate(post: dict[str, Any], followers: int | None) -> float:
    """Likes/comments/shares weighted vs reach (or followers as fallback)."""
    likes = int(post.get("likes") or 0)
    comments = int(post.get("comments") or 0)
    shares = int(post.get("shares") or 0)
    saves = int(post.get("saves") or 0)
    reach = int(post.get("reach") or 0)
    denom = reach or followers or 100
    return (likes + 2 * comments + 3 * shares + 2 * saves) / max(denom, 1)


class AutoLearner:
    """Learns WITHOUT user clicks: scrapes the business's own posts, matches
    them to previously sent prompts (by date + caption overlap), scores the
    real engagement, and feeds good/bad outcomes into the Bayesian loop."""

    def __init__(self) -> None:
        self.updater = BayesianUpdater()

    def run_for_business(
        self,
        business_id: int,
        social_data: dict[str, Any],
        db: Any,
    ) -> dict[str, Any]:
        posts = social_data.get("posts", [])
        followers = social_data.get("follower_count")
        if not posts:
            logger.info("Business %s: no posts available for auto-learning", business_id)
            return {"matched": 0, "learned": 0}

        benchmark = self._benchmark_median(posts, followers)
        sent_prompts = self._recent_prompts(db, business_id)
        matched = self._match(posts, sent_prompts)

        learned = 0
        details = []
        for prompt_id, prompt_type, prompt_text, post in matched:
            rate = engagement_rate(post, followers)
            outcome = "good" if rate >= benchmark else "bad"
            self._record(db, business_id, prompt_id, prompt_type, post, rate, outcome)
            self.updater.record(
                business_id,
                outcome,
                prompt_type=prompt_type,
                format=post.get("format"),
            )
            learned += 1
            details.append(
                {"prompt_id": prompt_id, "post_id": post.get("post_id"), "rate": round(rate, 4), "outcome": outcome}
            )

        logger.info(
            "Business %s auto-learned: %d matched, %d recorded (benchmark rate=%.4f)",
            business_id, len(matched), learned, benchmark,
        )
        return {"matched": len(matched), "learned": learned, "benchmark": round(benchmark, 4), "details": details}

    def _recent_prompts(self, db: Any, business_id: int) -> list[dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(days=MATCH_LOOKBACK_DAYS)
        rows = (
            db.query(Prompt)
            .filter(
                Prompt.business_id == business_id,
                Prompt.sent_at >= cutoff,
                Prompt.feedback.is_(None),
            )
            .all()
        )
        return [
            {
                "id": p.id,
                "prompt_type": p.prompt_type.value if p.prompt_type else None,
                "prompt_text": p.prompt_text,
                "sent_at": p.sent_at,
            }
            for p in rows
        ]

    def _benchmark_median(self, posts: list[dict[str, Any]], followers: int | None) -> float:
        """Median engagement rate of the business's own recent posts."""
        rates = sorted(engagement_rate(p, followers) for p in posts if engagement_rate(p, followers) > 0)
        if not rates:
            return 0.01
        n = len(rates)
        mid = n // 2
        return rates[mid] if n % 2 else (rates[mid - 1] + rates[mid]) / 2

    def _match(
        self, posts: list[dict[str, Any]], prompts: list[dict[str, Any]]
    ) -> list[tuple[int, str | None, str, dict[str, Any]]]:
        """Greedy best-match: each prompt matches its most similar post posted
        within POST_MATCH_WINDOW_DAYS after it was sent; each post used once."""
        matched: list[tuple[int, str | None, str, dict[str, Any]]] = []
        used_post_ids: set[str] = set()

        for prompt in sorted(prompts, key=lambda p: p["sent_at"] or datetime.min):
            if not prompt.get("sent_at"):
                continue
            window_end = prompt["sent_at"] + timedelta(days=POST_MATCH_WINDOW_DAYS)

            best: tuple[float, dict[str, Any]] | None = None
            for post in posts:
                post_id = str(post.get("post_id") or post.get("url") or "")
                if post_id in used_post_ids:
                    continue
                posted_at = self._parse_date(post.get("posted_at"))
                if posted_at is None:
                    continue
                if posted_at < prompt["sent_at"] or posted_at > window_end:
                    continue
                sim = jaccard_similarity(
                    post.get("caption") or "", prompt["prompt_text"]
                )
                if sim >= MIN_SIMILARITY and (best is None or sim > best[0]):
                    best = (sim, post)

            if best:
                post = best[1]
                post_id = str(post.get("post_id") or post.get("url") or "")
                if post_id:
                    used_post_ids.add(post_id)
                matched.append((prompt["id"], prompt["prompt_type"], prompt["prompt_text"], post))

        return matched

    def _record(
        self,
        db: Any,
        business_id: int,
        prompt_id: int,
        prompt_type: str | None,
        post: dict[str, Any],
        rate: float,
        outcome: str,
    ) -> None:
        prompt = db.get(Prompt, prompt_id)
        if prompt is None:
            return

        from src.storage.models import FeedbackType

        prompt.feedback = FeedbackType.GOOD if outcome == "good" else FeedbackType.BAD
        prompt.feedback_at = datetime.utcnow()
        db.add(prompt)

        platform = Platform.LINKEDIN if post.get("platform") == "linkedin" else Platform.INSTAGRAM
        existing = (
            db.query(Engagement)
            .filter(Engagement.prompt_id == prompt_id, Engagement.post_id == str(post.get("post_id") or ""))
            .first()
        )
        if existing:
            db.delete(existing)
            db.flush()

        engagement = Engagement(
            prompt_id=prompt_id,
            platform=platform,
            post_id=str(post.get("post_id") or "")[:100],
            post_url=str(post.get("url") or "")[:500] or None,
            likes=int(post.get("likes") or 0),
            comments=int(post.get("comments") or 0),
            shares=int(post.get("shares") or 0),
            saves=int(post.get("saves") or 0),
            reach=int(post.get("reach") or 0),
            impressions=int(post.get("impressions") or 0),
            fetched_at=datetime.utcnow(),
        )
        db.add(engagement)
        db.commit()

    @staticmethod
    def _parse_date(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        if isinstance(value, (int, float)) and value > 1e9:
            return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pass
        return None


def run_auto_learning() -> dict[str, Any]:
    """Scrape every active business's own posts and run the auto-learner."""
    import asyncio

    from src.fetch.social_scraper import SocialScraper

    db = SessionLocal()
    scraper = SocialScraper()
    results: dict[str, Any] = {}

    try:
        for config in load_businesses():
            if not config.get("is_active", True):
                continue
            biz = db.query(Business).filter(Business.name == config["name"]).first()
            if biz is None:
                continue
            handle = config.get("instagram_handle") or ""
            if not handle:
                continue
            social = asyncio.run(scraper.fetch_instagram(handle))
            results[config["name"]] = AutoLearner().run_for_business(biz.id, social, db)
    finally:
        db.close()
    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.shared.config import settings

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name, res in run_auto_learning().items():
        logger.info("AUTO-LEARN %s: %s", name, res)