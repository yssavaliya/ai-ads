from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func

from src.learning.bayesian_updater import BayesianUpdater
from src.storage.models import Business, Engagement, Platform, Prompt, SessionLocal
from src.shared.utils import load_businesses

logger = logging.getLogger(__name__)

MIN_SIMILARITY = 0.30          # min caption/prompt token overlap to count as a match
POST_MATCH_WINDOW_DAYS = 10    # how many days after a prompt the user may have posted
MATCH_LOOKBACK_DAYS = 14       # how far back to look for sent prompts
BENCHMARK_LOOKBACK_DAYS = 30   # median engagement window
NO_HISTORY_BENCHMARK = 0.01    # fallback "good post" bar when a business has no
                                 # positive-engagement posts yet to compute a real median from
MIN_POSTS_FOR_BENCHMARK = 3    # need this many non-zero posts before trusting the
                                 # own-median benchmark; below it we use an absolute bar


def tokenize(text: str) -> set[str]:
    return {
        w for w in re.split(r"[^a-z0-9#]+", (text or "").lower()) if len(w) > 2
    }


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def containment(a: str, b: str) -> float:
    """Overlap relative to the SMALLER caption — catches cases where one post
    is a shortened/paraphrased version of the other (common when a user trims
    the suggested caption before posting)."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def caption_similarity(post_caption: str, prompt_text: str) -> float:
    return max(jaccard_similarity(post_caption, prompt_text), containment(post_caption, prompt_text))


_POSITIVE = {
    "love", "great", "amazing", "awesome", "fire", "best", "cool", "nice", "want",
    "need", "please", "superb", "helpful", "inspiring", "beautiful", "perfect",
    "yes", "🔥", "❤", "💜", "💯", "👏", "😍", "🙏", "💪",
}
_NEGATIVE = {
    "bad", "worst", "hate", "boring", "ugly", "fake", "scam", "disappoint",
    "lazy", "weak", "cringe", "trash", "avoid", "no", "not", "why", "annoying",
    "confusing", "hard", "difficult", "useless",
}


def comment_sentiment(texts: list[str]) -> int:
    """Net sentiment of a list of comment strings. Positive = more love than hate."""
    net = 0
    for t in texts or []:
        toks = {w for w in re.split(r"[^a-z0-9#]+", (t or "").lower()) if len(w) > 1}
        net += sum(1 for w in toks if w in _POSITIVE)
        net -= sum(1 for w in toks if w in _NEGATIVE)
    return net


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
        self._db: Any = None

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

        benchmark, bench_n = self._benchmark_median(posts, followers)
        sent_prompts = self._recent_prompts(db, business_id)
        for p in sent_prompts:
            p["audience_segment"] = None
            if db is not None:
                from src.storage.models import Prompt

                row = db.get(Prompt, p["id"])
                if row is not None:
                    p["audience_segment"] = row.audience_segment
        self._db = db
        matched = self._match(posts, sent_prompts)

        learned = 0
        details = []
        for prompt_id, prompt_type, prompt_text, post, segment in matched:
            rate = engagement_rate(post, followers)
            outcome = self._decide_outcome(post, followers, benchmark, bench_n, post.get("comments_text"))
            self._record(db, business_id, prompt_id, prompt_type, post, rate, outcome)
            self.updater.record(
                business_id,
                outcome,
                prompt_type=prompt_type,
                format=post.get("format"),
                audience_segment=segment,
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

    def _benchmark_median(self, posts: list[dict[str, Any]], followers: int | None) -> tuple[float, int]:
        """Median engagement rate of the business's own recent posts.
        Returns (median, count_of_nonzero_posts) so callers can fall back to an
        absolute bar when there isn't enough history to trust the median."""
        rates = sorted(engagement_rate(p, followers) for p in posts if engagement_rate(p, followers) > 0)
        if not rates:
            return NO_HISTORY_BENCHMARK, 0
        n = len(rates)
        mid = n // 2
        return (rates[mid] if n % 2 else (rates[mid - 1] + rates[mid]) / 2), n

    @staticmethod
    def _decide_outcome(
        post: dict[str, Any], followers: int | None, benchmark: float, bench_n: int,
        comments_text: list[str] | None = None,
    ) -> str:
        """good/bad for a matched post.

        With enough history we compare against the business's own median
        engagement rate. With too few posts (typical for a brand-new account)
        we use an absolute bar so the loop still gets a real negative signal
        instead of everything reading 'good' (the old 0.01 floor did).

        Comment text is a second signal: a wall of negative comments on a
        post that merely cleared the like bar flips the outcome to 'bad' so
        the learner doesn't reward content people disliked."""
        rate = engagement_rate(post, followers)
        likes = int(post.get("likes") or 0)
        comments = int(post.get("comments") or 0)
        if bench_n >= MIN_POSTS_FOR_BENCHMARK:
            base = "good" if rate >= benchmark else "bad"
        # Not enough history: a post that drew any real reaction is good; a
        # total flop is bad. Avoids penalizing a tiny account's first tries
        # while still learning from genuine duds.
        elif likes >= 2 or comments >= 1:
            base = "good"
        else:
            base = "bad"

        sentiment = comment_sentiment(comments_text or [])
        if sentiment <= -2:
            return "bad"
        # Strongly positive comments can rescue a post that only barely missed
        # the like/rate bar — the audience clearly liked it.
        if sentiment >= 2 and base == "bad":
            return "good"
        return base

    def _match(
        self, posts: list[dict[str, Any]], prompts: list[dict[str, Any]]
    ) -> list[tuple[int, str | None, str, dict[str, Any], str | None]]:
        """Match published posts to sent prompts.

        Priority 1 — explicit links: posts whose post_id matches an Engagement
        row created via the 📎 'I posted this' button bypass similarity entirely.
        Priority 2 — similarity: caption/prompt Jaccard >= MIN_SIMILARITY within
        the post window; each post used once.
        Returns (prompt_id, prompt_type, prompt_text, post, audience_segment)."""
        from src.storage.models import Engagement

        matched: list[tuple[int, str | None, str, dict[str, Any], str | None]] = []
        used_post_ids: set[str] = set()

        prompt_ids = [p["id"] for p in prompts]
        linked_post_ids: set[str] = set()
        if prompt_ids and self._db is not None:
            rows = (
                self._db.query(Engagement)
                .filter(Engagement.prompt_id.in_(prompt_ids))
                .filter(Engagement.post_id.isnot(None))
                .all()
            )
            linked_post_ids = {r.post_id for r in rows if r.post_id}

        # Pass 1: exact-ID matches (human-confirmed links).
        for prompt in prompts:
            for post in posts:
                post_id = str(post.get("post_id") or post.get("url") or "")
                if not post_id or post_id in used_post_ids or post_id not in linked_post_ids:
                    continue
                used_post_ids.add(post_id)
                matched.append(
                    (prompt["id"], prompt.get("prompt_type"), prompt["prompt_text"], post, prompt.get("audience_segment"))
                )

        # Pass 2: fuzzy similarity matches.
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
                sim = caption_similarity(
                    post.get("caption") or "", prompt["prompt_text"]
                )
                if sim >= MIN_SIMILARITY and (best is None or sim > best[0]):
                    best = (sim, post)

            if best:
                post = best[1]
                post_id = str(post.get("post_id") or post.get("url") or "")
                if post_id:
                    used_post_ids.add(post_id)
                matched.append(
                    (prompt["id"], prompt.get("prompt_type"), prompt["prompt_text"], post, prompt.get("audience_segment"))
                )

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
        self._log_engagement(prompt_id, post, rate, outcome)

    @staticmethod
    def _log_engagement(prompt_id: int, post: dict[str, Any], rate: float, outcome: str) -> None:
        """Append a durable, git-tracked engagement record so learning history
        survives agent.db cache eviction (raw Engagement rows live only in the
        cache; the Bayesian weights persist separately in data/weights.json)."""
        try:
            from pathlib import Path

            path = Path(settings.data_dir) / "engagement_log.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "ts": datetime.utcnow().isoformat(),
                            "prompt_id": prompt_id,
                            "post_id": str(post.get("post_id") or "")[:100],
                            "rate": round(rate, 4),
                            "likes": int(post.get("likes") or 0),
                            "comments": int(post.get("comments") or 0),
                            "outcome": outcome,
                        }
                    )
                    + "\n"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to append engagement log: %s", exc)

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