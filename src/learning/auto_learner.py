from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, or_

from src.learning.bayesian_updater import BayesianUpdater
from src.shared.config import settings
from src.shared.privacy import redact_text
from src.shared.utils import load_businesses, utc_now_naive
from src.storage.models import Business, Engagement, Platform, Prompt, SessionLocal

logger = logging.getLogger(__name__)

MIN_SIMILARITY = 0.30          # min caption/prompt token overlap to count as a match
POST_MATCH_WINDOW_DAYS = 21    # how many days after a prompt the user may have posted
                                  # (manual posting is async; 10d missed late posts)
MATCH_LOOKBACK_DAYS = 14       # how far back to look for sent prompts
REFRESH_LOOKBACK_DAYS = 30     # an already-learned prompt keeps getting its engagement refreshed this long
MATURITY_HOURS = 48            # a post younger than this is not judged yet: its likes are still arriving
BENCHMARK_LOOKBACK_DAYS = 30   # median engagement window
NO_HISTORY_BENCHMARK = 0.01    # fallback "good post" bar when a business has no
                                 # positive-engagement posts yet to compute a real median from
MIN_POSTS_FOR_BENCHMARK = 3    # need this many non-zero posts before trusting the
                                 # own-median benchmark; below it we use an absolute bar


_SHORTCODE_URL_RE = re.compile(r"/(?:p|reel|reels|tv)/([A-Za-z0-9_-]+)")


def shortcode_from_url(url: str | None) -> str:
    match = _SHORTCODE_URL_RE.search(url or "")
    return match.group(1) if match else ""


def post_keys(post: dict[str, Any]) -> set[str]:
    """All identifiers a scraped post can be linked by: Apify/internal id, the
    full URL, and the URL shortcode (what the "I posted this" button stores)."""
    keys = {
        str(post.get("post_id") or ""),
        str(post.get("url") or ""),
        shortcode_from_url(post.get("url")),
    }
    return {k for k in keys if k}


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
            # say WHY: a scrape error (no token, blocked, actor failed) and a genuinely empty profile
            # look identical as "0 learned" otherwise
            return {"posts": 0, "matched": 0, "learned": 0, "scrape_error": social_data.get("error"), "followers": followers}

        biz_row = db.get(Business, business_id) if db is not None else None
        weights_key = biz_row.weights_key if biz_row is not None else str(business_id)
        benchmark, bench_n = self._benchmark_median(posts, followers)
        sent_prompts = self._recent_prompts(db, business_id)  # already carries audience_segment (was an N+1)
        self._db = db
        matched = self._match(posts, sent_prompts)

        learned = deferred = unusable = refreshed = 0
        details = []
        already = {p["id"]: p["learned"] for p in sent_prompts}
        prompt_map = {p["id"]: p for p in sent_prompts}
        now = utc_now_naive()
        for prompt_id, prompt_type, prompt_text, post, segment, fmt in matched:
            rate = engagement_rate(post, followers)
            outcome = self._decide_outcome(post, followers, benchmark, bench_n, post.get("comments_text"))

            # DATA QUALITY before the single, permanent weight update:
            posted_at = self._parse_date(post.get("posted_at"))
            mature = posted_at is not None and (now - posted_at) >= timedelta(hours=MATURITY_HOURS)
            # Instagram hides like counts on some posts (reported as -1): a 'zero' there is missing
            # data, and scoring it would teach the system that the content flopped.
            usable = not post.get("likes_hidden")
            learn = mature and usable and not already.get(prompt_id)
            if not already.get(prompt_id):
                if posted_at is None or not usable:
                    unusable += 1
                elif not mature:
                    deferred += 1

            recorded = self._record(db, business_id, prompt_id, prompt_type, post, rate, outcome, learn=learn)
            # The weights move ONLY the first time a (prompt, post) pair is judged; every later scrape
            # refreshes the stored engagement without re-counting (engagement.yml runs every 6h).
            if recorded:
                hook_type = prompt_map.get(prompt_id, {}).get("hook_type")
                self.updater.record(
                    weights_key,
                    outcome,
                    prompt_type=prompt_type,
                    format=fmt or post.get("format"),
                    audience_segment=segment,
                    hook_type=hook_type,
                    source="auto",
                )
                learned += 1
            elif already.get(prompt_id):
                refreshed += 1
            details.append(
                {"prompt_id": prompt_id, "post_id": post.get("post_id"), "rate": round(rate, 4), "outcome": outcome,
                 "learned": recorded, "deferred": (not mature) and not already.get(prompt_id)}
            )

        logger.info(
            "Business %s auto-learned: %d matched, %d learned, %d deferred (immature), %d unusable, %d refreshed (benchmark rate=%.4f)",
            business_id, len(matched), learned, deferred, unusable, refreshed, benchmark,
        )
        return {
            "posts": len(posts), "candidates": len(sent_prompts),
            "matched": len(matched), "unmatched": len(posts) - len(matched),
            "learned": learned, "deferred": deferred, "unusable": unusable, "refreshed": refreshed,
            "hidden_likes": sum(1 for p in posts if p.get("likes_hidden")),
            "benchmark": round(benchmark, 4), "details": details,
        }

    def _recent_prompts(self, db: Any, business_id: int) -> list[dict[str, Any]]:
        cutoff = utc_now_naive() - timedelta(days=MATCH_LOOKBACK_DAYS)
        refresh_cutoff = utc_now_naive() - timedelta(days=REFRESH_LOOKBACK_DAYS)
        # Eligible: prompts nobody judged yet (feedback IS NULL), plus prompts the learner already
        # counted (auto_learned_at) so their stored engagement can keep growing. Prompts judged by a
        # human click have feedback but no auto_learned_at: they were counted once already - skip.
        rows = [
            p
            for p in db.query(Prompt)
            .filter(
                Prompt.business_id == business_id,
                Prompt.sent_at >= refresh_cutoff,
                or_(Prompt.feedback.is_(None), Prompt.auto_learned_at.isnot(None)),
            )
            .all()
            if p.auto_learned_at is not None or p.sent_at >= cutoff
        ]
        return [
            {
                "learned": p.auto_learned_at is not None,
                "id": p.id,
                "prompt_type": p.prompt_type.value if p.prompt_type else None,
                "prompt_text": p.prompt_text,
                "caption": p.caption or "",
                "format": p.format or "",
                "audience_segment": p.audience_segment,
                "hook_type": getattr(p, "hook_type", None),
                "sent_at": p.sent_at,
            }
            for p in rows
        ]

    def _benchmark_median(self, posts: list[dict[str, Any]], followers: int | None) -> tuple[float, int]:
        """Median engagement rate of the business's own recent posts.
        Returns (median, count_of_nonzero_posts) so callers can fall back to an
        absolute bar when there isn't enough history to trust the median."""
        now = utc_now_naive()
        rates = []
        for post in posts:
            posted_at = self._parse_date(post.get("posted_at"))
            if post.get("likes_hidden") or posted_at is None:
                continue
            if now - posted_at < timedelta(hours=MATURITY_HOURS):
                continue
            rate = engagement_rate(post, followers)
            if rate > 0:
                rates.append(rate)
        rates.sort()
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
    ) -> list[tuple[int, str | None, str, dict[str, Any], str | None, str | None]]:
        """Match published posts to sent prompts.

        Priority 1 — explicit links: posts matching an Engagement row created
        via the 📎 'I posted this' button are attributed to THAT prompt and
        bypass similarity entirely.
        Priority 2 — similarity: caption/prompt Jaccard >= MIN_SIMILARITY within
        the post window; each post and each prompt is used once.
        Returns (prompt_id, prompt_type, prompt_text, post, audience_segment, format)."""
        from src.storage.models import Engagement

        matched: list[tuple[int, str | None, str, dict[str, Any], str | None, str | None]] = []
        used_post_keys: set[str] = set()
        used_prompt_ids: set[int] = set()
        by_id = {p["id"]: p for p in prompts}

        # key (post id / url / shortcode) -> prompt id it was explicitly linked to
        linked: dict[str, int] = {}
        if by_id and self._db is not None:
            rows = (
                self._db.query(Engagement)
                .filter(Engagement.prompt_id.in_(list(by_id)))
                .filter(Engagement.post_id.isnot(None))
                .all()
            )
            for r in rows:
                for key in (r.post_id, r.post_url, shortcode_from_url(r.post_url)):
                    if key:
                        linked[str(key)] = r.prompt_id

        # Pass 1: exact-ID matches (human-confirmed links).
        for post in posts:
            keys = post_keys(post)
            if not keys or keys & used_post_keys:
                continue
            prompt_id = next((linked[k] for k in keys if k in linked), None)
            if prompt_id is None or prompt_id in used_prompt_ids:
                continue
            prompt = by_id[prompt_id]
            used_post_keys |= keys
            used_prompt_ids.add(prompt_id)
            matched.append(
                (prompt_id, prompt.get("prompt_type"), prompt["prompt_text"], post,
                 prompt.get("audience_segment"), prompt.get("format"))
            )

        # Pass 2: fuzzy similarity matches.
        for prompt in sorted(prompts, key=lambda p: p["sent_at"] or datetime.min):
            if prompt["id"] in used_prompt_ids or not prompt.get("sent_at"):
                continue
            window_end = prompt["sent_at"] + timedelta(days=POST_MATCH_WINDOW_DAYS)

            # The human posts the email's `caption` (hook + hashtags), which is
            # NOT the creative `prompt_text`. Match the scraped post against the
            # actually-posted caption first, falling back to the brief.
            ref_caption = prompt.get("caption") or ""
            ref_brief = prompt.get("prompt_text") or ""

            best: tuple[float, dict[str, Any]] | None = None
            for post in posts:
                if post_keys(post) & used_post_keys:
                    continue
                posted_at = self._parse_date(post.get("posted_at"))
                if posted_at is None:
                    continue
                if posted_at < prompt["sent_at"] or posted_at > window_end:
                    continue
                scraped = post.get("caption") or ""
                sim = max(
                    caption_similarity(scraped, ref_caption) if ref_caption else 0.0,
                    caption_similarity(scraped, ref_brief) if ref_brief else 0.0,
                )
                if sim >= MIN_SIMILARITY and (best is None or sim > best[0]):
                    best = (sim, post)

            if best:
                post = best[1]
                used_post_keys |= post_keys(post)
                used_prompt_ids.add(prompt["id"])
                matched.append(
                    (prompt["id"], prompt.get("prompt_type"), prompt["prompt_text"], post,
                     prompt.get("audience_segment"), prompt.get("format"))
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
        learn: bool = True,
    ) -> bool:
        """Store the post's engagement and, when `learn` is set and the prompt has not been counted
        yet, record the outcome on the prompt. Returns True only on that first counting - later
        re-scrapes refresh the stored engagement but must NOT re-run the Bayesian weight update, or
        trials (and thus the weights) get inflated every run."""
        prompt = db.get(Prompt, prompt_id)
        if prompt is None:
            return False

        from src.storage.models import FeedbackType

        # `auto_learned_at` (not `feedback`) says whether the learner has counted this prompt. It used
        # to be `feedback`, which also made the prompt vanish from every later scrape, so engagement
        # was frozen at the first sighting (audit H-03).
        already_learned = prompt.auto_learned_at is not None
        do_learn = learn and not already_learned
        if do_learn:
            prompt.feedback = FeedbackType.GOOD if outcome == "good" else FeedbackType.BAD
            prompt.feedback_at = utc_now_naive()
            prompt.auto_learned_at = utc_now_naive()
            db.add(prompt)

        platform = Platform.LINKEDIN if post.get("platform") == "linkedin" else Platform.INSTAGRAM
        post_id = str(post.get("post_id") or "")[:100]
        post_url = str(post.get("url") or "")[:500]
        short = shortcode_from_url(post_url)

        # A post may have been explicitly linked via the "I posted this" button
        # (process_feedback stores post_id == URL shortcode), but the same post
        # re-scraped from Instagram carries Apify's internal id, so a naive
        # post_id match misses it -> would double-count the weight and create a
        # duplicate Engagement row. Match on post_id OR the URL shortcode.
        existing = None
        for cand in db.query(Engagement).filter(Engagement.prompt_id == prompt_id).all():
            if cand.post_id == post_id:
                existing = cand
                break
            if short and cand.post_url and shortcode_from_url(cand.post_url) == short:
                existing = cand
                break
        is_new = do_learn
        if existing:
            # Refresh stored engagement (likes/comments grow over time) but do
            # not treat it as a new learning event.
            db.delete(existing)
            db.flush()

        engagement = Engagement(
            prompt_id=prompt_id,
            platform=platform,
            post_id=post_id,
            post_url=str(post.get("url") or "")[:500] or None,
            likes=int(post.get("likes") or 0),
            comments=int(post.get("comments") or 0),
            shares=int(post.get("shares") or 0),
            saves=int(post.get("saves") or 0),
            reach=int(post.get("reach") or 0),
            impressions=int(post.get("impressions") or 0),
            fetched_at=utc_now_naive(),
        )
        db.add(engagement)
        db.commit()
        if is_new:
            self._log_engagement(
                prompt_id, post, rate, outcome, prompt.caption or prompt.prompt_text,
                meta={
                    "business": business_id, "business_key": prompt.business.weights_key,
                    "prompt_type": prompt_type, "format": prompt.format,
                    "virality_score": prompt.virality_score,
                },
            )
        return is_new

    @staticmethod
    def _log_engagement(
        prompt_id: int, post: dict[str, Any], rate: float, outcome: str, caption: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Append a durable, git-tracked engagement record so learning history
        survives agent.db cache eviction (raw Engagement rows live only in the
        cache; the Bayesian weights persist separately in data/weights.json).
        Stores the posted caption so the record is self-contained even if the
        Prompt row is later deleted."""
        try:
            from pathlib import Path

            path = Path(settings.data_dir) / "engagement_log.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "ts": utc_now_naive().isoformat(),
                            "prompt_id": prompt_id,
                            "post_id": str(post.get("post_id") or "")[:100],
                            "caption": redact_text((caption or "")[:500]),
                            "rate": round(rate, 4),
                            "likes": int(post.get("likes") or 0),
                            "comments": int(post.get("comments") or 0),
                            "outcome": outcome,
                            **(meta or {}),
                        }
                    )
                    + "\n"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to append engagement log: %s", exc)

    @staticmethod
    def _parse_date(value: Any) -> datetime | None:
        """Any scraped timestamp -> NAIVE UTC (the convention of every DB column).

        Timezone-aware values (an ISO string with Z or an offset, an epoch) are CONVERTED to UTC.
        The old code simply dropped the tzinfo, so 14:00Z and 19:30+05:30 produced different
        'hours' and every window comparison / best-time inference was off by the UTC offset.
        Naive strings are taken to be UTC (instaloader's date_utc)."""
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1e9:
            dt = datetime.fromtimestamp(value, tz=UTC)
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC).replace(tzinfo=None)
        return dt


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
            try:
                biz = db.query(Business).filter(Business.name == config["name"]).first()
                if biz is None:
                    results[config["name"]] = {"skipped": "not in the database yet (no daily run has created it)"}
                    continue
                handle = config.get("instagram_handle") or ""
                if not handle:
                    results[config["name"]] = {"skipped": "no instagram_handle configured"}
                    continue
                social = asyncio.run(scraper.fetch_instagram(handle))
                results[config["name"]] = AutoLearner().run_for_business(biz.id, social, db)
            except Exception as exc:  # noqa: BLE001
                # One business's scrape/learning failure must not abort learning
                # for the rest (and fail the workflow step before weights commit).
                logger.error("::warning::Auto-learning failed for %s: %s", config.get("name"), exc)
                results[config.get("name", "?")] = {"matched": 0, "learned": 0, "error": str(exc)}
    finally:
        db.close()
    return results


SUMMARY_COLUMNS = (
    ("posts", "Posts found"), ("candidates", "Prompts eligible"), ("matched", "Matched"),
    ("deferred", f"Too new (<{MATURITY_HOURS}h)"), ("hidden_likes", "Likes hidden"), ("unusable", "Unusable"),
    ("refreshed", "Refreshed"), ("learned", "Learned"),
)


def format_summary(results: dict[str, Any]) -> str:
    """Markdown for the run page. 'Learned 0' alone cannot tell a quiet day from a broken scrape, so
    every count that explains a zero is shown, plus the reason a business was skipped or failed."""
    lines = ["### Engagement sync", ""]
    if not results:
        return "\n".join(lines + ["No active business to process."])
    lines += ["| Business | " + " | ".join(label for _, label in SUMMARY_COLUMNS) + " |",
              "|---|" + "---:|" * len(SUMMARY_COLUMNS)]
    notes: list[str] = []
    for name, res in results.items():
        cells = [str(res.get(key, "-")) if key in res else "-" for key, _ in SUMMARY_COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
        if res.get("skipped"):
            notes.append(f"- **{name}** skipped: {res['skipped']}")
        if res.get("error"):
            notes.append(f"- **{name}** failed: {res['error']}")
        if res.get("scrape_error"):
            notes.append(f"- **{name}** scrape returned no posts: {res['scrape_error']}")
        elif res.get("posts") == 0 and "followers" in res:
            seen = f"profile found ({res['followers']} followers)" if res["followers"] is not None else "no profile data came back"
            notes.append(f"- **{name}** has no posts to learn from yet: {seen}.")
    total = sum(int(r.get("learned", 0)) for r in results.values())
    lines += ["", f"**Total learned this run: {total}**"]
    if total == 0:
        lines.append("Nothing was learned: see the columns above (posts missing, none matched a prompt, still younger than "
                     f"{MATURITY_HOURS} h, likes hidden) and the notes below.")
    return "\n".join(lines + ([""] + notes if notes else []))


def write_step_summary(results: dict[str, Any]) -> bool:
    """Append the summary to the GitHub run page. No-op (False) outside Actions."""
    import os

    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return False
    try:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(format_summary(results) + "\n")
        return True
    except OSError as exc:
        logger.warning("Could not write the step summary: %s", exc)
        return False


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.shared.config import settings

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    outcome = run_auto_learning()
    for name, res in outcome.items():
        logger.info("AUTO-LEARN %s: %s", name, {k: v for k, v in res.items() if k != "details"})
        if res.get("scrape_error") or res.get("error"):
            logger.warning("::warning::Engagement sync for %s learned nothing: %s", name, res.get("scrape_error") or res.get("error"))
    write_step_summary(outcome)
