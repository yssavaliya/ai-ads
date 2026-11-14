"""Evidence-based posting window.

Old behaviour: take the hour an LLM wrote in the audience map, else the hour the business happened to post most
often (which is where it POSTED, not where posts PERFORMED). This module weighs real performance by local time of day:

- every scraped post (own posts weight 1.0, verified competitors' posts weight 0.5) becomes a sample;
- each post's performance is relative to its own account's median, so big and small accounts are comparable;
- hours are grouped into 3-hour blocks to keep the estimate stable with few posts;
- the answer carries a confidence level and its evidence, and falls back (labelled) to the model's estimate or a
  default when there is not enough data. It never presents a guess as measured.
"""
from __future__ import annotations

import statistics
from collections import Counter
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.intelligence.research import verified_competitors

BLOCK_HOURS = 3
OWN_WEIGHT = 1.0
COMPETITOR_WEIGHT = 0.5
MIN_SAMPLES = 8          # below this there is no measured recommendation
MEDIUM_SAMPLES = 20
MIN_BLOCK_SAMPLES = 3
DEFAULT_HOUR = 19


def _parse(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _interactions(post: dict[str, Any]) -> float:
    likes = post.get("likes")
    if post.get("likes_hidden") or (isinstance(likes, (int, float)) and likes < 0):
        return -1.0  # hidden counts are missing data, never zero performance
    return float(int(likes or 0) + 2 * int(post.get("comments") or 0))


def _samples(posts: list[dict[str, Any]], tz: ZoneInfo, weight: float) -> list[tuple[int, float, float]]:
    """(local_hour, relative_performance, weight) per usable post."""
    usable = [(p, _interactions(p)) for p in posts if isinstance(p, dict)]
    usable = [(p, v) for p, v in usable if v >= 0 and _parse(p.get("posted_at")) is not None]
    if len(usable) < 3:
        return []
    median = statistics.median(v for _, v in usable) or 1.0
    out = []
    for post, value in usable:
        local = _parse(post.get("posted_at")).astimezone(tz)  # type: ignore[union-attr]
        out.append((local.hour, value / median, weight))
    return out


def _block(hour: int) -> int:
    return hour // BLOCK_HOURS


def _label(block: int) -> str:
    start = block * BLOCK_HOURS
    end = start + BLOCK_HOURS - 1
    return f"{start:02d}:00-{end:02d}:59"


def recommend_window(
    own_posts: list[dict[str, Any]] | None,
    competitor_accounts: list[list[dict[str, Any]]] | None,
    tz_name: str = "Asia/Kolkata",
    audience_hint_hour: int | None = None,
) -> dict[str, Any]:
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001 - bad timezone name in config
        tz = ZoneInfo("Asia/Kolkata")

    samples = _samples(own_posts or [], tz, OWN_WEIGHT)
    for account_posts in competitor_accounts or []:  # medians are per account so big and small are comparable
        samples += _samples(account_posts, tz, COMPETITOR_WEIGHT)
    blocks: dict[int, list[tuple[float, float]]] = {}
    for hour, score, weight in samples:
        blocks.setdefault(_block(hour), []).append((score, weight))

    eligible = {
        b: sum(s * w for s, w in items) / sum(w for _, w in items)
        for b, items in blocks.items() if len(items) >= MIN_BLOCK_SAMPLES
    }
    if len(samples) >= MIN_SAMPLES and eligible:
        best = max(eligible, key=lambda b: eligible[b])
        n_best = len(blocks[best])
        others = [v for b, v in eligible.items() if b != best]
        lift = (eligible[best] / statistics.mean(others) - 1.0) if others and statistics.mean(others) > 0 else None
        confidence = "medium" if len(samples) >= MEDIUM_SAMPLES and n_best >= 5 else "low"
        evidence = (
            f"{_label(best)} had the strongest relative engagement across {len(samples)} scraped posts "
            f"({n_best} in that window"
            + (f", {lift:+.0%} vs other windows" if lift is not None else "")
            + "). Competitor posts count half; local times assume the business's timezone."
        )
        return {
            "hour": best * BLOCK_HOURS + 1, "block": _label(best), "confidence": confidence,
            "source": "scraped post performance", "evidence": evidence, "samples": len(samples),
        }

    if audience_hint_hour is not None:
        return {
            "hour": audience_hint_hour, "block": None, "confidence": "none",
            "source": "audience map (model estimate, not measured)",
            "evidence": f"Only {len(samples)} usable posts (need {MIN_SAMPLES}); using the model's estimate.",
            "samples": len(samples),
        }
    own_hours = [
        _parse(p.get("posted_at")).astimezone(tz).hour  # type: ignore[union-attr]
        for p in (own_posts or []) if isinstance(p, dict) and _parse(p.get("posted_at")) is not None
    ]
    if len(own_hours) >= 3:
        habit = Counter(own_hours).most_common(1)[0][0]
        return {
            "hour": habit, "block": None, "confidence": "none", "source": "usual posting hour (habit, not performance)",
            "evidence": f"Only {len(samples)} usable posts with engagement (need {MIN_SAMPLES}); using the hour this "
                        f"account posts most often, which says nothing about when posts perform best.",
            "samples": len(samples),
        }
    return {
        "hour": DEFAULT_HOUR, "block": None, "confidence": "none", "source": "default evening slot",
        "evidence": f"Only {len(samples)} usable posts (need {MIN_SAMPLES}) and no audience estimate.",
        "samples": len(samples),
    }


def competitor_accounts(competitors: list[dict[str, Any]] | None) -> list[list[dict[str, Any]]]:
    """One list of posts per verified competitor account."""
    return [
        [p for p in (comp.get("top_posts") or []) if isinstance(p, dict)]
        for comp in verified_competitors(competitors)
    ]
