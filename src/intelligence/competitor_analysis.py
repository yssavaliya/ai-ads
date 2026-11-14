"""Competitor benchmarks computed from posts we actually scraped (no LLM, no guessing).

An agency's competitor analyst answers: how often do they post, what formats, what openers and CTAs, what hashtags,
which posts beat their own average and what do those have in common. This module computes exactly that from the
verified competitors' recent posts, and turns it into a short list of observations (each tied to its numbers) that the
strategy and trend prompts receive as FACTS, instead of a wall of raw captions.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from src.intelligence.research import verified_competitors

_HASHTAG = re.compile(r"#\w+")
_CTA = re.compile(
    r"\b(?:dm|comment|save|share|tag|follow|link in bio|book|order|shop|call|whatsapp|click|sign up|learn more|try)\b",
    re.IGNORECASE,
)
MIN_POSTS = 3


def _parse(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _interactions(post: dict[str, Any]) -> int:
    return int(post.get("likes") or 0) + 2 * int(post.get("comments") or 0)


def _opener(caption: str, words: int = 8) -> str:
    first_line = (caption or "").strip().splitlines()[0] if (caption or "").strip() else ""
    return " ".join(first_line.split()[:words])


def _is_video(post: dict[str, Any]) -> bool:
    kind = f"{post.get('media_type') or ''} {post.get('product_type') or ''}".lower()
    return "video" in kind or "clips" in kind or "reel" in kind or bool(post.get("video_views"))


def analyze_competitor(competitor: dict[str, Any]) -> dict[str, Any] | None:
    """Metrics for one competitor, or None when too few posts to say anything."""
    posts = [p for p in (competitor.get("top_posts") or []) if isinstance(p, dict) and p.get("caption") is not None]
    if len(posts) < MIN_POSTS:
        return None
    profile = competitor.get("profile") or {}
    followers = profile.get("follower_count") or competitor.get("follower_count")
    dates = sorted(d for d in (_parse(p.get("posted_at")) for p in posts) if d)
    span_days = (dates[-1] - dates[0]).days if len(dates) >= 2 else 0
    per_week = round(len(dates) / (span_days / 7.0), 1) if span_days >= 7 else None
    inter = [_interactions(p) for p in posts]
    median_inter = statistics.median(inter)
    ranked = sorted(posts, key=_interactions, reverse=True)
    captions = [str(p.get("caption") or "") for p in posts]
    cta_share = sum(1 for c in captions if _CTA.search(c)) / len(captions)
    return {
        "name": competitor.get("name") or competitor.get("handle") or "?",
        "handle": competitor.get("handle"),
        "followers": followers,
        "posts_seen": len(posts),
        "posts_per_week": per_week,
        "video_share": round(sum(1 for p in posts if _is_video(p)) / len(posts), 2),
        "median_interactions": median_inter,
        "median_engagement_rate": round(median_inter / followers, 4) if followers else None,
        "median_caption_words": int(statistics.median(len(c.split()) for c in captions)),
        "cta_share": round(cta_share, 2),
        "hashtags": [tag for tag, _ in Counter(t.lower() for c in captions for t in _HASHTAG.findall(c)).most_common(5)],
        "best_openers": [
            {"opener": _opener(p.get("caption") or ""), "interactions": _interactions(p)}
            for p in ranked[:3] if _opener(p.get("caption") or "")
        ],
    }


def observations(analyses: list[dict[str, Any]]) -> list[str]:
    """Plain-language findings, each grounded in the numbers above. Empty when there is nothing solid to say."""
    notes: list[str] = []
    if not analyses:
        return notes
    video = statistics.mean(a["video_share"] for a in analyses)
    notes.append(f"Video/Reel share across verified competitors: {video:.0%}"
                 + (" (Reels are the norm here)" if video >= 0.5 else " (mostly still images: Reels can stand out)" if video < 0.25 else ""))
    cadence = [a["posts_per_week"] for a in analyses if a["posts_per_week"] is not None]
    if cadence:
        notes.append(f"Competitors post about {statistics.median(cadence):.1f} times per week (median)")
    cta = statistics.mean(a["cta_share"] for a in analyses)
    notes.append(f"{cta:.0%} of competitor captions contain an explicit call to action"
                 + (" - a clear next step is a differentiator" if cta < 0.4 else ""))
    words = statistics.median(a["median_caption_words"] for a in analyses)
    notes.append(f"Typical caption length: {words:.0f} words")
    tags = Counter(t for a in analyses for t in a["hashtags"]).most_common(4)
    if tags:
        notes.append("Most-used hashtags: " + ", ".join(t for t, _ in tags))
    rated = [a for a in analyses if a["median_engagement_rate"] is not None]
    if len(rated) >= 2:
        top = max(rated, key=lambda a: a["median_engagement_rate"])
        notes.append(f"Strongest engagement relative to audience: {top['name']} ({top['median_engagement_rate']:.1%} of followers)")
    return notes


def benchmark(competitors: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Verified-only competitor benchmarks for prompts and the hub. ``available`` is False when there is no evidence."""
    analyses = [a for a in (analyze_competitor(c) for c in verified_competitors(competitors)) if a]
    return {
        "available": bool(analyses),
        "competitors": analyses[:5],
        "observations": observations(analyses),
        "source": "scraped public posts (verified competitors only)",
    }
