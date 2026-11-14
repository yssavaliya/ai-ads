"""Research provenance: tell downstream prompts how much of the 'research' is real.

Competitors are first suggested by an LLM (from its memory), then scraped. A suggestion whose handle could not be
scraped may not exist, so it must not be presented to later stages as market evidence. Trends come from model
knowledge, not a live feed, and are labelled as hypotheses.
"""
from __future__ import annotations

from typing import Any

TREND_PROVENANCE = (
    "DATA PROVENANCE: trends below come from model knowledge, not a live trend feed or analytics. "
    "Treat them as hypotheses to adapt to this brand, never as proven facts."
)


LIVE_PROVENANCE = (
    "DATA PROVENANCE: the LIVE SIGNALS block below is real, dated and attributed (news, Hacker News, Instagram "
    "hashtag posts). Ground trending_topics in it where it is relevant and mark them basis=\"live_signal\"; "
    "anything you add from your own knowledge must be marked basis=\"model_knowledge\" and treated as a hypothesis."
)


def verified_competitors(competitors: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Only competitors whose public posts were actually retrieved."""
    return [c for c in (competitors or []) if any(p.get("caption") for p in (c.get("top_posts") or []))]


def summarize_competitors(competitors: list[dict[str, Any]] | None) -> str:
    suggested = [c for c in (competitors or []) if isinstance(c, dict)]
    real = verified_competitors(suggested)
    if not real:
        return (
            "No verified competitor data (the suggested competitors could not be scraped, so their existence and "
            "activity are unconfirmed). Do not cite competitor behaviour."
            if suggested else "No competitor data available."
        )
    lines = []
    for comp in real[:5]:
        captions = [p.get("caption", "") for p in comp.get("top_posts", []) if p.get("caption")]
        lines.append(f"- {comp.get('name', comp.get('handle', '?'))}: " + " || ".join(captions[:4])[:2000])
    unverified = len(suggested) - len(real)
    header = f"Verified from public posts: {len(real)} of {len(suggested)} suggested competitors."
    if unverified:
        header += f" {unverified} could not be verified and are excluded."
    return header + "\n" + "\n".join(lines)


def research_quality(
    competitors: list[dict[str, Any]] | None, social_data: dict[str, Any] | None, live_signals: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compact data-quality record for logs and the hub: what the strategy was actually based on."""
    suggested = [c for c in (competitors or []) if isinstance(c, dict)]
    return {
        "competitors_suggested": len(suggested),
        "competitors_verified": len(verified_competitors(suggested)),
        "own_posts_seen": len((social_data or {}).get("posts") or []),
        "trend_source": (
            f"live signals ({len(live_signals['signals'])} from {', '.join(live_signals.get('sources', []))}) + model knowledge"
            if live_signals and live_signals.get("signals") else "model knowledge (no live feed)"
        ),
    }
