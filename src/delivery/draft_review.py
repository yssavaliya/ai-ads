"""Unattended review of Buffer drafts: analyse, then schedule at the right time, or hold with a reason.

A draft nobody reviewed must not sit forever, and it must not be published blindly either. After the review window
this module re-checks each draft the pipeline created (quality score, unverified claims, duplicates, media) and, if it
passes, picks an exact publish time (measured window or experiment arm, kept clear of the queue). Anything that fails
a check is HELD for a human with a stated reason; the existing max-age rule still removes it eventually.

Pure functions only: the network calls live in scripts/promote_drafts.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.delivery.buffer_state import find_duplicate
from src.delivery.slot_planner import plan_slot
from src.generation.claims import unsupported_claims

DEFAULT_MIN_SCORE = 70


@dataclass(frozen=True)
class Decision:
    action: str            # "schedule" | "hold"
    reason: str
    due_at: datetime | None = None
    hour: int | None = None
    timezone: str = ""


def index_records(published_log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Buffer post id -> the pipeline's record of the creative behind it."""
    out: dict[str, dict[str, Any]] = {}
    for record in published_log.get("posted") or []:
        if not isinstance(record, dict):
            continue
        for post in record.get("buffer_posts") or []:
            if isinstance(post, dict) and post.get("id"):
                out[str(post["id"])] = record
    return out


def _hour(record: dict[str, Any]) -> int | None:
    hour = record.get("schedule_hour")
    if hour is None:
        hour = (record.get("window") or {}).get("hour")
    try:
        return None if hour is None else int(hour)
    except (TypeError, ValueError):
        return None


def review_draft(
    draft: dict[str, Any],
    record: dict[str, Any] | None,
    queue_texts: list[str],
    taken: list[datetime],
    now: datetime,
    grounding: str = "",
    min_score: int | None = None,
) -> Decision:
    if record is None:
        return Decision("hold", "not created by this pipeline (no creative record)")

    score = record.get("score")
    threshold = int(min_score if min_score is not None else (record.get("threshold") or DEFAULT_MIN_SCORE))
    if record.get("below_threshold"):
        return Decision("hold", f"flagged below its quality bar ({score}/{threshold}); needs a human decision")
    if not isinstance(score, (int, float)):
        return Decision("hold", "never scored (the virality predictor was unavailable); needs a human decision")
    if score < threshold:
        return Decision("hold", f"score {score} is under the quality bar {threshold}")

    text = str(draft.get("text") or record.get("caption") or "")
    bad = unsupported_claims({"caption": text, "title": record.get("title") or ""}, grounding)
    if bad:
        return Decision("hold", f"unverified claims in the caption: {', '.join(bad[:3])}")

    repeated = find_duplicate(text, queue_texts)
    if repeated:
        return Decision("hold", f"near-duplicate of a post already in Buffer ({repeated[:50]!r})")

    hour, tz = _hour(record), str(record.get("timezone") or "")
    due = plan_slot(now, hour, tz or None, taken)
    if due is None:
        return Decision("hold", "no free publish slot inside the planning horizon", hour=hour, timezone=tz)
    basis = (record.get("window") or {}).get("source") or "default evening slot"
    if (record.get("experiment") or {}).get("variable") == "post_block":
        basis = f"timing experiment arm {record['experiment'].get('arm')}"
    return Decision("schedule", f"passed review (score {score}); timing basis: {basis}", due, hour, tz)
