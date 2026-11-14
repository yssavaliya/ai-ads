"""Conversion tracking: which creative, angle and hook actually produced inquiries and leads.

Instagram offers no public API for "someone DMed us because of this post", so the one honest, automatable path is:
every published creative on the hub gets a signed "Got a lead from this post?" link. One click opens a pre-filled
GitHub issue; the feedback job verifies the signature, writes an append-only event here, and credits the creative.
Nothing is inferred: a lead is only counted when the owner says so, and each issue is counted once.

Events live in ``data/conversions.jsonl`` (append-only, replayable). The weekly report summarises them by angle,
hook, format and audience, next to the vanity metrics, so results that matter are visible beside reach and likes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENT_TYPES = ("lead", "inquiry", "booking", "sale")
_VALUE = re.compile(r"value[:\s]+(?:₹|rs\.?|inr|\$|usd|eur|€)?\s*([\d][\d,]*(?:\.\d+)?)", re.IGNORECASE)
_TYPE = re.compile(r"\btype[:\s]+(lead|inquiry|enquiry|booking|sale)\b", re.IGNORECASE)
DIMENSIONS = ("angle", "hook_type", "format", "audience_segment", "prompt_type")


def ledger_path(data_dir: Path) -> Path:
    return Path(data_dir) / "conversions.jsonl"


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("id"):
            events.append(event)
    return events


def angle_for_prompt(data_dir: Path, prompt_id: Any) -> str:
    """The campaign angle the pipeline recorded when it published this prompt ('' when unknown)."""
    try:
        posted = json.loads((Path(data_dir) / "published_log.json").read_text(encoding="utf-8")).get("posted") or []
    except (OSError, ValueError):
        return ""
    for entry in reversed(posted):
        if isinstance(entry, dict) and str(entry.get("prompt_id")) == str(prompt_id) and entry.get("angle"):
            return str(entry["angle"])
    return ""


def parse_details(body: str) -> dict[str, Any]:
    """Optional ``Type:`` and ``Value:`` lines the owner may add to the issue (never required)."""
    text = body or ""
    kind = "lead"
    match = _TYPE.search(text)
    if match:
        kind = "inquiry" if match.group(1).lower() == "enquiry" else match.group(1).lower()
    value = None
    match = _VALUE.search(text)
    if match:
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            value = None
    return {"type": kind, "value": value}


def record(
    path: Path,
    event_id: str,
    prompt: dict[str, Any],
    body: str = "",
    source: str = "issue",
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Append one conversion event unless ``event_id`` was already recorded. Returns the event, or None if a duplicate."""
    existing = read_events(path)
    if any(e.get("id") == event_id for e in existing):
        return None
    details = parse_details(body)
    event = {
        "id": event_id,
        "ts": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "type": details["type"],
        "value": details["value"],
        "source": source,
        "prompt_uid": prompt.get("uid"),
        "prompt_id": prompt.get("prompt_id"),
        "business": prompt.get("business") or "",
        "angle": prompt.get("angle") or "",
        "hook_type": prompt.get("hook_type") or "",
        "format": prompt.get("format") or "",
        "audience_segment": prompt.get("audience_segment") or "",
        "prompt_type": prompt.get("prompt_type") or "",
        "first_for_prompt": not any(e.get("prompt_uid") == prompt.get("uid") for e in existing),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def summarize(events: list[dict[str, Any]], business: str | None = None) -> dict[str, Any]:
    """Totals and the best performers per dimension, from the ledger only."""
    rows = [e for e in events if business is None or not e.get("business") or e.get("business") == business]
    summary: dict[str, Any] = {
        "total": len(rows),
        "by_type": dict(Counter(e.get("type") or "lead" for e in rows)),
        "value": round(sum(float(e["value"]) for e in rows if isinstance(e.get("value"), (int, float))), 2),
        "creatives": len({e.get("prompt_uid") for e in rows if e.get("prompt_uid")}),
    }
    for dimension in DIMENSIONS:
        counts = Counter(str(e.get(dimension)) for e in rows if e.get(dimension))
        summary[f"by_{dimension}"] = counts.most_common(5)
    return summary


def lines_for_report(summary: dict[str, Any]) -> list[str]:
    """Plain-language findings for the weekly report; one line says so when there is nothing yet."""
    if not summary.get("total"):
        return ["No leads have been logged against any post yet. Use the 'Got a lead?' link on a post when someone enquires."]
    lines = [f"{summary['total']} conversion(s) from {summary['creatives']} creative(s)"
             + (f", worth {summary['value']:g} in total" if summary.get("value") else "")]
    labels = {"angle": "angle", "hook_type": "hook", "format": "format", "audience_segment": "audience", "prompt_type": "content type"}
    for dimension in DIMENSIONS:
        top = summary.get(f"by_{dimension}") or []
        if top:
            lines.append(f"Most conversions by {labels[dimension]}: " + ", ".join(f"{name} ({count})" for name, count in top[:3]))
    return lines
