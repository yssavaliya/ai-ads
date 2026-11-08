"""Verified facts about the business, supplied by its owner in ``data/brand_facts.json``.

The website scrape of an Instagram URL yields almost nothing, so without this file the generator knows only a
name and a handle and fills the gaps with invented services, results and turnaround times. Everything in the
file is treated as true: the generator may state it, and numbers that appear in it pass the claims guard.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BRAND_FACTS_PATH = Path("data/brand_facts.json")

# Fields that carry real content. Keys starting with "_" are instructions for the human editing the file.
CONTENT_FIELDS = ("offers", "audience", "voice", "proof_points", "allowed_claims", "do_not_say")


def load_brand_facts(path: Path | str | None = None) -> dict[str, Any]:
    """The facts the owner wrote down; {} when the file is missing, unreadable or empty."""
    target = Path(path) if path else BRAND_FACTS_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if not key.startswith("_") and value not in ("", [], None)}


def has_content(facts: dict[str, Any]) -> bool:
    return any(facts.get(field) for field in CONTENT_FIELDS)


def merge_into_profile(profile: dict[str, Any] | None, facts: dict[str, Any]) -> dict[str, Any]:
    """The business profile the generator sees, with the verified facts attached under one clear key."""
    merged = dict(profile or {})
    verified = {key: value for key, value in facts.items() if key != "content_strategy"}
    strategy = facts.get("content_strategy")
    if verified:
        merged["verified_facts"] = verified
    if isinstance(strategy, dict) and strategy:
        # Keep editorial strategy separate from verified business claims. The
        # generator should follow it, but it is not evidence for factual claims.
        merged["content_strategy"] = strategy
    return merged


def grounding_text(facts: dict[str, Any]) -> str:
    """Text the claims guard may treat as supporting a number (empty when the owner wrote nothing)."""
    verified = {key: value for key, value in facts.items() if key != "content_strategy"}
    return json.dumps(verified, ensure_ascii=False) if verified else ""


def warn_if_empty(facts: dict[str, Any]) -> bool:
    """Log a visible warning when there are no facts; returns True when the file has content."""
    if has_content(facts):
        return True
    logger.warning(
        "::warning::data/brand_facts.json has no offers or proof points, so the generator knows only the business "
        "name. Copy will stay generic (no invented services, results or turnaround times). Fill the file in."
    )
    return False
