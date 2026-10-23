"""Is the learning loop actually learning? Computes monitoring metrics from the durable logs.

Inputs are the git-tracked files the pipelines already write:
  data/engagement_log.jsonl   one record per judged post (outcome, virality score, type, format)
  data/learning_ledger.jsonl  every weight update (source: auto | manual)

Metrics (all descriptive - with few posts they are noisy, and the report says so):
  hit_rate            share of judged posts that beat the business's own benchmark
  trend               hit rate of the first half vs the second half of the history
  by_type             hit rate and count per prompt type (what the weights are reacting to)
  score_separation    mean predicted virality of hits minus misses. > 0 means the predictor ranks well;
                      ~0 or negative means the score carries no signal for this business
  sources             how many weight updates came from engagement vs. human clicks
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.learning.bayesian_updater import effective_updates, read_ledger

MIN_EVENTS_FOR_TREND = 8


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _rate(records: list[dict[str, Any]]) -> float | None:
    return round(sum(1 for r in records if r.get("outcome") == "good") / len(records), 3) if records else None


def evaluate(log_path: Path, ledger_path: Path) -> dict[str, Any]:
    records = [r for r in _read_jsonl(log_path) if r.get("outcome") in ("good", "bad")]
    # one judgement per post: a duplicated record must not count twice
    unique: dict[str, dict[str, Any]] = {}
    for r in records:
        unique[str(r.get("post_id") or r.get("prompt_id"))] = r
    records = list(unique.values())

    half = len(records) // 2
    by_type: dict[str, dict[str, Any]] = {}
    for r in records:
        bucket = by_type.setdefault(str(r.get("prompt_type") or "unknown"), {"n": 0, "good": 0})
        bucket["n"] += 1
        bucket["good"] += 1 if r["outcome"] == "good" else 0
    for bucket in by_type.values():
        bucket["hit_rate"] = round(bucket["good"] / bucket["n"], 3)

    scored = [r for r in records if isinstance(r.get("virality_score"), (int, float))]
    good = [r["virality_score"] for r in scored if r["outcome"] == "good"]
    bad = [r["virality_score"] for r in scored if r["outcome"] == "bad"]
    separation = round(sum(good) / len(good) - sum(bad) / len(bad), 2) if good and bad else None

    _, updates = effective_updates(read_ledger(ledger_path))
    sources: dict[str, int] = {}
    for u in updates:
        sources[u.get("source", "unknown")] = sources.get(u.get("source", "unknown"), 0) + 1

    enough = len(records) >= MIN_EVENTS_FOR_TREND
    return {
        "judged_posts": len(records),
        "hit_rate": _rate(records),
        "trend": {"first_half": _rate(records[:half]), "second_half": _rate(records[half:])} if enough else None,
        "by_type": by_type,
        "score_separation": separation,
        "ledger_version": len(updates),
        "sources": sources,
        "caveat": None if enough else f"fewer than {MIN_EVENTS_FOR_TREND} judged posts: no trend can be claimed yet",
    }
