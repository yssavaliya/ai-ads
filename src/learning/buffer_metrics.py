"""Sync Buffer post metrics and learn from exact Buffer-to-creative attribution.

Buffer's personal-key post metrics are experimental and refresh daily. Missing
metric types stay null; they are never coerced to zero. A post is not judged
until it is at least 72 hours old, its metrics are fresh, and five earlier
comparable posts provide a baseline.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import httpx

from src.delivery.buffer_publisher import BUFFER_GQL_URL
from src.learning.auto_learner import AutoLearner
from src.learning.bayesian_updater import BayesianUpdater
from src.shared.config import settings
from src.storage.models import Prompt, SessionLocal

logger = logging.getLogger(__name__)

PUBLISHED_LOG = Path(settings.data_dir) / "published_log.json"
METRICS_LOG = Path(settings.data_dir) / "buffer_metrics.jsonl"
MIN_MATURITY_HOURS = 72
MIN_BASELINE_POSTS = 5
METRICS_MAX_AGE_HOURS = 48
REFRESH_AFTER_HOURS = 20

POST_METRICS_QUERY = """
query GetPostMetrics {
  post(input: { id: \"{post_id}\" }) {
    id
    status
    sentAt
    channelId
    metrics {
      type
      name
      value
      unit
    }
    metricsUpdatedAt
  }
}
"""

# Interactions with stronger downstream value get more weight. Rates are divided
# by reach, so they remain comparable when posts are shown to different numbers
# of accounts. Retention is evaluated separately when Buffer provides both fields.
INTERACTION_WEIGHTS = {
    "reactions": 1.0,
    "comments": 2.0,
    "shares": 3.0,
    "saves": 3.0,
    "follows": 4.0,
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_date(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _score(metrics: dict[str, float]) -> tuple[float | None, tuple[str, ...], float | None]:
    """Return weighted interactions/reach, metric signature, and average watch seconds."""
    reach = metrics.get("reach")
    components = tuple(sorted(key for key in INTERACTION_WEIGHTS if key in metrics))
    if reach is None or reach <= 0 or not components:
        return None, components, None
    interactions = sum(INTERACTION_WEIGHTS[key] * metrics[key] for key in components)
    rate = interactions / reach
    watched = metrics.get("totalTimeWatched")
    views = metrics.get("views")
    # Buffer reports Instagram Reels' averageTimeWatched directly in seconds.
    # totalTimeWatched is reported in minutes on networks such as LinkedIn.
    avg_watch_seconds = metrics.get("averageTimeWatched")
    if avg_watch_seconds is None and watched is not None and views and views > 0:
        avg_watch_seconds = watched * 60.0 / views
    return round(rate, 6), components, round(avg_watch_seconds, 3) if avg_watch_seconds is not None else None


def _metrics_map(raw_metrics: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in raw_metrics:
        name = str(metric.get("type") or "")
        value = metric.get("value")
        if name and isinstance(value, (int, float)) and not isinstance(value, bool):
            values[name] = float(value)
    return values


def _fetch_post(client: httpx.Client, token: str, post_id: str) -> dict[str, Any] | None:
    # JSON string encoding escapes quotes and control characters for GraphQL.
    query = POST_METRICS_QUERY.replace("{post_id}", json.dumps(post_id, ensure_ascii=False))
    response = None
    for attempt in range(3):
        response = client.post(
            BUFFER_GQL_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"query": query},
        )
        if response.status_code != 429 and response.status_code < 500:
            break
        if attempt < 2:
            time.sleep(2 ** attempt)
    assert response is not None
    if response.is_error:
        raise RuntimeError(f"Buffer API HTTP {response.status_code}: {response.text[:500]}")
    response.raise_for_status()
    payload = response.json()
    errors = payload.get("errors") or []
    if errors:
        raise RuntimeError("; ".join(str(error.get("message") or "Buffer GraphQL error") for error in errors))
    data = payload.get("data") or {}
    post = data.get("post")
    return post if isinstance(post, dict) else None


def _latest_by_post(events: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str], dict[str, datetime]]:
    snapshots: dict[str, dict[str, Any]] = {}
    learned: set[str] = set()
    last_checked: dict[str, datetime] = {}
    for event in events:
        post_id = str(event.get("buffer_post_id") or "")
        if not post_id:
            continue
        checked = _parse_date(event.get("checked_at"))
        if checked and (post_id not in last_checked or checked > last_checked[post_id]):
            last_checked[post_id] = checked
        if event.get("type") == "learned":
            learned.add(post_id)
        elif event.get("type") == "snapshot":
            updated = _parse_date(event.get("metrics_updated_at"))
            previous = snapshots.get(post_id)
            prev_updated = _parse_date(previous.get("metrics_updated_at")) if previous else None
            if previous is None or (updated and (prev_updated is None or updated >= prev_updated)):
                snapshots[post_id] = event
    return snapshots, learned, last_checked


def _summary(message: str) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(message + "\n")
        except OSError:
            logger.warning("Could not write Buffer metrics summary")


def sync_buffer_metrics() -> dict[str, int]:
    token = getattr(settings, "buffer_access_token", None) or os.getenv("BUFFER_ACCESS_TOKEN")
    if not token:
        logger.warning("::warning::Buffer metrics skipped: BUFFER_ACCESS_TOKEN is not configured")
        return {"mapped": 0, "refreshed": 0, "learned": 0, "skipped": 0}

    published = _read_json(PUBLISHED_LOG).get("posted") or []
    mappings: dict[str, dict[str, Any]] = {}
    for entry in published:
        if not isinstance(entry, dict):
            continue
        for buffer_post in entry.get("buffer_posts") or []:
            if isinstance(buffer_post, dict) and buffer_post.get("id"):
                mappings[str(buffer_post["id"])] = entry

    if not mappings:
        logger.info("Buffer metrics: no exact Buffer post IDs are linked yet")
        _summary("### Buffer post metrics\nNo Buffer post IDs are linked yet.")
        return {"mapped": 0, "refreshed": 0, "learned": 0, "skipped": 0}

    events = _read_events(METRICS_LOG)
    snapshots, learned_ids, last_checked = _latest_by_post(events)
    now = datetime.now(UTC)
    refreshed = skipped = 0
    current: dict[str, dict[str, Any]] = {}

    with httpx.Client(timeout=30) as client:
        for post_id, entry in mappings.items():
            previous = snapshots.get(post_id)
            checked = last_checked.get(post_id)
            if checked and now - checked < timedelta(hours=REFRESH_AFTER_HOURS) and previous:
                current[post_id] = previous
                continue
            try:
                post = _fetch_post(client, token, post_id)
            except Exception as exc:  # noqa: BLE001 - missing scope should not corrupt other learning
                skipped += 1
                _append_event(METRICS_LOG, {
                    "type": "error", "buffer_post_id": post_id, "checked_at": now.isoformat(),
                    "error": str(exc)[:300],
                })
                logger.warning(
                    "::warning::Buffer metrics unavailable for %s: %s. Confirm BUFFER_ACCESS_TOKEN is a personal key with insightsRead access.",
                    post_id, exc,
                )
                continue

            if not post:
                skipped += 1
                _append_event(METRICS_LOG, {
                    "type": "check", "buffer_post_id": post_id, "checked_at": now.isoformat(),
                    "status": "not_found",
                })
                continue

            sent_at = post.get("sentAt")
            metrics_updated_at = post.get("metricsUpdatedAt")
            raw_metrics = post.get("metrics")
            if not sent_at or not metrics_updated_at or not isinstance(raw_metrics, list):
                skipped += 1
                event = {
                    "type": "check", "buffer_post_id": post_id, "checked_at": now.isoformat(),
                    "status": post.get("status"), "sent_at": sent_at,
                    "metrics_updated_at": metrics_updated_at,
                }
                _append_event(METRICS_LOG, event)
                if previous:
                    current[post_id] = previous
                continue

            sent_time = _parse_date(sent_at)
            updated_time = _parse_date(metrics_updated_at)
            if not sent_time or not updated_time:
                skipped += 1
                continue
            raw_values = _metrics_map(raw_metrics)
            score, components, avg_watch = _score(raw_values)
            snapshot = {
                "type": "snapshot",
                "buffer_post_id": post_id,
                "creative_id": entry.get("id"),
                "prompt_id": entry.get("prompt_id"),
                "business_name": entry.get("business_name") or "",
                "sent_at": sent_time.isoformat(),
                "metrics_updated_at": updated_time.isoformat(),
                "checked_at": now.isoformat(),
                "metrics": {key: raw_values.get(key) for key in (
                    "reach", "impressions", "views", "reactions", "comments", "shares", "saves",
                    "follows", "totalTimeWatched", "averageTimeWatched", "engagementRate",
                )},
                "available_metrics": sorted(raw_values),
                "interaction_components": list(components),
                "quality_rate": score,
                "avg_watch_seconds": avg_watch,
            }
            if previous is None or previous.get("metrics_updated_at") != snapshot["metrics_updated_at"]:
                _append_event(METRICS_LOG, snapshot)
                refreshed += 1
            current[post_id] = snapshot

    # Read newly appended snapshots so every decision is replayable and idempotent.
    events = _read_events(METRICS_LOG)
    snapshots, learned_ids, _ = _latest_by_post(events)
    db = SessionLocal()
    updater = BayesianUpdater()
    recorder = AutoLearner()
    learned_count = 0
    try:
        ordered = sorted(
            snapshots.values(),
            key=lambda item: (_parse_date(item.get("sent_at")) or datetime.max.replace(tzinfo=UTC),
                              str(item.get("buffer_post_id"))),
        )
        for snapshot in ordered:
            post_id = str(snapshot.get("buffer_post_id") or "")
            if not post_id or post_id in learned_ids:
                continue
            sent_at = _parse_date(snapshot.get("sent_at"))
            metrics_at = _parse_date(snapshot.get("metrics_updated_at"))
            score = snapshot.get("quality_rate")
            reach = (snapshot.get("metrics") or {}).get("reach")
            components = tuple(snapshot.get("interaction_components") or ())
            if (
                not sent_at
                or not metrics_at
                or now - sent_at < timedelta(hours=MIN_MATURITY_HOURS)
                or now - metrics_at > timedelta(hours=METRICS_MAX_AGE_HOURS)
                or score is None
                or not isinstance(reach, (int, float))
                or reach <= 0
                or not components
            ):
                continue

            business_name = str(snapshot.get("business_name") or "")
            baseline = []
            retention_baseline = []
            for other in snapshots.values():
                other_sent = _parse_date(other.get("sent_at"))
                if (
                    other.get("buffer_post_id") == post_id
                    or str(other.get("business_name") or "") != business_name
                    or not other_sent
                    or other_sent >= sent_at
                    or now - other_sent < timedelta(hours=MIN_MATURITY_HOURS)
                    or tuple(other.get("interaction_components") or ()) != components
                    or other.get("quality_rate") is None
                    or (_parse_date(other.get("metrics_updated_at")) is None)
                    or now - _parse_date(other.get("metrics_updated_at")) > timedelta(hours=METRICS_MAX_AGE_HOURS)
                ):
                    continue
                baseline.append(float(other["quality_rate"]))
                other_retention = other.get("avg_watch_seconds")
                if other_retention is not None:
                    retention_baseline.append(float(other_retention))
            if len(baseline) < MIN_BASELINE_POSTS:
                continue

            boundary = median(baseline)
            outcome = "good" if float(score) >= boundary else "bad"
            avg_watch = snapshot.get("avg_watch_seconds")
            if avg_watch is not None and len(retention_baseline) >= MIN_BASELINE_POSTS:
                outcome = "good" if outcome == "good" and float(avg_watch) >= median(retention_baseline) else "bad"

            prompt_id = snapshot.get("prompt_id")
            try:
                prompt = db.get(Prompt, int(prompt_id)) if prompt_id is not None else None
            except (TypeError, ValueError):
                prompt = None
            if prompt is None:
                logger.warning("Buffer metrics: Prompt %r is unavailable; recording metrics without a weight update", prompt_id)
                continue

            metrics = snapshot.get("metrics") or {}
            post_data = {
                "post_id": f"buffer:{post_id}",
                "posted_at": sent_at.isoformat(),
                "platform": "instagram",
                "likes": int(metrics.get("reactions") or 0),
                "comments": int(metrics.get("comments") or 0),
                "shares": int(metrics.get("shares") or 0),
                "saves": int(metrics.get("saves") or 0),
                "reach": int(reach),
                "impressions": int(metrics.get("impressions") or 0),
                "views": metrics.get("views"),
                "follows": metrics.get("follows"),
                "totalTimeWatched": metrics.get("totalTimeWatched"),
                "averageTimeWatched": metrics.get("averageTimeWatched"),
                "avg_watch_seconds": avg_watch,
            }
            recorded = recorder._record(
                db, prompt.business_id, prompt.id,
                prompt.prompt_type.value if prompt.prompt_type else None,
                post_data, float(score), outcome, learn=True,
            )
            if recorded:
                updater.record(
                    prompt.business.weights_key,
                    outcome,
                    prompt_type=prompt.prompt_type.value if prompt.prompt_type else None,
                    format=prompt.format,
                    audience_segment=prompt.audience_segment,
                    hook_type=prompt.hook_type,
                    source="buffer_metrics",
                )
                learned_count += 1

            _append_event(METRICS_LOG, {
                "type": "learned", "buffer_post_id": post_id, "prompt_id": prompt.id,
                "business_name": business_name, "outcome": outcome,
                "quality_rate": score, "baseline_median": round(boundary, 6),
                "baseline_posts": len(baseline), "avg_watch_seconds": avg_watch,
                "retention_baseline_median": round(median(retention_baseline), 3) if retention_baseline else None,
                "learned_at": now.isoformat(), "weight_updated": recorded,
            })
            learned_ids.add(post_id)
            logger.info(
                "Buffer metrics learning: post=%s outcome=%s score=%.5f baseline=%.5f n=%d weight_updated=%s",
                post_id, outcome, score, boundary, len(baseline), recorded,
            )
    finally:
        db.close()

    logger.info(
        "Buffer metrics sync: mapped=%d refreshed=%d learned=%d skipped=%d",
        len(mappings), refreshed, learned_count, skipped,
    )
    _summary(
        "### Buffer post metrics\n"
        f"Mapped posts: {len(mappings)} · snapshots refreshed: {refreshed} · new learning updates: {learned_count} · skipped/not mature: {skipped}\n\n"
        f"Learning requires 72-hour maturity, fresh per-post metrics, reach, at least one reported interaction type, and {MIN_BASELINE_POSTS} earlier comparable posts."
    )
    return {"mapped": len(mappings), "refreshed": refreshed, "learned": learned_count, "skipped": skipped}


def _buffer_managed_prompt_ids() -> set[int]:
    """Don't let the lower-fidelity scraper learn prompts with exact Buffer IDs."""
    if not (getattr(settings, "buffer_access_token", None) or os.getenv("BUFFER_ACCESS_TOKEN")):
        return set()
    records = _read_json(PUBLISHED_LOG).get("posted") or []
    ids = set()
    for entry in records:
        if not isinstance(entry, dict) or not entry.get("buffer_posts"):
            continue
        try:
            ids.add(int(entry.get("prompt_id")))
        except (TypeError, ValueError):
            continue
    return ids


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sync_buffer_metrics()
