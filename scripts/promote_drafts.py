#!/usr/bin/env python3
"""Unattended review and scheduling of Buffer drafts.

The pipeline saves drafts so a human can review them in Buffer. A draft nobody touches must neither rot nor be
published blindly, so this job (run on a schedule) does what a social media manager would:

  1. Leave drafts younger than DRAFT_MIN_AGE_HOURS alone (the human review window).
  2. Delete drafts whose media is gone, or older than DRAFT_MAX_AGE_HOURS (no longer timely).
  3. Review every remaining draft (src/delivery/draft_review.py): quality score, unverified claims, duplicates.
  4. Schedule the ones that pass at an exact time (measured posting window or experiment arm, clear of the queue)
     with Buffer's custom schedule; fall back to the next automatic queue slot if Buffer rejects a custom time.
  5. HOLD the rest, with the reason printed and written to the Actions summary.

Environment: BUFFER_ACCESS_TOKEN, BUFFER_PROFILE_IDS (required); DRAFT_MIN_AGE_HOURS (24), DRAFT_MAX_AGE_HOURS (168),
BUFFER_QUEUE_LIMIT (3), AUTO_SCHEDULE_MIN_SCORE (the creative's own threshold, else 70), DRY_RUN=true (report only).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.delivery.buffer_state import (  # noqa: E402
    check_asset_url,
    delete_post,
    fetch_drafts,
    fetch_queue_state,
    promote_draft,
    schedule_draft,
    supports_custom_schedule,
)
from src.delivery.draft_review import index_records, review_draft  # noqa: E402
from src.delivery.slot_planner import describe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _grounding() -> str:
    try:
        from src.shared.brand_facts import grounding_text, load_brand_facts

        return str(grounding_text(load_brand_facts()))
    except Exception as exc:  # noqa: BLE001 - claims are then checked against nothing, which is stricter
        logger.warning("Brand facts unavailable (%s); claims are checked with no verified facts", exc)
        return ""


def _published_log() -> dict:
    try:
        return json.loads((ROOT / "data" / "published_log.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _summary(lines: list[str]) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + "\n")
        except OSError:
            pass


def main() -> None:
    token = os.getenv("BUFFER_ACCESS_TOKEN")
    profile_ids = [p.strip() for p in (os.getenv("BUFFER_PROFILE_IDS") or os.getenv("BUFFER_PROFILE_ID") or "").split(",") if p.strip()]
    if not token or not profile_ids:
        print("⚠️  BUFFER_ACCESS_TOKEN and BUFFER_PROFILE_IDS required. Skipping.")
        sys.exit(0)

    min_age = float(os.getenv("DRAFT_MIN_AGE_HOURS") or "24")
    max_age = float(os.getenv("DRAFT_MAX_AGE_HOURS") or "168")
    queue_limit = int(os.getenv("BUFFER_QUEUE_LIMIT") or "3")
    min_score = int(os.getenv("AUTO_SCHEDULE_MIN_SCORE")) if os.getenv("AUTO_SCHEDULE_MIN_SCORE") else None
    dry_run = (os.getenv("DRY_RUN") or "").lower() in {"1", "true", "yes"}

    records = index_records(_published_log())
    grounding = _grounding()
    custom_ok = supports_custom_schedule(token)  # True / False / None (unknown: try, then fall back)
    print(f"Buffer custom scheduling supported: {custom_ok if custom_ok is not None else 'unknown (will try)'}")

    now = datetime.now(UTC)
    scheduled = deleted = held = young = 0
    report: list[str] = ["### Draft review", "", "| Draft | Age | Decision |", "|---|---|---|"]

    for channel_id in profile_ids:
        try:
            queue = fetch_queue_state(token, channel_id)
            slots = max(0, queue_limit - queue.scheduled)
            queue_texts, taken = queue.texts(), queue.due_times()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read the queue for %s (%s); scheduling nothing there", channel_id, exc)
            slots, queue_texts, taken = 0, [], []

        drafts = fetch_drafts(token, channel_id)
        logger.info("Channel %s: %d drafts, %d queue slots free", channel_id, len(drafts), slots)

        for draft in drafts:
            post_id = str(draft.get("id"))
            preview = (draft.get("text") or "")[:50].replace("\n", " ")
            try:
                created = datetime.fromisoformat(str(draft.get("createdAt") or "").replace("Z", "+00:00"))
            except ValueError:
                logger.warning("Draft %s: unreadable createdAt; skipping", post_id)
                continue
            age = (now - created).total_seconds() / 3600

            if age < min_age:
                young += 1
                report.append(f"| `{preview}…` | {age:.0f}h | waiting for human review ({min_age:.0f}h window) |")
                continue

            urls = [a.get("url") for a in (draft.get("assets") or []) if a.get("url")]
            if urls and not all(check_asset_url(u) for u in urls):
                reason = "media expired or broken"
            elif age > max_age:
                reason = f"older than {max_age:.0f}h (no longer timely)"
            else:
                reason = ""
            if reason:
                if not dry_run:
                    try:
                        delete_post(token, post_id)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Could not delete %s: %s", post_id, exc)
                        continue
                deleted += 1
                report.append(f"| `{preview}…` | {age:.0f}h | 🗑️ deleted: {reason} |")
                continue

            if slots <= 0:
                held += 1
                report.append(f"| `{preview}…` | {age:.0f}h | held: queue is full ({queue_limit}) |")
                continue

            decision = review_draft(draft, records.get(post_id), queue_texts, taken, now, grounding, min_score)
            if decision.action != "schedule" or decision.due_at is None:
                held += 1
                logger.info("Draft %s held: %s", post_id, decision.reason)
                report.append(f"| `{preview}…` | {age:.0f}h | ⏸️ held: {decision.reason} |")
                continue

            when = describe(decision.due_at, decision.timezone or None)
            if dry_run:
                report.append(f"| `{preview}…` | {age:.0f}h | would schedule {when} ({decision.reason}) |")
                scheduled += 1
                continue
            try:
                if custom_ok is False:
                    raise RuntimeError("custom scheduling is not available")
                post = schedule_draft(token, post_id, decision.due_at)
                how = f"scheduled {when}"
            except Exception as exc:  # noqa: BLE001 - never lose an approved draft over a custom time
                logger.warning("Custom schedule failed for %s (%s); using the next automatic slot", post_id, exc)
                try:
                    post = promote_draft(token, post_id)
                    how = "queued at the next automatic slot (custom time unavailable)"
                except Exception as exc2:  # noqa: BLE001
                    logger.error("Could not schedule draft %s: %s", post_id, exc2)
                    report.append(f"| `{preview}…` | {age:.0f}h | ❌ error: {exc2} |")
                    continue
            scheduled += 1
            slots -= 1
            taken.append(decision.due_at)
            queue_texts.append(draft.get("text") or "")
            print(f"✅ {how}: {preview}… (Buffer says due {post.get('dueAt') or 'auto'})")
            report.append(f"| `{preview}…` | {age:.0f}h | ✅ {how} — {decision.reason} |")

    print("\n--- Draft Review Summary ---")
    print(f"Scheduled: {scheduled}{' (dry run)' if dry_run else ''}\nHeld     : {held}\nDeleted  : {deleted}\nIn review window: {young}")
    print("----------------------------")
    _summary(report)


if __name__ == "__main__":
    main()
