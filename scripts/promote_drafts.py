#!/usr/bin/env python3
"""Auto-promote Buffer drafts to the active queue, or clean up stale ones.

Runs as a scheduled GitHub Actions job so the human-review gate does not
become a dead-letter queue.

Logic:
  1. Fetch all drafts for the configured Buffer channel.
  2. For each draft:
     a. If it is older than MAX_DRAFT_AGE_HOURS, check its media URL.
        - If the media is still reachable  → promote it to the queue.
        - If the media is broken (404/expired) → delete the stale draft.
     b. If it is younger than MIN_DRAFT_AGE_HOURS, leave it alone
        (give the human a window to review first).
  3. Respect the queue limit: do not promote if the scheduled queue is
     already full.

Environment variables:
  BUFFER_ACCESS_TOKEN      – required
  BUFFER_PROFILE_IDS       – required (comma-separated channel IDs)
  DRAFT_MIN_AGE_HOURS      – minimum age before auto-promoting (default: 24)
  DRAFT_MAX_AGE_HOURS      – maximum age before the draft is stale (default: 168 = 7 days)
  BUFFER_QUEUE_LIMIT       – max scheduled posts in the live queue (default: 3)
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime, timedelta

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.delivery.buffer_state import (  # noqa: E402
    check_asset_url,
    delete_post,
    fetch_drafts,
    fetch_queue_state,
    promote_draft,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    token = os.getenv("BUFFER_ACCESS_TOKEN")
    profile_ids_raw = os.getenv("BUFFER_PROFILE_IDS") or os.getenv("BUFFER_PROFILE_ID") or ""
    profile_ids = [pid.strip() for pid in profile_ids_raw.split(",") if pid.strip()]

    if not token or not profile_ids:
        print("⚠️  BUFFER_ACCESS_TOKEN and BUFFER_PROFILE_IDS required. Skipping.")
        sys.exit(0)

    min_age_hours = float(os.getenv("DRAFT_MIN_AGE_HOURS") or "24")
    max_age_hours = float(os.getenv("DRAFT_MAX_AGE_HOURS") or "168")  # 7 days
    queue_limit = int(os.getenv("BUFFER_QUEUE_LIMIT") or "3")

    now = datetime.now(UTC)
    promoted = 0
    deleted = 0
    skipped_young = 0

    for channel_id in profile_ids:
        # Check current queue capacity
        try:
            queue = fetch_queue_state(token, channel_id)
            queue_slots = max(0, queue_limit - queue.scheduled)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read queue for %s (%s); assuming 0 slots", channel_id, exc)
            queue_slots = 0

        drafts = fetch_drafts(token, channel_id)
        logger.info("Channel %s: found %d drafts, %d queue slots available", channel_id, len(drafts), queue_slots)

        for draft in drafts:
            post_id = draft.get("id")
            text_preview = (draft.get("text") or "")[:60]
            created_at_str = draft.get("createdAt") or ""

            # Parse draft age
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                logger.warning("  Draft %s: cannot parse createdAt %r — skipping", post_id, created_at_str)
                continue

            age = now - created_at
            age_hours = age.total_seconds() / 3600

            # Too young — give the human time to review
            if age_hours < min_age_hours:
                skipped_young += 1
                logger.info("  Draft %s (%.0fh old): too young for auto-action — skipping [%s…]", post_id, age_hours, text_preview)
                continue

            # Check whether the media asset is still reachable
            assets = draft.get("assets") or []
            asset_urls = [a.get("url") for a in assets if a.get("url")]
            media_ok = all(check_asset_url(url) for url in asset_urls) if asset_urls else True  # text-only posts are fine

            # Stale: media broken → delete
            if not media_ok:
                logger.warning("  Draft %s (%.0fh old): media expired/broken — deleting [%s…]", post_id, age_hours, text_preview)
                try:
                    delete_post(token, post_id)
                    deleted += 1
                    print(f"🗑️  Deleted stale draft {post_id}: {text_preview}…")
                except Exception as exc:  # noqa: BLE001
                    logger.error("  Could not delete draft %s: %s", post_id, exc)
                continue

            # Too old even with good media — also delete (content is no longer timely)
            if age_hours > max_age_hours:
                logger.warning("  Draft %s (%.0fh old): exceeded max age — deleting [%s…]", post_id, age_hours, text_preview)
                try:
                    delete_post(token, post_id)
                    deleted += 1
                    print(f"🗑️  Deleted expired draft {post_id}: {text_preview}…")
                except Exception as exc:  # noqa: BLE001
                    logger.error("  Could not delete draft %s: %s", post_id, exc)
                continue

            # Within the review window and media is good — promote if queue has room
            if queue_slots <= 0:
                logger.info("  Draft %s (%.0fh old): queue full — leaving in drafts [%s…]", post_id, age_hours, text_preview)
                continue

            logger.info("  Draft %s (%.0fh old): promoting to queue [%s…]", post_id, age_hours, text_preview)
            try:
                result = promote_draft(token, post_id)
                promoted += 1
                queue_slots -= 1
                print(f"✅ Promoted draft {post_id} → queue (due: {result.get('dueAt', 'auto')}): {text_preview}…")
            except Exception as exc:  # noqa: BLE001
                logger.error("  Could not promote draft %s: %s", post_id, exc)

    print("\n--- Draft Review Summary ---")
    print(f"Promoted : {promoted}")
    print(f"Deleted  : {deleted} (stale/expired)")
    print(f"Skipped  : {skipped_young} (too young, still in review window)")
    print("----------------------------")


if __name__ == "__main__":
    main()
