"""CLI and GitHub Actions runner to schedule/publish generated ad creatives to Buffer.

Usage:
    # Publish today's top scoring ad from data/site_manifests/
    python scripts/publish_buffer.py

    # Publish a specific prompt ID:
    python scripts/publish_buffer.py --prompt-id 17

    # Send as a draft to Buffer for mobile review:
    python scripts/publish_buffer.py --draft

    # Post immediately to social channels (skip queue):
    python scripts/publish_buffer.py --now
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.delivery.buffer_publisher import BufferPublisher  # noqa: E402
from src.delivery.buffer_state import fetch_queue_state, find_duplicate  # noqa: E402
from src.delivery.slot_planner import describe, plan_slot  # noqa: E402
from src.generation.claims import unsupported_claims  # noqa: E402
from src.shared.brand_facts import grounding_text, load_brand_facts  # noqa: E402
from src.shared.config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("publish_buffer")


def get_latest_prompts() -> list[dict]:
    manifest_dir = ROOT / "data" / "site_manifests"
    all_entries = []

    for path in manifest_dir.glob("*.json"):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            business_name = (data.get("business") or {}).get("name") or ""
            for entry in data.get("entries", []):
                timezone = (data.get("business") or {}).get("timezone") or ""
                all_entries.append((entry.get("date", ""), {**entry, "_business_name": business_name, "_timezone": timezone}))
        except Exception:
            continue

    if not all_entries:
        return []

    all_entries.sort(key=lambda x: x[0], reverse=True)
    latest_date = all_entries[0][0]
    return [entry for (date, entry) in all_entries if date == latest_date]


PUBLISHED_LOG = ROOT / "data" / "published_log.json"
MAX_LOG_ENTRIES = 300
DEFAULT_QUEUE_LIMIT = 3  # keep few unreviewed posts in the live queue (Buffer's free plan holds 10)
DEFAULT_MIN_HOURS_BETWEEN_POSTS = 20  # at most one auto-queued post per day, whatever triggers the run


def posted_ids() -> set[str]:
    try:
        data = json.loads(PUBLISHED_LOG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {str(item.get("id")) for item in data.get("posted", []) if isinstance(item, dict) and item.get("id")}


def choose_due(prompt: dict, taken: list | None = None, now: datetime | None = None):
    """The publish time for this creative: its experiment/measured hour in the business timezone, clear of the
    times Buffer already holds. None -> let Buffer pick the next automatic slot."""
    hour = prompt.get("schedule_hour")
    if hour is None:
        hour = (prompt.get("post_time_evidence") or {}).get("hour")
    return plan_slot(now or datetime.now(UTC), hour, prompt.get("_timezone") or None, taken or [])


def hours_since_last_post(now: datetime | None = None) -> float | None:
    """Hours since the newest entry in the published log, or None when nothing was ever posted."""
    try:
        data = json.loads(PUBLISHED_LOG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    newest: datetime | None = None
    for item in data.get("posted", []):
        try:
            at = datetime.fromisoformat(str(item.get("at")))
        except (TypeError, ValueError):
            continue
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        newest = at if newest is None or at > newest else newest
    if newest is None:
        return None
    return ((now or datetime.now(UTC)) - newest).total_seconds() / 3600


def record_posted(entry: dict, buffer_posts: list[dict] | None = None, scheduled_for: str | None = None) -> None:
    """Remember the creative and exact Buffer IDs for later per-post analytics."""
    try:
        data = json.loads(PUBLISHED_LOG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    posted = [item for item in data.get("posted", []) if isinstance(item, dict)]
    stable_id = str(entry.get("id") or "")
    record = next((item for item in posted if str(item.get("id") or "") == stable_id), None)
    if record is None:
        record = {"id": stable_id}
        posted.append(record)
    record.update({
        "title": entry.get("title"),
        "prompt_id": entry.get("prompt_id"),
        "business_name": entry.get("_business_name") or entry.get("business_name") or "",
        "prompt_type": entry.get("prompt_type") or "",
        "format": entry.get("format") or "",
        "hook_type": entry.get("hook_type") or "",
        "audience_segment": entry.get("audience_segment") or "",
        "caption": entry.get("caption") or "",
        "at": datetime.now(UTC).isoformat(),
        # what the later (unattended) scheduling step needs to judge and time this creative
        "score": entry.get("score"),
        "threshold": entry.get("threshold"),
        "below_threshold": bool(entry.get("below_threshold")),
        "window": entry.get("post_time_evidence") or {},
        "schedule_hour": entry.get("schedule_hour"),
        "timezone": entry.get("_timezone") or "",
        "experiment": entry.get("experiment") or {},
        "angle": (entry.get("strategy") or {}).get("angle") or "",
        "scheduled_for": scheduled_for or "",
    })
    record["buffer_posts"] = [
        {
            "id": str(post.get("id")),
            "channel_id": str(post.get("channelId") or ""),
            "status": str(post.get("status") or ""),
        }
        for post in (buffer_posts or [])
        if post.get("id")
    ]
    PUBLISHED_LOG.parent.mkdir(parents=True, exist_ok=True)
    PUBLISHED_LOG.write_text(json.dumps({"posted": posted[-MAX_LOG_ENTRIES:]}, indent=2), encoding="utf-8")


def is_degraded(entry: dict) -> bool:
    """True when provider failure forced a branded card or procedural Reel artwork."""
    return any(
        marker in str(w).lower()
        for w in (entry.get("warnings") or [])
        for marker in ("branded card", "locally illustrated reel scene")
    )


def is_verified_minute_reel(entry: dict) -> bool:
    asset = entry.get("asset") or {}
    asset_type = str(asset.get("type") or entry.get("asset_type") or "").lower()
    try:
        duration = float(entry.get("video_duration_seconds") or 0)
    except (TypeError, ValueError):
        return False
    return asset_type in {"reel", "video"} and duration >= 60


def auto_publish_block_reason(prompt: dict) -> str | None:
    """Fail closed on missing, malformed, or known-degraded assets before Buffer scheduling."""
    asset = prompt.get("asset") or {}
    asset_type = str(prompt.get("asset_type") or asset.get("type") or "").lower()
    prompt_asset_path = prompt.get("asset_path")
    is_video = asset_type in {"reel", "video"} or str(prompt_asset_path or "").lower().endswith(".mp4")
    video_path = asset.get("video") or (prompt_asset_path if is_video else None)
    image_path = (prompt_asset_path if not video_path else None) or asset.get("image") or asset.get("thumb")
    asset_path = video_path or image_path
    if not asset_path:
        return "no rendered media asset"

    path_lower = str(asset_path).lower()
    is_video = bool(video_path) or asset_type in {"reel", "video"} or path_lower.endswith(".mp4")
    if is_video and not path_lower.endswith(".mp4"):
        return "Buffer Reel publishing requires a rendered MP4 asset"

    warnings = prompt.get("warnings") or []
    if isinstance(warnings, str):
        warnings = [warnings]
    warning_text = " ".join(str(item).lower() for item in warnings)
    for fragment in (
        "all ai image providers failed",
        "used a branded card background",
        "locally illustrated reel scene",
        "reel render failed",
        "falling back to single image",
        "asset render failed",
        "watermark",
    ):
        if fragment in warning_text:
            return f"render QA failed: {fragment}"
    return None


def pick_prompt(
    prompts: list[dict],
    posted: set[str] | frozenset[str] = frozenset(),
    grounding: str = "",
    require_publishable_asset: bool = False,
    reel_only: bool = False,
) -> dict | None:
    """Highest-scoring creative that has a real AI image and no unverified statistics.

    Today's manifest also holds creatives from earlier runs (made before the claims guard, or before the
    image fix), so the publish step re-checks instead of trusting that everything in it is clean.
    None when nothing usable is left.
    """
    usable = []
    for p in prompts:
        if is_degraded(p) or str(p.get("id")) in posted:
            continue
        if reel_only and not is_verified_minute_reel(p):
            continue
        if require_publishable_asset and auto_publish_block_reason(p):
            continue
        claims = unsupported_claims(p, grounding)
        if claims:
            logger.warning("Skipping %r: unverified claims %s", p.get("title"), claims)
            continue
        usable.append(p)
    if not usable:
        return None
    # the creative carrying this week's designed experiment is preferred: it was only tagged because it already
    # cleared the quality bar, and the test needs it published
    return max(usable, key=lambda x: (bool(x.get("experiment")), int(x.get("score") or x.get("virality_score") or 0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish generated creative to Buffer.")
    parser.add_argument("--prompt-id", type=str, help="Specific prompt ID to publish")
    parser.add_argument("--draft", action="store_true", help="Create as a Buffer draft instead of scheduling directly")
    parser.add_argument("--publish-live", action="store_true", help="Publish directly to live queue, overriding default draft mode")
    parser.add_argument("--now", action="store_true", help="Post immediately to social media")
    parser.add_argument("--token", type=str, help="Override Buffer Access Token")
    parser.add_argument("--profile-id", type=str, help="Override Buffer Profile ID")
    parser.add_argument("--reel-only", action="store_true", help="Publish only a verified Reel with at least 60 seconds of narration")
    parser.add_argument("--replace-post-id", help="Replace the media and caption on this existing Buffer post, preserving its schedule")

    args = parser.parse_args()

    token = args.token or getattr(settings, "buffer_access_token", None) or os.getenv("BUFFER_ACCESS_TOKEN")
    profile_id = args.profile_id or getattr(settings, "buffer_profile_ids", None) or os.getenv("BUFFER_PROFILE_IDS") or os.getenv("BUFFER_PROFILE_ID")

    if not token or not profile_id:
        print("⚠️ Buffer credentials not found.")
        print("Required:")
        print("  BUFFER_ACCESS_TOKEN in env / secrets")
        print("  BUFFER_PROFILE_ID   in env / secrets")
        print("Skipping Buffer publication.")
        sys.exit(0)

    draft_env = os.getenv("BUFFER_DRAFT_MODE", "true").lower()
    is_draft = not (args.now or args.publish_live) and (args.draft or draft_env not in ("false", "0", "no"))
    is_now = args.now or os.getenv("BUFFER_NOW", "").lower() in ("true", "1", "yes")

    publisher = BufferPublisher(
        access_token=token,
        profile_ids=[profile_id],
        draft_mode=is_draft,
        now=is_now,
    )

    prompts = get_latest_prompts()
    if not prompts:
        print("No prompts found in data/site_manifests/.")
        sys.exit(0)

    target_prompt = None
    if args.prompt_id:
        for p in prompts:
            if str(p.get("prompt_id")) == str(args.prompt_id) or str(p.get("id")) == str(args.prompt_id):
                target_prompt = p
                break
        if not target_prompt:
            print(f"Error: Prompt ID '{args.prompt_id}' not found in recent prompts.")
            sys.exit(1)
    else:
        target_prompt = pick_prompt(
            prompts,
            posted_ids(),
            grounding_text(load_brand_facts()),
            require_publishable_asset=True,
            reel_only=args.reel_only or os.getenv("BUFFER_REEL_ONLY", "").lower() in ("1", "true", "yes"),
        )
        if target_prompt is None:
            print("::warning::No creative today meets the publishing requirements (fallback media, unverified claims, "
                  "or no verified one-minute Reel); nothing was posted.")
            sys.exit(0)

    block_reason = auto_publish_block_reason(target_prompt)
    if not block_reason and (args.reel_only or os.getenv("BUFFER_REEL_ONLY", "").lower() in ("1", "true", "yes")):
        if not is_verified_minute_reel(target_prompt):
            block_reason = "Reel is missing verified narration duration of at least 60 seconds"
    if block_reason:
        print(f"Automated render QA blocked publication: {block_reason}. Skipping Buffer publication.")
        sys.exit(0)

    title = target_prompt.get("title", "Ad")
    score = target_prompt.get("score") or target_prompt.get("virality_score") or 0
    caption = target_prompt.get("caption") or target_prompt.get("prompt_text", "")
    hashtags = target_prompt.get("hashtags", [])
    if hashtags:
        caption = f"{caption.strip()}\n\n{' '.join(hashtags)}"

    asset_info = target_prompt.get("asset") or {}
    asset_type = str(target_prompt.get("asset_type") or asset_info.get("type") or "").lower()
    prompt_asset_path = target_prompt.get("asset_path")
    is_video = asset_type in {"reel", "video"} or str(prompt_asset_path or "").lower().endswith(".mp4")
    video_path = asset_info.get("video") or (prompt_asset_path if is_video else None)
    image_path = (prompt_asset_path if not video_path else None) or asset_info.get("image") or asset_info.get("thumb")

    def public_url(asset_path: str | None) -> str | None:
        if not asset_path:
            return None
        if "assets/" in asset_path:
            asset_path = asset_path[asset_path.index("assets/"):]
        return publisher.get_public_asset_url(asset_path)

    video_url = public_url(video_path)
    image_url = public_url(image_path) if not video_url else None

    print("\n--- Publishing to Buffer ---")
    print(f"Title     : {title} (Virality Score: {score})")
    print(f"Media URL : {video_url or image_url}")
    print(f"Mode      : {'Immediate Post' if args.now else ('Draft' if args.draft else 'Scheduled Queue')}")
    print(f"Profiles  : {publisher.profile_ids}")
    print("----------------------------\n")

    # Daily cap: repeated manual runs must not stack several unreviewed posts into the live queue.
    if not (args.prompt_id or is_now or is_draft or args.replace_post_id):
        min_hours = float(os.getenv("BUFFER_MIN_HOURS") or DEFAULT_MIN_HOURS_BETWEEN_POSTS)
        since = hours_since_last_post()
        if since is not None and since < min_hours:
            print(f"::warning::Not posting: the last post was {since:.1f}h ago (minimum {min_hours:g}h between auto-queued posts).")
            sys.exit(0)

    # Other projects feed the same Buffer channel: do not repeat what it already holds or crowd its queue.
    queue_limit = int(os.getenv("BUFFER_QUEUE_LIMIT") or DEFAULT_QUEUE_LIMIT)
    try:
        queue = fetch_queue_state(token, str(publisher.profile_ids[0]))
    except Exception as exc:  # noqa: BLE001 - the check is a safeguard, not a reason to lose the day's post
        print(f"::warning::Could not read the Buffer queue ({exc}); posting without the duplicate and queue check.")
        queue = None
    if queue is not None and not args.replace_post_id:
        repeated = find_duplicate(caption, queue.texts())
        if repeated:
            print(f"::warning::Not posting: Buffer already holds a near-identical post ({repeated[:70]!r}).")
            sys.exit(0)
        if not (is_draft or is_now) and queue.scheduled >= queue_limit:
            print(f"::warning::Not posting: Buffer already has {queue.scheduled} scheduled posts (limit {queue_limit}).")
            sys.exit(0)

    try:
        if args.replace_post_id:
            if not video_url or not is_verified_minute_reel(target_prompt):
                raise ValueError("Replacing an existing Buffer post requires a verified Reel of at least 60 seconds.")
            res = publisher.replace_scheduled_reel(args.replace_post_id, caption, video_url)
            post = res.get("post", {})
            print("✅ Replaced the existing Buffer Reel; its schedule was preserved.")
            record_posted(target_prompt, [post])
            print(f"Post ID   : {post.get('id')}")
            print(f"Status    : {post.get('status')}")
            print(f"Due At    : {post.get('dueAt') or 'Existing Buffer queue slot'}")
        else:
            due_at = None
            if not (is_draft or is_now):
                due_at = choose_due(target_prompt, queue.due_times() if queue is not None else [])
            try:
                res = publisher.schedule_post(text=caption, image_url=image_url, video_url=video_url, due_at=due_at)
            except Exception as exc:  # noqa: BLE001 - a custom time must never cost the day's post
                if due_at is None:
                    raise
                print(f"::warning::Custom schedule rejected ({exc}); falling back to the next automatic queue slot.")
                due_at = None
                res = publisher.schedule_post(text=caption, image_url=image_url, video_url=video_url)
            print("✅ Successfully published to Buffer!")
            if due_at is not None:
                print(f"Planned   : {describe(due_at, target_prompt.get('_timezone') or None)}")
            buffer_posts = res.get("posts", [])
            record_posted(target_prompt, buffer_posts, due_at.isoformat() if due_at else None)
            for post in buffer_posts:
                print(f"Post ID   : {post.get('id')}")
                print(f"Status    : {post.get('status')}")
                print(f"Due At    : {post.get('dueAt') or 'Auto Queue'}")
    except Exception as exc:
        print(f"❌ Failed to publish to Buffer: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
