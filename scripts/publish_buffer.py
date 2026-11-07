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
            for entry in data.get("entries", []):
                all_entries.append((entry.get("date", ""), entry))
        except Exception:
            continue

    if not all_entries:
        return []

    all_entries.sort(key=lambda x: x[0], reverse=True)
    latest_date = all_entries[0][0]
    return [entry for (date, entry) in all_entries if date == latest_date]


PUBLISHED_LOG = ROOT / "data" / "published_log.json"
MAX_LOG_ENTRIES = 300
DEFAULT_QUEUE_LIMIT = 8  # Buffer's free plan holds 10 scheduled posts; leave room for the other project


def posted_ids() -> set[str]:
    try:
        data = json.loads(PUBLISHED_LOG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {str(item.get("id")) for item in data.get("posted", []) if isinstance(item, dict) and item.get("id")}


def record_posted(entry: dict) -> None:
    """Remember what was sent so a re-run the same day cannot post the same creative again."""
    try:
        data = json.loads(PUBLISHED_LOG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    posted = [item for item in data.get("posted", []) if isinstance(item, dict)]
    posted.append({"id": str(entry.get("id")), "title": entry.get("title"), "at": datetime.now(UTC).isoformat()})
    PUBLISHED_LOG.parent.mkdir(parents=True, exist_ok=True)
    PUBLISHED_LOG.write_text(json.dumps({"posted": posted[-MAX_LOG_ENTRIES:]}, indent=2), encoding="utf-8")


def is_degraded(entry: dict) -> bool:
    """True when the creative shipped with the flat fallback card because every AI image provider failed."""
    return any("branded card" in str(w).lower() for w in (entry.get("warnings") or []))


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
        "reel render failed",
        "falling back to single image",
        "asset render failed",
        "watermark",
    ):
        if fragment in warning_text:
            return f"render QA failed: {fragment}"
    return None


def pick_prompt(prompts: list[dict], posted: set[str] | frozenset[str] = frozenset(), grounding: str = "") -> dict | None:
    """Highest-scoring creative that has a real AI image and no unverified statistics.

    Today's manifest also holds creatives from earlier runs (made before the claims guard, or before the
    image fix), so the publish step re-checks instead of trusting that everything in it is clean.
    None when nothing usable is left.
    """
    usable = []
    for p in prompts:
        if is_degraded(p) or auto_publish_block_reason(p) or str(p.get("id")) in posted:
            continue
        claims = unsupported_claims(p, grounding)
        if claims:
            logger.warning("Skipping %r: unverified claims %s", p.get("title"), claims)
            continue
        usable.append(p)
    if not usable:
        return None
    return max(usable, key=lambda x: int(x.get("score") or x.get("virality_score") or 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish generated creative to Buffer.")
    parser.add_argument("--prompt-id", type=str, help="Specific prompt ID to publish")
    parser.add_argument("--draft", action="store_true", help="Create as a Buffer draft instead of scheduling directly")
    parser.add_argument("--now", action="store_true", help="Post immediately to social media")
    parser.add_argument("--token", type=str, help="Override Buffer Access Token")
    parser.add_argument("--profile-id", type=str, help="Override Buffer Profile ID")

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

    is_draft = args.draft or os.getenv("BUFFER_DRAFT_MODE", "").lower() in ("true", "1", "yes")
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
        target_prompt = pick_prompt(prompts, posted_ids(), grounding_text(load_brand_facts()))
        if target_prompt is None:
            print("::warning::No creative today is safe to post (flat fallback card, or an unverified statistic in the "
                  "copy); nothing was posted.")
            sys.exit(0)

    block_reason = auto_publish_block_reason(target_prompt)
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

    # Other projects feed the same Buffer channel: do not repeat what it already holds or crowd its queue.
    queue_limit = int(os.getenv("BUFFER_QUEUE_LIMIT") or DEFAULT_QUEUE_LIMIT)
    try:
        queue = fetch_queue_state(token, str(publisher.profile_ids[0]))
    except Exception as exc:  # noqa: BLE001 - the check is a safeguard, not a reason to lose the day's post
        print(f"::warning::Could not read the Buffer queue ({exc}); posting without the duplicate and queue check.")
        queue = None
    if queue is not None:
        repeated = find_duplicate(caption, queue.texts())
        if repeated:
            print(f"::warning::Not posting: Buffer already holds a near-identical post ({repeated[:70]!r}).")
            sys.exit(0)
        if not (is_draft or is_now) and queue.scheduled >= queue_limit:
            print(f"::warning::Not posting: Buffer already has {queue.scheduled} scheduled posts (limit {queue_limit}).")
            sys.exit(0)

    try:
        res = publisher.schedule_post(text=caption, image_url=image_url, video_url=video_url)
        print("✅ Successfully published to Buffer!")
        record_posted(target_prompt)
        for post in res.get("posts", []):
            print(f"Post ID   : {post.get('id')}")
            print(f"Status    : {post.get('status')}")
            print(f"Due At    : {post.get('dueAt') or 'Auto Queue'}")
    except Exception as exc:
        print(f"❌ Failed to publish to Buffer: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
