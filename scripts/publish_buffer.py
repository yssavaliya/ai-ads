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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.delivery.buffer_publisher import BufferPublisher  # noqa: E402
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

    publisher = BufferPublisher(
        access_token=token,
        profile_ids=[profile_id],
        draft_mode=args.draft,
        now=args.now,
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
        prompts.sort(key=lambda x: int(x.get("score") or x.get("virality_score") or 0), reverse=True)
        target_prompt = prompts[0]

    title = target_prompt.get("title", "Ad")
    score = target_prompt.get("score") or target_prompt.get("virality_score") or 0
    caption = target_prompt.get("caption") or target_prompt.get("prompt_text", "")
    hashtags = target_prompt.get("hashtags", [])
    if hashtags:
        caption = f"{caption.strip()}\n\n{' '.join(hashtags)}"

    asset_info = target_prompt.get("asset") or {}
    asset_path = target_prompt.get("asset_path") or asset_info.get("image") or asset_info.get("thumb")

    image_url = None
    if asset_path:
        if "assets/" in asset_path:
            rel = asset_path[asset_path.index("assets/"):]
            image_url = publisher.get_public_asset_url(rel)
        else:
            image_url = publisher.get_public_asset_url(asset_path)

    print("\n--- Publishing to Buffer ---")
    print(f"Title     : {title} (Virality Score: {score})")
    print(f"Image URL : {image_url}")
    print(f"Mode      : {'Immediate Post' if args.now else ('Draft' if args.draft else 'Scheduled Queue')}")
    print(f"Profiles  : {publisher.profile_ids}")
    print("----------------------------\n")

    try:
        res = publisher.schedule_post(text=caption, image_url=image_url)
        print("✅ Successfully published to Buffer!")
        for post in res.get("posts", []):
            print(f"Post ID   : {post.get('id')}")
            print(f"Status    : {post.get('status')}")
            print(f"Due At    : {post.get('dueAt') or 'Auto Queue'}")
    except Exception as exc:
        print(f"❌ Failed to publish to Buffer: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
