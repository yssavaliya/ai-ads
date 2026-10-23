"""Diagnose why the Instagram scrape of one handle returns nothing.

Calls the Instagram actor with the input shapes the actor family accepts and prints, per shape, the HTTP
status, how many items came back, which fields the first item has and any `error` the actor reports.
The token is only sent as a header and never printed.

    python scripts/probe_apify_instagram.py aimagix.studio

Exit code 1 when no shape returned a post.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from src.fetch.social_scraper import SocialScraper


def variants(handle: str) -> dict[str, dict[str, Any]]:
    url = f"https://www.instagram.com/{handle}/"
    return {
        "A usernames (what the pipeline sends today)": {"usernames": [handle], "resultsLimit": 5, "commentsLimit": 3},
        "B directUrls + posts": {"directUrls": [url], "resultsType": "posts", "resultsLimit": 5, "addParentData": True},
        "C directUrls + details": {"directUrls": [url], "resultsType": "details", "resultsLimit": 1},
    }


def describe(items: Any) -> dict[str, Any]:
    if not isinstance(items, list):
        return {"items": "not a list", "type": type(items).__name__}
    first = items[0] if items and isinstance(items[0], dict) else {}
    normalized = SocialScraper._normalize_instagram(items)
    return {
        "items": len(items),
        "first_item_keys": sorted(first)[:14],
        "actor_error": first.get("error") or first.get("errorDescription"),
        "posts_after_normalizing": len(normalized["posts"]),
        "followers": normalized["follower_count"],
        "normalizer_error": normalized.get("error"),
    }


async def main() -> int:
    handle = sys.argv[1] if len(sys.argv) > 1 else "aimagix.studio"
    scraper = SocialScraper()
    if not scraper.apify_token:
        print("APIFY_TOKEN is not set")
        return 1
    found = False
    async with httpx.AsyncClient(timeout=SocialScraper.APIFY_TIMEOUT, headers=scraper._auth_headers()) as client:
        for label, body in variants(handle).items():
            try:
                resp = await client.post(f"{SocialScraper.APIFY_BASE}/acts/{SocialScraper.APIFY_INSTAGRAM_ACTOR}/run-sync-get-dataset-items", json=body)
                report: dict[str, Any] = {"http": resp.status_code}
                if resp.status_code < 400:
                    report.update(describe(resp.json()))
                    found = found or report.get("posts_after_normalizing", 0) > 0
                else:
                    report["body"] = resp.text[:300]
            except Exception as exc:  # noqa: BLE001 - a probe must report, not crash
                report = {"exception": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"[{label}]\n  {json.dumps(report, default=str)}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
