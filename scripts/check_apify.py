"""Verify APIFY_TOKEN works: calls both actors exactly like the pipeline does.

Usage:
    set APIFY_TOKEN=your_token   (Windows)
    python scripts/check_apify.py [query]

Prints result counts per actor. Exit code 1 if any actor fails.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import httpx

from src.fetch.social_scraper import SocialScraper


async def main() -> None:
    token = SocialScraper().apify_token
    query = sys.argv[1] if len(sys.argv) > 1 else "bakery Ahmedabad"
    handle = sys.argv[2] if len(sys.argv) > 2 else "vickybakery"

    if not token:
        print("APIFY_TOKEN not set (env or .env)")
        raise SystemExit(1)

    print(f"Token: {token[:8]}... (len={len(token)})")
    timeout = SocialScraper.APIFY_TIMEOUT

    async with httpx.AsyncClient(timeout=timeout) as client:
        t0 = asyncio.get_event_loop().time()
        resp = await client.post(
            f"{SocialScraper.APIFY_BASE}/acts/apify/instagram-search-scraper/run-sync-get-dataset-items",
            params={"token": token},
            json={"searchType": "user", "search": [query], "resultsLimit": 5},
        )
        dt = asyncio.get_event_loop().time() - t0
        print(f"\n[search actor] {resp.status_code} in {dt:.0f}s")
        if resp.status_code >= 400:
            print(resp.text[:500])
            raise SystemExit(1)
        items = resp.json()
        print(f"results: {len(items)}")
        for it in items[:3]:
            print(f"  @{it.get('username')} — {it.get('followersCount')} followers — {(it.get('fullName') or '')[:40]}")

        t0 = asyncio.get_event_loop().time()
        resp = await client.post(
            f"{SocialScraper.APIFY_BASE}/acts/{SocialScraper.APIFY_INSTAGRAM_ACTOR}/run-sync-get-dataset-items",
            params={"token": token},
            json={
                "username": [handle],
                "resultsLimit": 5,
                "shouldDownloadVideos": False,
                "shouldDownloadCovers": False,
                "shouldDownloadSubtitles": False,
                "shouldDownloadCaptions": False,
            },
        )
        dt = asyncio.get_event_loop().time() - t0
        print(f"\n[profile actor @{handle}] {resp.status_code} in {dt:.0f}s")
        if resp.status_code >= 400:
            print(resp.text[:500])
            raise SystemExit(1)
        items = resp.json()
        print(f"results: {len(items)}")
        for it in items[:1]:
            print(f"  @{it.get('username')} — {it.get('followersCount')} followers — {it.get('postsCount')} posts")


if __name__ == "__main__":
    asyncio.run(main())
