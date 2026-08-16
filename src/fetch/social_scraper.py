from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from src.shared.config import settings

logger = logging.getLogger(__name__)


class SocialScraper:
    """Fetches public Instagram/LinkedIn profile + recent post data.
    Primary: Apify free tier (REST API). Fallback: Playwright for public pages.
    Never requires local CPU models; runs entirely in GitHub Actions."""

    APIFY_INSTAGRAM_ACTOR = "apify/instagram-scraper"
    APIFY_LINKEDIN_ACTOR = "curious_coder/linkedin-profile-scraper"
    APIFY_BASE = "https://api.apify.com/v2"
    POST_LIMIT = 20

    def __init__(self, apify_token: str | None = None) -> None:
        self.apify_token = apify_token or settings.apify_token

    async def fetch_instagram(self, handle: str, limit: int = POST_LIMIT) -> dict[str, Any]:
        handle = handle.lstrip("@")
        if not handle:
            return {"handle": handle, "error": "empty handle", "posts": []}

        if self.apify_token:
            try:
                return await self._apify_instagram(handle, limit)
            except Exception as exc:
                logger.warning("Apify Instagram failed for %s: %s; falling back to Playwright", handle, exc)
        else:
            logger.info("APIFY_TOKEN not set; falling back to Playwright for @%s", handle)

        try:
            return await self._playwright_instagram(handle)
        except Exception as exc:
            logger.warning("Playwright Instagram failed for %s: %s", handle, exc)
            return {"handle": handle, "error": str(exc), "posts": []}

    async def fetch_linkedin(self, handle: str, limit: int = POST_LIMIT) -> dict[str, Any]:
        handle = handle.strip()
        if not handle:
            return {"handle": handle, "error": "empty handle", "posts": []}

        if not self.apify_token:
            logger.info("APIFY_TOKEN not set; LinkedIn public scraping unavailable")
            return {"handle": handle, "error": "no apify token", "posts": []}

        try:
            return await self._apify_linkedin(handle, limit)
        except Exception as exc:
            logger.warning("Apify LinkedIn failed for %s: %s", handle, exc)
            return {"handle": handle, "error": str(exc), "posts": []}

    async def _apify_instagram(self, handle: str, limit: int) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120) as client:
            run_resp = await client.post(
                f"{self.APIFY_BASE}/acts/{self.APIFY_INSTAGRAM_ACTOR}/run-sync-get-dataset-items",
                params={"token": self.apify_token},
                json={
                    "username": [handle],
                    "resultsLimit": limit,
                    "shouldDownloadVideos": False,
                    "shouldDownloadCovers": False,
                    "shouldDownloadSubtitles": False,
                    "shouldDownloadCaptions": False,
                },
            )
            run_resp.raise_for_status()
            items = run_resp.json()
        return self._normalize_instagram(items)

    async def _apify_linkedin(self, handle: str, limit: int) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120) as client:
            run_resp = await client.post(
                f"{self.APIFY_BASE}/acts/{self.APIFY_LINKEDIN_ACTOR}/run-sync-get-dataset-items",
                params={"token": self.apify_token},
                json={"handles": [handle], "resultsLimit": limit},
            )
            run_resp.raise_for_status()
            items = run_resp.json()
        return self._normalize_linkedin(items)

    async def _playwright_instagram(self, handle: str) -> dict[str, Any]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError("playwright not installed")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = await browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            url = f"https://www.instagram.com/{handle}/"
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)

            meta_description = ""
            meta = page.locator('meta[name="description"]').first
            if await meta.count():
                meta_description = await meta.get_attribute("content") or ""

            await page.wait_for_timeout(2500)
            body_text = await page.evaluate("() => document.body ? document.body.innerText.slice(0, 4000) : ''")
            await browser.close()

            return {
                "handle": handle,
                "meta_description": meta_description,
                "body_text": body_text,
                "posts": [],
                "source": "playwright",
            }

    @staticmethod
    def _normalize_instagram(items: list[dict[str, Any]]) -> dict[str, Any]:
        posts = []
        for item in items:
            caption = (
                item.get("caption") or ""
                or (item.get("alt") or "")
                or (item.get("description") or "")
            )
            posts.append(
                {
                    "post_id": item.get("id") or item.get("shortCode") or "",
                    "caption": caption,
                    "likes": item.get("likesCount") or 0,
                    "comments": item.get("commentsCount") or 0,
                    "video_views": item.get("videoViewCount") or 0,
                    "url": item.get("url") or "",
                    "posted_at": item.get("timestamp") or item.get("postedAt") or None,
                    "platform": "instagram",
                }
            )
        return {
            "handle": items[0].get("username") if items else "",
            "follower_count": items[0].get("followersCount") if items else None,
            "following_count": items[0].get("followingCount") if items else None,
            "posts_count": items[0].get("postsCount") if items else None,
            "full_name": items[0].get("fullName") if items else None,
            "biography": items[0].get("biography") if items else None,
            "posts": posts[:20],
            "source": "apify",
        }

    @staticmethod
    def _normalize_linkedin(items: list[dict[str, Any]]) -> dict[str, Any]:
        posts = []
        for item in items:
            caption = item.get("text") or item.get("caption") or item.get("summary") or ""
            posts.append(
                {
                    "post_id": item.get("id") or "",
                    "caption": caption,
                    "likes": item.get("likesCount") or 0,
                    "comments": item.get("commentsCount") or 0,
                    "shares": item.get("sharesCount") or 0,
                    "url": item.get("url") or "",
                    "posted_at": item.get("postedAt") or item.get("date") or None,
                    "platform": "linkedin",
                }
            )
        return {
            "handle": items[0].get("handle") if items else "",
            "follower_count": items[0].get("followersCount") if items else None,
            "headline": items[0].get("headline") if items else None,
            "about": items[0].get("about") if items else None,
            "posts": posts[:20],
            "source": "apify",
        }


def fetch_instagram_sync(handle: str) -> dict[str, Any]:
    return asyncio.run(SocialScraper().fetch_instagram(handle))


def fetch_linkedin_sync(handle: str) -> dict[str, Any]:
    return asyncio.run(SocialScraper().fetch_linkedin(handle))