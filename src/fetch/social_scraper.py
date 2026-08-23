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

    async def fetch_instagram_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search Instagram accounts by keyword (Apify free tier)."""
        if not self.apify_token:
            return []
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                run_resp = await client.post(
                    f"{self.APIFY_BASE}/acts/apify/instagram-search-scraper/run-sync-get-dataset-items",
                    params={"token": self.apify_token},
                    json={"searchType": "user", "search": [query], "resultsLimit": limit},
                )
                run_resp.raise_for_status()
                items = run_resp.json()
            results = []
            for item in items:
                results.append(
                    {
                        "username": item.get("username") or "",
                        "fullName": item.get("fullName") or item.get("full_name") or "",
                        "followersCount": item.get("followersCount") or item.get("followers_count") or 0,
                        "externalUrl": item.get("externalUrl") or item.get("external_url") or "",
                        "biography": item.get("biography") or "",
                    }
                )
            return results
        except Exception as exc:
            logger.warning("Instagram search failed for %r: %s", query, exc)
            return []

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
        """Fallback when APIFY_TOKEN is unset: instaloader (open-source,
        MIT-licensed) fetches public profile metadata + recent post metadata.
        Loudly degraded output vs Apify; raises on failure so callers log it."""
        try:
            import asyncio as _asyncio
            from concurrent.futures import ThreadPoolExecutor

            def _fetch() -> dict[str, Any]:
                import instaloader

                L = instaloader.Instaloader(
                    download_comments=False,
                    download_geotags=False,
                    download_pictures=False,
                    download_videos=False,
                    download_video_thumbnails=False,
                    save_metadata=False,
                    compress_json=False,
                    quiet=True,
                )
                profile = instaloader.Profile.from_username(L.context, handle)
                posts = []
                for i, post in enumerate(profile.get_posts()):
                    if i >= 12:
                        break
                    posts.append(
                        {
                            "post_id": post.shortcode,
                            "caption": post.caption or "",
                            "likes": post.likes or 0,
                            "comments": post.comments or 0,
                            "url": f"https://www.instagram.com/p/{post.shortcode}/",
                            "posted_at": post.date_utc.isoformat() if post.date_utc else None,
                            "platform": "instagram",
                        }
                    )
                return {
                    "handle": handle,
                    "follower_count": profile.followers,
                    "following_count": profile.followees,
                    "posts_count": profile.mediacount,
                    "full_name": profile.full_name,
                    "biography": profile.biography,
                    "posts": posts,
                    "source": "instaloader",
                }

            loop = _asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                return await loop.run_in_executor(pool, _fetch)
        except ImportError as exc:
            raise RuntimeError("instaloader not installed") from exc

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