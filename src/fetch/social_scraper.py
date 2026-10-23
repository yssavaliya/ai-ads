from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from src.shared.config import settings

logger = logging.getLogger(__name__)


class SocialScraper:
    """Fetches public Instagram/LinkedIn profile + recent post data.
    Primary: Apify free tier (REST API). Fallback: instaloader for public profiles.
    Never requires local CPU models; runs entirely in GitHub Actions."""

    # Actor IDs in Apify API URL paths MUST use '~' between owner and name
    # ('/' in the path 404s with "page-not-found" — verified empirically).
    APIFY_INSTAGRAM_ACTOR = "apify~instagram-scraper"
    APIFY_SEARCH_ACTOR = "apify~instagram-search-scraper"
    APIFY_LINKEDIN_ACTOR = "curious_coder~linkedin-profile-scraper"
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
                logger.warning("Apify Instagram failed for %s: %s; falling back to instaloader", handle, exc)
        else:
            logger.info("APIFY_TOKEN not set; falling back to instaloader for @%s", handle)

        try:
            return await self._instaloader_instagram(handle)
        except Exception as exc:
            logger.warning("instaloader Instagram failed for %s: %s", handle, exc)
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
            logger.warning("fetch_instagram_search skipped: no APIFY_TOKEN")
            return []
        # The actor validates 'search' against a pattern that forbids most
        # punctuation; raw web-search titles ("Homes.com: Homes for Sale...")
        # must be reduced to plain words or the call 400s.
        query = re.sub(r"[^\w\s]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        if not query:
            return []
        try:
            async with httpx.AsyncClient(timeout=self.APIFY_TIMEOUT, headers=self._auth_headers()) as client:
                run_resp = await client.post(
                    f"{self.APIFY_BASE}/acts/{self.APIFY_SEARCH_ACTOR}/run-sync-get-dataset-items",
                        # 'search' must be a plain string for this actor (400
                    # "Field input.search must be string" when sent as array).
                    json={"searchType": "user", "search": query, "resultsLimit": limit},
                )
                if run_resp.status_code >= 400:
                    logger.error(
                        "::error::Apify Instagram search actor failed for %r: %s",
                        query, self._apify_error_detail(run_resp),
                    )
                run_resp.raise_for_status()
                items = run_resp.json()
            if not items:
                logger.warning("Apify Instagram search returned 0 results for %r", query)
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

    # Apify run-sync waits for the actor to finish server-side (up to 300s);
    # a client timeout below that silently kills every run with a TimeoutError.
    APIFY_TIMEOUT = httpx.Timeout(320, connect=30.0)

    def _auth_headers(self) -> dict[str, str]:
        """Bearer header keeps the token out of URLs (httpx logs request URLs)."""
        return {"Authorization": f"Bearer {self.apify_token}"}

    def _apify_error_detail(self, resp: httpx.Response) -> str:
        body = ""
        try:
            body = (resp.text or "")[:300]
        except Exception:
            pass
        return f"status={resp.status_code} body={body!r}"

    async def _apify_instagram(self, handle: str, limit: int) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.APIFY_TIMEOUT, headers=self._auth_headers()) as client:
            run_resp = await client.post(
                f"{self.APIFY_BASE}/acts/{self.APIFY_INSTAGRAM_ACTOR}/run-sync-get-dataset-items",
                json={
                    "usernames": [handle],
                    "resultsLimit": limit,
                    "commentsLimit": 3,
                },
            )
            if run_resp.status_code >= 400:
                logger.error(
                    "::error::Apify Instagram actor failed for @%s: %s",
                    handle, self._apify_error_detail(run_resp),
                )
            run_resp.raise_for_status()
            items = run_resp.json()
        if not items:
            logger.warning("Apify Instagram actor returned 0 items for @%s", handle)
        return self._normalize_instagram(items)

    async def _apify_linkedin(self, handle: str, limit: int) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.APIFY_TIMEOUT, headers=self._auth_headers()) as client:
            run_resp = await client.post(
                f"{self.APIFY_BASE}/acts/{self.APIFY_LINKEDIN_ACTOR}/run-sync-get-dataset-items",
                json={"handles": [handle], "resultsLimit": limit},
            )
            if run_resp.status_code >= 400:
                logger.error(
                    "::error::Apify LinkedIn actor failed for %s: %s",
                    handle, self._apify_error_detail(run_resp),
                )
            run_resp.raise_for_status()
            items = run_resp.json()
        return self._normalize_linkedin(items)

    async def _instaloader_instagram(self, handle: str) -> dict[str, Any]:
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
                    # Fail fast: default unlimited retries sleep ~11 minutes per
                    # profile on 429s, which hangs CI jobs. GitHub runner IPs
                    # are heavily rate-limited by IG anyway — if it doesn't
                    # work first try, let Apify handle it instead.
                    max_connection_attempts=1,
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
        """The Apify actor returns either profile objects (with `latestPosts`)
        or flat post items (with `ownerUsername`), depending on its mode. Accept
        both so follower counts and posts are never silently dropped."""
        items = [i for i in items if isinstance(i, dict)]
        profile: dict[str, Any] = next(
            (i for i in items if i.get("followersCount") is not None or isinstance(i.get("latestPosts"), list)),
            {},
        )
        raw_posts: list[dict[str, Any]] = []
        for item in items:
            latest = item.get("latestPosts")
            if isinstance(latest, list):
                raw_posts.extend(p for p in latest if isinstance(p, dict))
            elif any(k in item for k in ("shortCode", "likesCount", "commentsCount", "caption")):
                raw_posts.append(item)

        def count(value: Any) -> int:
            try:
                return max(int(value or 0), 0)  # Instagram reports hidden likes as -1
            except (TypeError, ValueError):
                return 0

        posts = []
        for item in raw_posts:
            caption = item.get("caption") or item.get("alt") or item.get("description") or ""
            comments_text: list[str] = []
            raw_comments = item.get("latestComments")
            if not isinstance(raw_comments, list):
                raw_comments = item.get("comments")
            if isinstance(raw_comments, list):
                for c in raw_comments[:5]:
                    if isinstance(c, dict):
                        txt_c = c.get("text") or c.get("caption") or ""
                    elif isinstance(c, str):
                        txt_c = c
                    else:
                        txt_c = ""
                    if txt_c:
                        comments_text.append(txt_c)
            url = item.get("url") or (
                f"https://www.instagram.com/p/{item['shortCode']}/" if item.get("shortCode") else ""
            )
            raw_likes = item.get("likesCount")
            posts.append(
                {
                    "post_id": item.get("id") or item.get("shortCode") or "",
                    "caption": caption,
                    "likes_hidden": isinstance(raw_likes, (int, float)) and raw_likes < 0,  # Instagram reports -1
                    "likes": count(raw_likes),
                    "comments": count(item.get("commentsCount")),
                    "comments_text": comments_text,
                    "video_views": count(item.get("videoViewCount") or item.get("videoPlayCount")),
                    "url": url,
                    "posted_at": item.get("timestamp") or item.get("postedAt") or None,
                    "platform": "instagram",
                }
            )
        first = items[0] if items else {}
        return {
            "handle": profile.get("username") or first.get("ownerUsername") or first.get("username") or "",
            "follower_count": profile.get("followersCount"),
            "following_count": profile.get("followingCount"),
            "posts_count": profile.get("postsCount"),
            "full_name": profile.get("fullName") or first.get("ownerFullName"),
            "biography": profile.get("biography"),
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
