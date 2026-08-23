from __future__ import annotations

import asyncio
import base64
import logging
import re
import urllib.parse
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from src.fetch.social_scraper import SocialScraper
from src.shared.config import settings

logger = logging.getLogger(__name__)

IGNORED_DOMAINS = {
    "instagram.com", "linkedin.com", "facebook.com", "twitter.com", "youtube.com",
    "google.com", "google.co.in", "wikipedia.org", "youtube.com", "amazon.com",
    "amazon.in", "flipkart.com", "justdial.com", "indiamart.com", "pinterest.com",
    "quora.com", "reddit.com", "medium.com", "businessinsider.com", "forbes.com",
    "economictimes.indiatimes.com", "yourstory.com", "inc42.com", "entrackr.com",
    "tripadvisor.com", "tripadvisor.in", "zomato.com", "datagemba.com",
    "yelp.com", "yelp.in", "urbanpro.com", "magicbricks.com", "99acres.com",
    "housing.com", "nobroker.in", "booking.com", "restaurantguru.com",
    "franchiseindia.com", "shopify.com", "etsy.com", "allbusiness.com",
    "indianexpress.com", "timesofindia.indiatimes.com", "livemint.com",
    "hindustantimes.com", "news18.com", "ndtv.com", "moneycontrol.com",
    "groww.in", "risehq.io", "wanderlog.com", "crazymasalafood.com",
    "webindia123.com", "duckduckgo.com", "lingoapp.com", "dineout.co.in",
    "eazydiner.com", "swiggy.com", "grocersindia.com", "whatshot.in",
    "socialsamosa.com", "adgully.com", "mobisn.com", "localstreet.in",
    "exploreandmore.in", "goodthings.com", "kooapp.com", "quora.com",
}

SOCIAL_LINK_PATTERNS = {
    "instagram": re.compile(r"instagram\.com/([a-zA-Z0-9_.]{1,30})"),
    "linkedin": re.compile(r"linkedin\.com/(?:company|in)/([a-zA-Z0-9\-_]+)"),
}


class LeadFinder:
    """Discovers candidate businesses automatically: free web search
    (DuckDuckGo HTML — no key), Apify Instagram search (free tier), and
    social links found on each candidate's website."""

    SEARCH_QUERIES_PER_NICHE = [
        "{niche} business {location}",
        "{niche} in {location}",
        "{niche} shop {location}",
    ]
    IG_SEARCH_QUERIES_PER_NICHE = [
        "{niche} {location} instagram",
        "{niche} startup {location} instagram",
    ]
    MAX_RESULTS_PER_QUERY = 8

    async def discover(
        self,
        niches: list[str],
        locations: list[str] | str = "India",
        count: int = 10,
        platform_focus: str = "",
    ) -> list[dict[str, Any]]:
        if isinstance(locations, str):
            locations = [locations]
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        # 1) Instagram search FIRST when available — these candidates already
        #    have handles + follower counts, so they are instantly scoreable.
        if settings.apify_token:
            try:
                ig_candidates = await self._search_instagram(niches, locations)
                for c in ig_candidates:
                    key = c.get("instagram_handle") or c.get("website_url") or ""
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(c)
            except Exception as exc:
                logger.warning("Apify Instagram search failed: %s", exc)

        # 2) Web search supplements (candidates get IG handles via enrichment).
        for niche in niches:
            for location in locations:
                query_templates = self.SEARCH_QUERIES_PER_NICHE
                if platform_focus == "instagram":
                    query_templates = query_templates + self.IG_SEARCH_QUERIES_PER_NICHE

                for query_template in query_templates:
                    if len(candidates) >= count:
                        break
                    await asyncio.sleep(2.0)  # throttle: avoid tripping search engines
                    query = query_template.format(niche=niche, location=location)
                    try:
                        results = await self._search_web(query)
                    except Exception as exc:
                        logger.warning("Search failed for %r: %s", query, exc)
                        results = []

                    for url, title in results:
                        if len(candidates) >= count:
                            break
                        domain = self._domain(url)
                        if not domain or domain in IGNORED_DOMAINS:
                            continue
                        if domain in seen:
                            continue
                        seen.add(domain)
                        candidates.append(
                            {
                                "name": self._clean_name(title, domain),
                                "website_url": url,
                                "instagram_handle": "",
                                "linkedin_handle": "",
                                "niche": niche,
                                "location": location,
                                "found_via": "search",
                            }
                        )

        logger.info("Lead discovery found %d candidate businesses", len(candidates))
        return candidates[:count]

    async def enrich_with_socials(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fetch each candidate's website and extract IG/LinkedIn handles.
        For web-found candidates without handles, also try a name-based
        Instagram lookup (Apify) so they become scoreable."""
        for candidate in candidates:
            try:
                from src.fetch.website_scraper import WebsiteScraper

                scraper = WebsiteScraper()
                profile = scraper.scrape(candidate.get("website_url") or "")
                handles = self._extract_social_handles(profile.get("pages", []), candidate.get("website_url"))
                candidate["instagram_handle"] = handles.get("instagram", candidate.get("instagram_handle", ""))
                candidate["linkedin_handle"] = handles.get("linkedin", candidate.get("linkedin_handle", ""))
                candidate["website_title"] = profile.get("title", "")
            except Exception as exc:
                logger.warning("Enrichment failed for %s: %s", candidate.get("name"), exc)

            if not candidate.get("instagram_handle") and settings.apify_token:
                try:
                    match = await self._find_instagram_by_name(
                        candidate.get("name", ""), candidate.get("niche", ""), candidate.get("location", "")
                    )
                    if match:
                        candidate["instagram_handle"] = match["handle"]
                        candidate.setdefault("follower_count", match.get("followers"))
                except Exception as exc:
                    logger.warning("Name-based IG lookup failed for %s: %s", candidate.get("name"), exc)
        return candidates

    async def _search_web(self, query: str) -> list[tuple[str, str]]:
        results = await self._search_duckduckgo(query)
        if not results:
            results = await self._search_bing(query)
        return results[: self.MAX_RESULTS_PER_QUERY]

    async def _search_duckduckgo(self, query: str) -> list[tuple[str, str]]:
        try:
            async with httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
                },
                timeout=30,
                follow_redirects=True,
            ) as client:
                resp = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
                resp.raise_for_status()
                if "anomaly" in resp.text.lower():
                    return []

            soup = BeautifulSoup(resp.text, "lxml")
            results = []
            for result in soup.select(".result"):
                link = result.select_one("a.result__a")
                if not link or not link.get("href"):
                    continue
                url = self._clean_search_url(link["href"])
                title = link.get_text(strip=True)
                if url:
                    results.append((url, title))
            return results
        except Exception as exc:
            logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
            return []

    async def _search_bing(self, query: str) -> list[tuple[str, str]]:
        try:
            async with httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                },
                timeout=30,
                follow_redirects=True,
            ) as client:
                resp = await client.get("https://www.bing.com/search", params={"q": query})
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")
            results = []
            for link in soup.select("li.b_algo h2 a"):
                href = link.get("href") or ""
                url = self._decode_bing_url(href)
                title = link.get_text(strip=True)
                if url:
                    results.append((url, title))
            return results
        except Exception as exc:
            logger.warning("Bing search failed for %r: %s", query, exc)
            return []

    @staticmethod
    def _decode_bing_url(href: str) -> str:
        """Bing wraps results in /ck/a redirect URLs; the real target is a
        base64 'u' query param."""
        if "bing.com/ck/a" not in href:
            return href if href.startswith("http") else ""
        match = re.search(r"[?&]u=([^&]+)", href)
        if not match:
            return ""
        encoded = urllib.parse.unquote(match.group(1))
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        try:
            return base64.b64decode(encoded).decode("utf-8", "ignore")
        except Exception:
            return ""

    async def _search_instagram(self, niches: list[str], locations: list[str]) -> list[dict[str, Any]]:
        scraper = SocialScraper()
        candidates = []
        for niche in niches:
            for location in locations:
                await asyncio.sleep(2.0)  # throttle Apify runs
                query = f"{niche} {location}"
                items = await scraper.fetch_instagram_search(query)
                for item in items:
                    handle = item.get("username") or ""
                    if not handle:
                        continue
                    candidates.append(
                        {
                            "name": item.get("fullName") or handle,
                            "website_url": item.get("externalUrl") or "",
                            "instagram_handle": handle,
                            "linkedin_handle": "",
                            "niche": niche,
                            "location": location,
                            "follower_count": item.get("followersCount") or 0,
                            "found_via": "instagram-search",
                        }
                    )
        return candidates

    async def _find_instagram_by_name(self, name: str, niche: str, location: str) -> dict[str, Any] | None:
        """Search Instagram for a business by its name (used for web-found candidates)."""
        query = name.strip() or niche
        if not query:
            return None
        items = await SocialScraper().fetch_instagram_search(query)
        for item in items:
            handle = item.get("username") or ""
            if not handle:
                continue
            full_name = (item.get("fullName") or "").lower()
            name_lower = name.lower()
            if full_name and name_lower and name_lower in full_name:
                return {
                    "handle": handle,
                    "followers": item.get("followersCount") or 0,
                    "full_name": item.get("fullName") or "",
                }
            if name_lower and name_lower.replace(" ", "") in handle.lower():
                return {
                    "handle": handle,
                    "followers": item.get("followersCount") or 0,
                    "full_name": item.get("fullName") or "",
                }
        return None

    @staticmethod
    def _clean_search_url(raw: str) -> str:
        match = re.search(r"uddg=([^&]+)", raw)
        if match:
            return urllib.parse.unquote(match.group(1))
        return raw if raw.startswith("http") else ""

    @staticmethod
    def _domain(url: str) -> str:
        try:
            return (urlparse(url).netloc or "").lower().replace("www.", "")
        except ValueError:
            return ""

    @staticmethod
    def _clean_name(title: str, domain: str) -> str:
        title = re.sub(r"\s*[|–—-].*$", "", title).strip()
        if title:
            return title[:80]
        return domain.replace(".com", "").replace(".in", "").replace(".", " ").title()[:80]

    @staticmethod
    def _extract_social_handles(pages: list[dict[str, Any]], fallback_url: str) -> dict[str, str]:
        handles: dict[str, str] = {}
        page_urls = [p.get("url", "") for p in pages] + [fallback_url]
        for page_url in page_urls:
            for platform, pattern in SOCIAL_LINK_PATTERNS.items():
                match = pattern.search(page_url or "")
                if match:
                    handle = match.group(1).rstrip("/")
                    if handle and not handles.get(platform):
                        handles[platform] = handle
        return handles