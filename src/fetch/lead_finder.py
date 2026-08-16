from __future__ import annotations

import asyncio
import logging
import re
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

        for niche in niches:
            for location in locations:
                query_templates = self.SEARCH_QUERIES_PER_NICHE
                if platform_focus == "instagram":
                    query_templates = query_templates + self.IG_SEARCH_QUERIES_PER_NICHE

                for query_template in query_templates:
                    query = query_template.format(niche=niche, location=location)
                    try:
                        results = await self._search_web(query)
                    except Exception as exc:
                        logger.warning("Search failed for %r: %s", query, exc)
                        results = []

                    for url, title in results:
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
                        if len(candidates) >= count:
                            return candidates

        if settings.apify_token:
            try:
                ig_candidates = await self._search_instagram(niches)
                for c in ig_candidates:
                    domain = c.get("website_url") and self._domain(c["website_url"])
                    if domain and domain not in seen:
                        seen.add(domain)
                        candidates.append(c)
                        if len(candidates) >= count:
                            break
            except Exception as exc:
                logger.warning("Apify Instagram search failed: %s", exc)

        logger.info("Lead discovery found %d candidate businesses", len(candidates))
        return candidates[:count]

    async def enrich_with_socials(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fetch each candidate's website and extract IG/LinkedIn handles."""
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
        return candidates

    async def _search_web(self, query: str) -> list[tuple[str, str]]:
        async with httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
            },
            timeout=30,
            follow_redirects=True,
        ) as client:
            resp = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
            resp.raise_for_status()

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
        return results[: self.MAX_RESULTS_PER_QUERY]

    async def _search_instagram(self, niches: list[str]) -> list[dict[str, Any]]:
        scraper = SocialScraper()
        candidates = []
        for niche in niches:
            items = await scraper.fetch_instagram_search(niche)
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
                        "location": "",
                        "found_via": "instagram-search",
                    }
                )
        return candidates

    @staticmethod
    def _clean_search_url(raw: str) -> str:
        match = re.search(r"uddg=([^&]+)", raw)
        if match:
            import urllib.parse

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