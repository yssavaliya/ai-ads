from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from typing import Any

import httpx

from src.fetch.social_scraper import SocialScraper
from src.intelligence.research import verified_competitors
from src.shared.config import settings
from src.shared.llm_client import LLMClient, extract_json_text, fence, fence_text

logger = logging.getLogger(__name__)


class CompetitorFinder:
    """Discovers competitors from a business profile using LLM knowledge,
    then pulls their top posts via social scraping."""

    COMPETITOR_COUNT = 5

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.social = SocialScraper()

    def discover(self, business_profile: dict[str, Any]) -> list[dict[str, Any]]:
        """Return list of {name, platform, handle, reason}."""
        prompt = f"""
You are a competitive intelligence analyst. Given this business profile, identify the
top {self.COMPETITOR_COUNT} direct competitors who are active on Instagram and/or LinkedIn.

Business profile:
{fence(business_profile, 8000)}

For each competitor provide:
- name: competitor brand name
- platform: "instagram" or "linkedin"
- handle: their public handle WITHOUT the @ symbol
- reason: one line explaining why they are a competitor

Respond ONLY with a JSON array:
[{{"name": "...", "platform": "instagram", "handle": "...", "reason": "..."}}]
"""
        data = self.llm.run_json(prompt, label="competitor-discovery")
        if isinstance(data, dict):
            data = data.get("competitors", [])
        # Instagram only: the LinkedIn actor needs a logged-in cookie and proxy (HTTP 400 without them), so a
        # LinkedIn competitor could never be verified and only added a failing call per run.
        competitors = [
            c for c in data
            if isinstance(c, dict) and c.get("handle") and str(c.get("platform") or "instagram").lower() == "instagram"
        ]
        logger.info("Discovered %d competitors", len(competitors))
        return competitors[: self.COMPETITOR_COUNT]

    async def fetch_competitor_posts(self, competitors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Scrape every competitor, `FETCH_CONCURRENCY` at a time (each Apify run can take minutes and
        they are independent). Order is preserved; one failure never loses the others."""
        gate = asyncio.Semaphore(max(1, settings.fetch_concurrency))

        async def one(comp: dict[str, Any]) -> dict[str, Any]:
            async with gate:
                try:
                    if comp["platform"] == "instagram":
                        data = await self.social.fetch_instagram(comp["handle"])
                    else:
                        data = await self.social.fetch_linkedin(comp["handle"])
                    comp["profile"] = data
                    comp["top_posts"] = data.get("posts", [])[:5]
                except Exception as exc:
                    logger.warning("Failed fetching competitor %s: %s", comp.get("handle"), exc)
                    comp["top_posts"] = []
                return comp

        return list(await asyncio.gather(*(one(c) for c in competitors)))

    # ------------------------------------------------------------------ search-based discovery
    MIN_COMPETITOR_FOLLOWERS = 500      # tiny/dormant accounts say nothing about a market
    MAX_COMPETITOR_FOLLOWERS = 500_000  # global brands are not peers
    WAVE_SIZE = 5
    TARGET_VERIFIED = 3

    @staticmethod
    def search_queries(business_profile: dict[str, Any], limit: int = 3) -> list[str]:
        """Plain-word Instagram search queries from what the owner says the business offers (verified facts)."""
        facts = business_profile.get("verified_facts") or {}
        offers = [str(o) for o in (facts.get("offers") or []) if str(o).strip()]
        queries: list[str] = []
        for offer in offers[:limit]:
            query = " ".join(re.sub(r"[^\w\s]", " ", offer).split()[:4])
            if query and query.lower() not in {q.lower() for q in queries}:
                queries.append(query)
        return queries

    @classmethod
    def rank_search_results(
        cls, results: list[dict[str, Any]], own_handle: str, queries: list[str]
    ) -> list[dict[str, Any]]:
        """Real accounts (the search returned them, so they exist) that are peers: right size, not us, ranked by how
        well their bio matches the query words and by audience size."""
        words = {w.lower() for q in queries for w in q.split() if len(w) > 2}
        ranked: list[tuple[float, dict[str, Any]]] = []
        seen: set[str] = set()
        for item in results:
            handle = str(item.get("username") or "").strip().lstrip("@").lower()
            followers = int(item.get("followersCount") or 0)
            if not handle or handle == own_handle.lower().lstrip("@") or handle in seen:
                continue
            if not cls.MIN_COMPETITOR_FOLLOWERS <= followers <= cls.MAX_COMPETITOR_FOLLOWERS:
                continue
            seen.add(handle)
            text = f"{item.get('fullName') or ''} {item.get('biography') or ''}".lower()
            relevance = sum(1 for w in words if w in text)
            ranked.append((relevance * 10 + math.log10(max(followers, 1)), {
                "name": item.get("fullName") or handle, "platform": "instagram", "handle": handle,
                "reason": f"found by Instagram search ({followers} followers)", "source": "search",
                "follower_count": followers,
            }))
        ranked.sort(key=lambda pair: -pair[0])
        return [c for _, c in ranked]

    def discover_by_search(self, business_profile: dict[str, Any], own_handle: str = "") -> list[dict[str, Any]]:
        queries = self.search_queries(business_profile)
        if not queries or not self.social.apify_token:
            return []

        async def run() -> list[dict[str, Any]]:
            found: list[dict[str, Any]] = []
            for query in queries:
                found.extend(await self.social.fetch_instagram_search(query, limit=10))
            return found

        try:
            return self.rank_search_results(asyncio.run(run()), own_handle, queries)
        except Exception as exc:  # noqa: BLE001 - search is an upgrade over LLM memory, never a hard dependency
            logger.warning("Competitor search failed: %s", exc)
            return []

    def discover_and_fetch(self, business_profile: dict[str, Any]) -> list[dict[str, Any]]:
        """Candidates from real Instagram search first, LLM memory second; scrape in waves until enough are
        verified (their posts actually came back). Unverifiable suggestions are dropped when anything verified."""
        own = str(business_profile.get("instagram_handle") or (business_profile.get("verified_facts") or {}).get("instagram_handle") or "")
        candidates = self.discover_by_search(business_profile, own) if settings.competitor_search_enabled else []
        try:
            llm_suggestions = self.discover(business_profile)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM competitor suggestions unavailable: %s", exc)
            llm_suggestions = []
        known = {c["handle"].lower() for c in candidates}
        for suggestion in llm_suggestions:
            handle = str(suggestion.get("handle") or "").lstrip("@").lower()
            if handle and handle not in known and handle != own.lower().lstrip("@"):
                known.add(handle)
                candidates.append({**suggestion, "handle": handle, "source": "llm"})
        if not candidates:
            return []

        limit = max(self.WAVE_SIZE, int(settings.competitor_probe_limit))
        scraped: list[dict[str, Any]] = []
        for start in range(0, min(limit, len(candidates)), self.WAVE_SIZE):
            wave = candidates[start:start + self.WAVE_SIZE]
            scraped.extend(asyncio.run(self.fetch_competitor_posts(wave)))
            if len(verified_competitors(scraped)) >= self.TARGET_VERIFIED:
                break
        verified = verified_competitors(scraped)
        logger.info(
            "Competitors: %d candidates probed, %d verified (%d from search)",
            len(scraped), len(verified), sum(1 for c in verified if c.get("source") == "search"),
        )
        return (verified or scraped)[: self.COMPETITOR_COUNT]
