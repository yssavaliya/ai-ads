from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from src.fetch.social_scraper import SocialScraper
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
        competitors = [c for c in data if isinstance(c, dict) and c.get("handle")]
        logger.info("Discovered %d competitors", len(competitors))
        return competitors[: self.COMPETITOR_COUNT]

    async def fetch_competitor_posts(self, competitors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched = []
        for comp in competitors:
            try:
                if comp["platform"] == "instagram":
                    data = await self.social.fetch_instagram(comp["handle"])
                else:
                    data = await self.social.fetch_linkedin(comp["handle"])
                comp["profile"] = data
                comp["top_posts"] = data.get("posts", [])[:5]
                enriched.append(comp)
            except Exception as exc:
                logger.warning("Failed fetching competitor %s: %s", comp.get("handle"), exc)
                comp["top_posts"] = []
                enriched.append(comp)
        return enriched

    def discover_and_fetch(self, business_profile: dict[str, Any]) -> list[dict[str, Any]]:
        competitors = self.discover(business_profile)
        if not competitors:
            return []
        return asyncio.run(self.fetch_competitor_posts(competitors))
