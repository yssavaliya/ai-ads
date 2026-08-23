from __future__ import annotations

import json
import logging
from typing import Any

from src.shared.llm_client import LLMClient, extract_json_text

logger = logging.getLogger(__name__)


class TrendRadar:
    """Scans trending topics/hashtags relevant to the business using LLM
    knowledge (no paid trend APIs) + competitor viral post patterns."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    def scan(
        self,
        business_profile: dict[str, Any],
        audience_map: dict[str, Any],
        competitor_posts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        competitor_summary = self._summarize_competitors(competitor_posts or [])
        themes = audience_map.get("content_themes", [])
        pain_points = audience_map.get("top_pain_points", [])

        prompt = f"""
You are a social media trend analyst for the Indian market (IST timezone). Given:

Business profile:
{json.dumps(business_profile, indent=2, default=str)[:5000]}

Audience content themes:
{json.dumps(themes, indent=2, default=str)[:3000]}

Audience pain points:
{json.dumps(pain_points, indent=2, default=str)[:2000]}

Competitor posts:
{competitor_summary}

Identify TODAY'S most relevant trending topics, hashtags, and content gaps.
Prefer real, current trends relevant to this industry. Do NOT invent fake viral claims.

Respond ONLY with JSON:
{{
  "trending_topics": [
    {{"topic": "topic name", "angle": "how the business can use it", "relevance": "high|medium|low"}}
  ],
  "trending_hashtags": ["#tag1", "#tag2", "#tag3"],
  "content_gaps": [
    {{"gap": "untapped topic", "evidence": "why it is untapped (competitor/audience based)"}}
  ],
  "viral_patterns": [
    {{"pattern": "hook style or format", "example": "description", "strength": "high|medium|low"}}
  ],
  "seasonal_events": ["relevant upcoming event/occasion in the next 7 days"],
  "recommended_formats": ["reel", "carousel", "story", "single image", "text post"]
}}
"""
        result = self.llm.run(prompt, label="trend-radar", temperature=0.5)
        data = json.loads(extract_json_text(result.stdout))
        if not isinstance(data, dict):
            raise ValueError("trend scan must be a JSON object")
        return data

    @staticmethod
    def _summarize_competitors(competitor_posts: list[dict[str, Any]]) -> str:
        if not competitor_posts:
            return "No competitor data available."
        lines = []
        for comp in competitor_posts[:5]:
            posts = comp.get("top_posts", [])
            captions = [p.get("caption", "") for p in posts if p.get("caption")]
            lines.append(
                f"- {comp.get('name', comp.get('handle', '?'))}: "
                + " || ".join(captions[:4])[:2000]
            )
        return "\n".join(lines) or "No competitor data available."