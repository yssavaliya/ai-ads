from __future__ import annotations

import json
import logging
from typing import Any

from src.shared.llm_client import LLMClient, extract_json_text

logger = logging.getLogger(__name__)


class AudienceMapper:
    """Builds audience personas, pain points, and content preferences from
    business profile + social data + competitor posts."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    def map_audience(
        self,
        business_profile: dict[str, Any],
        social_data: dict[str, Any] | None = None,
        competitor_posts: list[dict[str, Any]] | None = None,
        previous_personas: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        social_summary = self._summarize_social(social_data)
        competitor_summary = self._summarize_competitors(competitor_posts or [])
        previous_json = json.dumps(previous_personas or {}, indent=2, default=str)[:3000]

        prompt = f"""
You are an audience intelligence analyst. Build a structured audience map for a business.

Business profile:
{json.dumps(business_profile, indent=2, default=str)[:8000]}

Social data (publicly available):
{social_summary}

Competitor activity:
{competitor_summary}

Previous audience map (use to detect drift, keep what still holds):
{previous_json}

Respond ONLY with JSON:
{{
  "personas": [
    {{
      "name": "persona label",
      "demographics": "age range, location, profession",
      "goals": ["goal 1", "goal 2"],
      "pain_points": ["pain 1", "pain 2"],
      "content_preferences": ["format or topic they engage with"],
      "best_posting_times": "e.g. Tue-Thu 6-9 PM IST",
      "share_of_audience": "e.g. 40%"
    }}
  ],
  "top_pain_points": ["global pain 1", "global pain 2", "global pain 3"],
  "content_themes": [
    {{"theme": "theme name", "evidence": "why this works", "strength": "high|medium|low"}}
  ],
  "engagement_triggers": ["emotional or practical trigger that drives engagement"],
  "language_tone": "describe the tone that resonates",
  "audience_insights": ["insight 1", "insight 2", "insight 3"],
  "drift_detected": "list of changes vs previous map, or 'none'"
}}
"""
        result = self.llm.run(prompt, label="audience-mapper", temperature=0.4)
        data = json.loads(extract_json_text(result.stdout))
        if not isinstance(data, dict):
            raise ValueError("audience map must be a JSON object")
        data["source"] = "llm"
        return data

    @staticmethod
    def _summarize_social(social_data: dict[str, Any] | None) -> str:
        if not social_data:
            return "No social data available."
        parts = []
        if social_data.get("biography"):
            parts.append(f"Bio: {social_data['biography']}")
        if social_data.get("follower_count") is not None:
            parts.append(f"Followers: {social_data['follower_count']}")
        captions = [p.get("caption", "") for p in social_data.get("posts", []) if p.get("caption")]
        if captions:
            parts.append("Recent captions: " + " || ".join(captions[:8])[:4000])
        return "\n".join(parts) or "No social data available."

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