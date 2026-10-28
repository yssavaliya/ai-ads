from __future__ import annotations

import json
import logging
from typing import Any

from src.shared.llm_client import LLMClient, extract_json_text, fence, fence_text

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
You are a viral social media growth strategist and trend analyst for the Indian & global digital market (IST timezone).
Analyze the business profile, audience persona, and competitor patterns to pinpoint scroll-stopping, high-reach content opportunities.

Business profile:
{fence(business_profile, 5000)}

Audience content themes:
{fence(themes, 3000)}

Audience pain points & desires:
{fence(pain_points, 2000)}

Competitor posts & market signals:
{fence_text(competitor_summary)}

Focus on what ACTUALLY drives virality on Instagram in 2026:
1. High-save triggers: Step-by-step blueprints, swipe files, prompt structures, ROI metrics.
2. High-share triggers: Relatable contrarian takes, hard truths, shocking before/after transformations.
3. 3-second hook psychology: Contrarian truth, curiosity gap, or specific numbers in the first 5 words.
4. Real, timely industry trends (e.g. AI workflows, automation tools, budget optimization, Indian business scaling).

Respond ONLY with JSON:
{{
  "trending_topics": [
    {{"topic": "topic name", "angle": "contrarian/ROI/blueprint angle", "hook_formula": "sample 10-word opening hook", "relevance": "high|medium|low"}}
  ],
  "trending_hashtags": ["#niche1", "#niche2", "#niche3", "#industry1", "#industry2", "#broad1", "#broad2"],
  "content_gaps": [
    {{"gap": "high-demand untapped problem", "evidence": "why competitors miss it and why audience saves it"}}
  ],
  "viral_patterns": [
    {{"pattern": "hook archetype (contrarian|roi_proof|curiosity_gap|before_after|framework)", "example": "concrete example", "strength": "high|medium|low"}}
  ],
  "seasonal_events": ["relevant upcoming event/occasion/milestone in the next 7-14 days"],
  "recommended_formats": ["single image"]
}}
"""
        data = self.llm.run_json(prompt, label="trend-radar", temperature=0.5, expect=(dict,))
        if not isinstance(data, dict):
            raise ValueError("trend scan must be a JSON object")
        # Format recommendations are consumed downstream. Keep legacy or hostile
        # model output from reintroducing video/carousel concepts into the flow.
        data["recommended_formats"] = ["single image"]
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
