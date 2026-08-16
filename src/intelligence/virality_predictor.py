from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.shared.llm_client import LLMClient, extract_json_text

logger = logging.getLogger(__name__)

VIRALITY_LABELS = {"LOW", "MEDIUM", "HIGH"}


class ViralityPredictor:
    """Scores a content prompt 0-100 for virality potential on Instagram/LinkedIn
    for the Indian market. Returns reasons + improvement hints so the generator
    can iterate until the threshold is met."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    def score(
        self,
        prompt_text: str,
        business_profile: dict[str, Any],
        audience_map: dict[str, Any],
        trend_data: dict[str, Any],
        prompt_type: str = "educational",
    ) -> dict[str, Any]:
        prompt = self._build_prompt(prompt_text, business_profile, audience_map, trend_data, prompt_type)
        try:
            result = self.llm.run(prompt, label="virality-predictor", temperature=0.2)
            payload = json.loads(extract_json_text(result.stdout))
            return self._validate(payload)
        except Exception as exc:
            logger.error("Virality prediction failed: %s", exc)
            return {
                "virality_score": 50,
                "virality_label": "MEDIUM",
                "virality_probability": 0.5,
                "key_factors": ["prediction failed; defaulted to medium"],
                "risks": ["unable to evaluate"],
                "reasoning": str(exc),
                "improvement_hints": ["retry prediction"],
                "timestamp": None,
            }

    def _build_prompt(
        self,
        prompt_text: str,
        business_profile: dict[str, Any],
        audience_map: dict[str, Any],
        trend_data: dict[str, Any],
        prompt_type: str,
    ) -> str:
        return f"""
You are a viral content strategist for the Indian social media market (Instagram + LinkedIn).
Predict the virality potential of a content prompt for this business.

Business profile:
{json.dumps(business_profile, indent=2, default=str)[:5000]}

Audience map:
{json.dumps(audience_map, indent=2, default=str)[:5000]}

Today's trends:
{json.dumps(trend_data, indent=2, default=str)[:5000]}

Prompt type: {prompt_type}

Content prompt to evaluate:
"""
        + f'"""{prompt_text}"""\n'
        + """
Evaluate like a senior growth marketer. Consider: hook strength (first 3 words),
emotional trigger, specificity, Indian cultural relevance, platform fit,
scroll-stopping potential, shareability, and fit with the audience map.

Respond ONLY with JSON:
{
  "virality_score": 0-100 integer,
  "virality_label": "LOW" | "MEDIUM" | "HIGH",
  "virality_probability": 0.0-1.0,
  "key_factors": ["top 3-5 factors driving the score"],
  "risks": ["top 2-4 risks that could kill virality"],
  "reasoning": "2-4 sentence explanation",
  "improvement_hints": ["2-4 concrete ways to make THIS prompt more viral"]
}
"""

    def _validate(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Virality prediction must be a JSON object")

        for field in ("virality_score", "virality_label", "virality_probability"):
            if field not in payload:
                raise ValueError(f"Missing virality field: {field}")

        score = payload["virality_score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("virality_score must be numeric")
        score = max(0, min(100, int(score)))

        label = str(payload["virality_label"]).upper()
        if label not in VIRALITY_LABELS:
            label = "MEDIUM"

        prob = payload["virality_probability"]
        if not isinstance(prob, (int, float)) or isinstance(prob, bool):
            prob = 0.5
        prob = max(0.0, min(1.0, float(prob)))

        for field in ("key_factors", "risks"):
            items = payload.get(field, [])
            if not isinstance(items, list):
                payload[field] = []
            else:
                payload[field] = [str(i) for i in items if str(i).strip()][:6]

        payload["virality_score"] = score
        payload["virality_label"] = label
        payload["virality_probability"] = prob
        payload["improvement_hints"] = [
            str(h) for h in payload.get("improvement_hints", []) if str(h).strip()
        ][:5]
        payload["reasoning"] = str(payload.get("reasoning", ""))[:2000]
        return payload