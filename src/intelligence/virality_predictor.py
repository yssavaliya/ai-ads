from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.shared.llm_client import LLMClient, extract_json_text, untrusted

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
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(
            prompt_text, business_profile, audience_map, trend_data, prompt_type, history
        )
        result = self.llm.run(prompt, label="virality-predictor", temperature=0.2)
        payload = json.loads(extract_json_text(result.stdout))
        return self._validate(payload)

    @staticmethod
    def _format_history(history: list[dict[str, Any]] | None) -> str:
        if not history:
            return "No historical performance data yet for this business."
        lines = []
        for h in history[:10]:
            text = (h.get("prompt_text") or "")[:200].replace("\n", " ")
            outcome = h.get("outcome") or "unknown"
            rate = h.get("rate")
            rate_s = f" (engagement rate {rate})" if rate is not None else ""
            lines.append(f"- [{outcome}]{rate_s}: {text}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        prompt_text: str,
        business_profile: dict[str, Any],
        audience_map: dict[str, Any],
        trend_data: dict[str, Any],
        prompt_type: str,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        return f"""
You are a viral content strategist for the Indian social media market (Instagram + LinkedIn).
Predict the virality potential of a content prompt for this business.

Business profile:
{untrusted(json.dumps(business_profile, indent=2, default=str)[:5000])}

Audience map:
{json.dumps(audience_map, indent=2, default=str)[:5000]}

Today's trends:
{json.dumps(trend_data, indent=2, default=str)[:5000]}

Prompt type: {prompt_type}

REAL PAST PERFORMANCE for this business (use to ground your judgment — favor
patterns similar to winners, penalize patterns similar to losers):
{self._format_history(history)}

Content prompt to evaluate:
"""
        + f'"""{prompt_text}"""\n'
        + """
Evaluate like a senior growth marketer. Consider: hook strength (first 3 words),
emotional trigger, specificity, Indian cultural relevance, platform fit,
scroll-stopping potential, shareability, and fit with the audience map. Weight
the REAL PAST PERFORMANCE above: if this prompt resembles a proven winner, score
it higher; if it resembles a proven loser, score it lower.

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
