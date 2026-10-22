from __future__ import annotations

import json
import logging
from typing import Any

from src.shared.llm_client import LLMClient, extract_json_text, untrusted

logger = logging.getLogger(__name__)

PROMPT_TYPE_DESCRIPTIONS = {
    "educational": "Teach one valuable lesson or insight, actionable, with a strong specific hook.",
    "storytelling": "Tell a relatable story with a turning point and a lesson tied to the brand.",
    "social_proof": "Showcase a real customer result, testimonial, or transformation with specifics.",
    "promotional": "Soft promotion with clear value and urgency, never hard-selling.",
    "engagement": "Ask a polarizing or thought-provoking question that invites comments.",
    "behind_scenes": "Reveal how the business works, mistakes, or the human side of the team.",
    "user_generated": "Spotlight a customer's content or experience, encouraging UGC.",
    "trend_jacking": "Ride a current trend/event with a brand-relevant twist.",
}


class PromptForge:
    """Generates content prompts per business. Each prompt instructs a human
    creator (or AI assistant) how to produce one post — the PROMPT itself is
    the deliverable, not an auto-posted caption."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    def generate_variants(
        self,
        business_profile: dict[str, Any],
        audience_map: dict[str, Any],
        trend_data: dict[str, Any],
        prompt_types: list[str],
        weights: dict[str, Any] | None = None,
        improvement_hints: list[str] | None = None,
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate N prompt variants. If improvement_hints given, focus on them."""
        weights = weights or {}
        type_descriptions = "\n".join(
            f"- {t}: {PROMPT_TYPE_DESCRIPTIONS.get(t, 'generic content prompt')}"
            for t in prompt_types
        )
        hints_text = (
            "STRICT REQUIREMENTS from the last prediction round - address ALL of these:\n"
            + "\n".join(f"- {h}" for h in improvement_hints)
            if improvement_hints
            else "No previous feedback."
        )

        prompt = f"""
You are a viral content strategist. Generate {count} distinct, ready-to-use content prompts
for a human creator. Each prompt must be specific enough that a creator can produce a post
from it without further research, but flexible enough to adapt to the platform.

Business profile:
{untrusted(json.dumps(business_profile, indent=2, default=str)[:6000])}

Audience map:
{json.dumps(audience_map, indent=2, default=str)[:6000]}

Today's trends:
{json.dumps(trend_data, indent=2, default=str)[:6000]}

Allowed prompt types:
{type_descriptions}

Learned weights (topic:format:audience → weight, higher = more proven):
{json.dumps(weights, indent=2, default=str)[:4000]}

{hints_text}

Each generated prompt must include: a scroll-stopping hook, the core message,
specific details (numbers, examples, Indian context), a call to action,
and platform guidance (Instagram: visual-first; LinkedIn: thought-leadership).

Respond ONLY with JSON:
[
  {{
    "prompt_type": "educational",
    "title": "short working title",
    "hook": "the exact opening hook (max 15 words)",
    "prompt_text": "the complete ready-to-use content prompt (150-350 words)",
    "hashtags": ["#tag1", "#tag2", "#tag3"],
    "platform_focus": "instagram" | "linkedin" | "both",
    "format": "reel" | "carousel" | "story" | "single image" | "text post",
    "angle": "one sentence on the strategic angle",
    "audience_segment": "which persona name from the audience map this targets",
    "reel_beats": ["3-5 short on-screen text lines for a reel version"]
  }}
]
"""
        data = self.llm.run_json(prompt, label="prompt-forge", temperature=0.8)
        if isinstance(data, dict):
            data = data.get("prompts", [])
        if not isinstance(data, list):
            raise ValueError("forge output must be a JSON array")
        variants = [
            {
                "prompt_type": str(v.get("prompt_type", "educational")),
                "title": str(v.get("title", "")),
                "hook": str(v.get("hook", "")),
                "prompt_text": str(v.get("prompt_text", "")),
                "hashtags": [str(h) for h in v.get("hashtags", []) if str(h).startswith("#")],
                "platform_focus": str(v.get("platform_focus", "both")),
                "format": str(v.get("format", "reel")).lower(),
                "angle": str(v.get("angle", "")),
                "audience_segment": str(v.get("audience_segment", "") or ""),
                "reel_beats": [str(b).strip() for b in v.get("reel_beats", []) if str(b).strip()][:5],
            }
            for v in data
            if isinstance(v, dict) and v.get("prompt_text")
        ]
        if not variants:
            raise ValueError("prompt forge produced no usable variants")
        logger.info("Generated %d prompt variants", len(variants))
        return variants[:count]

    def refine_with_hints(
        self,
        best_variant: dict[str, Any],
        improvement_hints: list[str],
        business_profile: dict[str, Any],
        audience_map: dict[str, Any],
        trend_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Take the best variant + hints and regenerate a stronger version."""
        hints = "\n".join(f"- {h}" for h in improvement_hints)
        prompt = f"""
You are a viral content strategist. Improve this content prompt to maximize virality.

Current prompt:
{json.dumps(best_variant, indent=2, default=str)}

Critique from the predictor (address ALL points):
{hints}

Business profile:
{untrusted(json.dumps(business_profile, indent=2, default=str)[:5000])}

Audience map:
{json.dumps(audience_map, indent=2, default=str)[:4000]}

Trends:
{json.dumps(trend_data, indent=2, default=str)[:4000]}

Respond ONLY with the improved prompt as JSON with the same fields:
{{
  "prompt_type": "...",
  "title": "...",
  "hook": "...",
  "prompt_text": "...",
  "hashtags": ["..."],
  "platform_focus": "...",
  "format": "...",
  "angle": "...",
  "audience_segment": "...",
  "reel_beats": ["..."]
}}
"""
        try:
            data = self.llm.run_json(prompt, label="prompt-forge-refine", temperature=0.6, expect=(dict,))
            if not isinstance(data, dict) or not data.get("prompt_text"):
                raise ValueError("refined prompt invalid")
            return {
                "prompt_type": str(data.get("prompt_type", best_variant.get("prompt_type", "educational"))),
                "title": str(data.get("title", best_variant.get("title", ""))),
                "hook": str(data.get("hook", best_variant.get("hook", ""))),
                "prompt_text": str(data.get("prompt_text", "")),
                "hashtags": [str(h) for h in data.get("hashtags", best_variant.get("hashtags", []))],
                "platform_focus": str(data.get("platform_focus", best_variant.get("platform_focus", "both"))),
                "format": str(data.get("format", best_variant.get("format", "reel"))).lower(),
                "angle": str(data.get("angle", best_variant.get("angle", ""))),
                "audience_segment": str(data.get("audience_segment", "") or best_variant.get("audience_segment", "")),
                "reel_beats": [
                    str(b).strip()
                    for b in data.get("reel_beats", best_variant.get("reel_beats", []))
                    if str(b).strip()
                ][:5],
            }
        except Exception as exc:
            # Refinement is an optimization: keep the previous honest variant
            # rather than fabricating, but make the degradation visible.
            logger.warning(
                "::warning::Prompt refinement failed (%s) — using unrefined variant %s",
                exc, best_variant.get("title", ""),
            )
            return best_variant
