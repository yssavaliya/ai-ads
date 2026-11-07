from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.shared.llm_client import LLMClient, extract_json_text, fence, fence_text

logger = logging.getLogger(__name__)

_FORMAT_WORDS = re.compile(r"\b(?:reels?|videos?|slideshows?)\b", re.IGNORECASE)
_CAROUSEL = re.compile(r"\b(?:video\s+)?carousels?\b", re.IGNORECASE)
_SLIDE_COUNT = re.compile(
    r"\b(?:(?:multi[- ]slide)|(?:multiple|several|many|a\s+few|\d+(?:\s*[-–]\s*\d+)?)\s+slides?)\b",
    re.IGNORECASE,
)
_TIMED_VIDEO = re.compile(
    r"\b(?:create|make|produce)\s+(?:a\s+)?\d+(?:[- ]second|[- ]minute)\s+(?:reel|video)\b",
    re.IGNORECASE,
)


def single_image_variant(variant: dict[str, Any]) -> dict[str, Any]:
    """Enforce the product's single static image format on generated prompts."""
    normalized = dict(variant)
    normalized["format"] = "single image"
    normalized["reel_beats"] = []
    text = str(normalized.get("prompt_text", "")).strip()
    text = _TIMED_VIDEO.sub("Create a single-image post", text)
    text = _SLIDE_COUNT.sub("one static composition", text)
    text = _CAROUSEL.sub("single-image layout", text)
    text = _FORMAT_WORDS.sub("single image", text)
    text = re.sub(r"\bon-screen text beats?\b", "text placed within the image", text, flags=re.IGNORECASE)
    text = re.sub(r"\bscreen[- ]capture\b", "single-frame illustration", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvisual flow\b", "visual hierarchy", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:film|record|shoot)\b", "depict", text, flags=re.IGNORECASE)
    text = re.sub(r"\bone static composition for a single-image layout\b", "one static composition", text, flags=re.IGNORECASE)
    requirement = "Visual deliverable: create exactly one finished static image, with the complete composition in a single frame."
    if text and requirement.casefold() not in text.casefold():
        normalized["prompt_text"] = f"{text}\n\n{requirement}"
    return normalized


def normalize_variant(variant: dict[str, Any]) -> dict[str, Any]:
    """Keep valid single-image and Reel briefs; degrade malformed Reel briefs to a still."""
    normalized = dict(variant)
    beats = [str(b).strip() for b in (normalized.get("reel_beats") or []) if str(b).strip()][:5]
    if str(normalized.get("format", "")).strip().lower() in {"reel", "video"} and len(beats) >= 3:
        normalized["format"] = "reel"
        normalized["reel_beats"] = beats
        overlays = [str(s).strip() for s in (normalized.get("reel_overlays") or []) if str(s).strip()][:5]
        normalized["reel_overlays"] = [" ".join(line.split()[:6]) for line in overlays]
        text = str(normalized.get("prompt_text", "")).strip()
        text = _TIMED_VIDEO.sub("Create a short Reel", text)
        text = _SLIDE_COUNT.sub("sequential scenes", text)
        text = _CAROUSEL.sub("Reel", text)
        normalized["prompt_text"] = text
        return normalized
    return single_image_variant(normalized)

PROMPT_TYPE_DESCRIPTIONS = {
    "educational": "Teach one valuable lesson or insight, actionable, with a strong specific hook.",
    "storytelling": "Tell a relatable story with a turning point and a lesson tied to the brand.",
    "social_proof": "Show the work itself: a before/after of the creative process or a sample output. Never invent customers, testimonials or results.",
    "promotional": "Soft promotion with clear value and urgency, never hard-selling.",
    "engagement": "Ask a polarizing or thought-provoking question that invites comments.",
    "behind_scenes": "Reveal how the business works, mistakes, or the human side of the team.",
    "user_generated": "Spotlight a customer's content or experience, encouraging UGC.",
    "trend_jacking": "Ride a current trend/event with a brand-relevant twist.",
}


class PromptForge:
    """Generate creator briefs for either a single image or an assembled Reel."""

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
You are a top-tier viral content strategist specializing in explosive social media growth (Instagram & LinkedIn).
Generate {count} distinct content prompts. Use a mix of single images and short Reels where suitable.
When count is 5, include at least 2 single images and 2 Reels so both render paths stay exercised. When count is smaller, choose the format that best fits the idea.

Business profile:
{fence(business_profile, 6000)}

Audience map:
{fence(audience_map, 6000)}

Today's trends & competitor insights:
{fence(trend_data, 6000)}

Allowed prompt types:
{type_descriptions}

Learned weights (dimension::item → weight, higher = more proven with this audience):
{fence(weights, 4000)}

{fence_text(hints_text)}

VIRAL HOOK PSYCHOLOGY:
Every prompt must choose an explicit "hook_type":
- "contrarian": Challenge conventional wisdom or reveal a common mistake ("Why Most AI Ads Look Generic").
- "roi_proof": Concrete, checkable value: what the studio delivers and how fast it works, using only facts from the business profile ("Brand visuals without the agency wait").
- "curiosity_gap": A surprising insight or hidden mechanism ("The 1 Prompt Structure Agencies Hide").
- "before_after": Dramatic contrast between old manual way vs new automated way ("Agency in 2024 vs Autonomous AI Studio in 2026").
- "framework": Actionable 3-4 step blueprint or checklist ("The 3-Step AI Creative Engine").

VISUAL SCENE FOR FLUX.1:
Provide a photographic, cinematic scene in "visual_scene" describing:
- Tangible physical subject & environment (e.g. minimalist high-tech studio desk, glowing indigo rim light, 35mm lens, f/1.8).
- ZERO text, ZERO letters, ZERO words inside the image. The image model draws gibberish on anything that carries writing, so do not put computer or phone screens, logos, signs, posters, labels, books or documents in the scene; use objects, materials, hands, light and architecture instead.

CAPTION ARCHITECTURE:
Provide a complete, ready-to-post "caption" with:
- Opening line repeating the hook
- 3-4 clear value points / insights separated by newlines
- Strong CTA to save or comment ("Bookmark this before your next campaign")
- Tiered hashtags: 3 hyper-niche, 3 industry, 2 broad discovery

FORMAT RULES:
- Choose either "single image" or "reel" for each idea. Never choose carousel.
- For "single image", describe one finished composition.
- For "reel", provide 3-5 sequential, visually distinct scene descriptions in reel_beats and 3-5 short on-screen lines in reel_overlays (max 6 words each). Start with the finished result or strongest visual in the first beat. Build a simple beginning, middle, and end that can be assembled from stills with gentle motion.
- Avoid relying on generated text inside an image. On-screen lines are overlaid by the renderer.

TRUTHFULNESS (strict):
- Never invent statistics, percentages, multipliers (3x), money amounts, client counts, client names, testimonials or results such as ROI, CTR or conversion lifts. These posts appear on a real business account.
- Do not refer to case studies, testimonials, awards, guarantees or "our clients": none are supplied. Do not tell viewers to look for proof you cannot show.
- Do not promise delivery times ("in 48 hours") or quantities ("10 concepts") either: the studio's real turnaround is not known to you.
- A number is allowed only if it appears in the business profile, audience map or trend data above.
- The only facts about the business you may state as true are those under "verified_facts" in the business profile. If there are none, write about the craft, the process and a clear point of view without describing specific services, prices, clients or results.
- Persuade with what the studio does, how the process works, a clear point of view, and a next step for the viewer.

Respond ONLY with JSON:
[
  {{
    "prompt_type": "educational",
    "hook_type": "contrarian" | "roi_proof" | "curiosity_gap" | "before_after" | "framework",
    "title": "short working title",
    "hook": "exact opening hook (max 12 words)",
    "visual_scene": "photographic scene description for image generation (no text)",
    "caption": "complete ready-to-post Instagram caption with value points and CTA",
    "prompt_text": "complete creator brief detailing strategy, execution, and audience psychology (150-350 words)",
    "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#tag6", "#tag7", "#tag8"],
    "platform_focus": "instagram" | "linkedin" | "both",
    "format": "single image" | "reel",
    "reel_beats": ["scene 1 visual", "scene 2 visual", "scene 3 visual"],
    "reel_overlays": ["short line 1", "short line 2", "short line 3"],
    "angle": "one sentence strategic angle",
    "audience_segment": "persona name from audience map"
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
                "hook_type": str(v.get("hook_type", "contrarian")),
                "title": str(v.get("title", "")),
                "hook": str(v.get("hook", "")),
                "visual_scene": str(v.get("visual_scene", "")),
                "caption": str(v.get("caption", "")),
                "prompt_text": str(v.get("prompt_text", "")),
                "hashtags": [str(h) for h in v.get("hashtags", []) if str(h).startswith("#")],
                "platform_focus": str(v.get("platform_focus", "both")),
                "format": str(v.get("format", "single image")),
                "angle": str(v.get("angle", "")),
                "audience_segment": str(v.get("audience_segment", "") or ""),
                "reel_beats": [str(b).strip() for b in v.get("reel_beats", []) if str(b).strip()][:5],
                "reel_overlays": [str(b).strip() for b in v.get("reel_overlays", []) if str(b).strip()][:5],
            }
            for v in data
            if isinstance(v, dict) and v.get("prompt_text")
        ]
        if not variants:
            raise ValueError("prompt forge produced no usable variants")
        variants = [normalize_variant(v) for v in variants]
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
Preserve the current format (single image or Reel). Do not change it to a carousel.
For a Reel, keep 3-5 sequential scene beats and short on-screen overlay lines.

Current prompt:
{fence(best_variant)}

Critique from the predictor (address ALL points):
{hints}

Business profile:
{fence(business_profile, 5000)}

Audience map:
{fence(audience_map, 4000)}

Trends:
{fence(trend_data, 4000)}

Respond ONLY with the improved prompt as JSON with the same fields:
{{
  "prompt_type": "...",
  "hook_type": "contrarian" | "roi_proof" | "curiosity_gap" | "before_after" | "framework",
  "title": "...",
  "hook": "...",
  "visual_scene": "photographic scene for FLUX.1 (no text)",
  "caption": "ready-to-post Instagram caption",
  "prompt_text": "...",
  "hashtags": ["..."],
  "platform_focus": "...",
  "format": "{best_variant.get('format', 'single image')}",
  "reel_beats": ["scene 1 visual", "scene 2 visual", "scene 3 visual"],
  "reel_overlays": ["short line 1", "short line 2", "short line 3"],
  "angle": "...",
  "audience_segment": "..."
}}
"""
        try:
            data = self.llm.run_json(prompt, label="prompt-forge-refine", temperature=0.6, expect=(dict,))
            if not isinstance(data, dict) or not data.get("prompt_text"):
                raise ValueError("refined prompt invalid")
            return normalize_variant({
                "prompt_type": str(data.get("prompt_type", best_variant.get("prompt_type", "educational"))),
                "hook_type": str(data.get("hook_type", best_variant.get("hook_type", "contrarian"))),
                "title": str(data.get("title", best_variant.get("title", ""))),
                "hook": str(data.get("hook", best_variant.get("hook", ""))),
                "visual_scene": str(data.get("visual_scene", best_variant.get("visual_scene", ""))),
                "caption": str(data.get("caption", best_variant.get("caption", ""))),
                "prompt_text": str(data.get("prompt_text", "")),
                "hashtags": [str(h) for h in data.get("hashtags", best_variant.get("hashtags", []))],
                "platform_focus": str(data.get("platform_focus", best_variant.get("platform_focus", "both"))),
                "format": str(best_variant.get("format", data.get("format", "single image"))),
                "angle": str(data.get("angle", best_variant.get("angle", ""))),
                "audience_segment": str(data.get("audience_segment", "") or best_variant.get("audience_segment", "")),
                "reel_beats": [
                    str(b).strip()
                    for b in data.get("reel_beats", best_variant.get("reel_beats", []))
                    if str(b).strip()
                ][:5],
                "reel_overlays": [
                    str(b).strip()
                    for b in data.get("reel_overlays", best_variant.get("reel_overlays", []))
                    if str(b).strip()
                ][:5],
            })
        except Exception as exc:
            # Refinement is an optimization: keep the previous honest variant
            # rather than fabricating, but make the degradation visible.
            logger.warning(
                "::warning::Prompt refinement failed (%s) — using unrefined variant %s",
                exc, best_variant.get("title", ""),
            )
            return best_variant
