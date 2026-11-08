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
MIN_REEL_NARRATION_WORDS = 150
MAX_REEL_SCENES = 6


def _word_count(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def single_image_variant(variant: dict[str, Any]) -> dict[str, Any]:
    """Enforce the product's single static image format on generated prompts."""
    normalized = dict(variant)
    normalized["format"] = "single image"
    normalized["reel_beats"] = []
    normalized["reel_narration"] = []
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
    beats = [str(b).strip() for b in (normalized.get("reel_beats") or []) if str(b).strip()][:MAX_REEL_SCENES]
    if str(normalized.get("format", "")).strip().lower() in {"reel", "video"} and len(beats) == MAX_REEL_SCENES:
        overlays = [str(s).strip() for s in (normalized.get("reel_overlays") or []) if str(s).strip()][:MAX_REEL_SCENES]
        overlays = [" ".join(line.split()[:6]) for line in overlays]
        narration = [str(s).strip() for s in (normalized.get("reel_narration") or []) if str(s).strip()][:MAX_REEL_SCENES]
        hook = str(normalized.get("hook") or "").strip()
        # Do not turn short overlay text into a pretend voiceover. A Reel is valid
        # only when it has a spoken hook and enough narration for a useful runtime.
        if (
            not hook
            or len(overlays) != len(beats)
            or len(narration) != len(beats)
            or _word_count(narration) < MIN_REEL_NARRATION_WORDS
            or not narration[0].casefold().startswith(hook.casefold())
        ):
            return single_image_variant(normalized)
        normalized["format"] = "reel"
        normalized["reel_beats"] = beats
        normalized["reel_overlays"] = overlays
        normalized["reel_narration"] = narration
        normalized["reel_overlays"][0] = " ".join(hook.split()[:6])
        text = str(normalized.get("prompt_text", "")).strip()
        text = _TIMED_VIDEO.sub("Create a narrated Reel of at least 60 seconds", text)
        text = _SLIDE_COUNT.sub("sequential scenes", text)
        text = _CAROUSEL.sub("Reel", text)
        normalized["prompt_text"] = text
        return normalized
    return single_image_variant(normalized)

PROMPT_TYPE_DESCRIPTIONS = {
    "educational": "Teach one useful product-ad or creative lesson and demonstrate it on a concrete spec concept.",
    "storytelling": "Show a product brief becoming an ad concept, with a clear creative decision and lesson.",
    "social_proof": "Show the creative work itself as a clearly labeled spec concept. Never invent customers, testimonials or results.",
    "promotional": "Explain one verified studio offer and invite a relevant product brief; never invent urgency.",
    "engagement": "Ask a focused product-ad question founders or marketers can answer from experience.",
    "behind_scenes": "Show a real creative decision or process step that turns a product brief into an ad concept.",
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
You are a practical Instagram growth strategist for a small creative studio. Optimize for qualified non-follower reach, saves, shares, profile visits, and inquiries, not vague "virality" claims.
Generate {count} distinct content prompts that all belong to the business's positioning and content strategy. Use a mix of single images and Reels.
When count is 5, include at least 3 Reels and 2 single images so the batch tests discovery and portfolio formats. When count is smaller, prefer a Reel unless a still image is clearly the better demonstration.

Business profile:
{fence(business_profile, 6000)}

Audience map:
{fence(audience_map, 6000)}

Today's trends & competitor insights:
{fence(trend_data, 6000)}

Editorial strategy (a targeting hypothesis and content mandate, NOT evidence for customer, performance, or product claims):
{fence(business_profile.get("content_strategy", {}), 3000)}

Allowed prompt types:
{type_descriptions}

Learned weights (dimension::item → weight, higher = more proven with this audience):
{fence(weights, 4000)}

{fence_text(hints_text)}

VIRAL HOOK PSYCHOLOGY:
Every prompt must choose an explicit "hook_type":
- "contrarian": Challenge conventional wisdom or reveal a common mistake ("Why Most AI Ads Look Generic").
- "roi_proof": A concrete product benefit or creative value, using only verified facts and no promised performance ("One clear product benefit makes a stronger opening").
- "curiosity_gap": A surprising insight or hidden mechanism ("The 1 Prompt Structure Agencies Hide").
- "before_after": Show the transformation from a plain product brief to a finished spec ad concept.
- "framework": A short actionable breakdown of the decisions in this specific ad concept.

VISUAL SCENE FOR FLUX.1:
Provide a photographic, cinematic scene in "visual_scene" describing:
- Tangible physical subject & environment (e.g. minimalist high-tech studio desk, glowing indigo rim light, 35mm lens, f/1.8).
- ZERO text, ZERO letters, ZERO words inside the image. The image model draws gibberish on anything that carries writing, so do not put computer or phone screens, logos, signs, posters, labels, books or documents in the scene; use objects, materials, hands, light and architecture instead.

CAPTION ARCHITECTURE:
Provide a complete, ready-to-post "caption" with:
- Opening line repeating the hook
- One useful, concrete explanation of the ad or creative decision, with clear line breaks
- One specific CTA that invites a relevant product brief by DMing "AD"; vary the CTA naturally
- Tiered hashtags: 3 hyper-niche, 3 industry, 2 broad discovery

ACCOUNT RELEVANCE (mandatory):
- Every post must clearly demonstrate AI video ads, product visuals, social creatives, or the process of turning a product brief into an ad for founders, marketers, or creative agencies.
- Use the editorial strategy's audience and pillars as constraints. Keep the feed recognizable as one studio account.
- For concept art, label the idea "spec concept" in caption/on-screen text; do not imitate a real brand's identity or imply a client relationship.
- Reject unrelated anime/fandom art, generic sunsets/nature, generic motivation, generic finance advice, and trend posts without an advertising-creative connection.
- Avoid vague engagement bait. Ask one focused question a product founder could answer, or invite a DM with their product and goal.
- The opening hook must be specific to a product advertising problem or show the finished ad concept immediately. Avoid generic hooks such as "You won't believe this", "This changes everything", and unsupported ROI claims.
- Each Reel must serve one audience and one business goal. Tell one coherent story: finished spec ad first, the product/audience problem, three concrete creative choices, the reveal, then one CTA. Make the first spoken words a specific hook that works without sound as an overlay in the opening frame. Use natural conversational English, contractions, short sentences, varied rhythm, and a human point of view; avoid generic AI buzzwords and formal essay language. Narration must explain the visual choices rather than repeat overlay text. Make the concept useful enough to save or share, without asking for multiple actions.

FORMAT RULES:
- Choose either "single image" or "reel" for each idea. Never choose carousel.
- For "single image", describe one finished composition.
- For "reel", provide exactly 6 sequential, visually distinct scene descriptions in reel_beats, one short on-screen line per beat in reel_overlays (max 6 words each), and a separate spoken script in reel_narration with 2-3 short, natural sentences per beat. The first spoken words MUST be the exact hook text, and the first overlay is a short version of that hook. Write 150-175 spoken words total, enough for a narrated Reel of at least 60 seconds. End with one clear spoken CTA. reel_beats are visual directions and must never be used as narration. Do not use overlays as a substitute for narration.
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
    "reel_beats": ["scene 1 visual", "scene 2 visual", "scene 3 visual", "scene 4 visual", "scene 5 visual", "scene 6 visual"],
    "reel_overlays": ["short hook", "short line 2", "short line 3", "short line 4", "short line 5", "short line 6"],
    "reel_narration": ["spoken hook and natural commentary for beat 1", "spoken commentary for beat 2", "spoken commentary for beat 3", "spoken commentary for beat 4", "spoken commentary for beat 5", "spoken commentary and one CTA for beat 6"],
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
                "reel_beats": [str(b).strip() for b in v.get("reel_beats", []) if str(b).strip()][:MAX_REEL_SCENES],
                "reel_overlays": [str(b).strip() for b in v.get("reel_overlays", []) if str(b).strip()][:MAX_REEL_SCENES],
                "reel_narration": [str(b).strip() for b in v.get("reel_narration", []) if str(b).strip()][:MAX_REEL_SCENES],
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
You are an Instagram growth strategist for a small creative studio. Improve this content prompt to attract relevant non-followers and potential buyers, while preserving the brand focus.
Preserve the current format (single image or Reel). Do not change it to a carousel.
Keep the idea within the business's editorial strategy. Remove unrelated fandom, landscape, motivation, or finance themes unless they are clearly part of a product ad concept. Use a specific product-ad hook and one concrete DM AD or product-category CTA. Do not invent proof or results.
For a Reel, keep exactly 6 sequential visual scene beats, show the finished ad concept first, add one overlay per beat, and use a separate 6-part conversational voiceover in reel_narration. The first spoken words start with the exact hook. Write 150-175 spoken words total so the finished voiceover exceeds 60 seconds; use short sentences, contractions, and varied cadence. End with one clear CTA. Never use visual scene beats or overlays as voiceover.

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
  "reel_beats": ["scene 1 visual", "scene 2 visual", "scene 3 visual", "scene 4 visual", "scene 5 visual", "scene 6 visual"],
  "reel_overlays": ["short hook", "short line 2", "short line 3", "short line 4", "short line 5", "short line 6"],
  "reel_narration": ["spoken hook and conversational commentary for beat 1", "commentary for beat 2", "commentary for beat 3", "commentary for beat 4", "commentary for beat 5", "commentary and one CTA for beat 6"],
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
                "reel_narration": [
                    str(b).strip()
                    for b in data.get("reel_narration", best_variant.get("reel_narration", []))
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
