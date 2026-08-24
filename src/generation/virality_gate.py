from __future__ import annotations

import logging
from typing import Any

from src.generation.prompt_forge import PromptForge
from src.intelligence.virality_predictor import ViralityPredictor
from src.shared.config import settings

logger = logging.getLogger(__name__)


class ViralityGate:
    """Iterative generation loop: generate variants → score → if best is below
    threshold, refine with predictor hints → repeat. Falls back to best available."""

    def __init__(
        self,
        threshold: int | None = None,
        max_attempts: int | None = None,
        forge: PromptForge | None = None,
        predictor: ViralityPredictor | None = None,
    ) -> None:
        self.threshold = threshold if threshold is not None else settings.virality_threshold
        self.max_attempts = max_attempts if max_attempts is not None else settings.max_regeneration_attempts
        self.forge = forge or PromptForge()
        self.predictor = predictor or ViralityPredictor()

    def run(
        self,
        business_profile: dict[str, Any],
        audience_map: dict[str, Any],
        trend_data: dict[str, Any],
        prompt_types: list[str],
        weights: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns up to 3 prompts that meet the threshold (diverse types),
        or the best available ones. Learned `weights` bias selection toward
        proven format/angle combos; `history` grounds the predictor in real
        past performance."""
        selected: dict[str, dict[str, Any]] = {}
        best_overall: dict[str, Any] | None = None
        best_score = -1
        hints: list[str] = []

        for attempt in range(1, self.max_attempts + 1):
            variants = self.forge.generate_variants(
                business_profile=business_profile,
                audience_map=audience_map,
                trend_data=trend_data,
                prompt_types=prompt_types,
                weights=weights,
                improvement_hints=hints,
                count=5,
            )
            if not variants:
                logger.warning("No variants generated on attempt %d", attempt)
                break

            scored = []
            for v in variants:
                prediction = self.predictor.score(
                    v["prompt_text"],
                    business_profile,
                    audience_map,
                    trend_data,
                    v.get("prompt_type", "educational"),
                    history=history,
                )
                v["virality_score"] = prediction["virality_score"]
                v["virality_label"] = prediction["virality_label"]
                v["virality_probability"] = prediction["virality_probability"]
                v["key_factors"] = prediction["key_factors"]
                v["risks"] = prediction["risks"]
                v["reasoning"] = prediction["reasoning"]
                v["improvement_hints"] = prediction.get("improvement_hints", [])
                v["generation_attempt"] = attempt
                scored.append(v)

            for v in scored:
                score = v["virality_score"]
                combined = score + self._weight_boost(weights, v)
                v["_combined"] = combined
                if combined > best_score:
                    best_score = combined
                    best_overall = v
                if score >= self.threshold:
                    ptype = v.get("prompt_type", "educational")
                    if ptype not in selected or selected[ptype].get("_combined", 0) < combined:
                        selected[ptype] = v

            if len(selected) >= 3:
                logger.info(
                    "Virality gate passed at attempt %d: %d types above threshold %d",
                    attempt, len(selected), self.threshold,
                )
                break

            hints = self._collect_hints(scored)
            if not hints:
                logger.warning("No improvement hints available; stopping refinement")
                break

            best_variant = max(scored, key=lambda v: v["virality_score"])
            refined = self.forge.refine_with_hints(
                best_variant, hints, business_profile, audience_map, trend_data
            )
            if refined:
                prediction = self.predictor.score(
                    refined["prompt_text"],
                    business_profile,
                    audience_map,
                    trend_data,
                    refined.get("prompt_type", "educational"),
                )
                refined["virality_score"] = prediction["virality_score"]
                refined["virality_label"] = prediction["virality_label"]
                refined["virality_probability"] = prediction["virality_probability"]
                refined["key_factors"] = prediction["key_factors"]
                refined["risks"] = prediction["risks"]
                refined["reasoning"] = prediction["reasoning"]
                refined["improvement_hints"] = prediction.get("improvement_hints", [])
                refined["generation_attempt"] = attempt
                refined_combined = refined["virality_score"] + self._weight_boost(weights, refined)
                refined["_combined"] = refined_combined
                if refined["virality_score"] >= self.threshold:
                    ptype = refined.get("prompt_type", "educational")
                    if ptype not in selected or selected[ptype].get("_combined", 0) < refined_combined:
                        selected[ptype] = refined
                if refined_combined > best_score:
                    best_score = refined_combined
                    best_overall = refined

        results = list(selected.values())
        if len(results) < 3 and best_overall:
            if best_overall.get("prompt_type") not in {r.get("prompt_type") for r in results}:
                results.append(best_overall)
            else:
                results = sorted(results + [best_overall], key=lambda v: v["virality_score"], reverse=True)[:3]
        if best_overall and len(results) < 3:
            fillers = sorted(
                [best_overall] + results, key=lambda v: v["virality_score"], reverse=True
            )
            results = fillers[:3]

        logger.info(
            "Virality gate final: %d prompts, best score=%d (threshold=%d)",
            len(results), max((v["virality_score"] for v in results), default=0), self.threshold,
        )
        return results[:3]

    @staticmethod
    def _weight_boost(
        weights: dict[str, Any] | None, v: dict[str, Any]
    ) -> float:
        """Small nudge so proven combos (weight > 1) win ties over unproven or
        penalized ones (weight < 1). Keeps the virality score as the primary
        signal but makes the self-learning loop actually steer generation."""
        if not weights:
            return 0.0
        total, n = 0.0, 0
        for dim, key in (
            ("formats", v.get("format")),
            ("topics", v.get("prompt_type")),
            ("audience_segments", v.get("audience_segment")),
        ):
            if key:
                val = weights.get(f"{dim}::{key}")
                if val is not None:
                    total += val
                    n += 1
        avg = (total / n) if n else 1.0
        return 6.0 * (avg - 1.0)

    @staticmethod
    def _collect_hints(scored: list[dict[str, Any]]) -> list[str]:
        hints: list[str] = []
        for v in scored:
            hints.extend(v.get("improvement_hints", []) or [])
        seen: set[str] = set()
        unique: list[str] = []
        for h in hints:
            key = h.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(h)
        return unique[:6]