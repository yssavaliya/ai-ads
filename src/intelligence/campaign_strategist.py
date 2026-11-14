"""Campaign strategy stage: decide WHAT to advertise and WHY before any creative is written.

An agency does not hand a copywriter "make 5 posts". A strategist first fixes the objective, the persona, the
customer problem, the emotional trigger and the differentiator, then weighs several candidate angles against an
explicit rubric and picks one. This module is that step.

Design rules:
- The LLM proposes angles and rates them 1-5 on a fixed rubric; the TOTAL and the WINNER are computed here, in code,
  so the decision is reproducible and explainable ("A beat B by 7 points: stronger audience fit").
- Angles that carry unverifiable numeric/proof claims are dropped (same guard as the creative stage).
- A bounded bonus/penalty from the learned hook weights lets real results nudge the choice, never decide it.
- On any failure the caller gets None and the existing pipeline runs unchanged: no invented strategy.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.generation.claims import unsupported_claims
from src.shared.llm_client import LLMClient, fence, fence_text

logger = logging.getLogger(__name__)

HOOK_TYPES = ("contrarian", "roi_proof", "curiosity_gap", "before_after", "framework")
OBJECTIVES = ("awareness", "engagement", "consideration", "lead_generation")
TRIGGERS = ("curiosity", "aspiration", "relief", "pride", "belonging", "fear_of_missing_out", "surprise")
FORMATS = ("reel", "single image")

# rubric criterion -> weight (sums to 1.0). Rated 1-5 by the model, combined here.
RUBRIC: dict[str, float] = {
    "audience_fit": 0.25,      # speaks to the chosen persona's real problem
    "conversion_pull": 0.20,   # makes the viewer want the stated next step
    "usp_fit": 0.20,           # leans on a verified differentiator, not a generic claim
    "novelty": 0.15,           # not what competitors and our own recent posts already did
    "claim_safety": 0.10,      # can be told truthfully from verified facts only
    "producibility": 0.10,     # can be rendered well with image/Reel tools we have
}
MAX_LEARNING_ADJUSTMENT = 4.0  # points on a 0-100 scale
MIN_VALID_ANGLES = 2


def _clean(value: Any, limit: int = 240) -> str:
    return " ".join(str(value or "").split())[:limit]


def _one_of(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = _clean(value, 40).lower().replace(" ", "_")
    return text if text in allowed else default


def _rate(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 1.0 <= number <= 5.0 else None


def rubric_total(scores: dict[str, Any]) -> float | None:
    """0-100 weighted rubric total, or None when any criterion is missing or out of range."""
    total = 0.0
    for criterion, weight in RUBRIC.items():
        rating = _rate((scores or {}).get(criterion))
        if rating is None:
            return None
        total += weight * ((rating - 1.0) / 4.0)
    return round(total * 100.0, 1)


def learning_adjustment(hook_type: str, weights: dict[str, Any] | None) -> float:
    """Bounded points from the learned hook weight (Beta mean: 0.5 = no evidence). Never decisive."""
    if not weights:
        return 0.0
    weight = weights.get(f"hook_types::{hook_type}")
    if not isinstance(weight, (int, float)):
        return 0.0
    return round(max(-MAX_LEARNING_ADJUSTMENT, min(MAX_LEARNING_ADJUSTMENT, (float(weight) - 0.5) * 20.0)), 2)


def _explain(chosen: dict[str, Any], runner_up: dict[str, Any] | None) -> str:
    best = sorted(chosen["scores"].items(), key=lambda kv: -float(kv[1]))[:2]
    strengths = " and ".join(name.replace("_", " ") for name, _ in best)
    text = f"Angle '{chosen['name']}' scored {chosen['total']:.0f}/100, strongest on {strengths}."
    if runner_up is not None:
        margin = chosen["total"] - runner_up["total"]
        text += f" It beat '{runner_up['name']}' ({runner_up['total']:.0f}) by {margin:.0f} points."
    if chosen.get("learning_adjustment"):
        text += f" Learned results moved it {chosen['learning_adjustment']:+.1f} points."
    return text


def _hypothesis(chosen: dict[str, Any], runner_up: dict[str, Any], kpis: list[str], problem: str) -> str:
    """Built from the WINNING angle in code. The model writes its hypothesis before the winner is known, so its
    own text often argues for an angle that lost (seen in production)."""
    metric = kpis[0] if kpis else "saves"
    text = (
        f"If we lead with the '{chosen['name']}' angle ({chosen['hook_type'].replace('_', ' ')} hook), "
        f"then {metric} per reach will beat this account's recent baseline"
    )
    if problem:
        text += f", because it speaks directly to: {problem}"
    return text + f". Runner-up to test next: '{runner_up['name']}'."


def score_angles(
    raw_angles: Any, grounding: str = "", weights: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Validate, score and rank the model's angles. Invalid or claim-carrying angles are dropped."""
    if not isinstance(raw_angles, list):
        return []
    angles: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_angles[:6]):
        if not isinstance(raw, dict):
            continue
        rated = {k: _rate((raw.get("scores") or {}).get(k)) for k in RUBRIC}
        scores = {k: v for k, v in rated.items() if v is not None}
        total = rubric_total(scores)
        name, pitch = _clean(raw.get("name"), 80), _clean(raw.get("pitch"), 300)
        if total is None or len(scores) != len(RUBRIC) or not name or not pitch:
            continue
        bad = unsupported_claims({"hook": pitch, "title": name, "caption": _clean(raw.get("proof_needed"), 200)}, grounding)
        if bad:
            logger.warning("::warning::Strategy angle %r dropped: unsupported claims %s", name, bad)
            continue
        hook_type = _one_of(raw.get("hook_type"), HOOK_TYPES, "framework")
        adjustment = learning_adjustment(hook_type, weights)
        angles.append({
            "id": _clean(raw.get("id") or chr(ord("A") + position), 4),
            "name": name,
            "hook_type": hook_type,
            "pitch": pitch,
            "scores": scores,
            "rubric_total": total,
            "learning_adjustment": adjustment,
            "total": round(total + adjustment, 1),
        })
    angles.sort(key=lambda a: (-a["total"], a["id"]))
    return angles


class CampaignStrategist:
    """One small LLM call -> a structured, explainable campaign brief."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    def plan(
        self,
        business_profile: dict[str, Any],
        audience_map: dict[str, Any],
        trend_data: dict[str, Any],
        weights: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        benchmarks: dict[str, Any] | None = None,
        experiment_hints: list[str] | None = None,
    ) -> dict[str, Any] | None:
        grounding = json.dumps([business_profile, audience_map, trend_data], default=str, ensure_ascii=False)
        try:
            payload = self.llm.run_json(
                self._prompt(business_profile, audience_map, trend_data, weights, history, benchmarks, experiment_hints),
                label="campaign-strategist", temperature=0.4, expect=(dict,),
            )
        except Exception as exc:  # noqa: BLE001 - strategy is an enhancement; the pipeline must still run
            logger.warning("::warning::Campaign strategy unavailable (%s); creative runs without a brief", exc)
            return None
        return self.build_brief(payload, grounding, weights)

    @staticmethod
    def build_brief(payload: Any, grounding: str = "", weights: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        angles = score_angles(payload.get("angles"), grounding, weights)
        if len(angles) < MIN_VALID_ANGLES:
            logger.warning("::warning::Campaign strategy produced %d valid angles (need %d); ignoring it",
                           len(angles), MIN_VALID_ANGLES)
            return None
        chosen, runner_up = angles[0], angles[1]
        kpis = [_clean(k, 40) for k in (payload.get("kpis") or []) if _clean(k, 40)][:4]
        return {
            "objective": _one_of(payload.get("objective"), OBJECTIVES, "engagement"),
            "persona": _clean(payload.get("target_persona"), 120),
            "customer_problem": _clean(payload.get("customer_problem")),
            "emotional_trigger": _one_of(payload.get("emotional_trigger"), TRIGGERS, "curiosity"),
            "differentiator": _clean(payload.get("differentiator")),
            "platform": "instagram",
            "format": _one_of(payload.get("format"), FORMATS, "reel").replace("_", " "),
            "cta": _clean(payload.get("cta"), 160),
            "kpis": kpis,
            "test_hypothesis": _hypothesis(chosen, runner_up, kpis, _clean(payload.get("customer_problem"), 120)),
            "angles": angles,
            "chosen": chosen,
            "rationale": _explain(chosen, runner_up),
            "source": "llm",
        }

    @staticmethod
    def _prompt(
        profile: dict[str, Any],
        audience: dict[str, Any],
        trends: dict[str, Any],
        weights: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
        benchmarks: dict[str, Any] | None = None,
        experiment_hints: list[str] | None = None,
    ) -> str:
        tested = "\n".join(f"- {h}" for h in (experiment_hints or [])) or "No conclusive experiments yet."
        measured = "\n".join(f"- {o}" for o in (benchmarks or {}).get("observations", [])[:6]) or (
            "No verified competitor data: do not assume anything about competitors."
        )
        recent = "\n".join(
            f"- [{h.get('outcome') or 'unknown'}] {(h.get('prompt_text') or '')[:120]}" for h in (history or [])[:5]
        ) or "No measured results yet."
        top_weights = dict(sorted((weights or {}).items(), key=lambda kv: -float(kv[1]) if isinstance(kv[1], (int, float)) else 0)[:6])
        return f"""
You are the head of strategy at a performance-minded advertising agency. Before any creative is written, decide the campaign.

Business profile (only "verified_facts" may be stated as true):
{fence(profile, 1800)}

Audience map:
{fence(audience, 1600)}

Trends (model knowledge, not live data - adapt, never copy blindly):
{fence(trends, 1200)}

Competitor benchmarks (MEASURED from scraped public posts; the only competitor facts you may use):
{fence_text(measured, 900)}

Conclusive experiments (designed tests on this account; weigh them heavily):
{fence_text(tested, 600)}

Recent results for this account:
{fence_text(recent, 700)}

Learned weights (higher = more proven):
{fence(top_weights, 400)}

Decide ONE campaign. Then propose exactly 3 different creative angles and rate each 1-5 (5 = best) on:
audience_fit, conversion_pull, usp_fit, novelty, claim_safety, producibility.
Be critical: rate down angles that are generic, repeat recent posts, or need proof we do not have.
Never invent statistics, client counts, results, prices or delivery times.

Respond ONLY with JSON:
{{
  "objective": "awareness|engagement|consideration|lead_generation",
  "target_persona": "name from the audience map",
  "customer_problem": "the specific problem this ad addresses",
  "emotional_trigger": "curiosity|aspiration|relief|pride|belonging|fear_of_missing_out|surprise",
  "differentiator": "what makes this studio different, from verified facts only",
  "format": "reel|single image",
  "cta": "the one thing the viewer should do next",
  "kpis": ["saves", "shares", "profile visits"],
  "angles": [
    {{"id": "A", "name": "short angle name", "hook_type": "contrarian|roi_proof|curiosity_gap|before_after|framework",
      "pitch": "one sentence: the hook and the idea",
      "scores": {{"audience_fit": 1, "conversion_pull": 1, "usp_fit": 1, "novelty": 1, "claim_safety": 1, "producibility": 1}}}}
  ]
}}
"""


def brief_for_creative(brief: dict[str, Any] | None) -> str:
    """The brief as instructions for the creative stage (empty string when there is none)."""
    if not brief:
        return ""
    chosen = brief["chosen"]
    lines = [
        "CAMPAIGN BRIEF (decided by the strategy team - every concept must serve it):",
        f"- Objective: {brief['objective']}; platform: {brief['platform']}; preferred format: {brief['format']}",
        f"- Target persona: {brief['persona']}",
        f"- Customer problem to address: {brief['customer_problem']}",
        f"- Emotional trigger: {brief['emotional_trigger']}",
        f"- Differentiator (verified): {brief['differentiator']}",
        f"- Winning angle: {chosen['name']} ({chosen['hook_type']}): {chosen['pitch']}",
        f"- Call to action: {brief['cta']}",
        "- At least two concepts must execute the winning angle in different ways; the rest may test the runner-up angle.",
    ]
    return "\n".join(line for line in lines if line.split(": ", 1)[-1].strip())


def strategy_summary(brief: dict[str, Any] | None) -> dict[str, Any]:
    """Compact, public-safe record attached to each creative (shown on the hub, kept for learning)."""
    if not brief:
        return {}
    chosen = brief["chosen"]
    return {
        "objective": brief["objective"],
        "persona": brief["persona"],
        "problem": brief["customer_problem"],
        "trigger": brief["emotional_trigger"],
        "angle": chosen["name"],
        "angle_hook_type": chosen["hook_type"],
        "angle_score": chosen["total"],
        "rationale": brief["rationale"],
        "cta": brief["cta"],
        "kpis": list(brief["kpis"]),
        "hypothesis": brief["test_hypothesis"],
        "alternatives": [{"name": a["name"], "score": a["total"]} for a in brief["angles"][1:]],
    }
