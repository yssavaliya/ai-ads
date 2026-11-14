"""Lead scoring: is this business worth our time, and WHY?

Every dimension is computed from data we actually have (follower count, recent captions and engagement, contact
details, niche) and returned with a plain-language note, so a human (or the digest page) can see why a lead was
selected. Nothing here is a guess: a dimension with no data scores neutral and is listed under ``data_gaps``.

Dimensions (100 points): engagement_gap 25, audience_size 15, activity 15, buying_intent 15,
content_potential 15, reachability 10, competition 5.
"""
from __future__ import annotations

import logging
import re
import statistics
from typing import Any

from src.learning.auto_learner import engagement_rate

logger = logging.getLogger(__name__)

MAX_POINTS = {
    "engagement_gap": 25,
    "audience_size": 15,
    "activity": 15,
    "buying_intent": 15,
    "content_potential": 15,
    "reachability": 10,
    "competition": 5,
}
TARGET_SCORE = 70
MAYBE_SCORE = 55

# Words in a business's own captions/bio that show it sells things to individual buyers (intent to convert).
_INTENT_RE = re.compile(
    r"\b(?:order|orders|dm to|dm us|book(?:ing)?|shop|buy|price|pricing|sale|offer|discount|delivery|whatsapp|"
    r"call us|enquir(?:y|ies)|custom|catalogue|catalog|link in bio|available|now open|pre-?order)\b",
    re.IGNORECASE,
)
# Products that photograph well: advertising creative has an immediate, visible effect for them.
_VISUAL_RE = re.compile(
    r"\b(?:bakery|cake|cafe|restaurant|food|sweets?|chocolate|jewel(?:l)?ery|jewel(?:l)?er|fashion|boutique|apparel|saree|"
    r"decor|furniture|interior|handmade|craft|gift|flower|florist|skincare|beauty|salon|studio|candle|ceramic|art|photograph|"
    r"toys?|bags?|shoes?|watch|perfume|organic)\b",
    re.IGNORECASE,
)


def _points(name: str, value: float, note: str) -> dict[str, Any]:
    return {"points": round(max(0.0, min(float(MAX_POINTS[name]), value)), 1), "max": MAX_POINTS[name], "note": note}


def _caption_text(posts: list[dict[str, Any]], social_data: dict[str, Any]) -> str:
    parts = [str(social_data.get("biography") or "")]
    parts.extend(str(p.get("caption") or "") for p in posts[:12])
    return " ".join(parts)


class LeadScorer:
    """Scores a candidate's advertising opportunity: a business with a real audience and a product people can buy,
    active on social media, whose engagement is weak (so better creative has room to move the numbers), and that
    we can actually reach."""

    def score(
        self,
        candidate: dict[str, Any],
        social_data: dict[str, Any],
        min_followers: int = 500,
        max_followers: int = 500_000,
        niche_competitors: int | None = None,
    ) -> dict[str, Any]:
        if niche_competitors is None:
            niche_competitors = candidate.get("niche_competitors")
        followers = int(social_data.get("follower_count") or 0)
        posts = [p for p in (social_data.get("posts") or []) if isinstance(p, dict)]

        if followers < min_followers:
            return self._rejected(followers, "too small: under min follower threshold")
        if followers > max_followers:
            return self._rejected(followers, "too big: likely already has an agency")

        breakdown: dict[str, dict[str, Any]] = {}
        reasons: list[str] = []
        gaps: list[str] = []

        # 1. audience size: 5x-30x the minimum is the sweet spot (big enough to matter, small enough to lack an agency)
        ratio = followers / max(1, min_followers)
        size = 15.0 if 3 <= ratio <= 30 else 11.0 if ratio < 3 else 9.0
        breakdown["audience_size"] = _points("audience_size", size, f"{followers} followers")
        reasons.append(f"{followers} followers (sweet spot)" if size == 15.0 else f"{followers} followers")

        # 2. activity: a business that posts regularly will actually use new creative
        active = [p for p in posts if p.get("caption")]
        if len(active) >= 3:
            breakdown["activity"] = _points("activity", 15, f"{len(active)} recent posts")
            reasons.append(f"active: {len(active)} recent posts")
        elif active:
            breakdown["activity"] = _points("activity", 8, f"only {len(active)} recent posts")
            reasons.append("limited recent activity")
        else:
            breakdown["activity"] = _points("activity", 3, "no recent captions seen")
            reasons.append("limited recent activity")
            gaps.append("no recent posts were visible")

        # 3. engagement gap: low engagement relative to followers = headroom for better creative
        rates = [r for r in (engagement_rate(p, followers) for p in posts) if r > 0]
        if rates:
            median_rate = statistics.median(rates)
            if median_rate < 0.03:
                gap, note = 25.0, f"low engagement rate ({median_rate:.2%}) — needs content help"
            elif median_rate < 0.06:
                gap, note = 17.0, f"below-average engagement ({median_rate:.2%})"
            else:
                gap, note = 5.0, f"healthy engagement ({median_rate:.2%})"
            reasons.append(note)
        else:
            gap, note = 12.0, "no engagement data yet"
            reasons.append(note)
            gaps.append("engagement could not be measured")
        breakdown["engagement_gap"] = _points("engagement_gap", gap, note)

        # 4. buying intent: do they sell to individual buyers and ask for the sale?
        text = _caption_text(posts, social_data)
        intent_hits = len({m.lower() for m in _INTENT_RE.findall(text)})
        has_phone = bool(candidate.get("phone"))
        intent = min(15.0, intent_hits * 3.0 + (3.0 if has_phone else 0.0))
        breakdown["buying_intent"] = _points(
            "buying_intent", intent if text.strip() else 6.0,
            f"{intent_hits} selling signals in captions/bio" if text.strip() else "no captions to judge intent",
        )
        if intent_hits:
            reasons.append(f"sells directly ({intent_hits} buying signals in captions)")
        if not text.strip():
            gaps.append("buying intent could not be judged")

        # 5. content potential: visual products + a site that shows what they sell
        niche_text = " ".join([str(candidate.get("niche") or ""), str(candidate.get("name") or ""), text[:800]])
        visual = bool(_VISUAL_RE.search(niche_text))
        potential = (10.0 if visual else 4.0) + (5.0 if candidate.get("website_url") else 0.0)
        breakdown["content_potential"] = _points(
            "content_potential", potential,
            ("visual product category" if visual else "category not obviously visual")
            + ("; has website" if candidate.get("website_url") else "; no website"),
        )
        if candidate.get("website_url"):
            reasons.append("has website (full profile available)")
        if visual:
            reasons.append("visual product category (ads show results quickly)")

        # 6. reachability: can we contact them?
        reach = (6.0 if candidate.get("email") or candidate.get("contact_email") else 0.0)
        reach += 2.0 if candidate.get("website_url") else 0.0
        reach += 2.0 if (candidate.get("instagram_handle") or candidate.get("linkedin_handle")) else 0.0
        breakdown["reachability"] = _points(
            "reachability", reach, "contact email found" if reach >= 6 else "no contact email found"
        )
        if candidate.get("instagram_handle") or candidate.get("linkedin_handle"):
            reasons.append("has social handle")
        if reach < 6:
            gaps.append("no contact email")

        # 7. competition: how crowded is this niche in the area (more local rivals = harder to stand out)
        if niche_competitors is None:
            comp, note = 3.0, "competition level unknown"
            gaps.append("local competition was not measured")
        else:
            comp = 5.0 if niche_competitors <= 3 else 3.0 if niche_competitors <= 8 else 1.0
            note = f"{niche_competitors} similar local businesses found"
        breakdown["competition"] = _points("competition", comp, note)

        total = round(sum(item["points"] for item in breakdown.values()))
        score = min(total, 100)
        return {
            "score": score,
            "followers": followers,
            "reasons": reasons,
            "eligible": True,
            "breakdown": breakdown,
            "decision": "target" if score >= TARGET_SCORE else "maybe" if score >= MAYBE_SCORE else "skip",
            "why_target": self._why(candidate, breakdown, score),
            "data_gaps": gaps,
        }

    @staticmethod
    def _rejected(followers: int, reason: str) -> dict[str, Any]:
        return {
            "score": 0, "followers": followers, "reasons": [reason], "eligible": False,
            "breakdown": {}, "decision": "skip", "why_target": reason, "data_gaps": [],
        }

    @staticmethod
    def _why(candidate: dict[str, Any], breakdown: dict[str, dict[str, Any]], score: int) -> str:
        ranked = sorted(breakdown.items(), key=lambda kv: kv[1]["points"] / kv[1]["max"], reverse=True)
        strong = [f"{name.replace('_', ' ')} ({item['note']})" for name, item in ranked[:2] if item["points"] / item["max"] >= 0.6]
        weak = ranked[-1]
        parts = [f"{candidate.get('name') or 'This business'} scores {score}/100."]
        if strong:
            parts.append("Strongest: " + "; ".join(strong) + ".")
        if weak[1]["points"] / weak[1]["max"] < 0.5:
            parts.append(f"Weakest: {weak[0].replace('_', ' ')} ({weak[1]['note']}).")
        return " ".join(parts)
