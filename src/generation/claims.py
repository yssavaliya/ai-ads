"""Guard against invented statistics in public ad copy.

The generator is an LLM with no access to real client results, so a caption such as "250% CTR lift in 48h",
"3x ROI" or "60 clients served" is fabricated unless the number appears in the inputs it was given. These
claims are published on a real business account, so a variant that contains an unsupported one is dropped
before it is scored or posted, and the generator is told what to avoid on the next attempt.
"""
from __future__ import annotations

import re
from typing import Any

# Fields that end up in front of the public: the caption, the image headline and the title.
PUBLIC_FIELDS = ("hook", "title", "caption")

_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_PATTERNS = (
    re.compile(rf"(?<![\w.]){_NUMBER}\s?%"),  # 35%, 2.5 %
    re.compile(rf"(?<![\w.]){_NUMBER}\s?[x×](?![A-Za-z0-9])"),  # 3x, 3 x, 3×
    re.compile(rf"(?:₹|\$|€|£|\bRs\.?|\bINR|\bUSD)\s?{_NUMBER}\s?(?:[kKmM]\b|lakhs?|crores?|million|billion)?"),
    re.compile(rf"(?<![\w.]){_NUMBER}\s?(?:[kKmM]\b|lakhs?|crores?|million|billion)?\s?(?:rupees|dollars|usd|inr)\b", re.IGNORECASE),
    re.compile(
        rf"(?<![\w.]){_NUMBER}\+?\s+(?:happy\s+|local\s+|global\s+)?"
        r"(?:clients?|brands?|customers?|businesses|studios|agencies|campaigns?|projects?|leads?|followers|sales|retailers?)\b",
        re.IGNORECASE,
    ),
)


def _canon(text: str) -> str:
    return re.sub(r"[\s,]+", "", text).casefold()


def claims_in(text: str) -> list[str]:
    """Numeric performance, money or customer-count claims found in ``text``, in order, without repeats."""
    found: list[str] = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text or ""):
            claim = " ".join(match.group(0).split())
            if claim not in found:
                found.append(claim)
    return found


def unsupported_claims(variant: dict[str, Any], grounding: str = "") -> list[str]:
    """Claims in the public fields that do not appear in ``grounding`` (the business, audience and trend data)."""
    known = _canon(grounding)
    bad: list[str] = []
    for field in PUBLIC_FIELDS:
        for claim in claims_in(str(variant.get(field) or "")):
            if _canon(claim) not in known and claim not in bad:
                bad.append(claim)
    return bad


def avoid_hint(claims: list[str]) -> str:
    shown = ", ".join(f'"{c}"' for c in claims[:6])
    return (
        f"Previous drafts were rejected for invented statistics ({shown}). Do not write any percentage, multiplier, "
        "money amount, client count or result such as ROI, CTR or conversion lift unless it appears in the business "
        "profile above. Persuade with what the studio does, how it works and what the viewer can do next."
    )
