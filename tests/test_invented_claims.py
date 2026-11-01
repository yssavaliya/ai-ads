"""Ad copy must not carry statistics nobody gave the generator."""
from __future__ import annotations

from src.generation.claims import claims_in, unsupported_claims
from src.generation.virality_gate import ViralityGate


def test_detects_percentages_multipliers_money_and_customer_counts():
    assert claims_in("250% CTR lift in 48h") == ["250%", "48h"]
    assert claims_in("48 hrs. 3× ROI") == ["48 hrs", "3×"]
    assert "2.5x" in claims_in("2.5x results")
    assert "₹75K" in claims_in("A ₹75K budget")
    assert claims_in("30 clients served, 120+ brands") == ["30 clients", "120+ brands"]
    assert "Rs. 5,000" in claims_in("Rs. 5,000 off")


def test_ordinary_numbers_are_not_claims():
    for text in ("Top 10 tips", "Version 3x3 grid", "Step 1 of 3", "The 5-Second Logo Test", "Spot it in the first hour"):
        assert claims_in(text) == [], text


def test_delivery_times_and_deliverable_counts_are_service_promises():
    assert claims_in("Watch a startup brand transform in 48 hours") == ["48 hours"]
    assert claims_in("48‑Hour Startup Brand Makeover") == ["48-Hour"]  # non-breaking hyphen is normalised
    assert claims_in("Refine it in under 24 hours") == ["24 hours"]
    assert claims_in("Generate 10 concepts instantly") == ["10 concepts"]
    assert claims_in("Get 3 logo options") == ["3 logo options"]


def test_only_public_fields_are_checked_and_grounded_numbers_pass():
    variant = {"hook": "Brand visuals without the wait", "title": "t", "caption": "We cut review time by 40% last year.", "prompt_text": "ignore 99% here"}
    assert unsupported_claims(variant) == ["40%"]  # prompt_text is an internal brief, not published
    assert unsupported_claims(variant, grounding="Studio fact: review time cut by 40 %") == []


class _Forge:
    def __init__(self):
        self.hints_seen: list[list[str]] = []

    def generate_variants(self, *, improvement_hints=None, **_):
        self.hints_seen.append(list(improvement_hints or []))
        if len(self.hints_seen) == 1:
            return [{"prompt_type": "educational", "title": "Fake", "hook": "250% CTR lift in 48h", "caption": "Proof: 3x ROI.",
                     "prompt_text": "brief one", "hashtags": ["#a"]}]
        return [{"prompt_type": "educational", "title": "Honest", "hook": "Brand visuals without the agency wait",
                 "caption": "See how the studio builds a visual in one sitting.", "prompt_text": "brief two", "hashtags": ["#a"]}]

    def refine_with_hints(self, best, hints, *args):
        return best


class _Predictor:
    def __init__(self):
        self.scored: list[str] = []

    def score(self, prompt_text, *args, **kwargs):
        self.scored.append(prompt_text)
        return {"virality_score": 90, "virality_label": "HIGH", "virality_probability": 0.9, "key_factors": [],
                "risks": [], "reasoning": "r", "improvement_hints": []}


def test_gate_drops_invented_claims_before_scoring_and_tells_the_next_attempt():
    forge, predictor = _Forge(), _Predictor()
    gate = ViralityGate(threshold=70, max_attempts=3, forge=forge, predictor=predictor)
    results = gate.run({"name": "Studio"}, {}, {}, ["educational"])
    assert [r["title"] for r in results] == ["Honest"]
    assert len(predictor.scored) == 1 and "brief one" not in predictor.scored[0]  # never scored the fabricated one
    assert forge.hints_seen[0] == []
    assert any("invented figures" in h and "250%" in h for h in forge.hints_seen[1])


def test_gate_keeps_a_claim_the_inputs_actually_contain():
    class Grounded(_Forge):
        def generate_variants(self, **kwargs):
            return [{"prompt_type": "educational", "title": "Real", "hook": "Delivered fast",
                     "caption": "Reviews arrive 40% faster.", "prompt_text": "brief", "hashtags": ["#a"]}]

    gate = ViralityGate(threshold=70, max_attempts=1, forge=Grounded(), predictor=_Predictor())
    assert gate.run({"name": "Studio", "fact": "Reviews arrive 40% faster with our workflow"}, {}, {}, ["educational"])[0]["title"] == "Real"


def test_proof_phrases_without_numbers_are_unverified_claims():
    assert claims_in("Link in bio for our latest case study.") == ["case study"]
    assert "our clients" in [c.lower() for c in claims_in("Our clients love it")]
    assert "trusted by" in [c.lower() for c in claims_in("Trusted by founders")]
    assert "award-winning" in [c.lower() for c in claims_in("An award-winning studio")]
    assert "guarantee" in [c.lower() for c in claims_in("We guarantee results")]
    assert claims_in("Nobody claps for the silent hours") == []
    assert claims_in("A study in light and shadow") == []  # 'study' alone is ordinary language


def test_a_proof_phrase_the_owner_wrote_themselves_is_allowed():
    variant = {"hook": "h", "title": "t", "caption": "See our case study on the site."}
    assert unsupported_claims(variant) == ["case study"]
    assert unsupported_claims(variant, grounding="We publish a case study on the website") == []
