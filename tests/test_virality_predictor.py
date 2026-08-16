from src.intelligence.virality_predictor import ViralityPredictor


def test_validate_payload_defaults():
    p = ViralityPredictor()
    payload = p._validate(
        {
            "virality_score": 85,
            "virality_label": "high",
            "virality_probability": 0.9,
            "key_factors": ["strong hook"],
            "risks": ["niche topic"],
            "reasoning": "Good hook, clear value",
            "improvement_hints": ["make it more specific"],
        }
    )
    assert payload["virality_score"] == 85
    assert payload["virality_label"] == "HIGH"
    assert payload["virality_probability"] == 0.9
    assert "improvement_hints" in payload


def test_validate_payload_clamps_bad_values():
    p = ViralityPredictor()
    payload = p._validate(
        {
            "virality_score": 150,
            "virality_label": "SUPER_VIRAL",
            "virality_probability": 3.5,
            "key_factors": [],
            "risks": "not a list",
            "reasoning": "",
        }
    )
    assert payload["virality_score"] == 100  # clamped
    assert payload["virality_label"] in ("LOW", "MEDIUM", "HIGH")
    assert payload["virality_probability"] == 1.0  # clamped


def test_validate_payload_normalizes_missing_lists():
    p = ViralityPredictor()
    payload = p._validate(
        {
            "virality_score": 60,
            "virality_label": "MEDIUM",
            "virality_probability": 0.5,
        }
    )
    assert payload["key_factors"] == []
    assert payload["risks"] == []