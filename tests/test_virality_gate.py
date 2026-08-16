import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.virality_gate import ViralityGate


class FakeResult:
    def __init__(self, payload):
        import json

        self.stdout = json.dumps(payload)


class FakeLLM:
    """Simulates the LLM: generates variants and scores that improve each attempt."""

    def __init__(self):
        self.calls = 0

    def run(self, prompt, label=None, temperature=0.7, max_retries=3):
        self.calls += 1
        import json

        if label == "prompt-forge":
            variants = []
            for i in range(5):
                score = 60 + i
                variants.append(
                    {
                        "prompt_type": "educational",
                        "title": f"Variant {i}",
                        "hook": f"Hook {i}",
                        "prompt_text": f"Full prompt text for variant {i}",
                        "hashtags": ["#test"],
                        "platform_focus": "instagram",
                        "format": "reel",
                        "angle": f"angle {i}",
                    }
                )
            return FakeResult(variants)
        if label == "virality-predictor":
            score = 78 if self.calls % 3 == 0 else 65
            return FakeResult(
                {
                    "virality_score": score,
                    "virality_label": "HIGH" if score >= 75 else "MEDIUM",
                    "virality_probability": score / 100,
                    "key_factors": ["hook"],
                    "risks": ["risk"],
                    "reasoning": "reason",
                    "improvement_hints": ["make hook stronger"],
                }
            )
        if label == "prompt-forge-refine":
            return FakeResult(
                {
                    "prompt_type": "educational",
                    "title": "Refined",
                    "hook": "Refined hook",
                    "prompt_text": "Refined prompt text",
                    "hashtags": ["#test"],
                    "platform_focus": "both",
                    "format": "carousel",
                    "angle": "refined",
                }
            )
        return FakeResult({})


def test_virality_gate_selects_above_threshold(monkeypatch):
    fake = FakeLLM()
    from src.generation.prompt_forge import PromptForge
    from src.intelligence.virality_predictor import ViralityPredictor

    class FakeForge(PromptForge):
        def __init__(self):
            self.llm = fake

    class FakePredictor(ViralityPredictor):
        def __init__(self):
            self.llm = fake

    gate = ViralityGate(threshold=75, max_attempts=3, forge=FakeForge(), predictor=FakePredictor())
    results = gate.run(
        {"name": "Test Biz"},
        {"personas": []},
        {"trending_topics": []},
        ["educational", "storytelling"],
    )
    assert 1 <= len(results) <= 3
    assert results[0]["virality_score"] >= 75


def test_virality_gate_falls_back_below_threshold(monkeypatch):
    class LowLLM(FakeLLM):
        def run(self, prompt, label=None, temperature=0.7, max_retries=3):
            import json

            if label == "prompt-forge":
                return FakeResult(
                    [
                        {
                            "prompt_type": "educational",
                            "title": f"V{i}",
                            "hook": f"H{i}",
                            "prompt_text": f"Full prompt text for variant {i}",
                            "hashtags": ["#t"],
                            "platform_focus": "instagram",
                            "format": "reel",
                            "angle": "a",
                        }
                        for i in range(5)
                    ]
                )
            return FakeResult(
                {
                    "virality_score": 30,
                    "virality_label": "LOW",
                    "virality_probability": 0.3,
                    "key_factors": ["x"],
                    "risks": ["y"],
                    "reasoning": "z",
                    "improvement_hints": ["improve"],
                }
            )

    fake = LowLLM()
    from src.generation.prompt_forge import PromptForge
    from src.intelligence.virality_predictor import ViralityPredictor

    class FakeForge(PromptForge):
        def __init__(self):
            self.llm = fake

    class FakePredictor(ViralityPredictor):
        def __init__(self):
            self.llm = fake

    gate = ViralityGate(threshold=75, max_attempts=3, forge=FakeForge(), predictor=FakePredictor())
    results = gate.run(
        {"name": "Test Biz"},
        {"personas": []},
        {"trending_topics": []},
        ["educational"],
    )
    # Should still return best-available (fallback), never empty
    assert len(results) >= 1
    assert all("virality_score" in r for r in results)