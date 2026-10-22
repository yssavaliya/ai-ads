"""Every prompt sent to a model must actually contain its inputs and output schema.

Real bug: ViralityPredictor._build_prompt ended its `return` after the first f-string, so
the content prompt and the JSON schema were dead code. Every model answered "Missing
Content Prompt" and every virality score in the project's history was meaningless — while
all tests (which stub the LLM) stayed green.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.shared.llm_client import LLMClient, LLMResult

ROOT = Path(__file__).resolve().parents[1]


class CapturingLLM(LLMClient):
    """Records every prompt and answers with a payload the callers accept."""

    def __init__(self, answers: dict[str, object] | None = None) -> None:
        super().__init__()
        self.prompts: dict[str, str] = {}
        self.answers = answers or {}

    def run(self, prompt, label=None, temperature=0.7, max_retries=3, exclude=None):
        self.prompts[label or ""] = prompt
        payload = self.answers.get(label or "", {})
        return LLMResult(stdout=json.dumps(payload), stderr="", exit_code=0, elapsed_seconds=0.0, provider="fake", model="m")


PREDICTION = {"virality_score": 80, "virality_label": "HIGH", "virality_probability": 0.8,
              "key_factors": [], "risks": [], "reasoning": "", "improvement_hints": []}
VARIANT = {"prompt_type": "educational", "title": "t", "hook": "h", "prompt_text": "p", "hashtags": ["#a"],
           "platform_focus": "both", "format": "reel", "angle": "a", "audience_segment": "s", "reel_beats": []}


def test_predictor_prompt_contains_the_text_the_history_and_the_schema():
    from src.intelligence.virality_predictor import ViralityPredictor

    llm = CapturingLLM({"virality-predictor": PREDICTION})
    predictor = ViralityPredictor.__new__(ViralityPredictor)
    predictor.llm = llm
    predictor.score("UNIQUE-CONTENT-PROMPT-777", {"name": "PROFILE-MARK"}, {"personas": ["AUDIENCE-MARK"]},
                    {"trending_topics": ["TREND-MARK"]}, "educational",
                    history=[{"prompt_text": "HISTORY-MARK", "outcome": "good"}])
    sent = llm.prompts["virality-predictor"]
    for marker in ("UNIQUE-CONTENT-PROMPT-777", "PROFILE-MARK", "AUDIENCE-MARK", "TREND-MARK", "HISTORY-MARK"):
        assert marker in sent, marker
    assert "Respond ONLY with JSON" in sent and '"virality_score"' in sent and '"improvement_hints"' in sent
    # the instructions come AFTER the thing to evaluate, as intended
    assert sent.index("UNIQUE-CONTENT-PROMPT-777") < sent.index("Respond ONLY with JSON")


def test_forge_prompts_contain_every_input_and_the_schema():
    from src.generation.prompt_forge import PromptForge

    llm = CapturingLLM({"prompt-forge": [VARIANT], "prompt-forge-refine": VARIANT})
    forge = PromptForge.__new__(PromptForge)
    forge.llm = llm
    forge.generate_variants({"name": "PROFILE-MARK"}, {"k": "AUDIENCE-MARK"}, {"k": "TREND-MARK"}, ["educational"],
                            weights={"topics::educational": 0.9, "w": "WEIGHT-MARK"}, improvement_hints=["HINT-MARK"])
    sent = llm.prompts["prompt-forge"]
    for marker in ("PROFILE-MARK", "AUDIENCE-MARK", "TREND-MARK", "WEIGHT-MARK", "HINT-MARK", "educational"):
        assert marker in sent, marker
    assert "Respond ONLY with JSON" in sent and '"reel_beats"' in sent

    forge.refine_with_hints({**VARIANT, "title": "BEST-MARK"}, ["CRITIQUE-MARK"], {"name": "PROFILE-MARK"},
                            {"k": "AUDIENCE-MARK"}, {"k": "TREND-MARK"})
    refined = llm.prompts["prompt-forge-refine"]
    for marker in ("BEST-MARK", "CRITIQUE-MARK", "PROFILE-MARK", "AUDIENCE-MARK", "TREND-MARK"):
        assert marker in refined, marker
    assert "Respond ONLY with the improved prompt as JSON" in refined


def test_audience_trend_and_competitor_prompts_contain_their_inputs():
    from src.fetch.competitor_finder import CompetitorFinder
    from src.intelligence.audience_mapper import AudienceMapper
    from src.intelligence.trend_radar import TrendRadar

    llm = CapturingLLM({"audience-mapper": {"personas": []}, "trend-radar": {}, "competitor-discovery": []})
    mapper = AudienceMapper.__new__(AudienceMapper)
    mapper.llm = llm
    mapper.map_audience({"name": "PROFILE-MARK"}, {"biography": "BIO-MARK", "posts": [{"caption": "CAPTION-MARK"}]},
                        [{"name": "COMP-MARK", "top_posts": [{"caption": "COMPCAP-MARK"}]}], {"old": "PREVIOUS-MARK"})
    sent = llm.prompts["audience-mapper"]
    for marker in ("PROFILE-MARK", "BIO-MARK", "CAPTION-MARK", "COMP-MARK", "COMPCAP-MARK", "PREVIOUS-MARK"):
        assert marker in sent, marker
    assert "Respond ONLY with JSON" in sent and '"personas"' in sent

    radar = TrendRadar.__new__(TrendRadar)
    radar.llm = llm
    radar.scan({"name": "PROFILE-MARK"}, {"content_themes": ["THEME-MARK"], "top_pain_points": ["PAIN-MARK"]},
               [{"name": "COMP-MARK", "top_posts": [{"caption": "COMPCAP-MARK"}]}])
    sent = llm.prompts["trend-radar"]
    for marker in ("PROFILE-MARK", "THEME-MARK", "PAIN-MARK", "COMPCAP-MARK"):
        assert marker in sent, marker
    assert "Respond ONLY with JSON" in sent and '"trending_topics"' in sent

    finder = CompetitorFinder.__new__(CompetitorFinder)
    finder.llm = llm
    finder.discover({"name": "PROFILE-MARK"})
    sent = llm.prompts["competitor-discovery"]
    assert "PROFILE-MARK" in sent and "Respond ONLY with a JSON array" in sent


def test_weekly_prompt_contains_prompts_engagement_weights_and_schema():
    from datetime import datetime
    from types import SimpleNamespace

    from src.scheduler.weekly_job import WeeklyPipeline

    llm = CapturingLLM({"weekly-report": {"top_performing": []}})
    pipe = WeeklyPipeline.__new__(WeeklyPipeline)
    pipe.llm = llm
    pipe.weights = SimpleNamespace(get_weights=lambda key: {"topics": {"WEIGHT-MARK": 1}})
    prompt = SimpleNamespace(id=1, prompt_text="PROMPT-MARK", prompt_type=SimpleNamespace(value="educational"),
                             virality_score=80, feedback=None, sent_at=datetime.utcnow())
    engagement = SimpleNamespace(likes=123456, comments=1, shares=0, saves=0, reach=10)
    biz = SimpleNamespace(weights_key="biz")
    pipe._build_report(biz, [prompt], {"name": "BUSINESS-MARK"}, [(prompt, engagement)])
    sent = llm.prompts["weekly-report"]
    for marker in ("BUSINESS-MARK", "PROMPT-MARK", "123456", "WEIGHT-MARK"):
        assert marker in sent, marker
    assert "Respond ONLY with JSON" in sent and '"top_performing"' in sent


def _dead_expression_statements() -> list[str]:
    hits = []
    for path in [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            for i, stmt in enumerate(body):
                if not isinstance(stmt, ast.Expr):
                    continue
                value = stmt.value
                is_docstring = i == 0 and isinstance(value, ast.Constant) and isinstance(value.value, str)
                is_ellipsis = isinstance(value, ast.Constant) and value.value is Ellipsis
                if not (is_docstring or is_ellipsis or isinstance(value, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom))):
                    hits.append(f"{path.relative_to(ROOT)}:{stmt.lineno}: {ast.unparse(value)[:60]}")
    return hits


def test_no_dead_expression_statements_anywhere_in_the_code():
    """A `+ "..."` on its own line after a `return` is silently discarded (the predictor bug)."""
    assert _dead_expression_statements() == []


def test_the_dead_expression_guard_actually_catches_the_original_bug(tmp_path, monkeypatch):
    bad = "def f(x):\n    return f'''a {x}'''\n    + 'b'\n"
    tree = ast.parse(bad)
    func = tree.body[0]
    assert isinstance(func.body[1], ast.Expr) and isinstance(func.body[1].value, ast.UnaryOp)  # the shape we scan for


@pytest.mark.parametrize(
    "prompt_text",
    ["short", "x" * 5000, 'quotes """ inside', "line1\nline2 {braces}"],
    ids=["short", "very-long", "triple-quotes", "newline-and-braces"],
)
def test_predictor_prompt_survives_awkward_content(prompt_text):
    from src.intelligence.virality_predictor import ViralityPredictor

    sent = ViralityPredictor.__new__(ViralityPredictor)._build_prompt(prompt_text, {}, {}, {}, "educational")
    assert prompt_text in sent and sent.rstrip().endswith("}")
