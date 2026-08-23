import json
from unittest.mock import MagicMock, patch

import pytest

from src.generation.prompt_forge import PromptForge
from src.intelligence.audience_mapper import AudienceMapper
from src.intelligence.trend_radar import TrendRadar
from src.intelligence.virality_predictor import ViralityPredictor
from src.fetch.competitor_finder import CompetitorFinder
from src.shared.llm_client import AllProvidersFailedError, LLMClient, LLMResult


def _fail(prompt: str, label: str | None = None, temperature: float = 0.7) -> LLMResult:
    raise AllProvidersFailedError(f"All LLM providers failed for {label}")


def test_llm_chain_raises_when_all_providers_fail():
    client = LLMClient()
    with (
        patch.object(client, "_try_groq", return_value=None),
        patch.object(client, "_try_gemini", return_value=None),
        patch.object(client, "_try_pollinations", return_value=None),
    ):
        with pytest.raises(AllProvidersFailedError):
            client.run("x", label="unit")


def test_prompt_forge_raises_instead_of_faking():
    forge = PromptForge()
    with patch.object(forge.llm, "run", side_effect=_fail):
        with pytest.raises(Exception):
            forge.generate_variants({}, {}, {}, ["educational"])


def test_virality_predictor_raises_instead_of_faking():
    predictor = ViralityPredictor()
    with patch.object(predictor.llm, "run", side_effect=_fail):
        with pytest.raises(Exception):
            predictor.score("some prompt", {}, {}, {})


def test_audience_mapper_raises_instead_of_faking():
    mapper = AudienceMapper()
    with patch.object(mapper.llm, "run", side_effect=_fail):
        with pytest.raises(Exception):
            mapper.map_audience({})


def test_trend_radar_raises_instead_of_faking():
    radar = TrendRadar()
    with patch.object(radar.llm, "run", side_effect=_fail):
        with pytest.raises(Exception):
            radar.scan({}, {})


def test_competitor_finder_raises_instead_of_faking():
    finder = CompetitorFinder()
    with patch.object(finder.llm, "run", side_effect=_fail):
        with pytest.raises(Exception):
            finder.discover({})


def test_forge_variants_include_reel_beats_and_segment():
    forge = PromptForge()
    payload = [
        {
            "prompt_type": "educational",
            "title": "T",
            "hook": "H",
            "prompt_text": "P",
            "hashtags": ["#a"],
            "format": "reel",
            "angle": "A",
            "audience_segment": "young couples",
            "reel_beats": ["beat one", "beat two", "", "beat three"],
        }
    ]

    def fake_run(self_prompt: str, label: str | None = None, temperature: float = 0.7) -> LLMResult:
        return LLMResult(stdout=json.dumps(payload), stderr="", exit_code=0, elapsed_seconds=0.1)

    with patch.object(forge.llm, "run", side_effect=fake_run):
        variants = forge.generate_variants({}, {}, {}, ["educational"])
    v = variants[0]
    assert v["audience_segment"] == "young couples"
    assert v["reel_beats"] == ["beat one", "beat two", "beat three"]
