import json
from unittest.mock import MagicMock, patch

import pytest

from src.fetch.competitor_finder import CompetitorFinder
from src.generation.prompt_forge import PromptForge
from src.intelligence.audience_mapper import AudienceMapper
from src.intelligence.trend_radar import TrendRadar
from src.intelligence.virality_predictor import ViralityPredictor
from src.shared.llm_client import AllProvidersFailedError, LLMClient, LLMResult


def _fail(prompt: str, label: str | None = None, temperature: float = 0.7) -> LLMResult:
    raise AllProvidersFailedError(f"All LLM providers failed for {label}")


def test_llm_chain_raises_when_all_providers_fail():
    client = LLMClient()
    with (
        patch.object(client, "_try_groq", return_value=None),
        patch.object(client, "_try_gemini", return_value=None),
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


def test_trend_radar_recommends_supported_image_and_reel_formats():
    radar = TrendRadar()
    radar.llm = MagicMock()
    radar.llm.run_json.return_value = {"recommended_formats": ["reel", "carousel", "single image"]}

    result = radar.scan({}, {})

    assert result["recommended_formats"] == ["single image", "reel"]
    assert '"recommended_formats": ["single image", "reel"]' in radar.llm.run_json.call_args.args[0]


def test_competitor_finder_raises_instead_of_faking():
    finder = CompetitorFinder()
    with patch.object(finder.llm, "run", side_effect=_fail):
        with pytest.raises(Exception):
            finder.discover({})


def test_forge_variants_keep_valid_reels_and_keep_segment():
    forge = PromptForge()
    payload = [
        {
            "prompt_type": "educational",
            "title": "T",
            "hook": "Show the finished blend",
            "prompt_text": "Create a 30-second Reel with on-screen text beats and several slides for a video carousel.",
            "hashtags": ["#a"],
            "format": "reel",
            "angle": "A",
            "audience_segment": "young couples",
            "reel_beats": ["finished chai first", "tea leaves close-up", "water meets leaves", "chai pours into a cup", "pack and cup reveal", "tabletop final frame"],
            "reel_overlays": ["Finished blend first", "Start with the leaves", "Watch the color", "Small choices matter", "See the full concept", "DM AD with your brief"],
            "reel_narration": [
                "Show the finished blend. We start with the cup, because people should see the idea before hearing how we made it. The warm color gives the story an inviting first impression.",
                "Then we move closer to the tea leaves. Their texture gives this concept something real to focus on, instead of filling the frame with decoration that distracts from the product.",
                "Watch how the water changes the scene. That simple movement creates a natural transition, and the changing color gives the viewer a reason to stay with the story.",
                "The pour becomes the center of the shot. We keep the background calm, use warm light, and let the product remain the most important thing on screen.",
                "Now the pack joins the cup, so the viewer can connect the finished drink with the product concept. The layout stays clear, with space around each detail.",
                "That is the whole creative idea: lead with the result, then show the choices behind it. If you want an ad concept for your product, DM us ‘AD’ with your brief.",
            ],
        }
    ]

    def fake_run(self_prompt: str, label: str | None = None, temperature: float = 0.7) -> LLMResult:
        return LLMResult(stdout=json.dumps(payload), stderr="", exit_code=0, elapsed_seconds=0.1)

    with patch.object(forge.llm, "run", side_effect=fake_run):
        variants = forge.generate_variants({}, {}, {}, ["educational"])
    v = variants[0]
    assert v["audience_segment"] == "young couples"
    assert v["format"] == "reel" and len(v["reel_beats"]) == 6
    assert len(v["reel_overlays"]) == 6
    text = v["prompt_text"].lower()
    assert "reel" in text
    assert "carousel" not in text

def test_groq_path_executes_without_nameerror(monkeypatch):
    """Regression: the Groq call referenced an undefined `max_retries` name
    and crashed before ever reaching the API — even with a valid key set."""
    from src.shared.config import settings

    monkeypatch.setattr(settings, "llm_provider_order", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_api_key", "")

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = '```json\n{"ok": true}\n```'

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp

    with patch("src.shared.llm_client.make_client", return_value=fake_client):
        result = LLMClient().run("say hi", label="regression")

    assert '"ok"' in result.stdout  # fences stripped, Groq answer used
    assert fake_client.chat.completions.create.call_count >= 1


def test_instagram_search_query_is_sanitized():
    import asyncio

    from src.fetch.social_scraper import SocialScraper

    captured = {}

    async def fake_post(self, **kwargs):
        captured["payload"] = None
        raise AssertionError("not used")

    scraper = SocialScraper(apify_token="t")

    class FakeResp:
        status_code = 200
        text = "[]"
        def json(self):
            return []
        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, params=None, json=None):
            captured["payload"] = json
            return FakeResp()

    with patch("src.fetch.social_scraper.httpx.AsyncClient", FakeClient):
        asyncio.run(scraper.fetch_instagram_search('Homes.com: Homes for Sale, Real Estate & Rent!'))

    assert captured["payload"]["search"] == "Homes com Homes for Sale Real Estate Rent"
