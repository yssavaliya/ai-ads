"""Free image providers fall through in order; a local branded card guarantees a visual."""
from __future__ import annotations

import base64
import io
import os

import httpx
import pytest
from PIL import Image

from src.generation import image_providers as ip
from src.generation.creative_forge import FEED_H, FEED_W, CreativeForge, CreativeForgeError
from src.shared.config import settings


def _png(size=(64, 64)) -> bytes:
    img = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))  # noise: > 1000 bytes
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def net(monkeypatch):
    """Route every httpx.Client through a handler; returns the list of requested hosts."""
    requested: list[str] = []
    routes: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.host)
        outcome = routes.get(request.url.host)
        if outcome is None:
            return httpx.Response(404)
        return outcome(request) if callable(outcome) else outcome

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(settings, "image_provider_order", "pollinations,gemini,huggingface,together,cloudflare")
    for key in ("gemini_api_key", "hf_token", "together_api_key", "cloudflare_account_id", "cloudflare_api_token"):
        monkeypatch.setattr(settings, key, None)
    requested_routes = type("Net", (), {"requested": requested, "routes": routes})
    return requested_routes


@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "logs_dir", tmp_path / "logs")
    return CreativeForge()


def test_rate_limited_pollinations_falls_through_to_gemini_and_is_cropped(net, forge, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "k")
    square = _png((256, 256))
    net.routes["image.pollinations.ai"] = httpx.Response(429, text="rate limit")
    net.routes["generativelanguage.googleapis.com"] = httpx.Response(200, json={"candidates": [{"content": {"parts": [
        {"text": "here you go"},
        {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(square).decode()}},
    ]}}]})
    path = forge.generate_image("a bakery", 270, 340, "x")
    assert Image.open(path).size == (270, 340)  # square output cover-cropped to the requested ratio
    assert net.requested == ["image.pollinations.ai", "generativelanguage.googleapis.com"]
    assert forge._take_warnings() == []  # a real AI image: nothing degraded


def test_limited_provider_is_skipped_on_the_next_image(net, forge, monkeypatch):
    monkeypatch.setattr(settings, "hf_token", "t")
    net.routes["image.pollinations.ai"] = httpx.Response(429, text="rate limit")
    net.routes["router.huggingface.co"] = httpx.Response(200, content=_png())
    forge.generate_image("one", 64, 64, "a")
    net.requested.clear()
    forge.generate_image("two", 64, 64, "b")
    assert net.requested == ["router.huggingface.co"]  # circuit breaker skipped pollinations


def test_unauthorised_provider_is_skipped_for_the_rest_of_the_run(net, forge, monkeypatch):
    monkeypatch.setattr(settings, "hf_token", "bad")
    monkeypatch.setattr(settings, "together_api_key", "t")
    net.routes["image.pollinations.ai"] = httpx.Response(500)
    net.routes["router.huggingface.co"] = httpx.Response(401)
    net.routes["api.hf-fallback.invalid"] = httpx.Response(401)
    net.routes["api-inference.huggingface.co"] = httpx.Response(401)
    net.routes["api.together.xyz"] = httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(_png()).decode()}]})
    forge.generate_image("one", 64, 64, "a")
    net.requested.clear()
    forge.generate_image("two", 64, 64, "b")
    assert "router.huggingface.co" not in net.requested and "api.together.xyz" in net.requested


def test_cloudflare_response_shape(net, forge, monkeypatch):
    monkeypatch.setattr(settings, "image_provider_order", "cloudflare")
    monkeypatch.setattr(settings, "cloudflare_account_id", "acct")
    monkeypatch.setattr(settings, "cloudflare_api_token", "tok")
    net.routes["api.cloudflare.com"] = httpx.Response(200, json={"result": {"image": base64.b64encode(_png()).decode()}})
    assert Image.open(forge.generate_image("p", 128, 64, "cf")).size == (128, 64)


def test_unconfigured_providers_are_never_called(net, forge):
    net.routes["image.pollinations.ai"] = httpx.Response(200, content=_png())
    forge.generate_image("p", 64, 64, "x")
    assert net.requested == ["image.pollinations.ai"]


def test_all_providers_down_renders_a_branded_card_and_says_so(net, forge):
    net.routes["image.pollinations.ai"] = httpx.Response(503)
    path = forge.generate_image("p", FEED_W // 4, FEED_H // 4, "bakery-post")
    assert path.endswith(".png") and Image.open(path).size == (FEED_W // 4, FEED_H // 4)
    warnings = forge._take_warnings()
    assert warnings and "branded card" in warnings[0]


def test_render_prompt_ships_an_image_with_a_warning_when_every_provider_fails(net, forge):
    net.routes["image.pollinations.ai"] = httpx.Response(503)
    result = forge.render_prompt({"hook": "Fresh bread", "title": "t", "format": "single image"}, {"name": "Sweet Bakery"})
    assert result["asset_type"] == "image" and os.path.exists(result["asset_path"])
    assert any("branded card" in w for w in result["warnings"])


def test_card_fallback_can_be_disabled(net, forge, monkeypatch):
    monkeypatch.setattr(settings, "image_card_fallback", False)
    net.routes["image.pollinations.ai"] = httpx.Response(503)
    with pytest.raises(CreativeForgeError):
        forge.generate_image("p", 64, 64, "x")


def test_branded_card_is_deterministic_and_varies_by_seed():
    a = ip.branded_card(120, 160, (31, 111, 104), "same")
    assert a == ip.branded_card(120, 160, (31, 111, 104), "same")
    assert a != ip.branded_card(120, 160, (31, 111, 104), "different")
    assert Image.open(io.BytesIO(a)).size == (120, 160)


def test_provider_order_is_configurable(net, forge, monkeypatch):
    monkeypatch.setattr(settings, "image_provider_order", "huggingface,pollinations")
    monkeypatch.setattr(settings, "hf_token", "t")
    net.routes["router.huggingface.co"] = httpx.Response(200, content=_png())
    forge.generate_image("p", 64, 64, "x")
    assert net.requested == ["router.huggingface.co"]


def test_gemini_image_tries_the_next_configured_model(net, forge, monkeypatch):
    monkeypatch.setattr(settings, "image_provider_order", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "k")
    monkeypatch.setattr(settings, "gemini_image_model", "img-new,img-old")
    png = base64.b64encode(_png((64, 64))).decode()

    def handler(request):
        if "img-new" in str(request.url):
            return httpx.Response(404, text="model not available")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"inlineData": {"data": png}}]}}]})

    net.routes["generativelanguage.googleapis.com"] = handler
    assert Image.open(forge.generate_image("p", 64, 64, "g")).size == (64, 64)
    assert net.requested == ["generativelanguage.googleapis.com"] * 2
