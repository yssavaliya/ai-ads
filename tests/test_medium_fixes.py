"""Regression tests for the audit's MEDIUM findings (M-01..M-03)."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from src.generation import image_providers as ip
from src.shared.config import settings

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- M-01: orphan weight keys
def test_committed_weights_have_no_orphan_numeric_business_keys():
    """Weights are keyed by business SLUG. Numeric ids ("1", "7") were left over from the old scheme and
    would silently never be read."""
    weights = json.loads((ROOT / "data" / "weights.json").read_text(encoding="utf-8"))
    assert weights, "weights.json must still hold the live businesses"
    assert not [k for k in weights if k.isdigit()], f"orphan numeric keys: {[k for k in weights if k.isdigit()]}"


def test_every_weights_key_is_a_known_business_slug():
    from src.storage.models import slugify

    names = [b["name"] for b in json.loads((ROOT / "data" / "businesses.json").read_text(encoding="utf-8"))]
    weights = json.loads((ROOT / "data" / "weights.json").read_text(encoding="utf-8"))
    assert set(weights) <= {slugify(n) for n in names}


# ---------------------------------------------------------------- M-02: onboarding validation
@pytest.fixture
def onboard_mod(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("onboard_business_m02", ROOT / "scripts" / "onboard_business.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["onboard_business_m02"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "delivery_mode", "site")
    monkeypatch.setattr(mod, "_store_profile", lambda business, profile: None)
    monkeypatch.setattr(mod.WebsiteScraper, "scrape", lambda self, url: {"title": "T", "page_count": 1})
    return mod


def _saved(tmp_path):
    f = tmp_path / "businesses.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


@pytest.mark.parametrize("email", ["env:OWNER_EMAIL", "owner@example.com"])
def test_onboarding_accepts_real_and_env_reference_emails(onboard_mod, tmp_path, email):
    result = onboard_mod.onboard("Good Co", "https://good.co", email, send_email=False)
    assert result["business"]["email"] == email
    assert [b["name"] for b in _saved(tmp_path)] == ["Good Co"]


@pytest.mark.parametrize("email", ["", "no-at-sign", "a@b", "env:lower_case", "two@@x.com", "sp ace@x.com"])
def test_onboarding_rejects_bad_emails_and_saves_nothing(onboard_mod, tmp_path, email):
    with pytest.raises(ValueError, match="email"):
        onboard_mod.onboard("Bad Co", "https://bad.co", email, send_email=False)
    assert _saved(tmp_path) == []


@pytest.mark.parametrize("threshold", [-1, 101, 1000])
def test_onboarding_rejects_out_of_range_thresholds(onboard_mod, tmp_path, threshold):
    with pytest.raises(ValueError, match="between 0 and 100"):
        onboard_mod.onboard("Edge Co", "https://edge.co", "o@edge.co", threshold=threshold, send_email=False)
    assert _saved(tmp_path) == []


@pytest.mark.parametrize("threshold", [0, 75, 100])
def test_onboarding_accepts_the_threshold_range_edges(onboard_mod, tmp_path, threshold):
    onboard_mod.onboard("Edge Co", "https://edge.co", "o@edge.co", threshold=threshold, send_email=False)
    assert _saved(tmp_path)[0]["virality_threshold"] == threshold


def test_onboarding_persists_only_after_the_scrape_step_cannot_fail(onboard_mod, tmp_path, monkeypatch):
    """Order matters: the business list is written LAST. A crash while fetching the profile used to leave
    a half-onboarded business in businesses.json."""
    order = []
    monkeypatch.setattr(onboard_mod.WebsiteScraper, "scrape", lambda self, url: order.append("scrape") or {"page_count": 1})
    real_save = onboard_mod.save_businesses
    monkeypatch.setattr(onboard_mod, "save_businesses", lambda b: (order.append("save"), real_save(b))[1])
    onboard_mod.onboard("Order Co", "https://order.co", "o@order.co", send_email=False)
    assert order == ["scrape", "save"]


def test_onboarding_rejects_a_duplicate_name_and_a_blank_name(onboard_mod, tmp_path):
    onboard_mod.onboard("Dup Co", "https://dup.co", "o@dup.co", send_email=False)
    with pytest.raises(ValueError, match="already exists"):
        onboard_mod.onboard("dup co", "https://dup.co", "o@dup.co", send_email=False)
    with pytest.raises(ValueError, match="name is required"):
        onboard_mod.onboard("  ", "https://x.co", "o@x.co", send_email=False)


# ---------------------------------------------------------------- M-03: undecodable image falls through
def _png(w=64, h=64):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 120, 200)).save(buf, "PNG")
    return buf.getvalue()


def test_corrupt_image_with_a_valid_header_falls_through_to_the_next_provider(monkeypatch):
    corrupt = b"\x89PNG\r\n\x1a\n" + b"not really a png" * 40  # passes is_image(), fails to decode
    calls = []

    def first(prompt, w, h):
        calls.append("first")
        return corrupt

    def second(prompt, w, h):
        calls.append("second")
        return _png()

    registry = {
        "a": ip.ImageProvider("a", first, lambda: True, False),   # needs fit_cover -> decode fails
        "b": ip.ImageProvider("b", second, lambda: True, False),
    }
    monkeypatch.setattr(ip, "REGISTRY", registry)
    monkeypatch.setattr(settings, "image_provider_order", "a,b")
    out = ip.generate("p", 128, 128)
    assert out is not None and out[1] == "b" and calls == ["first", "second"]
    assert Image.open(io.BytesIO(out[0])).size == (128, 128)
    # and the broken provider is put on cool-down rather than hammered on every image
    assert not ip.HEALTH.available("img:a")


def test_all_providers_returning_undecodable_images_yields_none_not_an_exception(monkeypatch):
    corrupt = b"\xff\xd8" + b"garbage" * 50
    monkeypatch.setattr(ip, "REGISTRY", {"a": ip.ImageProvider("a", lambda p, w, h: corrupt, lambda: True, False)})
    monkeypatch.setattr(settings, "image_provider_order", "a")
    assert ip.generate("p", 128, 128) is None


# ---------------------------------------------------------------- H-09: DNS pinning (no rebinding window)
def test_request_goes_to_the_validated_ip_not_to_a_second_dns_answer(monkeypatch):
    """First lookup (the check) answers with a public IP; a rebinding attacker answers differently the
    second time. The connection must use the checked address and still present the real hostname."""
    import socket

    import httpx

    from src.fetch.website_scraper import WebsiteScraper

    answers = iter(["93.184.216.34", "169.254.169.254", "169.254.169.254"])
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *a, **k: [(2, 1, 6, "", (next(answers), 0))])
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(host=request.url.host, header=request.headers["host"], sni=request.extensions.get("sni_hostname"))
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html>ok</html>")

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    assert "ok" in WebsiteScraper()._fetch("https://rebind.example/page")
    assert seen == {"host": "93.184.216.34", "header": "rebind.example", "sni": "rebind.example"}


def test_pinning_keeps_port_ipv6_and_literal_urls_intact():
    from src.fetch.website_scraper import WebsiteScraper

    pin = WebsiteScraper._pin
    assert pin("http://site.example:8080/a?b=1", "93.184.216.34")[0] == "http://93.184.216.34:8080/a?b=1"
    assert pin("https://site.example/x", "2606:2800:220:1::1")[0] == "https://[2606:2800:220:1::1]/x"
    assert pin("http://8.8.8.8/x", "8.8.8.8") == ("http://8.8.8.8/x", {}, {})
    assert pin("https://user:pw@site.example/x", "1.2.3.4")[1] == {"Host": "site.example"}  # credentials never become the Host


# ---------------------------------------------------------------- no invented scores
class _Forge:
    def generate_variants(self, **kw):
        return [{"prompt_type": f"t{i}", "prompt_text": f"text {i}", "format": "reel"} for i in range(3)]

    def refine_with_hints(self, best, *a):
        return best


class _FlakyPredictor:
    """Scores everything except the text in `fail_on`."""

    def __init__(self, fail_on=()):
        self.fail_on = fail_on

    def score(self, text, *a, **k):
        if text in self.fail_on:
            raise RuntimeError("predictor down")
        return {"virality_score": 40, "virality_label": "LOW", "virality_probability": 0.2, "key_factors": ["k"],
                "risks": [], "reasoning": "measured", "improvement_hints": []}


def _gate(predictor, threshold):
    from src.generation.virality_gate import ViralityGate

    return ViralityGate(threshold=threshold, max_attempts=1, forge=_Forge(), predictor=predictor)


def test_a_failed_prediction_is_unscored_not_an_invented_50():
    """Predictor completely down: content still ships (a dead day is worse), but with NO made-up number."""
    down = _FlakyPredictor(fail_on={"text 0", "text 1", "text 2"})
    out = _gate(down, threshold=30).run({}, {}, {}, ["t0", "t1", "t2"])
    assert out, "the gate must still return the real variants"
    for p in out:
        assert p["virality_score"] is None and p["virality_label"] == "UNSCORED" and p["virality_probability"] is None
        assert any("Not scored" in w for w in p["warnings"])
        assert p["below_threshold"] is False  # nothing was measured, so nothing is "below the bar"


def test_an_unscored_variant_never_takes_a_slot_from_a_measured_one():
    """With a bar of 30 the old fake 50 beat the real 40s. Measured scores must always win."""
    out = _gate(_FlakyPredictor(fail_on={"text 1"}), threshold=30).run({}, {}, {}, ["t0", "t1", "t2"])
    assert {p["prompt_text"] for p in out} == {"text 0", "text 2"}
    assert all(p["virality_score"] == 40 for p in out)
    below = _gate(_FlakyPredictor(fail_on={"text 1"}), threshold=90).run({}, {}, {}, ["t0", "t1", "t2"])
    assert below and all(p["virality_score"] == 40 and p["below_threshold"] is True for p in below)


def test_a_real_score_of_zero_is_kept_and_outranks_unscored():
    from src.generation.virality_gate import rank

    assert rank({"virality_score": 0}) > rank({"virality_score": None})
    assert rank({"virality_score": 0}) == 0.0


def test_unscored_prompts_survive_the_policies_and_the_site(tmp_path, monkeypatch):
    from src.generation.virality_gate import apply_threshold_policy

    prompt = {"title": "x", "virality_score": None, "below_threshold": False, "warnings": []}
    assert apply_threshold_policy([dict(prompt)], "block") == [prompt]  # block removes low SCORES only
    assert apply_threshold_policy([dict(prompt)], "flag") == [prompt]
