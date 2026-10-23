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
