"""The post sent to Buffer must have a real AI image, and the overlay must not draw empty glyph boxes."""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path

from PIL import Image

from src.generation.creative_forge import CreativeForge

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FLAT = ["All AI image providers failed or are rate limited — used a branded card background instead"]


def _publish_module():
    spec = importlib.util.spec_from_file_location("publish_buffer_under_test", SCRIPTS / "publish_buffer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_flat_card_is_never_picked_even_with_the_top_score():
    pb = _publish_module()
    flat_best = {"title": "flat", "score": 78, "warnings": FLAT}
    real = {"title": "real", "score": 60, "warnings": []}
    assert pb.is_degraded(flat_best) and not pb.is_degraded(real)
    assert pb.pick_prompt([flat_best, real])["title"] == "real"


def test_nothing_is_picked_when_only_flat_cards_remain():
    pb = _publish_module()
    assert pb.pick_prompt([{"title": "a", "score": 70, "warnings": FLAT}]) is None
    assert pb.pick_prompt([]) is None


def test_a_creative_with_an_unverified_statistic_is_never_picked_even_with_the_top_score():
    pb = _publish_module()
    invented = {"title": "invented", "score": 82, "warnings": [], "hook": "250% CTR in 48h", "caption": "Proof: 3x ROI."}
    clean = {"title": "clean", "score": 60, "warnings": [], "hook": "Brand visuals without the agency wait", "caption": "See how it works."}
    assert pb.pick_prompt([invented, clean])["title"] == "clean"
    assert pb.pick_prompt([invented]) is None


def test_highest_score_wins_among_real_images():
    pb = _publish_module()
    picked = pb.pick_prompt([{"title": "a", "score": 40}, {"title": "b", "score": 90}, {"title": "c", "virality_score": 70}])
    assert picked["title"] == "b"


def test_reel_only_requires_verified_duration_of_at_least_one_minute():
    pb = _publish_module()
    image = {"id": "image", "title": "image", "score": 99, "asset": {"type": "image"}}
    short_reel = {"id": "short", "title": "short", "score": 98, "asset": {"type": "reel"},
                  "video_duration_seconds": 59.9}
    minute_reel = {"id": "minute", "title": "minute", "score": 70, "asset": {"type": "reel"},
                   "video_duration_seconds": 60}
    assert pb.pick_prompt([image, short_reel, minute_reel], reel_only=True)["id"] == "minute"
    assert pb.pick_prompt([image, short_reel], reel_only=True) is None


def test_font_safe_replaces_hyphens_and_drops_emoji():
    safe = CreativeForge._font_safe
    assert safe("48 hrs to VC‑ready") == "48 hrs to VC-ready"
    assert safe("Before ✨ after \U0001F525") == "Before after"
    assert safe("a b c") == "a b c"
    assert safe("Budget ≠ results — ok") == "Budget ≠ results — ok"  # glyphs Poppins has stay


def test_overlay_draws_text_without_raising_for_unsupported_characters(tmp_path):
    base = tmp_path / "base.png"
    Image.new("RGB", (540, 675), (30, 40, 90)).save(base)
    forge = CreativeForge()
    out = forge.overlay_text(str(base), "VC‑ready ✨ branding", category="before_after", brand_handle="aimagix.studio",
                             out_path=str(tmp_path / "out.png"))
    assert Image.open(io.BytesIO(Path(out).read_bytes())).size == (540, 675)
