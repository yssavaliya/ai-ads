import base64
import io
from unittest.mock import MagicMock, patch

import pytest

from src.generation.creative_forge import FEED_H, FEED_W, REEL_H, CreativeForge, CreativeForgeError


def _real_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (FEED_W, FEED_H), (120, 120, 120)).save(buf, "PNG")
    return buf.getvalue()


PNG_1PX = _real_png()


def _fake_png_response(content: bytes):
    resp = MagicMock()
    resp.content = content
    resp.raise_for_status = lambda: None
    client = MagicMock()
    client.get.return_value = resp
    client.__enter__ = lambda s: client
    client.__exit__ = lambda s, *a: None
    return client


@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setattr("src.shared.config.settings.logs_dir", str(tmp_path / "logs"))
    f = CreativeForge()
    f.output_dir = tmp_path / "assets"
    f.output_dir.mkdir(parents=True, exist_ok=True)
    return f


def test_generate_image_writes_png(forge):
    with patch("httpx.Client", return_value=_fake_png_response(PNG_1PX)):
        path = forge.generate_image("a bakery ad", FEED_W, FEED_H, "test")
    assert path.endswith(".png")
    with open(path, "rb") as fh:
        assert fh.read(4) == b"\x89PNG"


def test_generate_image_raises_when_all_providers_fail(forge):
    with (
        patch("httpx.Client", side_effect=Exception("down")),
        patch.object(forge, "_hf_image", return_value=None),
    ):
        with pytest.raises(CreativeForgeError):
            forge.generate_image("x", FEED_W, FEED_H, "fail")


def test_overlay_text_produces_file(tmp_path, forge):
    src = tmp_path / "in.png"

    from PIL import Image

    Image.new("RGB", (FEED_W, FEED_H), (200, 200, 200)).save(src)
    out = forge.overlay_text(str(src), "Fresh cakes daily", out_path=str(tmp_path / "out.png"))
    assert out == str(tmp_path / "out.png")
    assert __import__("os").path.getsize(out) > 0


def test_render_prompt_image_happy_path(forge):
    with (
        patch.object(forge, "generate_image", return_value=str(forge.output_dir / "raw.png")),
        patch.object(
            forge,
            "overlay_text",
            return_value=str(forge.output_dir / "final.png"),
        ) as overlay,
    ):
        result = forge.render_prompt(
            {"hook": "Hook line", "title": "T", "prompt_text": "P", "format": "single image"},
            {"name": "Sweet Bakery"},
        )
    assert result["asset_type"] == "image"
    assert result["warnings"] == []
    overlay.assert_called_once()


def test_render_prompt_reel_falls_back_to_image_when_beats_fail(forge):
    calls = {"n": 0}

    def fake_gen(prompt, w, h, name):
        calls["n"] += 1
        if h == REEL_H:
            raise CreativeForgeError("beat image failed")
        return str(forge.output_dir / f"img{calls['n']}.png")

    def fake_overlay(src, text, brand_color=(0, 0, 0), out_path=None):
        return out_path or (src or "final.png")

    with (
        patch.object(forge, "generate_image", side_effect=fake_gen),
        patch.object(forge, "overlay_text", side_effect=fake_overlay),
        patch.object(CreativeForge, "_tts_beats", side_effect=CreativeForgeError("no tts")),
        patch.object(CreativeForge, "_assemble_video"),
    ):
        result = forge.render_prompt(
            {
                "hook": "H",
                "title": "Reel T",
                "prompt_text": "P",
                "format": "reel",
                "reel_beats": ["b1"],
            },
            {"name": "Bakery"},
        )
    # Reel failed -> fell back to a single feed image
    assert result["asset_type"] == "image"
    assert any("Reel render failed" in w for w in result["warnings"])
