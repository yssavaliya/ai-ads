"""Reel (MP4) assembly. In production every Reel silently became a still image because:
  1. the xfade command never included its output file (ffmpeg: "unconnected output"), and
  2. the fallback concat list held RELATIVE paths, which ffmpeg resolves against the list's
     own folder (logs/assets/logs/assets/x.mp4 -> not found), and
  3. the error text hid ffmpeg's real message behind "exit status 4294967294".
"""
from __future__ import annotations

import re
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest
from PIL import Image

from src.generation import creative_forge as cf
from src.generation import image_providers as ip
from src.shared.config import settings

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "logs_dir", tmp_path / "logs")
    monkeypatch.setattr(cf, "FONT_PATH", Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Poppins-Bold.ttf")
    return cf.CreativeForge()


def _stills(forge, n=3):
    out = []
    for i in range(n):
        raw = forge.output_dir / f"b{i}.png"
        raw.write_bytes(ip.branded_card(cf.REEL_W // 4, cf.REEL_H // 4, (31, 111, 104), seed=str(i)))
        out.append(forge.overlay_text(str(raw), f"Beat {i}", out_path=str(forge.output_dir / f"b{i}-final.png")))
    return out


# ------------------------------------------------------------ unit (no ffmpeg)
def test_concat_list_uses_absolute_escaped_paths(tmp_path):
    odd = tmp_path / "it's here" / "seg.mp4"
    odd.parent.mkdir()
    odd.write_bytes(b"x")
    lst = tmp_path / "list.txt"
    cf.CreativeForge._write_concat_list([str(odd), "relative/seg.mp4"], lst)
    lines = lst.read_text(encoding="utf-8").splitlines()
    assert all(Path(re.match(r"file '(.*)'$", line).group(1).replace("'\\''", "'")).is_absolute() for line in lines)
    assert "it'\\''s here" in lines[0]  # a quote in a folder name cannot break the list


def test_xfade_command_names_its_output_file(forge, monkeypatch):
    commands: list[list[str]] = []

    def fake_run(cmd, timeout, what):
        commands.append(cmd)
        if "xfade" in " ".join(cmd):  # create the file the code then checks for
            Path(cmd[-1]).write_bytes(b"video")

    monkeypatch.setattr(forge, "_run_ffmpeg", fake_run)
    out = forge.output_dir / "r.mp4"
    forge._assemble_video(_stills(forge), [3.0, 3.0, 3.0], None, str(out))
    xfade = next(c for c in commands if "-filter_complex" in c)
    assert xfade[-1].endswith("-xfade.mp4") and xfade[-1] != "[v2]"  # previously: no output at all
    assert "libx264" in xfade and "yuv420p" in xfade
    assert xfade[xfade.index("-map") + 1] == "[v2]"


def test_audio_reel_extends_final_still_to_preserve_last_spoken_words(forge, monkeypatch):
    commands: list[list[str]] = []

    def fake_run(cmd, timeout, what):
        commands.append(cmd)
        if what == "xfade":
            Path(cmd[-1]).write_bytes(b"video")

    monkeypatch.setattr(forge, "_run_ffmpeg", fake_run)
    audio = forge.output_dir / "voice.wav"
    audio.write_bytes(b"wav")
    forge._assemble_video(_stills(forge), [4.0, 4.0, 4.0], str(audio), str(forge.output_dir / "voiced.mp4"))
    beat_commands = [cmd for cmd in commands if "-t" in cmd]
    assert [cmd[cmd.index("-t") + 1] for cmd in beat_commands] == ["4.00", "4.00", "4.70"]


def test_failed_xfade_falls_back_to_concat_with_absolute_paths(forge, monkeypatch):
    lists: list[str] = []

    def fake_run(cmd, timeout, what):
        if what == "xfade":
            raise cf.CreativeForgeError("xfade: ffmpeg exit 1: boom")
        if what == "plain concat":
            lists.append(Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8"))
            Path(cmd[-1]).write_bytes(b"video")

    monkeypatch.setattr(forge, "_run_ffmpeg", fake_run)
    forge._assemble_video(_stills(forge), [3.0, 3.0, 3.0], None, str(forge.output_dir / "r.mp4"))
    assert lists and all(Path(re.match(r"file '(.*)'", line).group(1)).is_absolute() for line in lists[0].splitlines())


def test_ffmpeg_failures_report_ffmpegs_own_message(monkeypatch):
    class Proc:
        returncode = 234
        stderr = b"banner\nsomething\n[fc#0] Filter xfade:default has an unconnected output\nError binding filtergraph"

    monkeypatch.setattr(cf.subprocess, "run", lambda *a, **k: Proc())
    with pytest.raises(cf.CreativeForgeError) as err:
        cf.CreativeForge._run_ffmpeg(["ffmpeg"], 5, "xfade")
    assert "unconnected output" in str(err.value) and "xfade" in str(err.value)


def test_single_beat_reel_needs_no_xfade(forge, monkeypatch):
    seen: list[str] = []

    def fake_run(cmd, timeout, what):
        seen.append(what)
        if what == "mux":
            Path(cmd[-1]).write_bytes(b"video")
        else:
            Path(cmd[-1]).write_bytes(b"seg")

    monkeypatch.setattr(forge, "_run_ffmpeg", fake_run)
    forge._assemble_video(_stills(forge, 1), [3.0], None, str(forge.output_dir / "one.mp4"))
    assert seen == ["beat 0 segment", "mux"]


# ----------------------------------------------- integration (real ffmpeg only)
def _info(path: Path) -> str:
    return subprocess.run(["ffmpeg", "-i", str(path)], capture_output=True, text=True).stderr


def _silent_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(struct.pack("<h", 0) * int(22050 * seconds))


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_real_ffmpeg_assembles_a_voiced_reel_with_relative_paths(tmp_path, monkeypatch):
    """Runs from a cwd with RELATIVE output paths — exactly the workflow's situation."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "logs_dir", Path("logs"))
    monkeypatch.setattr(cf, "FONT_PATH", Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Poppins-Bold.ttf")
    forge = cf.CreativeForge()
    wav = tmp_path / "vo.wav"
    _silent_wav(wav, 9.0)
    out = forge.output_dir / "reel.mp4"
    forge._assemble_video(_stills(forge), [3.0, 3.0, 3.0], str(wav), str(out))

    info = _info(out)
    assert "Video: h264" in info and f"{cf.REEL_W}x{cf.REEL_H}" in info
    assert "Audio: aac" in info
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
    assert 8.8 < int(h) * 3600 + int(m) * 60 + float(s) < 9.5  # final still extends to preserve all 9s of audio
    assert not list(forge.output_dir.glob("*-seg*.mp4"))  # temp segments cleaned up
    assert forge._extract_thumbnail(str(out), "reel") and (forge.output_dir / "reel-thumb.png").exists()


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_real_ffmpeg_silent_reel_and_xfade_not_concat(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "logs_dir", Path("logs"))
    monkeypatch.setattr(cf, "FONT_PATH", Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Poppins-Bold.ttf")
    forge = cf.CreativeForge()
    out = forge.output_dir / "silent.mp4"
    forge._assemble_video(_stills(forge), [3.0, 3.0, 3.0], None, str(out))
    assert "Audio:" not in _info(out)
    assert "falling back to plain concat" not in caplog.text  # the crossfade path really works now


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_real_render_prompt_falls_back_to_image_when_voiceover_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "logs_dir", Path("logs"))
    monkeypatch.setattr(cf, "FONT_PATH", Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Poppins-Bold.ttf")
    monkeypatch.setattr(settings, "image_provider_order", "")  # no network: local branded cards
    forge = cf.CreativeForge()
    monkeypatch.setattr(forge, "_tts_beats", lambda beats, name: (_ for _ in ()).throw(cf.CreativeForgeError("no piper here")))
    result = forge.render_prompt(
        {
            "format": "reel", "title": "T", "hook": "H",
            "reel_beats": ["one", "two", "three"],
            "reel_overlays": ["H", "Step two", "Try it today"],
            "reel_narration": [
                "H. Start with the useful result, then tell viewers what detail they will learn today.",
                "Explain the process with specific visual choices and a clear point anyone can follow.",
                "Close with a practical thought and invite viewers to save or share this idea today too.",
            ],
        },
        {"name": "Reel Biz"},
    )
    assert result["asset_type"] == "image"  # never publish a Reel when required speech is unavailable
    assert any("Reel render failed" in w for w in result["warnings"])


def test_short_voiceover_fails_closed_instead_of_trimming_the_reel(forge, monkeypatch):
    monkeypatch.setattr(settings, "image_provider_order", "")
    monkeypatch.setattr(settings, "image_card_fallback", True)
    monkeypatch.setattr(forge, "_tts_beats", lambda lines, _name: ("voice.wav", [2.0] * len(lines)))
    monkeypatch.setattr(forge, "_assemble_video", lambda *_args: pytest.fail("short audio must not be muxed"))
    result = forge.render_prompt(
        {
            "format": "reel", "title": "short audio", "hook": "Start with the result",
            "reel_beats": ["first", "second", "third"],
            "reel_overlays": ["Start with result", "Show the process", "Save this idea"],
            "reel_narration": [
                "Start with the result. Show the final creative first, then explain why its opening works for this audience and what makes the message memorable.",
                "Walk through the choices clearly and show how each detail supports the central message, from the colors and framing to the product benefit.",
                "Close with a useful takeaway and invite viewers to save the Reel for later reference, then try this approach in their next campaign.",
            ],
        },
        {"name": "Reel Biz"},
    )
    assert result["asset_type"] == "image"
    assert any("only 6.0s" in warning for warning in result["warnings"])


def test_wav_duration_needs_no_ffprobe(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    _silent_wav(wav, 2.5)
    monkeypatch.setattr(cf.subprocess, "run", lambda *a, **k: pytest.fail("must not shell out"))
    assert cf.CreativeForge._wav_duration(str(wav)) == pytest.approx(2.5, abs=0.01)
    (tmp_path / "bad.wav").write_bytes(b"not a wav")
    assert cf.CreativeForge._wav_duration(str(tmp_path / "bad.wav")) == 3.0  # safe default
    assert cf.CreativeForge._wav_duration(str(tmp_path / "missing.wav")) == 3.0


def test_piper_voice_download_uses_real_onnx_and_json_paths(tmp_path, monkeypatch):
    requested = []

    class Response:
        content = b"voice-data"

        @staticmethod
        def raise_for_status():
            return None

    def fake_get(url, **_kwargs):
        requested.append(url)
        return Response()

    monkeypatch.setattr(cf, "VOICE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(cf.httpx, "get", fake_get)
    path = cf.CreativeForge()._ensure_voice_model()
    assert path.name == "en_US-lessac-medium.onnx"
    assert requested[0].endswith("en_US-lessac-medium.onnx")
    assert requested[1].endswith("en_US-lessac-medium.onnx.json")
    assert path.read_bytes() == b"voice-data"
    assert path.with_name(path.name + ".json").read_bytes() == b"voice-data"


def test_reel_render_uses_distinct_scene_beats_and_overlay_lines(monkeypatch):
    forge = cf.CreativeForge()
    seen_scenes = []
    seen_overlays = []
    spoken_lines = []
    monkeypatch.setattr(cf, "parallel_map", lambda fn, items, _limit: [type("Outcome", (), {"ok": True, "value": fn(item)})() for item in items])
    monkeypatch.setattr(forge, "_image_prompt", lambda prompt, _biz: prompt["visual_scene"])
    monkeypatch.setattr(forge, "generate_image", lambda prompt, *_args: seen_scenes.append(prompt) or "still.png")
    monkeypatch.setattr(forge, "overlay_text", lambda path, text, **_kw: seen_overlays.append(text) or path)
    monkeypatch.setattr(forge, "_tts_beats", lambda lines, *_args: spoken_lines.extend(lines) or ("voice.wav", [5.2] * len(lines)))
    monkeypatch.setattr(forge, "_assemble_video", lambda *_args: None)
    monkeypatch.setattr(forge, "_extract_thumbnail", lambda *_args: None)

    forge._render_reel(
        {
            "format": "reel",
            "title": "Chai",
            "visual_scene": "cinematic product photography",
            "reel_beats": ["tea leaves macro", "chai pouring", "pack reveal"],
            "hook": "Start with fresh tea leaves.",
            "reel_overlays": ["Not the hook", "A slow pour", "Meet your chai"],
            "reel_narration": [
                "Start with fresh tea leaves. Then show the finished cup to give viewers a reason to stay.",
                "Let the chai pour slowly while you explain how aroma and color shape the ritual.",
                "Make this your next ritual. Save the idea and tell us how you take your chai.",
            ],
        },
        {"name": "Aimagix"},
        "chai",
    )

    assert sorted(seen_scenes) == sorted(
        [
            "cinematic product photography. Scene: tea leaves macro",
            "cinematic product photography. Scene: chai pouring",
            "cinematic product photography. Scene: pack reveal",
        ]
    )
    assert seen_overlays == ["Start with fresh tea leaves.", "A slow pour", "Meet your chai"]
    assert spoken_lines == [
        "Start with fresh tea leaves. Then show the finished cup to give viewers a reason to stay.",
        "Let the chai pour slowly while you explain how aroma and color shape the ritual.",
        "Make this your next ritual. Save the idea and tell us how you take your chai.",
    ]
    assert not set(spoken_lines) & {"tea leaves macro", "chai pouring", "pack reveal"}
