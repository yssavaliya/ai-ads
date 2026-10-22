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
    assert 7.5 < int(h) * 3600 + int(m) * 60 + float(s) < 9.5  # 3 beats x 3s minus two 0.35s crossfades
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
def test_real_render_prompt_produces_a_reel_not_an_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "logs_dir", Path("logs"))
    monkeypatch.setattr(cf, "FONT_PATH", Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Poppins-Bold.ttf")
    monkeypatch.setattr(settings, "image_provider_order", "")  # no network: local branded cards
    forge = cf.CreativeForge()
    monkeypatch.setattr(forge, "_tts_beats", lambda beats, name: (_ for _ in ()).throw(cf.CreativeForgeError("no piper here")))
    result = forge.render_prompt(
        {"format": "reel", "title": "T", "hook": "H", "reel_beats": ["one", "two", "three"]}, {"name": "Reel Biz"}
    )
    assert result["asset_type"] == "reel" and result["asset_path"].endswith(".mp4")
    assert result["thumbnail"] and Image.open(result["thumbnail"]).size[0] > 0
    assert any("Voiceover unavailable" in w for w in result["warnings"])  # degraded, but still a Reel


def test_wav_duration_needs_no_ffprobe(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    _silent_wav(wav, 2.5)
    monkeypatch.setattr(cf.subprocess, "run", lambda *a, **k: pytest.fail("must not shell out"))
    assert cf.CreativeForge._wav_duration(str(wav)) == pytest.approx(2.5, abs=0.01)
    (tmp_path / "bad.wav").write_bytes(b"not a wav")
    assert cf.CreativeForge._wav_duration(str(tmp_path / "bad.wav")) == 3.0  # safe default
    assert cf.CreativeForge._wav_duration(str(tmp_path / "missing.wav")) == 3.0
