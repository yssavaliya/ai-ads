from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from src.shared.config import settings

logger = logging.getLogger(__name__)

FEED_W = 1080
FEED_H = 1350
REEL_W = 1080
REEL_H = 1920

FONT_PATH = Path("assets/fonts/Poppins-Bold.ttf")
VOICE_MODEL_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium/en_US-lessac-medium"
)
VOICE_CACHE_DIR = Path.home() / ".cache" / "piper-voices"

IMAGE_STYLE_SUFFIX = (
    ", professional advertising photography, high quality, vibrant, "
    "eye-catching composition, instagram aesthetic, no text, no words, no letters"
)


class CreativeForgeError(RuntimeError):
    pass


class CreativeForge:
    """Renders approved text prompts into real creative assets.

    Images: Pollinations (keyless, open models) → optional HF Inference
    fallback (FLUX.1-schnell) → Pillow text overlay with a bundled OFL font.
    Reels: motion-poster pipeline — one still per beat + Piper TTS voiceover +
    FFmpeg Ken Burns/xfade assembly into a 1080x1920 H.264 MP4.
    Every degradation is logged loudly and reported back to the caller."""

    def __init__(self) -> None:
        self.output_dir = Path(settings.logs_dir) / "assets"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render_prompt(
        self,
        prompt: dict[str, Any],
        business_profile: dict[str, Any],
        index: int = 0,
    ) -> dict[str, Any]:
        """Render one approved prompt into an asset. Returns
        {asset_path, asset_type, warnings: [..]} — raises CreativeForgeError
        only when NOTHING could be produced."""
        warnings: list[str] = []
        is_reel = str(prompt.get("format", "")).lower() == "reel" and prompt.get("reel_beats")
        base_name = self._slug(f"{business_profile.get('name', 'biz')}-{prompt.get('title', 'post')}")[:60]

        if is_reel:
            try:
                result = self._render_reel(prompt, business_profile, base_name)
                result["warnings"] += warnings
                return result
            except CreativeForgeError as exc:
                warnings.append(f"Reel render failed ({exc}); falling back to single image")
                logger.warning("::warning::%s", warnings[-1])

        image_path = self.generate_image(
            self._image_prompt(prompt, business_profile), FEED_W, FEED_H, f"{base_name}-{index}"
        )
        final_path = self.overlay_text(
            image_path,
            prompt.get("hook") or prompt.get("title") or "",
            brand_color=self._brand_color(business_profile),
            out_path=str(self.output_dir / f"{base_name}-{index}-final.png"),
        )
        return {"asset_path": str(final_path), "asset_type": "image", "warnings": warnings}

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    def generate_image(self, image_prompt: str, width: int, height: int, name: str) -> str:
        png = self._pollinations_image(image_prompt, width, height)
        if png is None and settings.hf_token:
            logger.warning("Pollinations image failed; trying Hugging Face FLUX.1-schnell fallback")
            png = self._hf_image(image_prompt, width, height)
        if png is None:
            raise CreativeForgeError(
                "All image providers failed (Pollinations + HF fallback)"
            )
        ext = ".jpg" if png[:2] == b"\xff\xd8" else ".png"
        out_path = self.output_dir / f"{self._slug(name)}-{width}x{height}{ext}"
        out_path.write_bytes(png)
        logger.info("Rendered image %s (%d bytes)", out_path, len(png))
        return str(out_path)

    @staticmethod
    def _image_prompt(prompt: dict[str, Any], business_profile: dict[str, Any]) -> str:
        parts = [
            prompt.get("hook", ""),
            prompt.get("angle", ""),
            f"for {business_profile.get('name', '')}",
            business_profile.get("meta_description", "")
            or " ".join(business_profile.get("headings", [])[:5]),
        ]
        return ". ".join(p for p in parts if p)[:800] + IMAGE_STYLE_SUFFIX

    def _pollinations_image(self, image_prompt: str, width: int, height: int) -> bytes | None:
        try:
            with httpx.Client(timeout=180, follow_redirects=True) as client:
                resp = client.get(
                    f"https://image.pollinations.ai/prompt/{quote(image_prompt, safe='')}",
                    params={"width": width, "height": height, "nologo": "true", "model": "flux"},
                )
                resp.raise_for_status()
                data = resp.content
            if len(data) < 1000 or not (data[:4] == b"\x89PNG" or data[:2] == b"\xff\xd8"):
                logger.error("Pollinations returned non-image payload (%d bytes)", len(data))
                return None
            return data
        except Exception as exc:
            logger.error("Pollinations image failed: %s", exc)
            return None

    def _hf_image(self, image_prompt: str, width: int, height: int) -> bytes | None:
        if not settings.hf_token:
            return None
        try:
            with httpx.Client(timeout=180) as client:
                resp = client.post(
                    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
                    headers={
                        "Authorization": f"Bearer {settings.hf_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "inputs": image_prompt,
                        "parameters": {"width": width, "height": height},
                    },
                )
                resp.raise_for_status()
                data = resp.content
            if not (data[:4] == b"\x89PNG" or data[:2] == b"\xff\xd8"):
                logger.error("HF inference returned non-image payload (%d bytes)", len(data))
                return None
            return data
        except Exception as exc:
            logger.error("HF image inference failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Text overlay
    # ------------------------------------------------------------------
    def overlay_text(
        self,
        image_path: str,
        text: str,
        brand_color: tuple[int, int, int] = (31, 111, 104),
        band_alpha: int = 170,
        out_path: str | None = None,
    ) -> str:
        """Composite hook text over a translucent brand-colored band near the
        bottom of the image, inside Instagram's safe area."""
        from PIL import Image, ImageDraw, ImageFont

        if not text:
            return image_path
        img = Image.open(image_path).convert("RGBA")
        w, h = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font_size = max(36, w // 16)
        font = self._load_font(font_size)
        lines = self._wrap_text(draw, text.upper(), font, max_width=int(w * 0.86))
        line_height = int(font_size * 1.28)
        pad_y = int(h * 0.05)

        block_h = line_height * len(lines) + 2 * pad_y
        band_top = h - int(h * 0.075) - block_h  # keep 7.5% bottom safe area clear

        draw.rectangle([0, band_top, w, band_top + block_h], fill=(*brand_color, band_alpha))
        y = band_top + pad_y
        for line in lines:
            text_w = draw.textlength(line, font=font)
            x = (w - text_w) // 2
            draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0, 140))
            draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
            y += line_height

        combined = Image.alpha_composite(img, overlay).convert("RGB")
        out = out_path or image_path.replace(".png", "-texted.png")
        combined.save(out, "PNG")
        logger.info("Overlaid hook text on %s -> %s", image_path, out)
        return out

    @staticmethod
    def _load_font(size: int):
        from PIL import ImageFont

        if FONT_PATH.exists():
            return ImageFont.truetype(str(FONT_PATH), size)
        logger.warning("::warning::Bundled font missing at %s; using default bitmap font", FONT_PATH)
        return ImageFont.load_default()

    @staticmethod
    def _wrap_text(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines[:6]

    # ------------------------------------------------------------------
    # Reels (motion poster)
    # ------------------------------------------------------------------
    def _render_reel(
        self, prompt: dict[str, Any], business_profile: dict[str, Any], base_name: str
    ) -> dict[str, Any]:
        beats: list[str] = [b for b in (prompt.get("reel_beats") or []) if b][:5]
        if not beats:
            raise CreativeForgeError("no reel beats provided")

        warnings: list[str] = []
        stills: list[str] = []
        for i, beat in enumerate(beats):
            try:
                raw = self.generate_image(
                    self._image_prompt({**prompt, "hook": beat}, business_profile),
                    REEL_W, REEL_H, f"{base_name}-beat{i}",
                )
                stills.append(
                    self.overlay_text(raw, beat, brand_color=self._brand_color(business_profile))
                )
            except CreativeForgeError as exc:
                warnings.append(f"Beat {i} image skipped: {exc}")
                logger.warning("::warning::%s", warnings[-1])
        if not stills:
            raise CreativeForgeError("every beat image failed")

        audio_path: str | None = None
        durations: list[float] = []
        try:
            audio_path, durations = self._tts_beats(beats, base_name)
        except CreativeForgeError as exc:
            warnings.append(f"Voiceover unavailable this run: {exc}")
            logger.warning("::warning::%s", warnings[-1])

        if not durations:
            durations = [3.2] * len(stills)  # silent fallback timing
        else:
            durations = durations[: len(stills)] + [3.0] * max(0, len(stills) - len(durations))

        out_path = self.output_dir / f"{base_name}-reel.mp4"
        self._assemble_video(stills, durations, audio_path, str(out_path))
        thumbnail = self._extract_thumbnail(str(out_path), base_name)
        return {
            "asset_path": str(out_path),
            "asset_type": "reel",
            "thumbnail": thumbnail,
            "warnings": warnings,
        }

    def _tts_beats(self, beats: list[str], base_name: str) -> tuple[str, list[float]]:
        """Piper TTS per beat. Returns (concatenated wav, per-beat durations).
        Raises CreativeForgeError when Piper/voice model unavailable."""
        piper = shutil.which("piper") or shutil.which("piper.exe")
        model = self._ensure_voice_model()
        work = self.output_dir / f"{self._slug(base_name)}-audio"
        work.mkdir(parents=True, exist_ok=True)

        wavs: list[str] = []
        for i, beat in enumerate(beats):
            wav = work / f"beat{i}.wav"
            proc = subprocess.run(  # noqa: S603 - fixed argv
                [piper, "--model", str(model), "--output_file", str(wav)],
                input=beat.encode("utf-8"),
                capture_output=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise CreativeForgeError(f"piper exit {proc.returncode}: {proc.stderr[-200:]!r}")
            wavs.append(str(wav))

        durations = [self._wav_duration(w) for w in wavs]
        concat_list = work / "list.txt"
        concat_list.write_text("\n".join(f"file '{w}'" for w in wavs), encoding="utf-8")
        full = work / "voiceover.wav"
        subprocess.run(  # noqa: S603 - fixed argv
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c", "copy", str(full)],
            capture_output=True, timeout=120, check=True,
        )
        return str(full), durations

    def _ensure_voice_model(self) -> Path:
        onnx = VOICE_CACHE_DIR / "en_US-lessac-medium.onnx"
        jsonf = VOICE_CACHE_DIR / "en_US-lessac-medium.onnx.json"
        if onnx.exists() and jsonf.exists():
            return onnx
        VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Piper voice model (~64MB) to %s", VOICE_CACHE_DIR)
        for suffix in ("", ".json"):
            target = Path(str(onnx) + suffix)
            resp = httpx.get(VOICE_MODEL_URL + suffix, timeout=300, follow_redirects=True)
            resp.raise_for_status()
            target.write_bytes(resp.content)
        return onnx

    @staticmethod
    def _wav_duration(path: str) -> float:
        proc = subprocess.run(  # noqa: S603 - fixed argv
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, timeout=60,
        )
        try:
            return float(proc.stdout.decode().strip())
        except ValueError:
            return 3.0

    def _assemble_video(
        self, stills: list[str], durations: list[float], audio: str | None, out_path: str
    ) -> None:
        """Ken Burns zoom on each still for its duration; xfade between beats;
        mux voiceover. Falls back to plain concat if xfade fails."""
        seg_files: list[str] = []
        try:
            for i, (still, dur) in enumerate(zip(stills, durations)):
                seg = out_path.replace(".mp4", f"-seg{i}.mp4")
                zoom = 1.0 + 0.08 * (dur / max(sum(durations), 1)) * len(stills)
                cmd = [
                    "ffmpeg", "-y", "-loop", "1", "-i", still, "-t", f"{dur:.2f}",
                    "-vf",
                    f"scale={REEL_W}:{REEL_H},zoompan=z='min(zoom+{zoom / max(dur, 0.5) / 25:.5f}',"
                    f"d={int(dur * 25)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={REEL_W}x{REEL_H},"
                    "fade=t=in:st=0:d=0.35,fade=t=out:st=%.2f:d=0.35,format=yuv420p" % max(dur - 0.35, 0),
                    "-r", "25", "-c:v", "libx264", "-preset", "veryfast", seg,
                ]
                subprocess.run(cmd, capture_output=True, timeout=300, check=True)  # noqa: S603
                seg_files.append(seg)

            if len(seg_files) == 1:
                tmp = seg_files[0]
            else:
                inputs: list[str] = []
                for seg in seg_files:
                    inputs += ["-i", seg]
                filter_parts = []
                offset = 0.0
                prev = "[0:v]"
                for idx in range(1, len(seg_files)):
                    offset += durations[idx - 1] - 0.35
                    out_lbl = f"[v{idx}]"
                    filter_parts.append(
                        f"{prev}[{idx}:v]xfade=transition=fade:duration=0.35:offset={offset:.2f}{out_lbl}"
                    )
                    prev = out_lbl
                fc = ";".join(filter_parts)
                xfade_cmd = (
                    ["ffmpeg", "-y"] + inputs
                    + ["-filter_complex", fc, "-map", prev]
                )
                tmp = out_path.replace(".mp4", "-xfade.mp4")
                proc = subprocess.run(xfade_cmd, capture_output=True, timeout=600)  # noqa: S603
                if proc.returncode != 0:
                    logger.warning("::warning::xfade failed; falling back to plain concat")
                    tmp = self._plain_concat(seg_files, out_path)
                elif os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                    pass
                else:
                    tmp = self._plain_concat(seg_files, out_path)

            mux_cmd = ["ffmpeg", "-y", "-i", tmp]
            if audio and os.path.exists(audio):
                mux_cmd += ["-i", audio, "-c:v", "copy", "-c:a", "aac", "-shortest"]
            else:
                mux_cmd += ["-c:v", "copy"]
            mux_cmd += [out_path]
            subprocess.run(mux_cmd, capture_output=True, timeout=300, check=True)  # noqa: S603
        except Exception as exc:
            raise CreativeForgeError(f"ffmpeg assembly failed: {exc}") from exc
        finally:
            for seg in seg_files:
                try:
                    os.remove(seg)
                except OSError:
                    pass

    @staticmethod
    def _plain_concat(seg_files: list[str], out_path: str) -> str:
        lst = out_path.replace(".mp4", "-list.txt")
        Path(lst).write_text("\n".join(f"file '{s}'" for s in seg_files), encoding="utf-8")
        tmp = out_path.replace(".mp4", "-concat.mp4")
        subprocess.run(  # noqa: S603 - fixed argv
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", tmp],
            capture_output=True, timeout=300, check=True,
        )
        return tmp

    def _extract_thumbnail(self, video_path: str, base_name: str) -> str | None:
        thumb = self.output_dir / f"{self._slug(base_name)}-thumb.png"
        try:
            subprocess.run(  # noqa: S603 - fixed argv
                ["ffmpeg", "-y", "-i", video_path, "-ss", "00:00:01", "-frames:v", "1", str(thumb)],
                capture_output=True, timeout=120, check=True,
            )
            return str(thumb)
        except Exception as exc:
            logger.warning("::warning::Thumbnail extraction failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _brand_color(business_profile: dict[str, Any]) -> tuple[int, int, int]:
        name = (business_profile.get("name") or "").lower()
        palette = {
            "bakery": (146, 64, 14),
            "cake": (190, 84, 20),
            "jewel": (88, 28, 105),
            "decor": (21, 94, 117),
            "fitness": (21, 128, 61),
        }
        for key, color in palette.items():
            if key in name:
                return color
        return (31, 111, 104)

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "asset"
