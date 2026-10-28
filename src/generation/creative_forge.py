from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from src.generation import image_providers
from src.shared.concurrency import parallel_map
from src.shared.config import settings

logger = logging.getLogger(__name__)

FEED_W = 1080
FEED_H = 1350
REEL_W = 1080
REEL_H = 1920
FPS = 25
DEFAULT_BRAND = (31, 111, 104)
MAX_ZOOM_GAIN = 0.08  # Ken Burns: end at 108% of the frame

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
        # Per-render state lives in thread-local storage: renders for several prompts (and several Reel
        # beats) run on worker threads, and a shared colour/warning list would leak between them.
        self._tl = threading.local()

    @property
    def _card_color(self) -> tuple[int, int, int]:
        return getattr(self._tl, "color", DEFAULT_BRAND)  # type: ignore[no-any-return]

    @_card_color.setter
    def _card_color(self, value: tuple[int, int, int]) -> None:
        self._tl.color = value

    @property
    def _pending_warnings(self) -> list[str]:
        if not hasattr(self._tl, "warnings"):
            self._tl.warnings = []
        return self._tl.warnings  # type: ignore[no-any-return]

    @_pending_warnings.setter
    def _pending_warnings(self, value: list[str]) -> None:
        self._tl.warnings = value

    @staticmethod
    def _run_ffmpeg(cmd: list[str], timeout: int, what: str) -> None:
        """Run ffmpeg; on failure raise with ffmpeg's own error (the bare CalledProcessError
        only said "exit status 4294967294", which hid the real cause in production)."""
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)  # noqa: S603 - fixed argv
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-4:]
            raise CreativeForgeError(f"{what}: ffmpeg exit {proc.returncode}: {' | '.join(tail)[:400]}")

    @staticmethod
    def _write_concat_list(paths: list[str], list_path: Path) -> None:
        """ffmpeg's concat demuxer resolves RELATIVE entries against the list file's own
        folder, so a list holding 'logs/assets/x.mp4' next to logs/assets/ looked for
        logs/assets/logs/assets/x.mp4. Always write absolute, quote-escaped paths."""
        lines = []
        for p in paths:
            absolute = Path(p).resolve().as_posix().replace("'", "'\\''")
            lines.append(f"file '{absolute}'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _take_warnings(self) -> list[str]:
        taken, self._pending_warnings = self._pending_warnings, []
        return taken

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
        self._take_warnings()  # discard leftovers from a previous render
        self._card_color = self._brand_color(business_profile)
        is_reel = str(prompt.get("format", "")).lower() == "reel" and prompt.get("reel_beats")
        base_name = self._slug(f"{business_profile.get('name', 'biz')}-{prompt.get('title', 'post')}")[:60]

        if is_reel:
            try:
                result = self._render_reel(prompt, business_profile, base_name)
                result["warnings"] += warnings
                return result
            except Exception as exc:  # noqa: BLE001 - any reel failure must fall back to an image
                warnings.append(f"Reel render failed ({exc}); falling back to single image")
                logger.warning("::warning::%s", warnings[-1])

        image_path = self.generate_image(
            self._image_prompt(prompt, business_profile), FEED_W, FEED_H, f"{base_name}-{index}"
        )
        category = prompt.get("hook_type") or prompt.get("prompt_type")
        brand_handle = business_profile.get("instagram_handle") or business_profile.get("name")
        out_target = str(self.output_dir / f"{base_name}-{index}-final.png")
        try:
            final_path = self.overlay_text(
                image_path,
                prompt.get("hook") or prompt.get("title") or "",
                brand_color=self._brand_color(business_profile),
                out_path=out_target,
                category=category,
                brand_handle=brand_handle,
            )
        except TypeError:
            final_path = self.overlay_text(
                image_path,
                prompt.get("hook") or prompt.get("title") or "",
                brand_color=self._brand_color(business_profile),
                out_path=out_target,
            )
        return {"asset_path": str(final_path), "asset_type": "image", "warnings": warnings + self._take_warnings()}

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    def generate_image(self, image_prompt: str, width: int, height: int, name: str) -> str:
        """Ask the free image providers in order; if every one is down or rate limited
        render a local branded card so the post still has a visual (and say so)."""
        generated = image_providers.generate(image_prompt, width, height)
        if generated is not None:
            data, provider = generated
        elif settings.image_card_fallback:
            data, provider = image_providers.branded_card(width, height, self._card_color, seed=name), "local-card"
            message = "All AI image providers failed or are rate limited — used a branded card background instead"
            self._pending_warnings.append(message)
            logger.warning("::warning::%s (%s)", message, name)
        else:
            raise CreativeForgeError("All image providers failed (" + ", ".join(image_providers.provider_order()) + ")")
        ext = ".jpg" if data[:2] == b"\xff\xd8" else ".png"
        out_path = self.output_dir / f"{self._slug(name)}-{width}x{height}{ext}"
        out_path.write_bytes(data)
        logger.info("Rendered image %s via %s (%d bytes)", out_path, provider, len(data))
        return str(out_path)

    # Thin shims kept for callers/tests that target a specific provider.
    def _pollinations_image(self, image_prompt: str, width: int, height: int) -> bytes | None:
        try:
            return image_providers.pollinations(image_prompt, width, height)
        except Exception as exc:  # noqa: BLE001
            logger.error("Pollinations image failed: %s", exc)
            return None

    def _hf_image(self, image_prompt: str, width: int, height: int) -> bytes | None:
        try:
            return image_providers.huggingface(image_prompt, width, height)
        except Exception as exc:  # noqa: BLE001
            logger.error("HF image inference failed: %s", exc)
            return None

    @staticmethod
    def _image_prompt(prompt: dict[str, Any], business_profile: dict[str, Any]) -> str:
        # Prioritize explicit visual scene instructions engineered for diffusion models
        visual_scene = str(prompt.get("visual_scene") or "").strip()
        if visual_scene:
            return (
                f"{visual_scene}, cinematic studio lighting, 35mm photography, high contrast, tactile textures, "
                f"hyper-detailed, award-winning composition, 8k resolution, no text, no words, no letters"
            )[:800]

        # Synthesize an editorial photographic scene rather than concatenating text hooks
        name = str(business_profile.get("name") or "")
        name_lower = name.lower()
        industry_visuals = {
            "aimagix": "modern minimalist creative AI tech agency, sleek floating glass desk, glowing indigo and violet rim lighting, cinematic 35mm photography, shallow depth of field, premium dark aesthetic",
            "fitness": "high-end modern athletic performance training gym, dynamic dramatic rim lighting, premium matte fitness equipment, cinematic editorial fitness photography, high contrast",
            "bakery": "artisanal gourmet sourdough and pastries on rustic dark marble counter, warm natural morning window light, shallow depth of field, editorial food magazine photography",
            "cake": "artisanal gourmet cake decorating studio, warm soft cinematic lighting, shallow depth of field, editorial food photography",
            "jewel": "luxury bespoke diamond jewelry on black velvet podium, focused directional macro spotlight, sparkling reflections, editorial commercial jewelry photography",
            "decor": "stylish Scandinavian interior living room, natural sunlight through sheer curtains, warm architectural textures, architectural digest photography",
        }
        base_visual = None
        for key, vis in industry_visuals.items():
            if key in name_lower:
                base_visual = vis
                break

        if not base_visual:
            base_visual = (
                f"editorial commercial photography for {name}, modern high-end studio setting, "
                f"cinematic rim lighting, 35mm lens, f/1.8, shallow depth of field, rich color grading"
            )

        hook = str(prompt.get("hook") or prompt.get("title") or "").strip()
        hook_ctx = f", {hook}" if hook else ""
        angle = str(prompt.get("angle") or "").strip()
        angle_ctx = f", depicting {angle}" if angle else ""
        return (
            f"{base_visual}{hook_ctx}{angle_ctx}, 8k resolution, award-winning composition, tactile textures, photorealistic, no text, no words, no letters"
        )[:800]

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
        category: str | None = None,
        brand_handle: str | None = None,
    ) -> str:
        """Composite hook text over a sleek, modern gradient scrim vignette with
        category badge and brand footer inside Instagram's safe area."""
        from PIL import Image, ImageDraw, ImageFont

        if not text:
            return image_path
        img = Image.open(image_path).convert("RGBA")
        w, h = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font_size = max(38, min(62, w // 17))
        font = self._load_font(font_size)
        cat_font = self._load_font(max(20, min(28, int(font_size * 0.44))))
        footer_font = self._load_font(max(18, min(24, int(font_size * 0.38))))

        lines = self._wrap_text(draw, text.upper(), font, max_width=int(w * 0.86))
        line_height = int(font_size * 1.25)
        text_block_h = line_height * len(lines)

        # Scrim: smooth quadratic gradient over bottom ~44-50%
        scrim_h = max(int(h * 0.44), min(int(h * 0.58), text_block_h + 360))
        scrim_top = h - scrim_h

        # Render gradient line by line
        for y in range(scrim_top, h):
            prog = (y - scrim_top) / scrim_h
            alpha = int(235 * (prog ** 1.6))
            # Deep rich dark tone matching brand subtle warmth/coolness
            r, g, b = brand_color
            scrim_rgb = (max(8, r // 10), max(10, g // 10), max(16, b // 8))
            draw.line([(0, y), (w, y)], fill=(*scrim_rgb, alpha))

        # 1. Category Pill Badge (if category given)
        cur_y = scrim_top + max(35, int((scrim_h - text_block_h - 140) * 0.35))
        if category:
            cat_label = f"✦ {str(category).replace('_', ' ').strip().upper()} ✦"
            cat_w = draw.textlength(cat_label, font=cat_font)
            pill_pad_x = 20
            pill_pad_y = 7
            pill_w = cat_w + 2 * pill_pad_x
            pill_h = int(getattr(cat_font, "size", 24)) + 2 * pill_pad_y
            pill_x1 = (w - pill_w) // 2
            pill_x2 = pill_x1 + pill_w
            pill_y1 = cur_y
            pill_y2 = pill_y1 + pill_h

            # Pill fill with brand color & subtle white outline
            draw.rounded_rectangle([pill_x1, pill_y1, pill_x2, pill_y2], radius=14, fill=(*brand_color, 215), outline=(255, 255, 255, 140), width=1)
            text_x = pill_x1 + pill_pad_x
            draw.text((text_x, pill_y1 + pill_pad_y - 1), cat_label, font=cat_font, fill=(255, 255, 255, 255))
            cur_y = pill_y2 + 22
        else:
            cur_y = scrim_top + max(40, (scrim_h - text_block_h - 80) // 2)

        # 2. Main Headline Hook Text (centered with crisp drop shadow)
        for line in lines:
            line_w = draw.textlength(line, font=font)
            x = (w - line_w) // 2
            # Drop shadow for high contrast & 3D pop
            draw.text((x + 2, cur_y + 3), line, font=font, fill=(0, 0, 0, 190))
            draw.text((x, cur_y), line, font=font, fill=(255, 255, 255, 255))
            cur_y += line_height

        # 3. Brand Handle Footer (keeps 6% bottom safe area clear)
        if brand_handle:
            clean_handle = brand_handle if brand_handle.startswith("@") else f"@{brand_handle}"
            foot_text = f"• {clean_handle} •"
            foot_w = draw.textlength(foot_text, font=footer_font)
            foot_x = (w - foot_w) // 2
            foot_y = h - int(h * 0.055)
            draw.text((foot_x + 1, foot_y + 1), foot_text, font=footer_font, fill=(0, 0, 0, 140))
            draw.text((foot_x, foot_y), foot_text, font=footer_font, fill=(215, 225, 235, 190))

        combined = Image.alpha_composite(img, overlay).convert("RGB")
        out = out_path or str(Path(image_path).with_name(Path(image_path).stem + "-texted.png"))
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
        card_color = self._card_color

        def make_still(job: tuple[int, str]) -> tuple[str, list[str]]:
            i, beat = job
            self._card_color = card_color  # this worker thread's own copy
            self._take_warnings()
            raw = self.generate_image(
                self._image_prompt({**prompt, "hook": beat}, business_profile),
                REEL_W, REEL_H, f"{base_name}-beat{i}",
            )
            return self.overlay_text(raw, beat, brand_color=self._brand_color(business_profile)), self._take_warnings()

        # The beat images are independent network calls: generate them concurrently (capped by
        # IMAGE_CONCURRENCY, order preserved). A failed beat is skipped exactly as before.
        jobs = list(enumerate(beats))
        for (i, _beat), outcome in zip(jobs, parallel_map(make_still, jobs, settings.image_concurrency)):
            if outcome.ok:
                still, beat_warnings = outcome.value
                stills.append(still)
                self._pending_warnings.extend(beat_warnings)
            elif isinstance(outcome.error, CreativeForgeError):
                warnings.append(f"Beat {i} image skipped: {outcome.error}")
                logger.warning("::warning::%s", warnings[-1])
            else:
                outcome.unwrap()  # anything unexpected still aborts the reel (-> image fallback)
        if not stills:
            raise CreativeForgeError("every beat image failed")

        audio_path: str | None = None
        durations: list[float] = []
        try:
            audio_path, durations = self._tts_beats(beats, base_name)
        except Exception as exc:  # noqa: BLE001 - voiceover is optional; never kill the reel
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
            "warnings": warnings + self._take_warnings(),
        }

    def _tts_beats(self, beats: list[str], base_name: str) -> tuple[str, list[float]]:
        """Piper TTS per beat. Returns (concatenated wav, per-beat durations).
        Raises CreativeForgeError when Piper/voice model unavailable."""
        piper = shutil.which("piper") or shutil.which("piper.exe")
        if not piper:
            raise CreativeForgeError("piper binary not found on PATH")
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
        self._write_concat_list(wavs, concat_list)
        full = work / "voiceover.wav"
        self._run_ffmpeg(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(full)],
            120, "voiceover concat",
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
        """Duration of a WAV in seconds using the stdlib (no ffprobe on the runner)."""
        try:
            with wave.open(path, "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except (wave.Error, OSError, EOFError, ZeroDivisionError):
            return 3.0

    @staticmethod
    def _segment_filter(dur: float) -> str:
        """ffmpeg -vf chain for one beat: Ken Burns zoom-in + fade in/out.

        zoompan turns ONE input frame into `d` output frames, so the still is
        fed in without -loop. The zoom expression is min(zoom+step, cap)."""
        frames = max(int(dur * FPS), 1)
        step = MAX_ZOOM_GAIN / frames
        cap = 1.0 + MAX_ZOOM_GAIN
        fade_out_at = max(dur - 0.35, 0)
        return (
            f"scale={REEL_W}:{REEL_H},"
            f"zoompan=z='min(zoom+{step:.6f},{cap:.2f})':d={frames}"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={REEL_W}x{REEL_H}:fps={FPS},"
            f"fade=t=in:st=0:d=0.35,fade=t=out:st={fade_out_at:.2f}:d=0.35,format=yuv420p"
        )

    def _assemble_video(
        self, stills: list[str], durations: list[float], audio: str | None, out_path: str
    ) -> None:
        """Ken Burns zoom on each still for its duration; xfade between beats;
        mux voiceover. Falls back to plain concat if xfade fails."""
        seg_files: list[str] = []
        try:
            for i, (still, dur) in enumerate(zip(stills, durations)):
                seg = out_path.replace(".mp4", f"-seg{i}.mp4")
                cmd = [
                    "ffmpeg", "-y", "-i", still,
                    "-vf", self._segment_filter(dur),
                    "-t", f"{dur:.2f}", "-r", str(FPS),
                    "-c:v", "libx264", "-preset", "veryfast", seg,
                ]
                self._run_ffmpeg(cmd, 300, f"beat {i} segment")
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
                tmp = out_path.replace(".mp4", "-xfade.mp4")
                xfade_cmd = (
                    ["ffmpeg", "-y"] + inputs
                    + ["-filter_complex", fc, "-map", prev,
                       "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", tmp]
                )
                try:
                    self._run_ffmpeg(xfade_cmd, 600, "xfade")
                    if not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
                        raise CreativeForgeError("xfade produced no output")
                except CreativeForgeError as exc:
                    logger.warning("::warning::xfade failed (%s); falling back to plain concat", exc)
                    tmp = self._plain_concat(seg_files, out_path)

            mux_cmd = ["ffmpeg", "-y", "-i", tmp]
            if audio and os.path.exists(audio):
                mux_cmd += ["-i", audio, "-c:v", "copy", "-c:a", "aac", "-shortest"]
            else:
                mux_cmd += ["-c:v", "copy"]
            mux_cmd += [out_path]
            self._run_ffmpeg(mux_cmd, 300, "mux")
        except Exception as exc:
            raise CreativeForgeError(f"ffmpeg assembly failed: {exc}") from exc
        finally:
            for seg in seg_files:
                try:
                    os.remove(seg)
                except OSError:
                    pass

    def _plain_concat(self, seg_files: list[str], out_path: str) -> str:
        lst = Path(out_path.replace(".mp4", "-list.txt"))
        self._write_concat_list(seg_files, lst)
        tmp = out_path.replace(".mp4", "-concat.mp4")
        self._run_ffmpeg(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", tmp],
            300, "plain concat",
        )
        return tmp

    def _extract_thumbnail(self, video_path: str, base_name: str) -> str | None:
        thumb = self.output_dir / f"{self._slug(base_name)}-thumb.png"
        try:
            self._run_ffmpeg(
                ["ffmpeg", "-y", "-i", video_path, "-ss", "00:00:01", "-frames:v", "1", str(thumb)],
                120, "thumbnail",
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
            "aimagix": (99, 102, 241),   # Electric indigo
            "studio": (79, 70, 229),    # Royal violet
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
