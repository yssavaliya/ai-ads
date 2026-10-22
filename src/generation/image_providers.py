"""Free image generators tried in order, plus a local branded-card renderer that
can never fail. Each provider returns image bytes or None; transport errors are
classified and fed into the shared health tracker so a rate-limited or
unauthorised provider is skipped for a while instead of being hammered.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from src.shared.config import settings
from src.shared.llm_providers import HEALTH, classify, csv

logger = logging.getLogger(__name__)

DEFAULT_ORDER = "pollinations,gemini,huggingface,together,cloudflare"

HF_IMAGE_URLS = (
    "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",  # legacy
)


def is_image(data: bytes | None) -> bool:
    return bool(data) and (data[:4] == b"\x89PNG" or data[:2] == b"\xff\xd8")  # type: ignore[index]


def _snap(value: int, multiple: int = 8, lo: int = 256, hi: int = 1440) -> int:
    return max(lo, min(hi, value // multiple * multiple))


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
def pollinations(prompt: str, width: int, height: int) -> bytes | None:
    """Keyless."""
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        resp = client.get(
            f"https://image.pollinations.ai/prompt/{quote(prompt, safe='')}",
            params={"width": width, "height": height, "nologo": "true", "model": "flux"},
        )
        resp.raise_for_status()
        data = resp.content
    if len(data) < 1000 or not is_image(data):
        logger.error("Pollinations returned a non-image payload (%d bytes)", len(data))
        return None
    return data


def huggingface(prompt: str, width: int, height: int) -> bytes | None:
    if not settings.hf_token:
        return None
    last: Exception | None = None
    for url in HF_IMAGE_URLS:
        try:
            with httpx.Client(timeout=180) as client:
                resp = client.post(
                    url,
                    headers={"Authorization": f"Bearer {settings.hf_token}", "Content-Type": "application/json"},
                    json={"inputs": prompt, "parameters": {"width": width, "height": height}},
                )
                resp.raise_for_status()
                data = resp.content
            if is_image(data):
                return data
            logger.error("HF inference (%s) returned a non-image payload (%d bytes)", url, len(data))
        except Exception as exc:  # noqa: BLE001 - try the legacy URL, then report
            last = exc
            logger.error("HF image inference failed at %s: %s", url, exc)
    if last is not None:
        raise last
    return None


def gemini_image(prompt: str, width: int, height: int) -> bytes | None:
    """Free tier of the Gemini image models (re-uses GEMINI_API_KEY); each model listed in
    GEMINI_IMAGE_MODEL is tried in order."""
    if not settings.gemini_api_key:
        return None
    last: Exception | None = None
    for model in csv(settings.gemini_image_model):
        try:
            with httpx.Client(timeout=180) as client:
                resp = client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": settings.gemini_api_key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:  # noqa: BLE001 - try the next model, then report
            last = exc
            logger.error("Gemini image model %s failed: %s", model, exc)
            continue
        for candidate in payload.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    data = base64.b64decode(inline["data"])
                    if is_image(data):
                        return data
        logger.error("Gemini image model %s returned no image", model)
    if last is not None:
        raise last
    return None


def together(prompt: str, width: int, height: int) -> bytes | None:
    if not settings.together_api_key:
        return None
    with httpx.Client(timeout=180) as client:
        resp = client.post(
            "https://api.together.xyz/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.together_api_key}"},
            json={
                "model": settings.together_image_model, "prompt": prompt[:1500], "steps": 4, "n": 1,
                "width": _snap(width), "height": _snap(height), "response_format": "b64_json",
            },
        )
        resp.raise_for_status()
        items = resp.json().get("data") or []
    if items and items[0].get("b64_json"):
        data = base64.b64decode(items[0]["b64_json"])
        return data if is_image(data) else None
    return None


def cloudflare(prompt: str, width: int, height: int) -> bytes | None:
    if not (settings.cloudflare_account_id and settings.cloudflare_api_token):
        return None
    with httpx.Client(timeout=180) as client:
        resp = client.post(
            f"https://api.cloudflare.com/client/v4/accounts/{settings.cloudflare_account_id}"
            "/ai/run/@cf/black-forest-labs/flux-1-schnell",
            headers={"Authorization": f"Bearer {settings.cloudflare_api_token}"},
            json={"prompt": prompt[:2048], "steps": 4},
        )
        resp.raise_for_status()
        encoded = (resp.json().get("result") or {}).get("image")
    if encoded:
        data = base64.b64decode(encoded)
        return data if is_image(data) else None
    return None


@dataclass(frozen=True)
class ImageProvider:
    name: str
    fn: Callable[[str, int, int], bytes | None]
    configured: Callable[[], bool]
    exact_size: bool  # False => output must be cover-cropped to the requested size


REGISTRY: dict[str, ImageProvider] = {
    p.name: p
    for p in (
        ImageProvider("pollinations", pollinations, lambda: True, True),
        ImageProvider("gemini", gemini_image, lambda: bool(settings.gemini_api_key), False),
        ImageProvider("huggingface", huggingface, lambda: bool(settings.hf_token), False),
        ImageProvider("together", together, lambda: bool(settings.together_api_key), False),
        ImageProvider("cloudflare", cloudflare,
                      lambda: bool(settings.cloudflare_account_id and settings.cloudflare_api_token), False),
    )
}


def provider_order() -> list[str]:
    return csv(settings.image_provider_order) or csv(DEFAULT_ORDER)


def generate(prompt: str, width: int, height: int) -> tuple[bytes, str] | None:
    """First image any available provider produces, as (bytes, provider). None if all fail."""
    for name in provider_order():
        provider = REGISTRY.get(name)
        if provider is None or not provider.configured() or not HEALTH.available(f"img:{name}"):
            continue
        try:
            data = provider.fn(prompt, width, height)
        except Exception as exc:  # noqa: BLE001 - network/SDK errors of every kind
            failure = classify(exc)
            HEALTH.mark(f"img:{name}", None, failure)
            logger.error("Image provider %s failed (%s): %s", name, failure.kind, exc)
            continue
        if not is_image(data):
            logger.warning("Image provider %s returned nothing usable", name)
            HEALTH.mark(f"img:{name}", None, classify(Exception("empty")))
            continue
        assert data is not None
        if not provider.exact_size:
            data = fit_cover(data, width, height)
        return data, name
    return None


# ---------------------------------------------------------------------------
# local rendering (no network, cannot fail)
# ---------------------------------------------------------------------------
def fit_cover(data: bytes, width: int, height: int) -> bytes:
    """Resize + centre-crop to exactly width x height (providers differ in aspect)."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.size == (width, height):
        return data
    scale = max(width / img.width, height / img.height)
    resized = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS)
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    out = io.BytesIO()
    resized.crop((left, top, left + width, top + height)).save(out, "PNG")
    return out.getvalue()


def branded_card(width: int, height: int, color: tuple[int, int, int], seed: str = "") -> bytes:
    """Deterministic, on-brand gradient background with soft shapes. Used when every
    AI image provider is down/limited so a post still ships with a clean visual
    (the hook text is overlaid afterwards exactly as for AI images)."""
    from PIL import Image, ImageDraw, ImageFilter

    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16))
    top = tuple(min(255, int(c * 1.35) + 25) for c in color)
    bottom = tuple(int(c * 0.55) for c in color)

    # vertical gradient, built 1px wide then stretched (fast at 1080x1920)
    column = Image.new("RGB", (1, height))
    for y in range(height):
        t = y / max(height - 1, 1)
        column.putpixel((0, y), tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    base = column.resize((width, height)).convert("RGBA")

    shapes = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shapes)
    for _ in range(7):
        radius = rng.randint(width // 8, width // 2)
        cx, cy = rng.randint(0, width), rng.randint(0, height)
        tint = tuple(min(255, c + rng.randint(30, 90)) for c in color)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(*tint, rng.randint(25, 60)))
    shapes = shapes.filter(ImageFilter.GaussianBlur(radius=max(width // 40, 4)))

    out = io.BytesIO()
    Image.alpha_composite(base, shapes).convert("RGB").save(out, "PNG")
    return out.getvalue()
