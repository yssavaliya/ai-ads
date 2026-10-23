"""Reproducible pipeline benchmark with SIMULATED network latency.

It runs the real DailyPipeline end to end (profile -> competitors -> audience -> trends -> virality
gate -> asset rendering -> delivery) but replaces every network edge with a stub that sleeps for a
fixed, realistic duration. Because the stubs are deterministic, the wall-clock difference between two
runs is purely the pipeline's own scheduling (sequential vs. concurrent), not provider noise.

CPU-bound media assembly (ffmpeg/Piper) is stubbed out and is NOT part of these numbers.

    python scripts/benchmark_pipeline.py                     # defaults below
    python scripts/benchmark_pipeline.py --llm 1.0 --image 2.0 --apify 3.0 --businesses 3
    LLM_CONCURRENCY=1 IMAGE_CONCURRENCY=1 FETCH_CONCURRENCY=1 python scripts/benchmark_pipeline.py   # sequential

Prints one JSON document so runs can be diffed.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_TMP = Path(tempfile.mkdtemp(prefix="ai-ads-bench-"))
os.environ.setdefault("DATABASE_PATH", str(_TMP / "bench.db"))
os.environ["DRY_RUN"] = "true"
os.environ["LLM_KEYLESS_FALLBACK"] = "false"
for _k in ("GROQ_API_KEY", "GEMINI_API_KEY", "HF_TOKEN", "GITHUB_TOKEN", "APIFY_TOKEN"):
    os.environ[_k] = ""

import logging  # noqa: E402

logging.disable(logging.CRITICAL)

from PIL import Image  # noqa: E402

from src.shared.config import settings  # noqa: E402

settings.data_dir = _TMP / "data"
settings.logs_dir = _TMP / "logs"
settings.site_dir = _TMP / "site"
settings.templates_dir = ROOT / "templates"
settings.data_dir.mkdir()


def _png(w: int = 270, h: int = 480) -> bytes:
    buf = io.BytesIO()
    Image.frombytes("RGB", (w, h), os.urandom(w * h * 3)).save(buf, "PNG")
    return buf.getvalue()


def _canned_llm(label: str) -> object:
    types = ["educational", "storytelling", "social_proof"]
    if label == "competitor-discovery":
        return [{"name": f"Rival {i}", "platform": "instagram", "handle": f"rival{i}", "reason": "r"} for i in range(5)]
    if label == "audience-mapper":
        return {"personas": [{"name": "Home bakers", "best_posting_times": "Tue 7 PM"}], "top_pain_points": [], "content_themes": []}
    if label == "trend-radar":
        return {"trending_topics": [], "trending_hashtags": ["#x"]}
    if label == "prompt-forge":
        return [
            {"prompt_type": types[i % 3], "title": f"Idea {i}", "hook": f"Hook {i}", "prompt_text": f"Brief {i} " + "detail " * 30,
             "hashtags": ["#a"], "platform_focus": "instagram", "format": "reel" if i == 0 else "single image",
             "angle": "a", "audience_segment": "bakers", "reel_beats": ["beat one", "beat two", "beat three", "beat four"] if i == 0 else []}
            for i in range(5)
        ]
    if label == "virality-predictor":
        return {"virality_score": 90, "virality_label": "HIGH", "virality_probability": 0.9, "key_factors": [], "risks": [],
                "reasoning": "", "improvement_hints": ["x"]}
    raise AssertionError(label)


def install_stubs(llm: float, image: float, apify: float) -> dict[str, int]:
    from src.fetch import social_scraper
    from src.fetch.website_scraper import WebsiteScraper
    from src.generation import creative_forge, image_providers
    from src.shared.llm_client import LLMClient, LLMResult

    calls = {"llm": 0, "image": 0, "apify": 0}

    def fake_run(self, prompt, label=None, temperature=0.7, max_retries=3, exclude=None):
        calls["llm"] += 1
        time.sleep(llm)
        return LLMResult(json.dumps(_canned_llm(label or "")), "", 0, llm, provider="stub", model="stub")

    LLMClient.run = fake_run  # type: ignore[method-assign]

    def fake_generate(prompt: str, width: int, height: int):
        calls["image"] += 1
        time.sleep(image)
        return _png(width // 8, height // 8), "stub"

    image_providers.generate = fake_generate  # type: ignore[assignment]

    async def fake_fetch(self, handle, limit=20):
        import asyncio

        calls["apify"] += 1
        await asyncio.sleep(apify)
        return {"handle": handle, "follower_count": 1000, "posts": []}

    social_scraper.SocialScraper.fetch_instagram = fake_fetch  # type: ignore[method-assign]
    WebsiteScraper.scrape = lambda self, url: {"title": "Bench", "headings": ["h"], "meta_description": "d"}  # type: ignore[method-assign]
    creative_forge.CreativeForge._tts_beats = lambda self, beats, name: (_ for _ in ()).throw(creative_forge.CreativeForgeError("stub"))  # type: ignore[method-assign]

    def fake_assemble(self, stills, durations, audio, out_path):
        Path(out_path).write_bytes(b"mp4")

    creative_forge.CreativeForge._assemble_video = fake_assemble  # type: ignore[method-assign]
    creative_forge.CreativeForge._extract_thumbnail = lambda self, video, name: None  # type: ignore[method-assign]
    return calls


def bench_daily(llm: float, image: float, apify: float, businesses: int) -> dict:
    import random

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import src.scheduler.daily_job as dj
    from src.storage.models import Base

    eng = create_engine(f"sqlite:///{_TMP / 'daily.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    dj.SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False)  # type: ignore[assignment]
    dj.random.random = lambda: 1.0  # no exploration variant: keep the workload identical across runs
    calls = install_stubs(llm, image, apify)
    random.seed(0)
    pipeline = dj.DailyPipeline()
    stages: dict[str, float] = {}
    started = time.perf_counter()
    for n in range(businesses):
        biz = {"name": f"Bench Biz {n}", "website_url": "https://bench.example", "instagram_handle": f"bench{n}",
               "email": "", "prompt_types": ["educational", "storytelling", "social_proof"], "virality_threshold": 75}
        t0 = time.perf_counter()
        result = pipeline.run_for_business(biz)
        stages[f"business_{n}"] = round(time.perf_counter() - t0, 2)
        assert result["prompts"] == 3 and result["delivered"], result
    total = time.perf_counter() - started
    return {"wall_seconds": round(total, 2), "per_business": stages, "calls": calls,
            "concurrency": {"llm": settings.__dict__.get("llm_concurrency", 1), "image": settings.__dict__.get("image_concurrency", 1),
                            "fetch": settings.__dict__.get("fetch_concurrency", 1)}}


def bench_site(businesses: int, days: int, per_day: int) -> dict:
    from src.delivery.site_publisher import SitePublisher

    site = _TMP / "site-bench"
    pub = SitePublisher(site)
    img = _TMP / "a.png"
    Image.frombytes("RGB", (540, 675), os.urandom(540 * 675 * 3)).save(img)
    # seed `businesses` manifests with `days` x `per_day` entries by publishing directly (not timed)
    from datetime import date, timedelta

    for b in range(businesses):
        slug = f"site-biz-{b}"
        entries = []
        for d in range(days):
            day = (date.today() - timedelta(days=d)).isoformat()
            for k in range(per_day):
                entries.append({
                    "id": f"{day}-{k}", "date": day, "created_at": f"{day}T10:00:0{k}", "prompt_id": k, "title": f"T{k}",
                    "prompt_type": "educational", "score": 80, "below_threshold": False, "threshold": 75, "label": "HIGH",
                    "hook": "hook", "angle": "a", "prompt_text": "brief " * 80, "caption": "caption " * 20, "hashtags": ["#a"] * 5,
                    "first_comment": "fc", "recommended_post_time": "Today 7 PM", "format": "single image", "platform_focus": "instagram",
                    "audience_segment": "s", "reasoning": "r" * 200, "key_factors": ["k"] * 3, "risks": ["r"] * 2, "warnings": [],
                    "asset": None, "feedback_links": {},
                })
        manifest = {"business": {"name": f"Site Biz {b}", "instagram_handle": "", "website_url": ""}, "entries": entries,
                    "weekly": [], "notices": [], "about": None, "path_key": slug}
        pub._save_manifest(slug, manifest)
    t0 = time.perf_counter()
    pub.rebuild()
    full = time.perf_counter() - t0
    biz = {"name": "Site Biz 0", "instagram_handle": "", "website_url": ""}
    t0 = time.perf_counter()
    pub.send_daily_prompts(biz, [{"prompt_id": 999, "title": "new", "prompt_type": "educational", "virality_score": 90,
                                  "prompt_text": "x", "caption": "c", "asset_path": str(img), "asset_type": "image"}])
    publish_one = time.perf_counter() - t0
    return {"businesses": businesses, "entries_each": days * per_day, "full_rebuild_ms": round(full * 1000),
            "publish_one_business_ms": round(publish_one * 1000)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=float, default=0.30, help="simulated seconds per LLM call")
    ap.add_argument("--image", type=float, default=0.40, help="simulated seconds per image-provider call")
    ap.add_argument("--apify", type=float, default=0.50, help="simulated seconds per scrape")
    ap.add_argument("--businesses", type=int, default=2)
    ap.add_argument("--site-businesses", type=int, default=10)
    ap.add_argument("--only", choices=["daily", "site"], default=None)
    args = ap.parse_args()
    report: dict = {"simulated_latency_s": {"llm": args.llm, "image": args.image, "apify": args.apify}}
    if args.only in (None, "daily"):
        report["daily"] = bench_daily(args.llm, args.image, args.apify, args.businesses)
    if args.only in (None, "site"):
        report["site"] = bench_site(args.site_businesses, days=45, per_day=3)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
