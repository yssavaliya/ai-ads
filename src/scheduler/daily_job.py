from __future__ import annotations

import json
import logging
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.delivery.channels import DeliveryResult, get_delivery
from src.delivery.github_feedback import GithubFeedback
from src.fetch.competitor_finder import CompetitorFinder
from src.fetch.website_scraper import WebsiteScraper
from src.generation.virality_gate import ViralityGate, apply_threshold_policy
from src.intelligence.audience_mapper import AudienceMapper
from src.intelligence.trend_radar import TrendRadar
from src.learning.bayesian_updater import BayesianUpdater
from src.shared.concurrency import parallel_map
from src.shared.config import settings
from src.shared.utils import load_businesses, utc_now_naive
from src.storage.models import Business, Prompt, PromptType, SessionLocal

logger = logging.getLogger(__name__)


class DailyPipeline:
    def __init__(self) -> None:
        self.website = WebsiteScraper()
        self.competitors = CompetitorFinder()
        self.audience = AudienceMapper()
        self.trends = TrendRadar()
        self.gate = ViralityGate()
        self.delivery = get_delivery()
        self.feedback = GithubFeedback()
        self.weights = BayesianUpdater()

    def run_for_business(self, business: dict[str, Any]) -> dict[str, Any]:
        name = business["name"]
        logger.info("=== Daily pipeline for %s ===", name)

        db = SessionLocal()
        try:
            biz = self._get_or_create_business(db, business)
            profile = self._load_profile(db, biz)

            if not self._profile_has_content(profile):
                # A failed first scrape leaves a truthy {"name", "source"} stub;
                # keep retrying until we have real website content.
                logger.warning("No usable profile for %s; attempting scrape", name)
                scraped = self._scrape_profile(business)
                profile = scraped if self._profile_has_content(scraped) else (profile or scraped)
                biz.profile_json = profile
                db.add(biz)
                db.commit()

            profile = profile or {}
            def _competitors() -> Any:
                return self._cached_intel(
                    business, "competitors", lambda: self.competitors.discover_and_fetch(profile)
                )

            def _own_posts() -> dict[str, Any]:
                if not business.get("instagram_handle"):
                    return {}
                from src.fetch.social_scraper import fetch_instagram_sync

                return fetch_instagram_sync(business["instagram_handle"])

            # Competitor discovery (an LLM call plus up to five scrapes) and the business's own
            # profile scrape do not depend on each other: overlap them.
            competitors_out, social_out = parallel_map(lambda job: job(), [_competitors, _own_posts], settings.fetch_concurrency)
            competitors = competitors_out.unwrap()
            social_data = social_out.unwrap()

            # AUTO-LEARN first: match the business's own recent posts to prompts
            # we sent before, score real engagement, update Bayesian weights.
            from src.learning.auto_learner import AutoLearner

            auto_result = AutoLearner().run_for_business(biz.id, social_data, db)
            logger.info(
                "Auto-learning for %s: matched=%s learned=%s",
                name, auto_result.get("matched"), auto_result.get("learned"),
            )

            audience_map = self._cached_intel(
                business, "audience",
                lambda: self.audience.map_audience(
                    profile, social_data, competitors, biz.audience_personas_json
                ),
            )
            # Never overwrite good, previously-learned personas with an empty
            # map from a transient audience-mapping outage (would erase learning).
            if audience_map:
                biz.audience_personas_json = audience_map
                db.add(biz)
                db.commit()

            trend_data = self._cached_intel(
                business, "trends",
                lambda: self.trends.scan(profile, audience_map, competitors),
            )
            weights = self.weights.to_forge_weights(biz.weights_key)
            history = self._load_history(db, biz)
            threshold = int(business.get("virality_threshold") or biz.virality_threshold or settings.virality_threshold)
            prompts = self.gate.run(
                profile,
                audience_map,
                trend_data,
                business.get("prompt_types", ["educational", "storytelling", "social_proof"]),
                weights,
                history=history,
                threshold=threshold,
            )

            # NOVELTY (#2): don't re-send a prompt we already sent in the last
            # 14 days — replace near-duplicates so the feed stays fresh.
            recent_texts = self._recent_prompt_texts(db, biz)
            if recent_texts:
                kept = self._dedupe_prompts(prompts, recent_texts)
                if kept:
                    logger.info("Dedup: %d/%d prompts kept (dropped near-duplicates)", len(kept), len(prompts))
                    prompts = kept

            # EXPLORATION (gap #5): with modest probability, generate one extra
            # variant in an under-tested format/angle so the learner keeps
            # gathering data on formats it would otherwise never try. Never
            # blocks the main flow — failures are swallowed.
            if random.random() < 0.25:
                try:
                    extra = self._explore_variant(
                        business, profile, audience_map, trend_data, weights, [str(p.get("prompt_type")) for p in prompts]
                    )
                    if extra:
                        prompts.append(extra)
                        logger.info("Exploration: added variant type=%s", extra.get("prompt_type"))
                except Exception as exc:
                    logger.warning("Exploration variant skipped: %s", exc)

            if not prompts:
                logger.error(
                    "::warning::No prompts generated for %s — the daily email will be empty. "
                    "Likely an LLM provider outage; the Drift Check workflow should alert separately.",
                    name,
                )

            # Policy for prompts that never reached THIS business's bar (flag by default, or drop).
            prompts = apply_threshold_policy(prompts)

            # Attach a ready-to-post brief (caption + best time) to every prompt.
            self._attach_posting_brief(business, prompts, social_data, audience_map)

            # Render real creative assets (image inline / Reel MP4 attachment).
            from src.generation.creative_forge import CreativeForge

            creative = CreativeForge()
            render_warnings: list[str] = []
            # Each prompt's asset is an independent image/Reel render: run them concurrently (the image
            # providers are additionally capped by IMAGE_CONCURRENCY so nested work stays bounded).
            jobs = list(enumerate(prompts))
            rendered = parallel_map(lambda job: creative.render_prompt(job[1], profile, index=job[0]), jobs, settings.image_concurrency)
            for (_, p), outcome in zip(jobs, rendered):
                try:
                    asset = outcome.unwrap()
                    p["asset_path"] = asset["asset_path"]
                    p["asset_type"] = asset["asset_type"]
                    p["thumbnail"] = asset.get("thumbnail")
                    if asset.get("warnings"):
                        p.setdefault("warnings", []).extend(asset["warnings"])
                        render_warnings.extend(asset["warnings"])
                except Exception as exc:
                    logger.warning(
                        "::warning::Asset render failed for %s/%s: %s",
                        name, p.get("title"), exc,
                    )
                    render_warnings.append(f"{p.get('title')}: {exc}")

            saved = []
            for p in prompts:
                prompt = Prompt(
                    business_id=biz.id,
                    prompt_text=p["prompt_text"],
                    caption=p.get("caption"),
                    prompt_type=self._prompt_type(p.get("prompt_type", "educational")),
                    virality_score=p.get("virality_score"),
                    virality_label=p.get("virality_label"),
                    virality_probability=p.get("virality_probability"),
                    key_factors=p.get("key_factors"),
                    risks=p.get("risks"),
                    reasoning=p.get("reasoning"),
                    generation_attempt=p.get("generation_attempt", 1),
                    audience_segment=(p.get("audience_segment") or None),
                    asset_path=p.get("asset_path"),
                    asset_type=p.get("asset_type"),
                    format=p.get("format"),
                    sent_at=None,  # stamped only once the email is really sent
                )
                db.add(prompt)
                db.flush()
                links = self.feedback.create_feedback_links(prompt.id, name, p.get("title", ""), prompt.uid)
                saved.append(
                    {
                        **p,
                        "prompt_id": prompt.id,
                        "feedback_links": links,
                    }
                )
            db.commit()

            try:
                delivered = self.delivery.send_daily_prompts(business, saved)
            except Exception as exc:
                # A transient delivery error (SMTP, disk) must not discard a successfully
                # generated batch — log it; prompts are already persisted and
                # the user can be re-notified by re-running the workflow.
                logger.error("::error::Delivery for %s failed: %s", name, exc)
                delivered = DeliveryResult({name: False}, {name: str(exc)[:200]})
            if delivered and saved:
                # Only delivered prompts count as "sent": they feed dedup,
                # history and auto-learning matching.
                now = utc_now_naive()
                for row in saved:
                    obj = db.get(Prompt, row["prompt_id"])
                    if obj is not None:
                        obj.sent_at = now
                db.commit()
            logger.info("Delivery for %s via %s: delivered=%s (%d prompts)", name, "+".join(self.delivery.names), delivered, len(saved))
            return {
                "business": name, "prompts": len(saved), "delivered": bool(delivered),
                "channels": getattr(delivered, "channels", {}),
            }
        finally:
            db.close()

    def _explore_variant(
        self,
        business: dict[str, Any],
        profile: dict[str, Any],
        audience_map: dict[str, Any],
        trend_data: dict[str, Any],
        weights: dict[str, Any],
        existing_types: list[str],
    ) -> dict[str, Any] | None:
        """Gap #5: generate one variant of an under-tested type to keep the
        Bayesian loop exploring the format/angle space instead of exploiting
        only what already works."""
        allowed = business.get("prompt_types", ["educational", "storytelling", "social_proof"])
        candidates = [t for t in allowed if t not in existing_types] or allowed
        # Information-gain: explore the LEAST-tried format first so the loop
        # fills its blind spots instead of randomly sampling proven ones.
        def _explore_score(t: str) -> float:
            w = weights.get(f"topics::{t}") if weights else None
            return 0.0 if w is None else float(w)  # unexplored (absent) ranked best

        chosen = min(candidates, key=_explore_score)
        extras = self.gate.run(profile, audience_map, trend_data, [chosen], weights,
                               threshold=int(business.get("virality_threshold") or settings.virality_threshold))
        for ex in extras:
            if ex.get("prompt_type") == chosen:
                return ex
        return extras[0] if extras else None

    def _attach_posting_brief(
        self,
        business: dict[str, Any],
        prompts: list[dict[str, Any]],
        social_data: dict[str, Any],
        audience_map: dict[str, Any] | None = None,
    ) -> None:
        """Gap #2/#9 + user request: every prompt carries a copy-paste caption
        and a recommended post time, so the human can post manually without
        re-deriving anything."""
        post_time = self._recommend_post_time(business, social_data, audience_map)
        handle = business.get("instagram_handle") or business.get("name") or ""
        for p in prompts:
            hook = (p.get("hook") or "").strip()
            hashtags = " ".join(p.get("hashtags") or [])
            caption = hook
            if hashtags:
                caption = f"{caption}\n\n{hashtags}".strip() if caption else hashtags
            p["caption"] = caption
            p["recommended_post_time"] = post_time
            # First-comment CTA: posting a pinned first comment reliably lifts
            # reach and comment rate, so we hand the user a ready one.
            p["first_comment"] = (
                f"💬 What's your take? Comment below.\n\n"
                f"Follow @{handle} for more."
            )

    def _recommend_post_time(
        self,
        business: dict[str, Any],
        social_data: dict[str, Any],
        audience_map: dict[str, Any] | None = None,
    ) -> str:
        """Recommend the next good slot to post, in the business timezone.
        Prefers the audience map's stated best_posting_times (#3); falls back to
        the hour the business's own posts performed best; else 7 PM local."""
        tz_name = business.get("timezone", "Asia/Kolkata")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("Asia/Kolkata")
        now = datetime.now(tz)

        best = self._audience_best_hour(audience_map)
        if best is None:
            from src.learning.auto_learner import AutoLearner

            hours: list[int] = []
            for post in social_data.get("posts", []):
                dt = AutoLearner._parse_date(post.get("posted_at"))  # naive UTC
                if dt is not None:
                    hours.append(dt.replace(tzinfo=UTC).astimezone(tz).hour)  # the business's local hour
            best = Counter(hours).most_common(1)[0][0] if len(hours) >= 3 else 19

        if now.hour >= best:
            target = (now + timedelta(days=1)).replace(hour=best, minute=0, second=0, microsecond=0)
        else:
            target = now.replace(hour=best, minute=0, second=0, microsecond=0)
        day = "Today" if target.date() == now.date() else "Tomorrow"
        return f"{day} at {target.strftime('%I:%M %p')} {tz_name.replace('_', ' ')}"

    @staticmethod
    def _audience_best_hour(audience_map: dict[str, Any] | None) -> int | None:
        """Parse a 'best_posting_times' like 'Tue-Thu 6-9 PM IST' into an hour."""
        if not audience_map:
            return None
        import re

        blob = " ".join(
            str(p.get("best_posting_times", ""))
            for p in audience_map.get("personas", []) or []
        )
        if not blob:
            return None
        matches = re.findall(r"(\d{1,2})\s*(AM|PM)", blob, re.I)
        if not matches:
            return None
        hours = []
        for h, ap in matches:
            h = int(h)
            if ap.upper() == "PM" and h != 12:
                h += 12
            elif ap.upper() == "AM" and h == 12:
                h = 0
            hours.append(h)
        return max(hours)  # catch the late-evening window, typically peak

    def _cached_intel(self, business: dict[str, Any], key: str, compute) -> Any:
        """Cache slow LLM-built intelligence (audience map, trends) per business
        for 24h so the daily job doesn't re-pay for it every run (#5)."""
        import json as _json

        cache_dir = Path(settings.data_dir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{self._slug(business.get('name', 'biz'))}-{key}.json"
        if path.exists():
            try:
                blob = _json.loads(path.read_text(encoding="utf-8"))
                if (time.time() - blob.get("ts", 0)) < 24 * 3600:
                    logger.info("Intel cache hit: %s/%s", business.get("name"), key)
                    return blob["data"]
            except Exception:
                pass
        try:
            data = compute()
        except Exception as exc:
            # A transient LLM/API outage must NOT kill the whole daily run.
            # Degrade to stale cached intel if we have it, else empty — the
            # generator and predictor still produce prompts (with less grounding),
            # and the Drift Check workflow alerts separately.
            logger.error(
                "::warning::Intel '%s' for %s failed: %s; falling back to stale/empty",
                key, business.get("name"), exc,
            )
            if path.exists():
                try:
                    blob = _json.loads(path.read_text(encoding="utf-8"))
                    logger.info("Using stale intel cache for %s/%s", business.get("name"), key)
                    return blob["data"]
                except Exception:
                    pass
            return {}
        try:
            path.write_text(
                _json.dumps({"ts": time.time(), "data": data}, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Intel cache write failed: %s", exc)
        return data

    def _load_history(self, db: Any, biz: Any) -> list[dict[str, Any]]:
        """Real past performance to ground the predictor (#1): recent prompts
        with their outcome. Combines (a) explicit human button feedback AND
        (b) auto-learned outcomes from scraped engagement — so the predictor
        stays grounded even when the human doesn't click the email buttons."""
        from datetime import timedelta

        from src.storage.models import FeedbackType, Prompt

        cutoff = utc_now_naive() - timedelta(days=30)
        prompts = (
            db.query(Prompt)
            .filter(Prompt.business_id == biz.id, Prompt.sent_at >= cutoff)
            .all()
        )
        # Ground the predictor on what was actually POSTED (the caption), not
        # the internal creative brief — that's the text real audiences saw.
        text_by_id = {p.id: (p.caption or p.prompt_text or "") for p in prompts}

        history: list[dict[str, Any]] = []
        seen: set[str] = set()
        for p in prompts:
            if not p.feedback:
                continue
            outcome = "good" if p.feedback == FeedbackType.GOOD else (
                "bad" if p.feedback in (FeedbackType.BAD, FeedbackType.REGENERATE) else "unknown"
            )
            posted_text = p.caption or p.prompt_text or ""
            history.append({"prompt_text": posted_text, "outcome": outcome})
            seen.add(posted_text[:120])

        # Enrich with AUTO-learned outcomes (real engagement scraped from the
        # business's own posts) — the richer, always-present signal. The log is
        # self-contained (it stores the posted caption), so it still contributes
        # even if the original Prompt row has since been deleted.
        for rec in self._read_engagement_log():
            logged_outcome = rec.get("outcome")
            if not logged_outcome:
                continue
            t = rec.get("caption") or text_by_id.get(rec.get("prompt_id")) or ""
            if not t:
                continue
            if t[:120] in seen:
                continue
            history.append({"prompt_text": t, "outcome": logged_outcome})
            seen.add(t[:120])

        return history[:15]

    @staticmethod
    def _read_engagement_log() -> list[dict[str, Any]]:
        """Durable, git-tracked auto-learning records (see auto_learner)."""
        path = Path(settings.data_dir) / "engagement_log.jsonl"
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        except Exception as exc:
            logger.warning("Failed to read engagement log: %s", exc)
        return out

    def _recent_prompt_texts(self, db: Any, biz: Any) -> list[str]:
        from datetime import timedelta

        from src.storage.models import Prompt

        cutoff = utc_now_naive() - timedelta(days=14)
        rows = (
            db.query(Prompt.prompt_text)
            .filter(Prompt.business_id == biz.id, Prompt.sent_at >= cutoff)
            .all()
        )
        return [r[0] for r in rows if r[0]]

    def _dedupe_prompts(
        self, prompts: list[dict[str, Any]], recent_texts: list[str], threshold: float = 0.6
    ) -> list[dict[str, Any]]:
        """Drop prompts too similar to something we already sent recently."""
        from src.learning.auto_learner import caption_similarity

        kept = []
        for p in prompts:
            text = p.get("prompt_text") or ""
            sim = max((caption_similarity(text, r) for r in recent_texts), default=0.0)
            if sim < threshold:
                kept.append(p)
            else:
                logger.info("Dedup: dropping near-duplicate (sim=%.2f): %s", sim, text[:60])
        # Never return empty — fall back to the single best if everything matched.
        if not kept and prompts:
            return [max(prompts, key=lambda x: x.get("virality_score", 0))]
        return kept

    @staticmethod
    def _slug(text: str) -> str:
        import re

        return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "biz"

    def run_all(self) -> list[dict[str, Any]]:
        businesses = [b for b in load_businesses() if b.get("is_active", True) and not b.get("demo")]
        if not businesses:
            logger.warning(
                "No active (non-demo) businesses configured; daily pipeline has nothing to do. "
                "Onboard a real business via the 'Onboard Business' workflow."
            )
            return []
        logger.info("Running daily pipeline for %d businesses", len(businesses))
        results = []
        failures = []
        for business in businesses:
            try:
                result = self.run_for_business(business)
                results.append(result)
                if not result.get("prompts"):
                    # "Ran green but produced nothing" must be visible.
                    failures.append({
                        "business": business.get("name"), "stage": "generation",
                        "error": "no prompts were generated (LLM chain down or all variants failed)",
                    })
            except Exception as exc:
                logger.error(
                    "::error::Daily pipeline failed for %s: %s", business.get("name"), exc
                )
                failures.append({"business": business.get("name"), "stage": "daily-pipeline", "error": str(exc)})
                results.append({"business": business.get("name"), "error": str(exc)})

        if failures:
            self.feedback.report_degraded_run(failures)
        return results

    # JSON config -> DB row. data/businesses.json is the source of truth for these fields; the
    # row used to be written once and never updated, so edited e-mails, handles, timezones and
    # thresholds silently had no effect (audit H-08).
    _CONFIG_FIELDS = (
        "website_url", "instagram_handle", "linkedin_handle", "email", "timezone",
        "virality_threshold", "prompt_types",
    )

    def _config_values(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "website_url": config.get("website_url"),
            "instagram_handle": config.get("instagram_handle"),
            "linkedin_handle": config.get("linkedin_handle"),
            "email": config.get("email") or "",
            "timezone": config.get("timezone", "Asia/Kolkata"),
            "virality_threshold": int(config.get("virality_threshold", settings.virality_threshold)),
            "prompt_types": config.get("prompt_types", ["educational", "storytelling", "social_proof"]),
        }

    def _get_or_create_business(self, db: Any, config: dict[str, Any]) -> Business:
        values = self._config_values(config)
        biz: Business | None = db.query(Business).filter(Business.name == config["name"]).first()
        if biz is None:
            biz = Business(name=config["name"], **values)
            db.add(biz)
            db.commit()
            db.refresh(biz)
            return biz
        changed = [f for f in self._CONFIG_FIELDS if getattr(biz, f) != values[f]]
        if changed:
            for f in changed:
                setattr(biz, f, values[f])
            db.add(biz)
            db.commit()
            # log field NAMES only: values include contact details
            logger.info("Business %s: config changed in businesses.json -> updated %s", config["name"], ", ".join(changed))
        return biz

    def _load_profile(self, db: Any, biz: Business) -> dict[str, Any] | None:
        return biz.profile_json

    @staticmethod
    def _profile_has_content(profile: dict[str, Any] | None) -> bool:
        """True when the profile carries real scraped content, not just the
        {"name", "source"} stub written after a failed scrape."""
        if not profile:
            return False
        return any(
            profile.get(k)
            for k in ("title", "meta_description", "headings", "text_blocks", "full_text_sample")
        )

    def _scrape_profile(self, business: dict[str, Any]) -> dict[str, Any]:
        profile = {"name": business["name"], "source": "website"}
        if business.get("website_url"):
            try:
                profile.update(self.website.scrape(business["website_url"]))
            except Exception as exc:
                logger.warning("Website scrape failed for %s: %s", business.get("name"), exc)
        return profile

    @staticmethod
    def _prompt_type(value: str) -> PromptType:
        try:
            return PromptType(value)
        except ValueError:
            return PromptType.EDUCATIONAL


def main() -> None:
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # request URLs can carry tokens
    pipeline = DailyPipeline()
    results = pipeline.run_all()
    for r in results:
        logger.info("RESULT: %s", r)

    # Fail the workflow step when every business errored — green CI must mean
    # the pipeline actually produced something.
    if results and all("error" in r for r in results):
        raise SystemExit(1)
    if results and not any(r.get("prompts") for r in results):
        logger.error("::error::The daily run generated no prompts for any business")
        raise SystemExit(1)
    if results and not any(r.get("delivered") for r in results):
        logger.error("::error::Nothing was delivered for any business (site publish failed, or SMTP down/unconfigured in email mode)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
