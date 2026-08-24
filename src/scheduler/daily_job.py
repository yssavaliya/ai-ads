from __future__ import annotations

import logging
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.delivery.email_sender import EmailSender
from src.delivery.github_feedback import GithubFeedback
from src.fetch.competitor_finder import CompetitorFinder
from src.fetch.website_scraper import WebsiteScraper
from src.generation.virality_gate import ViralityGate
from src.intelligence.audience_mapper import AudienceMapper
from src.intelligence.trend_radar import TrendRadar
from src.learning.bayesian_updater import BayesianUpdater
from src.shared.config import settings
from src.shared.utils import load_businesses
from src.storage.models import Business, Prompt, PromptType, SessionLocal

logger = logging.getLogger(__name__)


class DailyPipeline:
    def __init__(self) -> None:
        self.website = WebsiteScraper()
        self.competitors = CompetitorFinder()
        self.audience = AudienceMapper()
        self.trends = TrendRadar()
        self.gate = ViralityGate()
        self.email = EmailSender()
        self.feedback = GithubFeedback()
        self.weights = BayesianUpdater()

    def run_for_business(self, business: dict[str, Any]) -> dict[str, Any]:
        name = business["name"]
        logger.info("=== Daily pipeline for %s ===", name)

        db = SessionLocal()
        try:
            biz = self._get_or_create_business(db, business)
            profile = self._load_profile(db, biz)

            if not profile:
                logger.warning("No profile for %s; attempting initial scrape", name)
                profile = self._scrape_profile(business)
                biz.profile_json = profile
                db.add(biz)
                db.commit()

            competitors = self.competitors.discover_and_fetch(profile)
            social_data = {}
            if business.get("instagram_handle"):
                from src.fetch.social_scraper import fetch_instagram_sync

                social_data = fetch_instagram_sync(business["instagram_handle"])

            # AUTO-LEARN first: match the business's own recent posts to prompts
            # we sent before, score real engagement, update Bayesian weights.
            from src.learning.auto_learner import AutoLearner

            auto_result = AutoLearner().run_for_business(biz.id, social_data, db)
            logger.info(
                "Auto-learning for %s: matched=%s learned=%s",
                name, auto_result.get("matched"), auto_result.get("learned"),
            )

            audience_map = self.audience.map_audience(
                profile, social_data, competitors, biz.audience_personas_json
            )
            biz.audience_personas_json = audience_map
            db.add(biz)
            db.commit()

            trend_data = self.trends.scan(profile, audience_map, competitors)
            weights = self.weights.to_forge_weights(biz.id)
            prompts = self.gate.run(
                profile,
                audience_map,
                trend_data,
                business.get("prompt_types", ["educational", "storytelling", "social_proof"]),
                weights,
            )

            # EXPLORATION (gap #5): with modest probability, generate one extra
            # variant in an under-tested format/angle so the learner keeps
            # gathering data on formats it would otherwise never try. Never
            # blocks the main flow — failures are swallowed.
            if random.random() < 0.25:
                try:
                    extra = self._explore_variant(
                        business, profile, audience_map, trend_data, weights, [p.get("prompt_type") for p in prompts]
                    )
                    if extra:
                        prompts.append(extra)
                        logger.info("Exploration: added variant type=%s", extra.get("prompt_type"))
                except Exception as exc:
                    logger.warning("Exploration variant skipped: %s", exc)

            # Attach a ready-to-post brief (caption + best time) to every prompt.
            self._attach_posting_brief(business, prompts, social_data)

            # Render real creative assets (image inline / Reel MP4 attachment).
            from src.generation.creative_forge import CreativeForge

            creative = CreativeForge()
            render_warnings: list[str] = []
            for i, p in enumerate(prompts):
                try:
                    asset = creative.render_prompt(p, profile, index=i)
                    p["asset_path"] = asset["asset_path"]
                    p["asset_type"] = asset["asset_type"]
                    p["thumbnail"] = asset.get("thumbnail")
                    if asset.get("warnings"):
                        p["warnings"] = asset["warnings"]
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
                    sent_at=datetime.utcnow(),
                )
                db.add(prompt)
                db.flush()
                links = self.feedback.create_feedback_links(prompt.id, name, p.get("title", ""))
                saved.append(
                    {
                        **p,
                        "prompt_id": prompt.id,
                        "feedback_links": links,
                    }
                )
            db.commit()

            email_ok = self.email.send_daily_prompts(business, saved)
            logger.info("Daily email for %s sent=%s (%d prompts)", name, email_ok, len(saved))
            return {"business": name, "prompts": len(saved), "email_sent": email_ok}
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
        chosen = random.choice(candidates)
        extras = self.gate.run(profile, audience_map, trend_data, [chosen], weights)
        for ex in extras:
            if ex.get("prompt_type") == chosen:
                return ex
        return extras[0] if extras else None

    def _attach_posting_brief(
        self,
        business: dict[str, Any],
        prompts: list[dict[str, Any]],
        social_data: dict[str, Any],
    ) -> None:
        """Gap #2/#9 + user request: every prompt carries a copy-paste caption
        and a recommended post time, so the human can post manually without
        re-deriving anything."""
        post_time = self._recommend_post_time(business, social_data)
        for p in prompts:
            hook = (p.get("hook") or "").strip()
            hashtags = " ".join(p.get("hashtags") or [])
            caption = hook
            if hashtags:
                caption = f"{caption}\n\n{hashtags}".strip() if caption else hashtags
            p["caption"] = caption
            p["recommended_post_time"] = post_time

    def _recommend_post_time(self, business: dict[str, Any], social_data: dict[str, Any]) -> str:
        """Recommend the next good slot to post, in the business timezone.
        Data-driven when we have >=3 timestamped posts (use the hour their own
        audience engaged most), otherwise default to 7 PM local."""
        tz_name = business.get("timezone", "Asia/Kolkata")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("Asia/Kolkata")
        now = datetime.now(tz)

        from src.learning.auto_learner import AutoLearner

        hours: list[int] = []
        for post in social_data.get("posts", []):
            dt = AutoLearner._parse_date(post.get("posted_at"))
            if dt is not None:
                hours.append(dt.hour)
        if len(hours) >= 3:
            best = Counter(hours).most_common(1)[0][0]
        else:
            best = 19

        if now.hour >= best:
            target = (now + timedelta(days=1)).replace(hour=best, minute=0, second=0, microsecond=0)
        else:
            target = now.replace(hour=best, minute=0, second=0, microsecond=0)
        day = "Today" if target.date() == now.date() else "Tomorrow"
        return f"{day} at {target.strftime('%I:%M %p')} {tz_name.replace('_', ' ')}"

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
                results.append(self.run_for_business(business))
            except Exception as exc:
                logger.error(
                    "::error::Daily pipeline failed for %s: %s", business.get("name"), exc
                )
                failures.append({"business": business.get("name"), "stage": "daily-pipeline", "error": str(exc)})
                results.append({"business": business.get("name"), "error": str(exc)})

        if failures:
            self.feedback.report_degraded_run(failures)
        return results

    def _get_or_create_business(self, db: Any, config: dict[str, Any]) -> Business:
        biz = db.query(Business).filter(Business.name == config["name"]).first()
        if biz is None:
            biz = Business(
                name=config["name"],
                website_url=config.get("website_url"),
                instagram_handle=config.get("instagram_handle"),
                linkedin_handle=config.get("linkedin_handle"),
                email=config["email"],
                timezone=config.get("timezone", "Asia/Kolkata"),
                virality_threshold=int(config.get("virality_threshold", 75)),
                prompt_types=config.get("prompt_types", ["educational", "storytelling", "social_proof"]),
            )
            db.add(biz)
            db.commit()
            db.refresh(biz)
        return biz

    def _load_profile(self, db: Any, biz: Business) -> dict[str, Any] | None:
        return biz.profile_json

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
    pipeline = DailyPipeline()
    results = pipeline.run_all()
    for r in results:
        logger.info("RESULT: %s", r)

    # Fail the workflow step when every business errored — green CI must mean
    # the pipeline actually produced something.
    if results and all("error" in r for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()