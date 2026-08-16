from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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

    def run_all(self) -> list[dict[str, Any]]:
        businesses = [b for b in load_businesses() if b.get("is_active", True)]
        logger.info("Running daily pipeline for %d businesses", len(businesses))
        results = []
        for business in businesses:
            try:
                results.append(self.run_for_business(business))
            except Exception as exc:
                logger.error("Daily pipeline failed for %s: %s", business.get("name"), exc)
                results.append({"business": business.get("name"), "error": str(exc)})
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


if __name__ == "__main__":
    main()