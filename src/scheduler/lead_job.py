from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.delivery.email_sender import EmailSender
from src.fetch.lead_finder import LeadFinder
from src.fetch.social_scraper import SocialScraper
from src.generation.virality_gate import ViralityGate
from src.intelligence.audience_mapper import AudienceMapper
from src.intelligence.lead_scorer import LeadScorer
from src.intelligence.trend_radar import TrendRadar
from src.shared.config import settings
from src.shared.utils import load_json, save_json

logger = logging.getLogger(__name__)


class LeadPipeline:
    """Discovers potential businesses automatically, scores their content
    opportunity, and generates viral prompt samples for the best leads."""

    def __init__(self) -> None:
        self.finder = LeadFinder()
        self.scorer = LeadScorer()
        self.gate = ViralityGate()
        self.audience = AudienceMapper()
        self.trends = TrendRadar()
        self.email = EmailSender()

    def load_config(self) -> dict[str, Any]:
        default = load_json(Path(settings.data_dir) / "lead_config.json", {})
        return {
            "niches": default.get("niches", ["fitness"]),
            "location": default.get("location", "India"),
            "min_followers": int(default.get("min_followers", 500)),
            "max_followers": int(default.get("max_followers", 500000)),
            "target_count": int(default.get("target_count", 5)),
            "min_opportunity_score": int(default.get("min_opportunity_score", 60)),
            "digest_email": default.get("digest_email", ""),
            "prompt_types": default.get(
                "prompt_types", ["educational", "storytelling", "social_proof", "promotional"]
            ),
        }

    def run(self) -> dict[str, Any]:
        config = self.load_config()
        logger.info(
            "Lead pipeline: niches=%s location=%s target=%d",
            config["niches"], config["location"], config["target_count"],
        )

        raw_candidates = asyncio.run(
            self.finder.discover(config["niches"], config["location"], config["target_count"] * 4)
        )
        candidates = asyncio.run(self.finder.enrich_with_socials(raw_candidates))
        logger.info("Discovered %d candidates with social handles", len(candidates))

        social = SocialScraper()
        scored_leads = []
        for candidate in candidates:
            handle = candidate.get("instagram_handle") or ""
            social_data = {}
            if handle:
                social_data = asyncio.run(social.fetch_instagram(handle))
                if social_data.get("follower_count"):
                    candidate["follower_count"] = social_data["follower_count"]

            assessment = self.scorer.score(
                candidate, social_data,
                min_followers=config["min_followers"],
                max_followers=config["max_followers"],
            )
            if not assessment.get("eligible"):
                continue

            lead = {
                **candidate,
                "opportunity_score": assessment["score"],
                "opportunity_reasons": assessment["reasons"],
                "followers": assessment["followers"],
            }
            scored_leads.append(lead)

        scored_leads.sort(key=lambda l: l["opportunity_score"], reverse=True)
        top_leads = scored_leads[: config["target_count"]]

        digest = []
        for lead in top_leads:
            try:
                prompts = self._generate_sample_prompts(lead, config)
                digest.append({**lead, "prompts": prompts})
            except Exception as exc:
                logger.error("Prompt generation failed for %s: %s", lead.get("name"), exc)
                digest.append({**lead, "prompts": []})

        self._save_leads(digest)
        email_ok = self._send_digest(digest, config)

        result = {
            "candidates_found": len(candidates),
            "eligible_scored": len(scored_leads),
            "top_leads": len(digest),
            "email_sent": email_ok,
            "generated_at": datetime.utcnow().isoformat(),
        }
        logger.info("Lead pipeline result: %s", result)
        return result

    def _generate_sample_prompts(self, lead: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
        profile = {
            "name": lead.get("name", ""),
            "website_url": lead.get("website_url", ""),
            "instagram_handle": lead.get("instagram_handle", ""),
            "niche": lead.get("niche", ""),
            "followers": lead.get("followers", 0),
        }
        audience_map = self.audience.map_audience(profile, None, None, None)
        trend_data = self.trends.scan(profile, audience_map, None)
        prompts = self.gate.run(
            profile,
            audience_map,
            trend_data,
            config.get("prompt_types", ["educational", "storytelling", "social_proof", "promotional"]),
        )
        return prompts

    def _save_leads(self, digest: list[dict[str, Any]]) -> None:
        path = Path(settings.data_dir) / "leads.json"
        save_json(path, {"generated_at": datetime.utcnow().isoformat(), "leads": digest})
        logger.info("Saved %d leads to %s", len(digest), path)

    def _send_digest(self, digest: list[dict[str, Any]], config: dict[str, Any]) -> bool:
        to = config.get("digest_email") or (settings.smtp_user or "")
        if not to:
            logger.warning("No digest_email configured; skipping lead digest email")
            return False
        return self.email.send_lead_digest(to, digest)


def main() -> None:
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = LeadPipeline().run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()