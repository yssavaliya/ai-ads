from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.delivery.email_sender import EmailSender
from src.intelligence.audience_mapper import AudienceMapper
from src.intelligence.trend_radar import TrendRadar
from src.learning.bayesian_updater import BayesianUpdater
from src.shared.config import settings
from src.shared.llm_client import LLMClient, extract_json_text
from src.shared.utils import load_businesses
from src.storage.models import Business, Engagement, Prompt, SessionLocal

logger = logging.getLogger(__name__)


class WeeklyPipeline:
    def __init__(self) -> None:
        self.email = EmailSender()
        self.audience = AudienceMapper()
        self.trends = TrendRadar()
        self.llm = LLMClient()
        self.weights = BayesianUpdater()

    def run_for_business(self, business: dict[str, Any]) -> dict[str, Any]:
        name = business["name"]
        logger.info("=== Weekly deep-dive for %s ===", name)

        db = SessionLocal()
        try:
            biz = db.query(Business).filter(Business.name == name).first()
            if biz is None:
                logger.warning("Business %s not in DB; skipping weekly", name)
                return {"business": name, "error": "not in db"}

            week_start = datetime.utcnow() - timedelta(days=7)
            prompts = (
                db.query(Prompt)
                .filter(Prompt.business_id == biz.id, Prompt.sent_at >= week_start)
                .order_by(Prompt.virality_score.desc())
                .all()
            )

            engagement_rows = (
                db.query(Prompt, Engagement)
                .join(Engagement, Engagement.prompt_id == Prompt.id)
                .filter(Prompt.business_id == biz.id, Prompt.sent_at >= week_start)
                .all()
            )

            report = self._build_report(biz, prompts, business, engagement_rows)
            email_ok = self.email.send_weekly_report(business, report)
            logger.info("Weekly report for %s sent=%s", name, email_ok)
            return {"business": name, "email_sent": email_ok, "prompts_analyzed": len(prompts)}
        finally:
            db.close()

    def run_all(self) -> list[dict[str, Any]]:
        businesses = [b for b in load_businesses() if b.get("is_active", True) and not b.get("demo")]
        logger.info("Running weekly pipeline for %d businesses", len(businesses))
        results = []
        for business in businesses:
            try:
                results.append(self.run_for_business(business))
            except Exception as exc:
                logger.error("Weekly pipeline failed for %s: %s", business.get("name"), exc)
                results.append({"business": business.get("name"), "error": str(exc)})
        return results

    def _build_report(
        self,
        biz: Business,
        prompts: list[Prompt],
        business_config: dict[str, Any],
        engagement_rows: list[tuple[Prompt, Engagement]] | None = None,
    ) -> dict[str, Any]:
        prompt_data = [
            {
                "prompt_text": p.prompt_text[:400],
                "prompt_type": p.prompt_type.value if p.prompt_type else "",
                "virality_score": p.virality_score,
                "feedback": p.feedback.value if p.feedback else None,
                "sent_at": p.sent_at.isoformat() if p.sent_at else None,
            }
            for p in prompts
        ]
        engagement_data = [
            {
                "prompt_id": p.id,
                "feedback": p.feedback.value if p.feedback else None,
                "likes": e.likes,
                "comments": e.comments,
                "shares": e.shares,
                "saves": e.saves,
                "reach": e.reach,
            }
            for p, e in (engagement_rows or [])
        ]
        weights = self.weights.get_weights(biz.id)

        prompt = f"""
You are a content intelligence analyst. Produce a weekly deep-dive report for a business.

Business:
{json.dumps(business_config, indent=2, default=str)[:4000]}

Prompts sent this week (with predicted virality + user feedback):
{json.dumps(prompt_data, indent=2, default=str)[:8000]}

REAL engagement measured from the business's own posts (auto-learned):
{json.dumps(engagement_data, indent=2, default=str)[:6000]}

Learned weights (topic/format/segment -> weight):
{json.dumps(weights, indent=2, default=str)[:4000]}

Respond ONLY with JSON:
{{
  "top_performing": [
    {{"prompt": "short label", "score": 0, "feedback": "good|bad|none", "why": "why it worked"}}
  ],
  "audience_insights": ["shift 1", "shift 2"],
  "competitor_moves": ["move 1", "move 2"],
  "content_pillars": [
    {{"pillar": "next week pillar", "rationale": "why", "suggested_frequency": "e.g. 2x/week"}}
  ],
  "calendar": ["Mon: ...", "Tue: ...", "Wed: ...", "Thu: ...", "Fri: ...", "Sat: ..."],
  "recommendations": ["prompt optimization 1", "optimization 2"],
  "risk_alerts": ["alert 1", "alert 2"]
}}
"""
        try:
            result = self.llm.run(prompt, label="weekly-report", temperature=0.4)
            report = json.loads(extract_json_text(result.stdout))
            if not isinstance(report, dict):
                raise ValueError("weekly report must be JSON object")
            return report
        except Exception as exc:
            logger.error("Weekly report generation failed: %s", exc)
            return {
                "top_performing": [],
                "audience_insights": ["Report generation failed; retry next week."],
                "competitor_moves": [],
                "content_pillars": [],
                "calendar": [],
                "recommendations": [],
                "risk_alerts": [str(exc)],
            }


def main() -> None:
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pipeline = WeeklyPipeline()
    results = pipeline.run_all()
    for r in results:
        logger.info("RESULT: %s", r)


if __name__ == "__main__":
    main()