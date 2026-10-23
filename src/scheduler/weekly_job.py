from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.delivery.channels import get_delivery
from src.intelligence.audience_mapper import AudienceMapper
from src.intelligence.trend_radar import TrendRadar
from src.learning.bayesian_updater import BayesianUpdater
from src.shared.config import settings
from src.shared.llm_client import LLMClient, extract_json_text, fence, fence_text
from src.shared.utils import load_businesses, utc_now_naive
from src.storage.models import Business, Engagement, Prompt, SessionLocal

logger = logging.getLogger(__name__)


class WeeklyPipeline:
    def __init__(self) -> None:
        self.delivery = get_delivery()
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

            week_start = utc_now_naive() - timedelta(days=7)
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

            report = self._build_report(biz, prompts, business, [(p, e) for p, e in engagement_rows])
            degraded = bool(report.pop("_degraded", False))
            delivered = self.delivery.send_weekly_report(business, report)
            logger.info("Weekly report for %s delivered=%s degraded=%s", name, bool(delivered), degraded)
            return {
                "business": name, "delivered": bool(delivered), "degraded": degraded,
                "channels": getattr(delivered, "channels", {}), "prompts_analyzed": len(prompts),
            }
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
        weights = self.weights.get_weights(biz.weights_key)

        prompt = f"""
You are a content intelligence analyst. Produce a weekly deep-dive report for a business.

Business:
{fence(business_config, 4000)}

Prompts sent this week (with predicted virality + user feedback):
{fence(prompt_data, 8000)}

REAL engagement measured from the business's own posts (auto-learned):
{fence(engagement_data, 6000)}

Learned weights (topic/format/segment -> weight):
{fence(weights, 4000)}

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
            report = self.llm.run_json(prompt, label="weekly-report", temperature=0.4, expect=(dict,))
            if not isinstance(report, dict):
                raise ValueError("weekly report must be JSON object")
            return report
        except Exception as exc:
            logger.error("Weekly report generation failed: %s", exc)
            return {
                "_degraded": True,  # placeholder: delivered so the owner sees it, but the run FAILS
                "top_performing": [],
                "audience_insights": ["Report generation failed; retry next week."],
                "competitor_moves": [],
                "content_pillars": [],
                "calendar": [],
                "recommendations": [],
                "risk_alerts": [str(exc)],
            }


def weekly_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every way a weekly run can be 'green but wrong': crash, placeholder report, nothing delivered."""
    failures = []
    for r in results:
        if r.get("error"):
            failures.append({"business": r.get("business"), "stage": "pipeline", "error": r["error"]})
        elif r.get("degraded"):
            failures.append({"business": r.get("business"), "stage": "report",
                             "error": "the LLM chain failed; a placeholder report was produced instead of real analysis"})
        elif not r.get("delivered"):
            failures.append({"business": r.get("business"), "stage": "delivery", "error": "no channel delivered the report"})
    return failures


def main() -> None:
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    pipeline = WeeklyPipeline()
    results = pipeline.run_all()
    for r in results:
        logger.info("RESULT: %s", r)

    failures = weekly_failures(results)
    if failures:
        for f in failures:
            logger.error("::error::Weekly %s failed for %s: %s", f["stage"], f["business"], f["error"])
        from src.delivery.github_feedback import GithubFeedback

        GithubFeedback().report_degraded_run(failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
