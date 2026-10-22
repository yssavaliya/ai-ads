from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.delivery.email_sender import EmailSender
from src.fetch.lead_finder import LeadFinder
from src.fetch.social_scraper import SocialScraper
from src.generation.virality_gate import ViralityGate
from src.intelligence.audience_mapper import AudienceMapper
from src.intelligence.lead_scorer import LeadScorer
from src.intelligence.trend_radar import TrendRadar
from src.learning.auto_learner import AutoLearner
from src.learning.bayesian_updater import BayesianUpdater
from src.shared.config import settings
from src.shared.utils import load_businesses, load_json, resolve_env_ref, save_json, utc_now_naive
from src.storage.models import Base, Business, Prompt, PromptType, SessionLocal
from src.storage.models import engine as db_engine

logger = logging.getLogger(__name__)


class LeadPipeline:
    """Discovers potential businesses automatically, scores their content
    opportunity, and generates viral prompt samples for the best leads.

    Self-learning: every run first re-scrapes previously discovered leads'
    Instagram posts and matches them against prompts we generated earlier —
    real engagement updates the Bayesian weights, which steer the next round
    of prompt generation."""

    def __init__(self) -> None:
        self.finder = LeadFinder()
        self.scorer = LeadScorer()
        self.gate = ViralityGate()
        self.audience = AudienceMapper()
        self.trends = TrendRadar()
        self.email = EmailSender()
        self.weights = BayesianUpdater()

    def _fetch_social(self, handle: str) -> dict[str, Any]:
        """One Apify/instaloader fetch per handle per run. The learning pass,
        candidate scoring and known-lead merge all need the same data."""
        cache: dict[str, dict[str, Any]] = self.__dict__.setdefault("_social_cache", {})
        key = handle.lstrip("@").lower()
        if key not in cache:
            cache[key] = asyncio.run(SocialScraper().fetch_instagram(handle))
        return cache[key]

    def load_config(self) -> dict[str, Any]:
        default = load_json(Path(settings.data_dir) / "lead_config.json", {})
        locations = default.get("locations") or default.get("location") or ["India"]
        if isinstance(locations, str):
            locations = [locations]
        return {
            "niches": default.get("niches", ["fitness"]),
            "locations": locations,
            "platform_focus": default.get("platform_focus", ""),
            "min_followers": int(default.get("min_followers", 500)),
            "max_followers": int(default.get("max_followers", 500000)),
            "target_count": int(default.get("target_count", 5)),
            "min_opportunity_score": int(default.get("min_opportunity_score", 60)),
            "enforce_location": bool(default.get("enforce_location", False)),
            "digest_email": resolve_env_ref(default.get("digest_email", "")),
            "prompt_types": default.get(
                "prompt_types", ["educational", "storytelling", "social_proof", "promotional"]
            ),
        }

    def run(self) -> dict[str, Any]:
        config = self.load_config()
        logger.info(
            "Lead pipeline: niches=%s locations=%s target=%d",
            config["niches"], config["locations"], config["target_count"],
        )

        # 1) Self-learning pass: re-scrape previously discovered leads, match
        #    their new posts to prompts we generated before, update weights.
        learning = self._learning_pass()

        raw_candidates = asyncio.run(
            self.finder.discover(
                config["niches"],
                config["locations"],
                config["target_count"] * 4,
                config["platform_focus"],
            )
        )
        candidates = asyncio.run(self.finder.enrich_with_socials(raw_candidates))
        logger.info("Discovered %d candidates with social handles", len(candidates))

        if config.get("enforce_location"):
            before = len(candidates)
            candidates = [c for c in candidates if self._is_local(c, config)]
            logger.info("Location filter dropped %d non-local candidates", before - len(candidates))

        deadline = time.monotonic() + settings.lead_time_budget_seconds
        scored_leads: list[dict[str, Any]] = []
        structurally_ineligible: list[dict[str, Any]] = []
        for candidate in candidates:
            if time.monotonic() > deadline:
                logger.warning(
                    "::warning::Lead time budget (%ds) exhausted; scoring stopped with %d candidates unscored",
                    settings.lead_time_budget_seconds, len(candidates) - len(scored_leads),
                )
                break
            handle = candidate.get("instagram_handle") or ""
            social_data = {}
            if handle:
                social_data = dict(self._fetch_social(handle))
                if social_data.get("follower_count"):
                    candidate["follower_count"] = social_data["follower_count"]
                elif candidate.get("follower_count"):
                    social_data["follower_count"] = candidate["follower_count"]

            assessment = self.scorer.score(
                candidate, social_data,
                min_followers=config["min_followers"],
                max_followers=config["max_followers"],
            )
            if not assessment.get("eligible"):
                # "Too big" is a durable verdict (a global brand won't shrink
                # into range next week) — cache it long-term so it stops
                # burning search budget every run. "Too small" is often just
                # an unverified/zero follower count from keyless discovery,
                # not a real tiny account, so it stays retryable.
                reasons = assessment.get("reasons") or []
                if any("too big" in r for r in reasons):
                    structurally_ineligible.append(candidate)
                continue

            lead = {
                **candidate,
                "opportunity_score": assessment["score"],
                "opportunity_reasons": assessment["reasons"],
                "followers": assessment["followers"],
            }
            scored_leads.append(lead)

        scored_leads = self._filter_min_score(scored_leads, config)
        scored_leads.sort(key=lambda lead: lead["opportunity_score"], reverse=True)
        top_leads = scored_leads[: config["target_count"]]

        db = SessionLocal()
        # Re-include previously-discovered eligible businesses so the digest
        # keeps delivering content instead of going empty once the good leads
        # have been delivered once and parked in the 21-day seen-cache.
        try:
            top_leads = self._merge_known_leads(db, top_leads, config)
        except Exception as exc:
            logger.warning("Could not merge known leads: %s", exc)

        # Only leads actually delivered enter the 21-day cooldown cache.
        # Raw candidates rejected by scoring stay retryable on future runs —
        # marking everything discovered would starve the pipeline after one
        # unlucky run.
        try:
            self.finder.mark_seen(top_leads)
        except Exception as exc:
            logger.warning("Could not update lead cooldown cache: %s", exc)

        if structurally_ineligible:
            try:
                self.finder.mark_rejected(structurally_ineligible)
                logger.info(
                    "Long-cooldown rejected %d structurally-ineligible candidates",
                    len(structurally_ineligible),
                )
            except Exception as exc:
                logger.warning("Could not update lead rejection cache: %s", exc)
        digest = []
        prompt_failures = 0
        failures = []
        creative = None
        try:
            for lead in top_leads:
                try:
                    biz = self._get_or_create_business(db, lead)
                    prompts = self._generate_sample_prompts(lead, config, biz.weights_key)
                    self._save_prompts(db, biz.id, prompts)

                    if creative is None:
                        from src.generation.creative_forge import CreativeForge

                        creative = CreativeForge()
                    profile = {
                        "name": lead.get("name", ""),
                        "meta_description": "",
                        "headings": [],
                    }
                    rendered = 0
                    for i, p in enumerate(prompts[:2]):  # top 2 assets per lead keep runtime sane
                        try:
                            asset = creative.render_prompt(p, profile, index=i)
                            p["asset_path"] = asset["asset_path"]
                            p["asset_type"] = asset["asset_type"]
                            p["thumbnail"] = asset.get("thumbnail")
                            if asset.get("warnings"):
                                p["warnings"] = asset["warnings"]
                            self._attach_asset(db, biz.id, p)
                            rendered += 1
                        except Exception as exc:
                            logger.warning(
                                "::warning::Asset render failed for %s/%s: %s",
                                lead.get("name"), p.get("title"), exc,
                            )
                    logger.info("Rendered %d assets for lead %s", rendered, lead.get("name"))
                    digest.append({**lead, "prompts": prompts})
                except Exception as exc:
                    prompt_failures += 1
                    failures.append(
                        {"lead": lead.get("name"), "stage": "prompt-generation", "error": str(exc)}
                    )
                    logger.error(
                        "::error::Prompt generation failed for %s: %s", lead.get("name"), exc
                    )
                    digest.append({**lead, "prompts": [], "prompt_error": str(exc)[:200]})

            if failures:
                from src.delivery.github_feedback import GithubFeedback

                GithubFeedback().report_degraded_run(failures)
        finally:
            db.close()

        self._save_leads(digest)
        email_ok = self._send_digest(digest, config)

        result = {
            "candidates_found": len(candidates),
            "eligible_scored": len(scored_leads),
            "top_leads": len(digest),
            "prompts_generated": sum(len(lead.get("prompts") or []) for lead in digest),
            "prompt_failures": prompt_failures,
            "learning": learning,
            "email_sent": email_ok,
            "generated_at": utc_now_naive().isoformat(),
        }
        logger.info("Lead pipeline result: %s", result)
        return result

    # Overseas markers that mean "not our market" — hard drop.
    _FOREIGN = [
        "california", "new york", "orange county", "hong kong", ".hk",
        "london", "united kingdom", "canada", "australia", "singapore",
        "france", "tokyo", "dubai", "germany", "usa", "u.s.", "united states",
    ]
    # Positive India/Gujarat/Ahmedabad signals — a business with one of these
    # (or a .in TLD) is treated as local even if the city isn't spelled out.
    _INDIA = ["ahmedabad", "gujarat", "india", "अहमदाबाद", "bharat", ".in", "co.in"]

    @classmethod
    def _is_local(cls, candidate: dict[str, Any], config: dict[str, Any]) -> bool:
        """Keep only leads that plausibly serve the target market.

        Web search returns global results, so without this the pool fills
        with franchises/portals (bakerycity.com = Orange County, etc.).
        We hard-drop overseas signals, then require at least one positive
        local signal (configured city, India/Gujarat marker, or .in TLD).
        Leads with neither are skipped — quality over quantity."""
        locations = [loc.lower() for loc in config.get("locations", [])]
        name = (candidate.get("name") or "").lower()
        domain = (candidate.get("website_url") or "").lower()
        handle = (candidate.get("instagram_handle") or "").lower()
        bio = (candidate.get("bio") or candidate.get("description") or "").lower()
        haystack = f"{name} {domain} {handle} {bio}"
        if any(f in haystack for f in cls._FOREIGN):
            return False
        local_signals = locations + cls._INDIA
        if any(sig in haystack for sig in local_signals):
            return True
        logger.debug("Location unconfirmed for '%s'; dropping (no local signal)", name)
        return False

    @staticmethod
    def _filter_min_score(leads: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
        floor = config.get("min_opportunity_score", 0)
        return [lead for lead in leads if lead.get("opportunity_score", 0) >= floor]

    @staticmethod
    def _client_names() -> set[str]:
        """Real onboarded clients (businesses.json) share the `businesses` table
        with discovered leads. They must never be scraped as, or listed in the
        digest as, sales leads."""
        return {(b.get("name") or "").lower() for b in load_businesses()}

    def _merge_known_leads(
        self, db: Any, top_leads: list[dict[str, Any]], config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Include previously-discovered eligible businesses (from the DB) in
        the digest so it keeps delivering content instead of going empty once
        the good leads have been delivered once and parked in the seen-cache."""
        exclude = {lead.get("instagram_handle", "").lower() for lead in top_leads}
        clients = self._client_names()
        try:
            businesses = (
                db.query(Business)
                .filter(Business.is_active.is_(True))
                .filter(Business.instagram_handle.isnot(None))
                .filter(Business.instagram_handle != "")
                .all()
            )
        except Exception as exc:
            logger.warning("Could not load known businesses: %s", exc)
            return top_leads

        known: list[dict[str, Any]] = []
        for biz in businesses:
            handle = (biz.instagram_handle or "").lstrip("@")
            if not handle or handle.lower() in exclude or (biz.name or "").lower() in clients:
                continue
            social_data: dict[str, Any] = {}
            try:
                sd = self._fetch_social(handle)
                if sd.get("follower_count"):
                    social_data = sd
            except Exception as exc:
                logger.warning("Known-lead social fetch failed for %s: %s", biz.name, exc)
            assessment = self.scorer.score(
                {
                    "name": biz.name,
                    "instagram_handle": handle,
                    "website_url": biz.website_url or "",
                    "niche": (biz.prompt_types or [""])[0],
                },
                social_data,
                min_followers=config["min_followers"],
                max_followers=config["max_followers"],
            )
            if not assessment.get("eligible") or assessment["score"] < config["min_opportunity_score"]:
                continue
            known.append(
                {
                    "name": biz.name,
                    "instagram_handle": handle,
                    "website_url": biz.website_url or "",
                    "niche": (biz.prompt_types or [""])[0],
                    "opportunity_score": assessment["score"],
                    "opportunity_reasons": assessment["reasons"],
                    "followers": assessment["followers"],
                }
            )
        merged = top_leads + known
        merged.sort(key=lambda lead: lead.get("opportunity_score", 0), reverse=True)
        return merged[: config["target_count"]]

    def _learning_pass(self) -> dict[str, Any]:
        """Re-scrape every known lead's Instagram, match new posts to prompts
        we generated earlier, and feed engagement outcomes into the weights."""
        Base.metadata.create_all(bind=db_engine)
        db = SessionLocal()
        learner = AutoLearner()
        totals = {"leads_checked": 0, "matched": 0, "learned": 0}
        try:
            businesses = (
                db.query(Business)
                .filter(Business.is_active.is_(True))
                .filter(Business.instagram_handle.isnot(None))
                .filter(Business.instagram_handle != "")
                .all()
            )
            clients = self._client_names()
            for biz in businesses:
                if (biz.name or "").lower() in clients:
                    continue  # clients are learned from by engagement.yml / daily job
                try:
                    social = self._fetch_social(biz.instagram_handle or "")
                    res = learner.run_for_business(biz.id, social, db)
                    totals["leads_checked"] += 1
                    totals["matched"] += int(res.get("matched") or 0)
                    totals["learned"] += int(res.get("learned") or 0)
                except Exception as exc:
                    logger.warning("Learning pass failed for %s: %s", biz.name, exc)
        finally:
            db.close()
        if totals["leads_checked"]:
            logger.info("Self-learning pass: %s", totals)
        else:
            logger.info("Self-learning pass: no known leads with IG handles yet")
        return totals

    def _get_or_create_business(self, db: Any, lead: dict[str, Any]) -> Business:
        """Persist the discovered lead so future runs can learn from it.
        Contact email only when actually found on their website — never the
        operator's own address or a fabricated placeholder."""
        handle = (lead.get("instagram_handle") or "").lstrip("@")
        contact_email = (lead.get("contact_email") or "").strip() or None
        biz = None
        if handle:
            biz = (
                db.query(Business)
                .filter(func.lower(Business.instagram_handle) == handle.lower())
                .first()
            )
        if biz is None and lead.get("website_url"):
            biz = (
                db.query(Business)
                .filter(Business.website_url == lead["website_url"])
                .first()
            )
        if biz is None:
            biz = Business(
                name=lead.get("name") or handle or "Unknown lead",
                website_url=lead.get("website_url") or None,
                instagram_handle=handle or None,
                linkedin_handle=lead.get("linkedin_handle") or None,
                email=contact_email or f"unverified+{handle or 'lead'}@leads.invalid",
                timezone="Asia/Kolkata",
                virality_threshold=settings.virality_threshold,
                prompt_types=["educational", "storytelling", "social_proof", "promotional"],
            )
            db.add(biz)
            db.commit()
            db.refresh(biz)
            logger.info("Persisted new lead business #%d %s", biz.id, biz.name)
        else:
            changed = False
            if handle and not biz.instagram_handle:
                biz.instagram_handle = handle
                changed = True
            if contact_email and "leads.invalid" in (biz.email or ""):
                biz.email = contact_email
                changed = True
            if changed:
                db.add(biz)
                db.commit()
        return biz

    def _save_prompts(self, db: Any, business_id: int, prompts: list[dict[str, Any]]) -> None:
        """Store generated sample prompts so auto-learning can match the
        lead's future posts against them."""
        now = utc_now_naive()
        for p in prompts:
            db.add(
                Prompt(
                    business_id=business_id,
                    prompt_text=p.get("prompt_text") or "",
                    caption=p.get("caption"),
                    prompt_type=self._prompt_type(p.get("prompt_type") or "educational"),
                    virality_score=p.get("virality_score"),
                    virality_label=p.get("virality_label"),
                    audience_segment=(p.get("audience_segment") or None),
                    sent_at=now,
                )
            )
        db.commit()

    def _attach_asset(self, db: Any, business_id: int, prompt: dict[str, Any]) -> None:
        """Record the rendered asset path on the newest matching Prompt row."""
        row = (
            db.query(Prompt)
            .filter(
                Prompt.business_id == business_id,
                Prompt.prompt_text == (prompt.get("prompt_text") or ""),
            )
            .order_by(Prompt.id.desc())
            .first()
        )
        if row is not None:
            row.asset_path = prompt.get("asset_path")
            row.asset_type = prompt.get("asset_type")
            db.add(row)
            db.commit()

    @staticmethod
    def _prompt_type(value: str) -> PromptType:
        try:
            return PromptType(value)
        except ValueError:
            return PromptType.EDUCATIONAL

    def _generate_sample_prompts(
        self, lead: dict[str, Any], config: dict[str, Any], weights_key: str | None = None
    ) -> list[dict[str, Any]]:
        profile = {
            "name": lead.get("name", ""),
            "website_url": lead.get("website_url", ""),
            "instagram_handle": lead.get("instagram_handle", ""),
            "niche": lead.get("niche", ""),
            "followers": lead.get("followers", 0),
        }
        audience_map = self.audience.map_audience(profile, {}, None, None)
        trend_data = self.trends.scan(profile, audience_map, [])
        weights = self.weights.to_forge_weights(weights_key) if weights_key else {}
        prompts = self.gate.run(
            profile,
            audience_map,
            trend_data,
            config.get("prompt_types", ["educational", "storytelling", "social_proof", "promotional"]),
            weights,
        )
        return prompts

    def _save_leads(self, digest: list[dict[str, Any]]) -> None:
        path = Path(settings.data_dir) / "leads.json"
        save_json(path, {"generated_at": utc_now_naive().isoformat(), "leads": digest})
        logger.info("Saved %d leads to %s", len(digest), path)

    def _send_digest(self, digest: list[dict[str, Any]], config: dict[str, Any]) -> bool:
        to = config.get("digest_email") or (settings.smtp_user or "")
        if not to:
            logger.warning("No digest_email configured; skipping lead digest email")
            return False
        return self.email.send_lead_digest(to, digest)


def main() -> None:
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # request URLs can carry tokens
    result = LeadPipeline().run()
    print(json.dumps(result, indent=2))

    # Leads were found but not a single prompt survived the LLM chain —
    # fail the workflow step so CI doesn't show green on a dead run.
    if result.get("top_leads", 0) > 0 and result.get("prompts_generated", 0) == 0:
        logger.error(
            "::error::%d leads found but 0 prompts generated — LLM chain is down. "
            "Check GROQ_API_KEY / GEMINI_API_KEY and the *_MODEL names.",
            result["top_leads"],
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
