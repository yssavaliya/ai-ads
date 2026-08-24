from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from src.learning.bayesian_updater import BayesianUpdater
from src.shared.config import settings
from src.shared.utils import load_businesses, load_weights, save_weights
from src.storage.models import Business, FeedbackType, Prompt, SessionLocal

logger = logging.getLogger(__name__)

FEEDBACK_PREFIX = "[content-intel-feedback]"
FEEDBACK_RE = re.compile(r"prompt #(\d+) \[(GOOD|BAD|REGENERATE|POSTED)")
IG_POST_RE = re.compile(r"https?://(?:www\.)?instagram\.com/(?:p|reel|reels|tv)/[A-Za-z0-9_-]+/?", re.I)
# Fallback for users who paste just the shortcode (e.g. "shortcode: ABC123").
SHORTCODE_RE = re.compile(r"shortcode[:\s]+([A-Za-z0-9_-]{5,30})", re.I)


def parse_feedback_issues() -> list[dict[str, Any]]:
    """Fetch open feedback issues from GitHub API and parse them."""
    if not (settings.github_token and settings.github_repo):
        logger.warning("GitHub feedback not configured (GITHUB_TOKEN/GITHUB_REPO)")
        return []
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{settings.github_repo}/issues",
            params={"state": "open", "per_page": 50},
            headers={
                "Authorization": f"Bearer {settings.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Failed to fetch issues: %s", exc)
        return []

    parsed = []
    for issue in resp.json():
        if "pull_request" in issue:
            continue
        title = issue.get("title", "")
        if not title.startswith(FEEDBACK_PREFIX):
            continue
        match = FEEDBACK_RE.search(title)
        if not match:
            continue
        parsed.append(
            {
                "issue_number": issue["number"],
                "prompt_id": int(match.group(1)),
                "feedback": match.group(2).lower(),
                "body": issue.get("body") or "",
            }
        )
    logger.info("Parsed %d feedback issues", len(parsed))
    return parsed


def close_issue(issue_number: int) -> None:
    try:
        httpx.patch(
            f"https://api.github.com/repos/{settings.github_repo}/issues/{issue_number}",
            headers={
                "Authorization": f"Bearer {settings.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"state": "closed"},
            timeout=30,
        )
        logger.info("Closed feedback issue #%s", issue_number)
    except Exception as exc:
        logger.error("Failed to close issue #%s: %s", issue_number, exc)


def process_feedback() -> int:
    """Process open feedback issues: update DB, Bayesian weights, close issues.
    POSTED issues write a direct Engagement row tied to the exact Prompt —
    no fuzzy matching needed for explicitly linked posts."""
    db = SessionLocal()
    updater = BayesianUpdater()
    issues = parse_feedback_issues()
    processed = 0

    for issue in issues:
        prompt_id = issue["prompt_id"]
        feedback = issue["feedback"]
        prompt = db.get(Prompt, prompt_id)
        if prompt is None:
            logger.warning("Feedback issue #%s references unknown prompt %s; closing anyway", issue["issue_number"], prompt_id)
            close_issue(issue["issue_number"])
            continue

        if feedback == "posted":
            url_match = IG_POST_RE.search(issue["body"])
            if url_match:
                post_url = url_match.group(0)
                post_id = post_url.rstrip("/").split("/")[-1]
            else:
                code_match = SHORTCODE_RE.search(issue["body"])
                if not code_match:
                    logger.warning(
                        "::warning::POSTED issue #%s has no Instagram URL/shortcode; leaving open for edit",
                        issue["issue_number"],
                    )
                    continue
                post_id = code_match.group(1)
                post_url = f"https://instagram.com/p/{post_id}"
            post_id = post_id.rstrip("/").split("/")[-1]
            from src.storage.models import Engagement, Platform
            from datetime import datetime

            existing = (
                db.query(Engagement)
                .filter(Engagement.prompt_id == prompt_id, Engagement.post_id == post_id)
                .first()
            )
            if existing is None:
                db.add(
                    Engagement(
                        prompt_id=prompt_id,
                        platform=Platform.INSTAGRAM,
                        post_id=post_id[:100],
                        post_url=post_url[:500],
                        fetched_at=datetime.utcnow(),
                    )
                )
            prompt.feedback = FeedbackType.GOOD  # human-confirmed posted
            prompt.feedback_at = datetime.utcnow()
            db.add(prompt)
            updater.record(
                prompt.business_id,
                "good",
                prompt_type=prompt.prompt_type.value if prompt.prompt_type else None,
                audience_segment=prompt.audience_segment or None,
            )
            db.commit()
            logger.info("Linked post %s to prompt %s", post_url, prompt_id)
            close_issue(issue["issue_number"])
            processed += 1
            continue

        # A regenerate request is a real negative signal AND must actually
        # produce a fresh replacement prompt (otherwise the button is a no-op).
        if feedback == "regenerate":
            prompt.feedback = FeedbackType.REGENERATE
            prompt.feedback_at = datetime.utcnow()
            db.add(prompt)
            updater.record(
                prompt.business_id,
                "bad",
                prompt_type=prompt.prompt_type.value if prompt.prompt_type else None,
                audience_segment=prompt.audience_segment or None,
            )
            db.commit()
            try:
                regenerate_prompt(prompt, db)
            except Exception as exc:  # noqa: BLE001
                logger.error("Regenerate failed for prompt %s: %s", prompt.id, exc)
            close_issue(issue["issue_number"])
            processed += 1
            continue

        fb = {"good": FeedbackType.GOOD, "bad": FeedbackType.BAD}.get(feedback)
        if fb is None:
            close_issue(issue["issue_number"])
            continue

        prompt.feedback = fb
        prompt.feedback_at = datetime.utcnow()
        db.add(prompt)

        updater.record(
            prompt.business_id,
            "good" if feedback == "good" else "bad",
            prompt_type=prompt.prompt_type.value if prompt.prompt_type else None,
            audience_segment=prompt.audience_segment or None,
        )
        db.commit()
        close_issue(issue["issue_number"])
        processed += 1

    db.close()
    logger.info("Processed %d feedback issues", processed)
    return processed


def regenerate_prompt(prompt: Any, db: Any) -> bool:
    """Generate a fresh variant for a prompt the user wanted redone, render its
    asset, persist it, and email it. Returns True if a replacement was sent."""
    from datetime import datetime

    from src.delivery.email_sender import EmailSender
    from src.generation.creative_forge import CreativeForge
    from src.generation.prompt_forge import PromptForge
    from src.generation.virality_gate import ViralityGate
    from src.learning.bayesian_updater import BayesianUpdater
    from src.scheduler.daily_job import DailyPipeline
    from src.shared.utils import load_businesses
    from src.storage.models import Business, Prompt, PromptType

    biz = db.get(Business, prompt.business_id)
    if biz is None:
        return False
    business = next((b for b in load_businesses() if b.get("name") == biz.name), {})
    if not business:
        return False

    profile = biz.profile_json or {}
    audience_map = biz.audience_personas_json or {}
    weights = BayesianUpdater().to_forge_weights(biz.id)
    ptype = prompt.prompt_type.value if prompt.prompt_type else "educational"

    forge = PromptForge()
    variants = forge.generate_variants(profile, audience_map, {}, [ptype], weights=weights, count=1)
    if not variants:
        logger.warning("Regenerate produced no variants for prompt %s", prompt.id)
        return False
    v = variants[0]
    v["prompt_type"] = ptype

    gate = ViralityGate()
    res = gate.predictor.score(v["prompt_text"], profile, audience_map, {}, ptype, history=[])
    v["virality_score"] = res.get("virality_score")
    v["virality_label"] = res.get("virality_label")
    v["virality_probability"] = res.get("virality_probability")
    v["key_factors"] = res.get("key_factors")
    v["risks"] = res.get("risks")
    v["reasoning"] = res.get("reasoning")

    dj = DailyPipeline()
    dj._attach_posting_brief(business, [v], {}, audience_map)
    try:
        creative = CreativeForge()
        asset = creative.render_prompt(v, profile, index=0)
        v["asset_path"] = asset.get("asset_path")
        v["asset_type"] = asset.get("asset_type")
        v["thumbnail"] = asset.get("thumbnail")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Regenerate asset render failed: %s", exc)

    new_prompt = Prompt(
        business_id=biz.id,
        prompt_text=v["prompt_text"],
        caption=v.get("caption"),
        prompt_type=PromptType(ptype),
        virality_score=v.get("virality_score"),
        virality_label=v.get("virality_label"),
        virality_probability=v.get("virality_probability"),
        key_factors=v.get("key_factors"),
        risks=v.get("risks"),
        reasoning=v.get("reasoning"),
        audience_segment=(v.get("audience_segment") or None),
        asset_path=v.get("asset_path"),
        asset_type=v.get("asset_type"),
        sent_at=datetime.utcnow(),
    )
    db.add(new_prompt)
    db.flush()
    links = dj.feedback.create_feedback_links(new_prompt.id, biz.name, v.get("title", ""))
    db.commit()

    try:
        EmailSender().send_daily_prompts(
            business,
            [{**v, "prompt_id": new_prompt.id, "feedback_links": links}],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Regenerate email failed: %s", exc)
    return True


def main() -> None:
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    process_feedback()


if __name__ == "__main__":
    main()