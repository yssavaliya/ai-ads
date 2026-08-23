from __future__ import annotations

import logging
import re
import sys
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
            if not url_match:
                logger.warning(
                    "::warning::POSTED issue #%s has no Instagram URL; leaving open for edit",
                    issue["issue_number"],
                )
                continue
            post_url = url_match.group(0)
            post_id = post_url.rstrip("/").split("/")[-1]
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

        fb = {"good": FeedbackType.GOOD, "bad": FeedbackType.BAD, "regenerate": FeedbackType.REGENERATE}.get(feedback)
        if fb is None:
            close_issue(issue["issue_number"])
            continue

        prompt.feedback = fb
        from datetime import datetime

        prompt.feedback_at = datetime.utcnow()
        db.add(prompt)

        # A regenerate request is a real negative signal for learning too.
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


def main() -> None:
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    process_feedback()


if __name__ == "__main__":
    main()