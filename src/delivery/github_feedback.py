from __future__ import annotations

import logging
from typing import Any

import httpx

from src.learning import feedback_tokens
from src.shared.config import settings

logger = logging.getLogger(__name__)

FEEDBACK_PREFIX = "[content-intel-feedback]"
DEGRADED_TITLE = "[pipeline] degraded run"


class GithubFeedback:
    """Creates GitHub issues for user feedback (👍/👎/🔄) on sent prompts.
    The feedback workflow (hourly cron) reads these issues and updates weights."""

    def __init__(self, token: str | None = None, repo: str | None = None) -> None:
        self.token = token or settings.github_token or ""
        self.repo = repo or settings.github_repo or ""
        self.base = "https://api.github.com"

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.repo)

    def create_feedback_links(
        self, prompt_id: int, business_name: str, prompt_title: str, prompt_uid: str | None = None
    ) -> dict[str, str]:
        """Signed feedback URLs for a prompt. Returns placeholder '#' links when GitHub is not
        configured OR no FEEDBACK_SIGNING_KEY exists: an unsigned link would be unverifiable,
        and feedback processing refuses anything without a valid token."""
        if not self.enabled or not feedback_tokens.enabled() or not prompt_uid:
            return {"good": "#", "bad": "#", "regenerate": "#", "posted": "#"}
        title = f"{FEEDBACK_PREFIX} prompt #{prompt_id}"
        body = (
            f"**Business:** {business_name}\n"
            f"**Prompt:** {prompt_title}\n"
            f"**Prompt ID:** {prompt_id}\n"
            f"**Feedback:** __FEEDBACK__"
        )
        base_title = title.replace("__FEEDBACK__", "").strip()
        posted_body = (
            f"**Business:** {business_name}\n"
            f"**Prompt:** {prompt_title}\n"
            f"**Prompt ID:** {prompt_id}\n\n"
            "**Instagram post URL** (paste the link to the post you published):\n\n"
            "https://www.instagram.com/p/...\n\n"
            "_The learner will attribute this post's engagement to the exact asset._"
        )
        urls = {}
        for label, word in (
            ("good", "GOOD 👍"),
            ("bad", "BAD 👎"),
            ("regenerate", "REGENERATE 🔄"),
            ("posted", "POSTED 📎"),
        ):
            issue_title = f"{base_title} [{word}] {feedback_tokens.sign(prompt_uid, label)}"
            issue_body = posted_body if label == "posted" else body.replace("__FEEDBACK__", word)
            urls[label] = (
                f"https://github.com/{self.repo}/issues/new?"
                f"title={self._quote(issue_title)}&body={self._quote(issue_body)}"
            )
        return urls

    def create_feedback_issue(self, prompt_id: int, business_name: str, prompt_title: str, feedback: str) -> int | None:
        """Programmatically open a feedback issue (used by webhook-less flows)."""
        if not self.enabled:
            return None
        label = {"good": "GOOD 👍", "bad": "BAD 👎", "regenerate": "REGENERATE 🔄"}.get(feedback, feedback)
        title = f"{FEEDBACK_PREFIX} prompt #{prompt_id} [{label}]"
        body = (
            f"**Business:** {business_name}\n"
            f"**Prompt:** {prompt_title}\n"
            f"**Prompt ID:** {prompt_id}\n"
            f"**Feedback:** {label}"
        )
        try:
            resp = httpx.post(
                f"{self.base}/repos/{self.repo}/issues",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={"title": title, "body": body},
                timeout=30,
            )
            resp.raise_for_status()
            number = resp.json().get("number")
            logger.info("Created feedback issue #%s for prompt %s (%s)", number, prompt_id, feedback)
            return int(number) if number is not None else None
        except Exception as exc:
            logger.error("Failed to create feedback issue: %s", exc)
            return None

    @staticmethod
    def _quote(text: str) -> str:
        import urllib.parse

        return urllib.parse.quote(text, safe="")

    def report_degraded_run(self, failures: list[dict[str, Any]]) -> int | None:
        """Create (or append to today's) tracking issue listing pipeline stages
        that failed. Makes 'ran green but produced nothing' visible without
        reading Actions logs. Never raises."""
        if not self.enabled or not failures:
            return None
        import datetime

        date_str = datetime.date.today().isoformat()
        title = f"{DEGRADED_TITLE} — {date_str}"
        entry_lines = []
        for f in failures[:40]:
            entry_lines.append(
                f"- **{f.get('business', f.get('lead', 'unknown'))}** · "
                f"stage `{f.get('stage', 'unknown')}`: {str(f.get('error', ''))[:300]}"
            )
        body = (
            f"Automated degraded-run report for {date_str}.\n\n"
            f"The workflow finished, but these stages failed and produced no real output:\n\n"
            + "\n".join(entry_lines)
            + "\n\n_Check secrets (GROQ_API_KEY etc.) and provider status. "
            "This issue updates in place as long as it stays open._"
        )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            existing = httpx.get(
                f"{self.base}/repos/{self.repo}/issues",
                params={"state": "open", "per_page": 50},
                headers=headers,
                timeout=30,
            )
            existing.raise_for_status()
            match = next(
                (i for i in existing.json() if i.get("title", "").startswith(DEGRADED_TITLE)),
                None,
            )
            if match:
                merged = (match.get("body") or "") + "\n\n---\n\n" + body
                resp = httpx.patch(
                    f"{self.base}/repos/{self.repo}/issues/{match['number']}",
                    headers=headers,
                    json={"body": merged[-60000:]},
                    timeout=30,
                )
                resp.raise_for_status()
                logger.info("Updated degraded-run issue #%s", match["number"])
                return int(match["number"])
            resp = httpx.post(
                f"{self.base}/repos/{self.repo}/issues",
                headers=headers,
                json={"title": title, "body": body},
                timeout=30,
            )
            resp.raise_for_status()
            number = resp.json().get("number")
            logger.info("Created degraded-run issue #%s (%d failures)", number, len(failures))
            return int(number) if number is not None else None
        except Exception as exc:
            logger.error("Failed to report degraded run: %s", exc)
            return None
