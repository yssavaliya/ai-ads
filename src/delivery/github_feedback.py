from __future__ import annotations

import logging
from typing import Any

import httpx

from src.shared.config import settings

logger = logging.getLogger(__name__)

FEEDBACK_PREFIX = "[content-intel-feedback]"


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

    def create_feedback_links(self, prompt_id: int, business_name: str, prompt_title: str) -> dict[str, str]:
        """Return feedback URLs for a prompt, or placeholder '#' if GitHub is not configured."""
        if not self.enabled:
            return {"good": "#", "bad": "#", "regenerate": "#"}
        title = f"{FEEDBACK_PREFIX} prompt #{prompt_id}"
        body = (
            f"**Business:** {business_name}\n"
            f"**Prompt:** {prompt_title}\n"
            f"**Prompt ID:** {prompt_id}\n"
            f"**Feedback:** __FEEDBACK__"
        )
        base_title = title.replace("__FEEDBACK__", "").strip()
        urls = {}
        for label, word in (("good", "GOOD 👍"), ("bad", "BAD 👎"), ("regenerate", "REGENERATE 🔄")):
            issue_title = f"{base_title} [{word}]"
            issue_body = body.replace("__FEEDBACK__", word)
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
            return number
        except Exception as exc:
            logger.error("Failed to create feedback issue: %s", exc)
            return None

    @staticmethod
    def _quote(text: str) -> str:
        import urllib.parse

        return urllib.parse.quote(text, safe="")