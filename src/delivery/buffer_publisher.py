"""Buffer delivery channel: schedules approved ad creatives to Instagram via Buffer's GraphQL API.

Uses public raw GitHub CDN asset URLs so Buffer can fetch and upload images directly to Instagram.
"""
from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urljoin

import httpx

from src.shared.config import settings

logger = logging.getLogger(__name__)

BUFFER_GQL_URL = "https://api.buffer.com"

CREATE_POST_MUTATION = """
mutation CreatePost($input: CreatePostInput!) {
  createPost(input: $input) {
    __typename
    ... on PostActionSuccess {
      post {
        id
        status
        dueAt
      }
    }
    ... on UnauthorizedError {
      message
    }
    ... on InvalidInputError {
      message
    }
    ... on LimitReachedError {
      message
    }
    ... on UnexpectedError {
      message
    }
  }
}
"""


class BufferPublisher:
    name = "buffer"

    def __init__(
        self,
        access_token: str | None = None,
        profile_ids: list[str] | None = None,
        draft_mode: bool | None = None,
        now: bool | None = None,
    ) -> None:
        self.access_token = (
            access_token
            or getattr(settings, "buffer_access_token", None)
            or os.getenv("BUFFER_ACCESS_TOKEN")
        )
        raw_ids = (
            profile_ids
            or getattr(settings, "buffer_profile_ids", None)
            or os.getenv("BUFFER_PROFILE_IDS")
            or os.getenv("BUFFER_PROFILE_ID")
        )
        if isinstance(raw_ids, str):
            self.profile_ids = [pid.strip() for pid in raw_ids.split(",") if pid.strip()]
        elif isinstance(raw_ids, list):
            self.profile_ids = [str(pid).strip() for pid in raw_ids if str(pid).strip()]
        else:
            self.profile_ids = []

        self.draft_mode = (
            draft_mode
            if draft_mode is not None
            else getattr(settings, "buffer_draft_mode", False)
        )
        self.now = (
            now
            if now is not None
            else getattr(settings, "buffer_now", False)
        )

    def is_configured(self) -> bool:
        return bool(self.access_token and self.profile_ids)

    def get_public_asset_url(self, relative_asset_path: str) -> str:
        """Resolve a relative asset path to a public GitHub raw CDN URL."""
        if relative_asset_path.startswith("http://") or relative_asset_path.startswith("https://"):
            return relative_asset_path

        repo = settings.github_repo or "yssavaliya/ai-ads"
        clean_path = relative_asset_path.replace("\\", "/").lstrip("/")
        return f"https://raw.githubusercontent.com/{repo}/gh-pages/{clean_path}"

    def schedule_post(
        self,
        text: str,
        image_url: str | None = None,
        profile_ids: list[str] | None = None,
        draft: bool | None = None,
        now: bool | None = None,
    ) -> dict[str, Any]:
        """Post or schedule a single update via Buffer GraphQL API."""
        if not self.access_token:
            raise ValueError("Buffer access token not configured.")

        targets = profile_ids or self.profile_ids
        if not targets:
            raise ValueError("No Buffer profile IDs specified.")

        is_draft = self.draft_mode if draft is None else draft
        is_now = self.now if now is None else now

        results = []
        for channel_id in targets:
            variables: dict[str, Any] = {
                "input": {
                    "channelId": channel_id,
                    "text": text,
                    "schedulingType": "automatic",
                    "mode": "shareNow" if is_now else "addToQueue",
                    "needsApproval": False,
                    "saveToDraft": is_draft,
                    "assets": [{"image": {"url": image_url}}] if image_url else [],
                    "metadata": {
                        "instagram": {
                            "type": "post",
                            "shouldShareToFeed": True,
                            "isAiGenerated": True,
                        }
                    },
                }
            }

            logger.info(
                "Posting to Buffer channel %s (image=%s, draft=%s, now=%s)",
                channel_id, image_url, is_draft, is_now,
            )

            with httpx.Client(timeout=45) as client:
                resp = client.post(
                    BUFFER_GQL_URL,
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    json={"query": CREATE_POST_MUTATION, "variables": variables},
                )
                if resp.status_code >= 400:
                    logger.error("Buffer API HTTP %s: %s", resp.status_code, resp.text)
                    resp.raise_for_status()

                payload = resp.json()
                data = payload.get("data", {}).get("createPost", {})
                typename = data.get("__typename")

                if typename != "PostActionSuccess":
                    err_msg = data.get("message") or str(data)
                    logger.error("Buffer GraphQL Error for channel %s: %s", channel_id, err_msg)
                    raise RuntimeError(f"Buffer GraphQL Error: {err_msg}")

                post_info = data.get("post", {})
                logger.info(
                    "Buffer post created successfully! ID: %s, Status: %s",
                    post_info.get("id"), post_info.get("status"),
                )
                results.append(post_info)

        return {"success": True, "posts": results}

    def send_daily_prompts(
        self,
        business: dict[str, Any],
        prompts: list[dict[str, Any]],
    ) -> bool:
        """Delivery channel interface: pick the top prompt and schedule to Buffer."""
        if not self.is_configured():
            logger.info("Buffer publisher skipped (BUFFER_ACCESS_TOKEN or BUFFER_PROFILE_ID not configured).")
            return True

        if not prompts:
            logger.info("No prompts available to deliver to Buffer.")
            return True

        # Sort prompts by virality score descending to post the strongest ad of the day
        approved = sorted(
            [p for p in prompts if not p.get("below_threshold")],
            key=lambda x: int(x.get("score") or x.get("virality_score") or 0),
            reverse=True,
        )
        if not approved:
            approved = sorted(
                prompts,
                key=lambda x: int(x.get("score") or x.get("virality_score") or 0),
                reverse=True,
            )

        top_prompt = approved[0]
        caption = top_prompt.get("caption") or top_prompt.get("prompt_text") or top_prompt.get("title", "")
        hashtags = top_prompt.get("hashtags", [])
        if hashtags:
            caption = f"{caption.strip()}\n\n{' '.join(hashtags)}"

        asset_info = top_prompt.get("asset") or {}
        asset_path = (
            top_prompt.get("asset_path")
            or asset_info.get("image")
            or asset_info.get("thumb")
        )

        image_url = None
        if asset_path:
            if "assets/" in asset_path:
                rel = asset_path[asset_path.index("assets/"):]
                image_url = self.get_public_asset_url(rel)
            else:
                image_url = self.get_public_asset_url(asset_path)

        try:
            self.schedule_post(text=caption, image_url=image_url)
            return True
        except Exception as exc:
            logger.error("Failed to schedule prompt to Buffer: %s", exc)
            return False

    def send_weekly_report(self, business: dict[str, Any], report: dict[str, Any]) -> bool:
        return True

    def send_onboarding(self, business: dict[str, Any], profile: dict[str, Any]) -> bool:
        return True

    def send_lead_digest(self, to: str, leads: list[dict[str, Any]]) -> bool:
        return True
