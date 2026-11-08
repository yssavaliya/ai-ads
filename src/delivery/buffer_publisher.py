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
        channelId
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

EDIT_POST_MUTATION = """
mutation EditPost($input: EditPostInput!) {
  editPost(input: $input) {
    __typename
    ... on PostActionSuccess {
      post { id channelId status dueAt }
    }
    ... on UnauthorizedError { message }
    ... on InvalidInputError { message }
    ... on LimitReachedError { message }
    ... on UnexpectedError { message }
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
        video_url: str | None = None,
    ) -> dict[str, Any]:
        """Post or schedule a single update via Buffer GraphQL API."""
        if not self.access_token:
            raise ValueError("Buffer access token not configured.")

        targets = profile_ids or self.profile_ids
        if not targets:
            raise ValueError("No Buffer profile IDs specified.")

        is_draft = self.draft_mode if draft is None else draft
        is_now = self.now if now is None else now

        if image_url and video_url:
            raise ValueError("Buffer posts must contain either an image or a video, not both.")

        asset: dict[str, Any] = {}
        instagram_type = "post"
        if video_url:
            asset = {"video": {"url": video_url}}
            instagram_type = "reel"
        elif image_url:
            asset = {"image": {"url": image_url}}

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
                    "assets": [asset] if asset else [],
                    "metadata": {
                        "instagram": {
                            "type": instagram_type,
                            "shouldShareToFeed": True,
                            "isAiGenerated": True,
                        }
                    },
                }
            }

            logger.info(
                "Posting to Buffer channel %s (asset=%s, draft=%s, now=%s)",
                channel_id, "video" if video_url else ("image" if image_url else "none"), is_draft, is_now,
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

    def replace_scheduled_reel(self, post_id: str, text: str, video_url: str) -> dict[str, Any]:
        """Replace media and caption on an existing scheduled Instagram Reel.

        The schedule is intentionally omitted from EditPostInput so Buffer keeps
        the existing queue slot and due time.
        """
        if not self.access_token:
            raise ValueError("Buffer access token not configured.")
        if not post_id or not video_url:
            raise ValueError("A Buffer post ID and Reel video URL are required.")
        variables = {"input": {
            "id": post_id,
            "text": text,
            "assets": [{"video": {"url": video_url}}],
            "metadata": {"instagram": {"type": "reel", "shouldShareToFeed": True, "isAiGenerated": True}},
        }}
        with httpx.Client(timeout=45) as client:
            resp = client.post(
                BUFFER_GQL_URL,
                headers={"Authorization": f"Bearer {self.access_token}"},
                json={"query": EDIT_POST_MUTATION, "variables": variables},
            )
            if resp.status_code >= 400:
                logger.error("Buffer API HTTP %s: %s", resp.status_code, resp.text)
                resp.raise_for_status()
            payload = resp.json()
        data = payload.get("data", {}).get("editPost", {})
        if data.get("__typename") != "PostActionSuccess":
            raise RuntimeError(f"Buffer GraphQL Error: {data.get('message') or data}")
        return {"success": True, "post": data.get("post", {})}

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
        asset_type = str(top_prompt.get("asset_type") or asset_info.get("type") or "").lower()
        prompt_asset_path = top_prompt.get("asset_path")
        is_video = asset_type in {"reel", "video"} or str(prompt_asset_path or "").lower().endswith(".mp4")
        video_path = asset_info.get("video") or (prompt_asset_path if is_video else None)
        image_path = (
            top_prompt.get("asset_path")
            if not video_path
            else None
        ) or asset_info.get("image") or asset_info.get("thumb")

        def public_url(asset_path: str | None) -> str | None:
            if not asset_path:
                return None
            if "assets/" in asset_path:
                asset_path = asset_path[asset_path.index("assets/"):]
            return self.get_public_asset_url(asset_path)

        video_url = public_url(video_path)
        image_url = public_url(image_path) if not video_url else None

        try:
            self.schedule_post(text=caption, image_url=image_url, video_url=video_url)
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
