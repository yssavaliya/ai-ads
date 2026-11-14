"""Read-only look at what Buffer already holds for a channel, so a new post does not duplicate or crowd it.

Never creates, edits or deletes anything. Several projects can feed one Buffer channel (the free plan holds only
10 scheduled posts), so each of them checks before adding.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

import httpx

from src.delivery.slot_planner import parse_due

logger = logging.getLogger(__name__)

BUFFER_GQL_URL = "https://api.buffer.com"
WAITING_STATUSES = {"scheduled", "buffer", "needs_approval"}  # drafts are not queued posts
DRAFT_STATUSES = {"draft"}

ORGANIZATIONS_QUERY = "query GetOrganizations { account { organizations { id name } } }"
# Buffer's schema has changed under us before (``Asset.url`` vanished and every queue/draft read failed). The asset
# sub-selection is therefore built from what the live schema actually offers, and the read never depends on it.
ASSET_FIELD_CANDIDATES = ("source", "url", "thumbnail", "mimeType", "type")
ASSET_FIELDS_QUERY = 'query AssetFields { __type(name: "Asset") { fields { name } } }'
_ASSET_SELECTION_CACHE: dict[str, str] = {}


def posts_query(asset_selection: str, include_times: bool = True) -> str:
    times = "dueAt sentAt " if include_times else ""
    return (
        "query GetPosts($input: PostsInput!, $first: Int) { posts(first: $first, input: $input) { edges { node { "
        f"id status {times}text channelId createdAt assets {{ {asset_selection} }} }} }} }} }}"
    )


def asset_selection(client: httpx.Client, token: str) -> str:
    """Field selection for ``assets``: the candidate fields that exist, else ``__typename`` (always valid)."""
    if token in _ASSET_SELECTION_CACHE:
        return _ASSET_SELECTION_CACHE[token]
    selection = "__typename"
    try:
        fields = ((_graphql(client, token, ASSET_FIELDS_QUERY).get("__type") or {}).get("fields")) or []
        names = {f.get("name") for f in fields}
        chosen = [name for name in ASSET_FIELD_CANDIDATES if name in names]
        if chosen:
            selection = " ".join(chosen)
    except Exception as exc:  # noqa: BLE001 - introspection off: fall back to the always-valid selection
        logger.info("Buffer asset schema probe unavailable (%s)", exc)
    logger.info("Buffer asset selection: %s", selection)
    _ASSET_SELECTION_CACHE[token] = selection
    return selection


def asset_urls(post: dict[str, Any]) -> list[str]:
    """Media URLs of a post, whichever field Buffer calls them (``url`` or ``source``)."""
    urls = []
    for asset in post.get("assets") or []:
        if isinstance(asset, dict):
            url = asset.get("url") or asset.get("source")
            if isinstance(url, str) and url.startswith("http"):
                urls.append(url)
    return urls

PROMOTE_DRAFT_MUTATION = """
mutation PromoteDraft($input: EditPostInput!) {
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

DELETE_POST_MUTATION = """
mutation DeletePost($input: DeletePostInput!) {
  deletePost(input: $input) {
    __typename
    ... on PostActionSuccess { post { id status } }
    ... on UnauthorizedError { message }
    ... on InvalidInputError { message }
    ... on UnexpectedError { message }
  }
}
"""

EDIT_INPUT_MODE_QUERY = """
query EditPostMode { __type(name: "EditPostInput") { inputFields { name type { name kind ofType { name kind } } } } }
"""
ENUM_VALUES_QUERY = "query ModeValues($name: String!) { __type(name: $name) { enumValues { name } } }"

SIMILARITY_THRESHOLD = 0.85


@dataclass(frozen=True)
class QueueState:
    posts: list[dict[str, Any]]

    @property
    def scheduled(self) -> int:
        return sum(1 for p in self.posts if str(p.get("status", "")).lower() in WAITING_STATUSES)

    def texts(self, include_drafts: bool = True) -> list[str]:
        """Caption texts Buffer holds. A draft being reviewed must not be compared with itself, so the reviewer
        asks for ``include_drafts=False`` (posts that are already scheduled or sent)."""
        return [
            str(p.get("text") or "") for p in self.posts
            if include_drafts or str(p.get("status", "")).lower() not in DRAFT_STATUSES
        ]

    def due_times(self) -> list[datetime]:
        """When the waiting (scheduled) posts will go out: the slots a new post must keep clear of."""
        times = [
            parse_due(p.get("dueAt")) for p in self.posts
            if str(p.get("status", "")).lower() in WAITING_STATUSES
        ]
        return [t for t in times if t is not None]


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def find_duplicate(caption: str, existing: list[str]) -> str | None:
    """The existing text a caption repeats (same opening or near-identical wording), if any."""
    new = _normalise(caption)
    if not new:
        return None
    for text in existing:
        old = _normalise(text)
        if not old:
            continue
        if new[:80] == old[:80] or SequenceMatcher(None, new, old).ratio() >= SIMILARITY_THRESHOLD:
            return text
    return None


def _graphql(client: httpx.Client, token: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    resp = client.post(
        BUFFER_GQL_URL,
        headers={"Authorization": f"Bearer {token}"},
        json={"query": query, "variables": variables or {}},
    )
    if resp.status_code >= 400:
        # Buffer answers some GraphQL errors with HTTP 500 and the real reason in the body: keep it.
        detail = " ".join((resp.text or "")[:400].split())
        raise RuntimeError(f"Buffer API HTTP {resp.status_code}: {detail}")
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError("; ".join(e.get("message", "unknown error") for e in payload["errors"]))
    return payload.get("data") or {}


def fetch_queue_state(token: str, channel_id: str, limit: int = 50, client: httpx.Client | None = None) -> QueueState:
    """The posts Buffer holds for ``channel_id`` (newest first, up to ``limit``). Raises on any API problem."""
    owns = client is None
    client = client or httpx.Client(timeout=30)
    try:
        orgs = ((_graphql(client, token, ORGANIZATIONS_QUERY).get("account") or {}).get("organizations")) or []
        if not orgs:
            raise RuntimeError("The Buffer token can see no organisations")
        last: Exception | None = None
        for org in orgs:
            try:
                data = _graphql(
                    client, token, posts_query(asset_selection(client, token)),
                    {"input": {"organizationId": org["id"], "filter": {"channelIds": [channel_id]}}, "first": limit},
                )
            except Exception as exc:  # noqa: BLE001 - try the next organisation
                last = exc
                continue
            edges = (data.get("posts") or {}).get("edges") or []
            return QueueState([(edge.get("node") or {}) for edge in edges])
        raise RuntimeError(str(last) if last else "no organisation returned posts")
    finally:
        if owns:
            client.close()


def fetch_drafts(token: str, channel_id: str, limit: int = 50, client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Return all draft posts for ``channel_id``."""
    owns = client is None
    client = client or httpx.Client(timeout=30)
    try:
        orgs = ((_graphql(client, token, ORGANIZATIONS_QUERY).get("account") or {}).get("organizations")) or []
        if not orgs:
            raise RuntimeError("The Buffer token can see no organisations")
        last: Exception | None = None
        for org in orgs:
            try:
                data = _graphql(
                    client, token, posts_query(asset_selection(client, token), include_times=False),
                    {"input": {"organizationId": org["id"], "filter": {"channelIds": [channel_id], "status": ["draft"]}}, "first": limit},
                )
            except Exception as exc:  # noqa: BLE001 - try the next organisation
                last = exc
                continue
            edges = (data.get("posts") or {}).get("edges") or []
            return [(edge.get("node") or {}) for edge in edges]
        if last is not None:
            # A schema change used to be swallowed here and read as "no drafts", so the scheduler silently did nothing.
            raise RuntimeError(f"Could not read Buffer drafts: {last}")
        return []
    finally:
        if owns:
            client.close()


def promote_draft(token: str, post_id: str, client: httpx.Client | None = None) -> dict[str, Any]:
    """Move a draft post into the active Buffer queue."""
    owns = client is None
    client = client or httpx.Client(timeout=30)
    try:
        data = _graphql(client, token, PROMOTE_DRAFT_MUTATION, {
            "input": {"postId": post_id, "saveToDraft": False, "mode": "addToQueue", "schedulingType": "automatic"},
        })
        result = data.get("editPost") or {}
        typename = result.get("__typename")
        if typename != "PostActionSuccess":
            raise RuntimeError(f"Buffer promote failed: {result.get('message') or result}")
        return result.get("post") or {}
    finally:
        if owns:
            client.close()


def delete_post(token: str, post_id: str, client: httpx.Client | None = None) -> bool:
    """Permanently delete a post (draft or scheduled) from Buffer."""
    owns = client is None
    client = client or httpx.Client(timeout=30)
    try:
        data = _graphql(client, token, DELETE_POST_MUTATION, {
            "input": {"postId": post_id},
        })
        result = data.get("deletePost") or {}
        return result.get("__typename") == "PostActionSuccess"
    finally:
        if owns:
            client.close()


def check_asset_url(url: str, timeout: float = 10) -> bool:
    """HEAD-check whether a media URL is still reachable (not expired or 404)."""
    if not url:
        return False
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.head(url)
            return resp.status_code < 400
    except Exception:  # noqa: BLE001
        return False


def supports_custom_schedule(token: str, client: httpx.Client | None = None) -> bool | None:
    """True/False when Buffer's schema says whether ``mode: customScheduled`` exists; None when it cannot be read
    (introspection off or a network error). Callers treat None as "try it, and fall back to the queue on error"."""
    owns = client is None
    client = client or httpx.Client(timeout=30)
    try:
        fields = ((_graphql(client, token, EDIT_INPUT_MODE_QUERY).get("__type") or {}).get("inputFields")) or []
        mode = next((f for f in fields if f.get("name") == "mode"), None)
        if mode is None:
            return None
        type_info = mode.get("type") or {}
        enum_name = type_info.get("name") or (type_info.get("ofType") or {}).get("name")
        if not enum_name:
            return None
        values = ((_graphql(client, token, ENUM_VALUES_QUERY, {"name": enum_name}).get("__type") or {}).get("enumValues")) or []
        return "customScheduled" in {v.get("name") for v in values} if values else None
    except Exception as exc:  # noqa: BLE001 - never block publishing on a schema probe
        logger.info("Buffer schema probe unavailable (%s)", exc)
        return None
    finally:
        if owns:
            client.close()


def schedule_draft(
    token: str, post_id: str, due_at: datetime, client: httpx.Client | None = None
) -> dict[str, Any]:
    """Move a draft into the queue at an exact time (``mode: customScheduled`` + ``dueAt``)."""
    owns = client is None
    client = client or httpx.Client(timeout=30)
    try:
        data = _graphql(client, token, PROMOTE_DRAFT_MUTATION, {
            "input": {
                "postId": post_id, "saveToDraft": False, "mode": "customScheduled",
                "schedulingType": "automatic", "dueAt": due_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            },
        })
        result = data.get("editPost") or {}
        if result.get("__typename") != "PostActionSuccess":
            raise RuntimeError(f"Buffer schedule failed: {result.get('message') or result}")
        return result.get("post") or {}
    finally:
        if owns:
            client.close()
