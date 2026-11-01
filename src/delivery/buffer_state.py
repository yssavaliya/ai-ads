"""Read-only look at what Buffer already holds for a channel, so a new post does not duplicate or crowd it.

Never creates, edits or deletes anything. Several projects can feed one Buffer channel (the free plan holds only
10 scheduled posts), so each of them checks before adding.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BUFFER_GQL_URL = "https://api.buffer.com"
WAITING_STATUSES = {"scheduled", "buffer", "needs_approval"}  # drafts are not queued posts

ORGANIZATIONS_QUERY = "query GetOrganizations { account { organizations { id name } } }"
POSTS_QUERY = """
query GetPosts($input: PostsInput!, $first: Int) {
  posts(first: $first, input: $input) {
    edges { node { id status dueAt sentAt text channelId } }
  }
}
"""

SIMILARITY_THRESHOLD = 0.85


@dataclass(frozen=True)
class QueueState:
    posts: list[dict[str, Any]]

    @property
    def scheduled(self) -> int:
        return sum(1 for p in self.posts if str(p.get("status", "")).lower() in WAITING_STATUSES)

    def texts(self) -> list[str]:
        return [str(p.get("text") or "") for p in self.posts]


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
    resp.raise_for_status()
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
                    client, token, POSTS_QUERY,
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
