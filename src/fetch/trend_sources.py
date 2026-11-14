"""Live market signals: what is actually being written and posted this week, instead of the model's memory.

Sources (all public, fetched with fixed hosts):
- Google News RSS search: headlines and dates for the business's own niche queries;
- Hacker News (Algolia) search: what technical/AI audiences are discussing;
- Instagram hashtag posts via Apify (only when a token exists): real posts under the niche hashtags, with likes.

Every signal carries its source and date so downstream prompts can cite it, and every source fails independently:
a dead feed means fewer signals, never a failed run.
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.shared.config import settings

logger = logging.getLogger(__name__)

NEWS_URL = "https://news.google.com/rss/search"
HN_URL = "https://hn.algolia.com/api/v1/search_by_date"
APIFY_HASHTAG_ACTOR = "apify~instagram-hashtag-scraper"
APIFY_BASE = "https://api.apify.com/v2"
MAX_FEED_BYTES = 600_000
USER_AGENT = "Mozilla/5.0 (compatible; content-intelligence trend reader)"


def _clean(text: Any, limit: int = 200) -> str:
    return " ".join(str(text or "").split())[:limit]


def build_queries(profile: dict[str, Any], audience_map: dict[str, Any] | None = None, limit: int = 3) -> list[str]:
    """Niche queries from what the owner says they offer, then the audience's own content themes."""
    queries: list[str] = []
    facts = profile.get("verified_facts") or {}
    candidates = [str(o) for o in (facts.get("offers") or [])]
    candidates += [str(t.get("theme") or "") for t in (audience_map or {}).get("content_themes", []) if isinstance(t, dict)]
    for text in candidates:
        query = " ".join(re.sub(r"[^\w\s]", " ", text).split()[:4])
        if query and query.lower() not in {q.lower() for q in queries}:
            queries.append(query)
        if len(queries) >= limit:
            break
    return queries


def build_hashtags(queries: list[str], limit: int = 2) -> list[str]:
    tags: list[str] = []
    for query in queries:
        tag = re.sub(r"[^a-z0-9]", "", query.lower())
        if 4 <= len(tag) <= 30 and tag not in tags:
            tags.append(tag)
        if len(tags) >= limit:
            break
    return tags


def parse_news_feed(xml_text: str, now: datetime | None = None, max_age_days: int = 14, limit: int = 6) -> list[dict[str, Any]]:
    now = now or datetime.now(UTC)
    soup = BeautifulSoup(xml_text[:MAX_FEED_BYTES], "xml")
    signals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in soup.find_all("item"):
        title = _clean(item.title.get_text() if item.title else "")
        if not title or title.lower() in seen:
            continue
        published = None
        if item.pubDate:
            try:
                published = parsedate_to_datetime(item.pubDate.get_text().strip()).astimezone(UTC)
            except (TypeError, ValueError):
                published = None
        if published is not None and now - published > timedelta(days=max_age_days):
            continue
        seen.add(title.lower())
        source = _clean(item.source.get_text() if item.source else "", 60)
        signals.append({
            "source": "news", "title": title, "detail": source,
            "date": published.date().isoformat() if published else "",
        })
        if len(signals) >= limit:
            break
    return signals


def fetch_news(query: str, client: httpx.Client | None = None, now: datetime | None = None) -> list[dict[str, Any]]:
    owns = client is None
    client = client or httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    try:
        resp = client.get(NEWS_URL, params={"q": f"{query} when:14d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
        resp.raise_for_status()
        return [{**s, "query": query} for s in parse_news_feed(resp.text, now)]
    except Exception as exc:  # noqa: BLE001
        logger.info("News signals unavailable for %r: %s", query, exc)
        return []
    finally:
        if owns:
            client.close()


def fetch_hacker_news(query: str, client: httpx.Client | None = None, now: datetime | None = None, limit: int = 4) -> list[dict[str, Any]]:
    now = now or datetime.now(UTC)
    since = int((now - timedelta(days=14)).timestamp())
    owns = client is None
    client = client or httpx.Client(timeout=20)
    try:
        resp = client.get(HN_URL, params={"query": query, "tags": "story", "numericFilters": f"created_at_i>{since},points>3", "hitsPerPage": limit})
        resp.raise_for_status()
        out = []
        for hit in resp.json().get("hits", [])[:limit]:
            title = _clean(hit.get("title"))
            if not title:
                continue
            out.append({
                "source": "hacker_news", "title": title, "query": query,
                "detail": f"{int(hit.get('points') or 0)} points, {int(hit.get('num_comments') or 0)} comments",
                "date": str(hit.get("created_at") or "")[:10],
            })
        return out
    except Exception as exc:  # noqa: BLE001
        logger.info("Hacker News signals unavailable for %r: %s", query, exc)
        return []
    finally:
        if owns:
            client.close()


def fetch_hashtag_posts(hashtag: str, token: str, client: httpx.Client | None = None, limit: int = 12) -> list[dict[str, Any]]:
    """Real recent Instagram posts under a niche hashtag, best-performing first (needs APIFY_TOKEN; costs credits)."""
    if not token:
        return []
    owns = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(200, connect=30))
    try:
        resp = client.post(
            f"{APIFY_BASE}/acts/{APIFY_HASHTAG_ACTOR}/run-sync-get-dataset-items",
            headers={"Authorization": f"Bearer {token}"},
            json={"hashtags": [hashtag], "resultsLimit": limit, "resultsType": "posts"},
        )
        resp.raise_for_status()
        items = [i for i in resp.json() if isinstance(i, dict) and not i.get("error") and i.get("caption")]
        items.sort(key=lambda i: int(i.get("likesCount") or 0) + 2 * int(i.get("commentsCount") or 0), reverse=True)
        out = []
        for item in items[:4]:
            caption = _clean(str(item["caption"]).splitlines()[0], 140)
            out.append({
                "source": "instagram_hashtag", "title": caption, "query": f"#{hashtag}",
                "detail": f"{max(int(item.get('likesCount') or 0), 0)} likes, {int(item.get('commentsCount') or 0)} comments"
                          + (f", {str(item.get('type') or '').lower()}" if item.get("type") else ""),
                "date": str(item.get("timestamp") or "")[:10],
            })
        return out
    except Exception as exc:  # noqa: BLE001
        logger.info("Hashtag signals unavailable for #%s: %s", hashtag, exc)
        return []
    finally:
        if owns:
            client.close()


def collect_signals(
    profile: dict[str, Any], audience_map: dict[str, Any] | None = None, apify_token: str | None = None
) -> dict[str, Any]:
    """All live signals for a business, with provenance. ``signals`` is empty when every source failed."""
    queries = build_queries(profile, audience_map)
    signals: list[dict[str, Any]] = []
    for query in queries:
        signals += fetch_news(query)
        signals += fetch_hacker_news(query)
    token = apify_token if apify_token is not None else settings.apify_token
    for tag in build_hashtags(queries) if token else []:
        signals += fetch_hashtag_posts(tag, token or "")
    return {
        "fetched_at": datetime.now(UTC).isoformat(timespec="minutes"),
        "queries": queries,
        "signals": signals[:24],
        "sources": sorted({s["source"] for s in signals}),
    }


def format_for_prompt(collected: dict[str, Any] | None, limit: int = 14) -> str:
    """One line per signal, dated and attributed; an explicit statement when there are none."""
    if not collected or not collected.get("signals"):
        return ""
    lines = [
        f"- [{s['source']}{', ' + s['date'] if s.get('date') else ''}] {s['title']}"
        + (f" ({s['detail']})" if s.get("detail") else "")
        for s in collected["signals"][:limit]
    ]
    return f"LIVE SIGNALS (fetched {collected.get('fetched_at', '')}; queries: {', '.join(collected.get('queries', []))}):\n" + "\n".join(lines)
