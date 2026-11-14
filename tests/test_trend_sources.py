"""Live trend signals: parsing, provenance, independent failure of each source."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import httpx

from src.fetch import trend_sources as ts
from src.intelligence.research import LIVE_PROVENANCE, TREND_PROVENANCE, research_quality
from src.intelligence.trend_radar import TrendRadar

NOW = datetime(2026, 11, 14, 12, tzinfo=UTC)

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>x</title>
<item><title>Amazon floods listings with AI video ads</title><pubDate>Thu, 12 Nov 2026 08:00:00 GMT</pubDate><source url="https://a.example">Retail Weekly</source></item>
<item><title>Amazon floods listings with AI video ads</title><pubDate>Thu, 12 Nov 2026 09:00:00 GMT</pubDate></item>
<item><title>Ancient story from last year</title><pubDate>Mon, 03 Nov 2025 08:00:00 GMT</pubDate></item>
<item><title>   </title></item>
<item><title>Brand launches generative ad tool</title></item>
</channel></rss>"""

PROFILE = {"verified_facts": {"offers": ["AI video ads", "Product visuals!", "AI video ads", "Social creatives", "Extra offer"]}}


def test_queries_and_hashtags_are_derived_from_verified_offers_then_audience_themes():
    queries = ts.build_queries(PROFILE, {"content_themes": [{"theme": "Prompt blueprints"}]})
    assert queries == ["AI video ads", "Product visuals", "Social creatives"]
    assert ts.build_queries({"verified_facts": {}}, {"content_themes": [{"theme": "Prompt blueprints"}, "junk"]}) == ["Prompt blueprints"]
    assert ts.build_queries({}) == []
    assert ts.build_hashtags(["AI video ads", "Product visuals", "ab"]) == ["aivideoads", "productvisuals"]


def test_news_feed_is_deduplicated_dated_and_age_limited():
    signals = ts.parse_news_feed(FEED, NOW)
    titles = [s["title"] for s in signals]
    assert titles == ["Amazon floods listings with AI video ads", "Brand launches generative ad tool"]
    assert signals[0]["date"] == "2026-11-12" and signals[0]["detail"] == "Retail Weekly" and signals[1]["date"] == ""
    assert ts.parse_news_feed("not xml at all", NOW) == []


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def get(self, url, params=None, **kw):
        self.calls.append((url, params))
        if self.error:
            raise self.error
        return self.response

    def post(self, url, **kw):
        self.calls.append((url, kw.get("json")))
        if self.error:
            raise self.error
        return self.response

    def close(self):
        pass


def response(text="", payload=None, status=200):
    req = httpx.Request("GET", "https://x")
    if payload is not None:
        return httpx.Response(status, json=payload, request=req)
    return httpx.Response(status, text=text, request=req)


def test_fetch_news_uses_a_recent_niche_query_and_survives_failure():
    client = FakeClient(response(FEED))
    out = ts.fetch_news("AI video ads", client, NOW)
    assert out[0]["query"] == "AI video ads" and "when:14d" in client.calls[0][1]["q"]
    assert ts.fetch_news("x", FakeClient(error=httpx.ConnectError("down"))) == []
    assert ts.fetch_news("x", FakeClient(response("", status=503))) == []


def test_fetch_hacker_news_formats_points_and_ignores_bad_hits():
    payload = {"hits": [{"title": "Show HN: AI ads", "points": 42, "num_comments": 7, "created_at": "2026-11-13T10:00:00Z"}, {"title": ""}]}
    out = ts.fetch_hacker_news("AI ads", FakeClient(response(payload=payload)), NOW)
    assert out == [{"source": "hacker_news", "title": "Show HN: AI ads", "query": "AI ads",
                    "detail": "42 points, 7 comments", "date": "2026-11-13"}]
    assert ts.fetch_hacker_news("x", FakeClient(error=RuntimeError("boom"))) == []


def test_hashtag_posts_need_a_token_skip_errors_and_rank_by_engagement():
    items = [
        {"caption": "Quiet post", "likesCount": 1, "commentsCount": 0, "type": "Image", "timestamp": "2026-11-14T05:00:00Z"},
        {"caption": "Big hook line\nsecond line", "likesCount": 90, "commentsCount": 4, "type": "Video", "timestamp": "2026-11-13T05:00:00Z"},
        {"error": "no_items", "caption": "ignored"},
        {"likesCount": 500},  # no caption
    ]
    assert ts.fetch_hashtag_posts("aivideoads", "") == []
    client = FakeClient(response(payload=items))
    out = ts.fetch_hashtag_posts("aivideoads", "tok", client)
    assert [o["title"] for o in out] == ["Big hook line", "Quiet post"] and out[0]["detail"] == "90 likes, 4 comments, video"
    assert client.calls[0][1] == {"hashtags": ["aivideoads"], "resultsLimit": 12, "resultsType": "posts"}
    assert ts.fetch_hashtag_posts("x", "tok", FakeClient(error=httpx.ReadTimeout("slow"))) == []


def test_collect_signals_merges_sources_and_one_dead_source_does_not_matter(monkeypatch):
    monkeypatch.setattr(ts, "fetch_news", lambda q, *a, **k: [{"source": "news", "title": f"news {q}", "date": "2026-11-12"}])
    monkeypatch.setattr(ts, "fetch_hacker_news", lambda q, *a, **k: [])
    monkeypatch.setattr(ts, "fetch_hashtag_posts", lambda tag, token, *a, **k: [{"source": "instagram_hashtag", "title": f"#{tag} post"}])
    collected = ts.collect_signals(PROFILE, None, apify_token="tok")
    assert collected["queries"] == ["AI video ads", "Product visuals", "Social creatives"]
    assert collected["sources"] == ["instagram_hashtag", "news"] and len(collected["signals"]) == 5
    no_token = ts.collect_signals(PROFILE, None, apify_token="")
    assert no_token["sources"] == ["news"]
    assert ts.collect_signals({}, None, apify_token="") ["signals"] == []


def test_prompt_formatting_is_attributed_and_empty_when_there_is_nothing():
    assert ts.format_for_prompt(None) == "" and ts.format_for_prompt({"signals": []}) == ""
    text = ts.format_for_prompt({"fetched_at": "2026-11-14T12:00", "queries": ["a"], "signals": [
        {"source": "news", "title": "T", "detail": "Retail Weekly", "date": "2026-11-12"}]})
    assert text.startswith("LIVE SIGNALS (fetched 2026-11-14T12:00") and "[news, 2026-11-12] T (Retail Weekly)" in text


def test_trend_prompt_switches_provenance_when_live_signals_exist():
    radar, seen = TrendRadar(), []

    def fake(prompt, label=None, **kw):
        seen.append(prompt)
        return {"trending_topics": []}

    signals = {"fetched_at": "now", "queries": ["q"], "signals": [{"source": "news", "title": "Headline X", "date": "2026-11-12"}]}
    with patch.object(radar.llm, "run_json", fake):
        radar.scan({"name": "B"}, {}, [], signals)
        radar.scan({"name": "B"}, {}, [], None)
    assert LIVE_PROVENANCE in seen[0] and "Headline X" in seen[0] and TREND_PROVENANCE not in seen[0]
    assert TREND_PROVENANCE in seen[1] and "LIVE SIGNALS" not in seen[1]
    assert "basis" in seen[0]


def test_research_quality_reports_the_trend_source():
    assert research_quality([], {}, None)["trend_source"] == "model knowledge (no live feed)"
    live = research_quality([], {}, {"signals": [1, 2], "sources": ["news", "hacker_news"]})
    assert live["trend_source"] == "live signals (2 from news, hacker_news) + model knowledge"
