"""Unverified (possibly hallucinated) competitors must not reach later prompts as market evidence."""
from src.intelligence.audience_mapper import AudienceMapper
from src.intelligence.research import (
    TREND_PROVENANCE,
    research_quality,
    summarize_competitors,
    verified_competitors,
)
from src.intelligence.trend_radar import TrendRadar

REAL = {"name": "Real Co", "handle": "realco", "top_posts": [{"caption": "Our new sneaker drop"}]}
GHOST = {"name": "Ghost Ltd", "handle": "ghostltd", "top_posts": []}


def test_only_competitors_with_scraped_posts_are_verified():
    assert verified_competitors([REAL, GHOST, {"name": "No key"}]) == [REAL]
    assert verified_competitors(None) == []


def test_summary_excludes_ghosts_and_says_how_much_was_verified():
    text = summarize_competitors([REAL, GHOST])
    assert "Real Co" in text and "Ghost Ltd" not in text
    assert "1 of 2" in text and "could not be verified" in text


def test_summary_warns_when_nothing_could_be_verified():
    text = summarize_competitors([GHOST])
    assert "No verified competitor data" in text and "Do not cite" in text
    assert summarize_competitors([]) == "No competitor data available."


def test_stages_use_the_shared_summary():
    assert AudienceMapper._summarize_competitors([REAL, GHOST]) == summarize_competitors([REAL, GHOST])
    assert TrendRadar._summarize_competitors([GHOST]) == summarize_competitors([GHOST])


def test_trend_prompt_labels_trends_as_hypotheses():
    from unittest.mock import patch

    radar = TrendRadar()
    seen = {}

    def fake(prompt, label=None, **kw):
        seen["prompt"] = prompt
        return {"trending_topics": []}

    with patch.object(radar.llm, "run_json", fake):
        radar.scan({"name": "B"}, {}, [GHOST])
    assert TREND_PROVENANCE in seen["prompt"] and "No verified competitor data" in seen["prompt"]


def test_research_quality_record():
    q = research_quality([REAL, GHOST], {"posts": [1, 2, 3]})
    assert q == {"competitors_suggested": 2, "competitors_verified": 1, "own_posts_seen": 3,
                 "trend_source": "model knowledge (no live feed)"}


def test_unscrapable_competitors_are_not_cached_for_a_day(tmp_path, monkeypatch):
    """Real run: a bad Apify token produced 5 competitors with no posts, and the 24h cache pinned that result."""
    from src.scheduler.daily_job import DailyPipeline
    from src.shared.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    pipeline = DailyPipeline.__new__(DailyPipeline)
    biz = {"name": "Cache Biz"}
    calls = []

    def compute():
        calls.append(1)
        return [GHOST]

    usable = lambda data: bool(verified_competitors(data))  # noqa: E731
    pipeline._cached_intel(biz, "competitors", compute, usable=usable)
    pipeline._cached_intel(biz, "competitors", compute, usable=usable)
    assert len(calls) == 2  # not cached: the second run retries

    good = lambda: [REAL]  # noqa: E731
    pipeline._cached_intel(biz, "competitors", good, usable=usable)
    assert pipeline._cached_intel(biz, "competitors", compute, usable=usable) == [REAL]  # cached once usable
