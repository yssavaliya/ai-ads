"""Competitor discovery by real Instagram search, scraped in waves until enough are verified."""
from __future__ import annotations

from unittest.mock import patch

from src.fetch.competitor_finder import CompetitorFinder
from src.shared.config import settings

PROFILE = {
    "name": "Studio", "instagram_handle": "ownstudio",
    "verified_facts": {"offers": ["AI video ads", "Product visuals!", "Social creatives", "Idea-to-ad creative work"]},
}


def hit(username, followers, name="", bio=""):
    return {"username": username, "followersCount": followers, "fullName": name, "biography": bio}


def posts(n=3):
    return [{"caption": f"post {i}", "likes": 5, "comments": 1, "posted_at": "2026-11-01T10:00:00Z"} for i in range(n)]


def test_queries_come_from_verified_offers_as_plain_words():
    assert CompetitorFinder.search_queries(PROFILE) == ["AI video ads", "Product visuals", "Social creatives"]
    assert CompetitorFinder.search_queries({"name": "x"}) == []
    assert CompetitorFinder.search_queries({"verified_facts": {"offers": ["Ads", "ads", " "]}}) == ["Ads"]


def test_search_results_are_filtered_to_real_peers_and_ranked_by_relevance():
    results = [
        hit("ownstudio", 800),                                  # us
        hit("tiny", 20, bio="AI video ads"),                    # too small to say anything
        hit("globalbrand", 9_000_000, bio="AI video ads"),     # not a peer
        hit("peer_a", 3000, "Peer A", "We make AI video ads and product visuals"),
        hit("peer_b", 90000, "Peer B", "Photography"),
        hit("peer_a", 3000),                                    # duplicate
        hit("", 3000),
    ]
    ranked = CompetitorFinder.rank_search_results(results, "@OwnStudio", ["AI video ads", "Product visuals"])
    assert [c["handle"] for c in ranked] == ["peer_a", "peer_b"]
    assert ranked[0]["source"] == "search" and ranked[0]["follower_count"] == 3000 and ranked[0]["platform"] == "instagram"


def _finder(monkeypatch, search_hits, llm_suggestions, scrape):
    monkeypatch.setattr(settings, "competitor_search_enabled", True)
    finder = CompetitorFinder()
    finder.social.apify_token = "t"

    async def fake_search(query, limit=10):
        return search_hits

    async def fake_scrape(competitors):
        return [scrape(dict(c)) for c in competitors]

    finder.social.fetch_instagram_search = fake_search
    finder.fetch_competitor_posts = fake_scrape
    return finder, patch.object(finder, "discover", return_value=llm_suggestions)


def test_search_candidates_are_scraped_first_and_unverifiable_ones_dropped(monkeypatch):
    hits = [hit(f"peer{i}", 2000 + i, bio="AI video ads") for i in range(6)]
    ghosts = {"peer1", "peer3"}
    finder, llm = _finder(
        monkeypatch, hits, [{"name": "Ghost", "platform": "instagram", "handle": "madeupbrand"}],
        lambda c: {**c, "top_posts": [] if c["handle"] in ghosts else posts()},
    )
    with llm:
        found = finder.discover_and_fetch(PROFILE)
    handles = [c["handle"] for c in found]
    assert 3 <= len(found) <= 5 and not ghosts & set(handles) and "madeupbrand" not in handles
    assert all(c["top_posts"] for c in found) and all(c["source"] == "search" for c in found)


def test_scraping_stops_after_the_first_wave_when_enough_are_verified(monkeypatch):
    waves = []
    hits = [hit(f"peer{i}", 2000 + i, bio="AI video ads") for i in range(9)]
    finder, llm = _finder(monkeypatch, hits, [], lambda c: {**c, "top_posts": posts()})
    original = finder.fetch_competitor_posts

    async def counting(competitors):
        waves.append(len(competitors))
        return await original(competitors)

    finder.fetch_competitor_posts = counting
    with llm:
        found = finder.discover_and_fetch(PROFILE)
    assert waves == [5] and len(found) == 5


def test_a_second_wave_runs_when_the_first_has_too_few_verified(monkeypatch):
    waves = []
    hits = [hit(f"peer{i}", 2000 + i, bio="AI video ads") for i in range(9)]
    first_wave_ghosts = {c["handle"] for c in CompetitorFinder.rank_search_results(hits, "", ["AI video ads"])[:4]}
    finder, llm = _finder(monkeypatch, hits, [], lambda c: {**c, "top_posts": [] if c["handle"] in first_wave_ghosts else posts()})
    original = finder.fetch_competitor_posts

    async def counting(competitors):
        waves.append(len(competitors))
        return await original(competitors)

    finder.fetch_competitor_posts = counting
    with llm:
        found = finder.discover_and_fetch(PROFILE)
    assert waves == [5, 4] and len(found) == 5 and not first_wave_ghosts & {c["handle"] for c in found}


def test_when_nothing_verifies_the_unverified_list_is_returned_so_it_is_not_cached(monkeypatch):
    finder, llm = _finder(monkeypatch, [hit("peer", 2000, bio="AI video ads")],
                          [{"name": "L", "platform": "instagram", "handle": "llmguess"}],
                          lambda c: {**c, "top_posts": []})
    with llm:
        found = finder.discover_and_fetch(PROFILE)
    assert {c["handle"] for c in found} == {"peer", "llmguess"} and not any(c["top_posts"] for c in found)


def test_llm_suggestions_still_work_without_an_apify_token_or_with_search_disabled(monkeypatch):
    finder, llm = _finder(monkeypatch, [hit("peer", 2000, bio="AI video ads")],
                          [{"name": "L", "platform": "instagram", "handle": "llmpeer"}], lambda c: {**c, "top_posts": posts()})
    finder.social.apify_token = ""
    with llm:
        assert [c["handle"] for c in finder.discover_and_fetch(PROFILE)] == ["llmpeer"]
    finder.social.apify_token = "t"
    monkeypatch.setattr(settings, "competitor_search_enabled", False)
    with llm:
        assert [c["handle"] for c in finder.discover_and_fetch(PROFILE)] == ["llmpeer"]


def test_search_and_llm_failures_degrade_to_empty_not_a_crash(monkeypatch):
    finder, llm = _finder(monkeypatch, [], [], lambda c: c)

    async def boom(query, limit=10):
        raise RuntimeError("apify down")

    finder.social.fetch_instagram_search = boom
    with patch.object(finder, "discover", side_effect=RuntimeError("llm down")):
        assert finder.discover_and_fetch(PROFILE) == []
    assert finder.discover_by_search({"name": "no offers"}) == []


def test_llm_suggestion_that_is_our_own_account_is_ignored(monkeypatch):
    finder, llm = _finder(monkeypatch, [], [{"name": "Us", "platform": "instagram", "handle": "OwnStudio"}], lambda c: {**c, "top_posts": posts()})
    with llm:
        assert finder.discover_and_fetch(PROFILE) == []
