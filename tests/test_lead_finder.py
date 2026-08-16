import asyncio
import base64
import urllib.parse
from unittest.mock import patch

import pytest

from src.fetch.lead_finder import LeadFinder


def _bing_redirect(target: str) -> str:
    encoded = "a1" + base64.b64encode(target.encode()).decode()
    return "https://www.bing.com/ck/a?!&&p=x&u=" + urllib.parse.quote(encoded) + "&ntb=1"


def test_decode_bing_url_direct():
    assert LeadFinder._decode_bing_url("https://example.com/") == "https://example.com/"


def test_decode_bing_url_redirect():
    assert LeadFinder._decode_bing_url(_bing_redirect("https://motibakery.com/")) == "https://motibakery.com/"


def test_decode_bing_url_garbage():
    assert LeadFinder._decode_bing_url("https://www.bing.com/ck/a?!&&u=notbase64!!!") == ""


def test_discover_runs_instagram_search_even_when_web_fills_count():
    finder = LeadFinder()

    async def fake_web_search(self, query):
        return [("https://shop-a.com/", "Shop A"), ("https://shop-b.com/", "Shop B")]

    async def fake_ig_search(self, query, limit=10):
        return [
            {
                "username": "ig_shop_c",
                "fullName": "IG Shop C",
                "followersCount": 1200,
                "externalUrl": "",
                "biography": "",
            }
        ]

    with (
        patch.object(finder, "_search_web", fake_web_search),
        patch("src.fetch.social_scraper.SocialScraper.fetch_instagram_search", fake_ig_search),
        patch("src.shared.config.settings.apify_token", "token"),
    ):
        candidates = asyncio.run(finder.discover(["bakery"], ["Ahmedabad"], count=3, platform_focus="instagram"))
    handles = [c.get("instagram_handle") for c in candidates]
    assert "ig_shop_c" in handles
    ig_candidate = next(c for c in candidates if c.get("instagram_handle") == "ig_shop_c")
    assert ig_candidate["follower_count"] == 1200
    assert ig_candidate["found_via"] == "instagram-search"


def test_enrich_looks_up_instagram_by_name_for_web_candidates():
    finder = LeadFinder()
    candidates = [
        {
            "name": "Sweet Bakery",
            "website_url": "https://sweetbakery.com/",
            "instagram_handle": "",
            "niche": "bakery",
            "location": "Ahmedabad",
        }
    ]

    async def fake_ig_search(self, query, limit=10):
        return [
            {
                "username": "sweetbakery",
                "fullName": "Sweet Bakery",
                "followersCount": 800,
                "externalUrl": "",
                "biography": "",
            }
        ]

    with (
        patch.object(finder, "_extract_social_handles", return_value={"instagram": "", "linkedin": ""}),
        patch("src.fetch.website_scraper.WebsiteScraper.scrape", return_value={"title": "x", "pages": []}),
        patch("src.fetch.social_scraper.SocialScraper.fetch_instagram_search", fake_ig_search),
        patch("src.shared.config.settings.apify_token", "token"),
    ):
        enriched = asyncio.run(finder.enrich_with_socials(candidates))

    assert enriched[0]["instagram_handle"] == "sweetbakery"
    assert enriched[0]["follower_count"] == 800