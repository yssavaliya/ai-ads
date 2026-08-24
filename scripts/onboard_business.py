from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fetch.website_scraper import WebsiteScraper
from src.shared.config import settings
from src.shared.utils import load_businesses, save_businesses

logger = logging.getLogger(__name__)


def onboard(
    name: str,
    website: str,
    email: str,
    instagram: str = "",
    linkedin: str = "",
    threshold: int = 75,
) -> dict:
    businesses = load_businesses()

    if any(b.get("name", "").lower() == name.lower() for b in businesses):
        raise ValueError(f"Business {name!r} already exists")

    if not email or "@" not in email:
        raise ValueError(f"Invalid email: {email!r}")

    business = {
        "id": max((b.get("id", 0) for b in businesses), default=0) + 1,
        "name": name,
        "website_url": website,
        "instagram_handle": instagram.lstrip("@"),
        "linkedin_handle": linkedin,
        "twitter_handle": "",
        "email": email,
        "timezone": "Asia/Kolkata",
        "virality_threshold": threshold,
        "prompt_types": ["educational", "storytelling", "social_proof"],
        "is_active": True,
    }
    businesses.append(business)
    save_businesses(businesses)

    profile = {}
    if website:
        try:
            scraper = WebsiteScraper()
            profile = scraper.scrape(website)
            logger.info("Website profile fetched: %s pages", profile.get("page_count"))
        except Exception as exc:
            logger.warning("Initial website scrape failed: %s", exc)

    logger.info("Business onboarded: %s (id=%s)", name, business["id"])
    return {"business": business, "profile": profile}


def main() -> None:
    parser = argparse.ArgumentParser(description="Onboard a business")
    parser.add_argument("--name", required=True)
    parser.add_argument("--website", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--instagram", default="")
    parser.add_argument("--linkedin", default="")
    parser.add_argument("--threshold", type=int, default=75)
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = onboard(args.name, args.website, args.email, args.instagram, args.linkedin, args.threshold)
    print(json.dumps(result["business"], indent=2))


if __name__ == "__main__":
    main()
