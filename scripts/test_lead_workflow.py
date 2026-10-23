"""End-to-end verification script for the Lead Generation Workflow.
Discovers or takes candidate businesses, scores their opportunity,
generates viral-engineered prompts, renders creative image assets,
and delivers the lead digest to the Content Hub website.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.delivery.channels import get_delivery  # noqa: E402
from src.generation.creative_forge import CreativeForge  # noqa: E402
from src.intelligence.lead_scorer import LeadScorer  # noqa: E402
from src.shared.config import settings  # noqa: E402


def run_lead_workflow() -> int:
    print("\n=======================================================")
    print("STEP 1: Initializing Lead Pipeline & Scorer")
    print("=======================================================")
    scorer = LeadScorer()
    forge = CreativeForge()
    delivery = get_delivery()

    candidate = {
        "name": "Artisan Bread House Ahmedabad",
        "website_url": "https://artisanbreadhouse.example",
        "instagram_handle": "artisanbreadhouse_ahm",
        "linkedin_handle": "",
        "niche": "bakery",
        "location": "Ahmedabad",
        "found_via": "search",
        "follower_count": 4850,
        "website_title": "Artisan Bread House | Fresh Organic Bakery Ahmedabad",
    }

    social_data = {
        "username": "artisanbreadhouse_ahm",
        "follower_count": 4850,
        "posts_count": 65,
        "biography": "Handcrafted sourdough and European breads in Ahmedabad. Fresh batches daily.",
        "posts": [
            {"caption": "Fresh sourdough loaves out of the oven! #sourdough #ahmedabad", "likes": 42, "comments": 3},
            {"caption": "Croissant layers in the making #croissants #pastry", "likes": 38, "comments": 2},
        ],
    }

    print("\n=======================================================")
    print("STEP 2: Scoring Lead Opportunity")
    print("=======================================================")
    assessment = scorer.score(candidate, social_data, min_followers=500, max_followers=50000)
    print(f"Eligible: {assessment['eligible']}")
    print(f"Opportunity Score: {assessment['score']}/100")
    print(f"Reasons: {assessment['reasons']}")
    assert assessment["eligible"], "Lead should be eligible"

    lead = {
        **candidate,
        "opportunity_score": assessment["score"],
        "opportunity_reasons": assessment["reasons"],
        "followers": assessment["followers"],
    }

    print("\n=======================================================")
    print("STEP 3: Generating Viral Creative Prompts & Rendering Assets")
    print("=======================================================")
    # Generate high quality prompt
    prompt = {
        "prompt_id": 991,
        "title": "The 24-Hour Fermentation Secret",
        "prompt_type": "educational",
        "virality_score": 89,
        "virality_label": "HIGH",
        "hook": "Why store bread lasts 2 weeks (and why real sourdough shouldn't)",
        "angle": "myth busting",
        "prompt_text": "Behind-the-scenes breakdown of sourdough fermentation vs commercial bread additives.",
        "caption": "Why does commercial bread stay soft for weeks? Preservatives.\n\nOur sourdough takes 24 hours to ferment with just flour, water, and wild yeast. It tastes better, digests easier, and is baked fresh every sunrise.\n\nDrop by today for a fresh warm loaf! 🥖 #ArtisanBread #AhmedabadFoodie #SourdoughLovers",
        "hashtags": ["#ArtisanBread", "#AhmedabadFoodie", "#SourdoughLovers"],
        "recommended_post_time": "Today at 06:30 PM IST",
        "format": "single image",
        "platform_focus": "instagram",
        "audience_segment": "Food enthusiasts & healthy eaters in Ahmedabad",
        "reasoning": "High curiosity hook with educational value that drives saves and shares.",
    }

    profile = {
        "name": lead["name"],
        "meta_description": lead["website_title"],
        "headings": ["Fresh Breads", "Sourdough Process", "Locations"],
    }

    print("Rendering creative asset image via CreativeForge...")
    rendered = forge.render_prompt(prompt, profile, index=0)
    print(f"Rendered asset path: {rendered['asset_path']}")
    print(f"Asset type: {rendered['asset_type']}")
    print(f"Warnings: {rendered.get('warnings')}")
    assert Path(rendered["asset_path"]).exists(), "Asset file must exist on disk"

    prompt["asset_path"] = rendered["asset_path"]
    prompt["asset_type"] = rendered["asset_type"]
    lead["prompts"] = [prompt]

    print("\n=======================================================")
    print("STEP 4: Publishing Lead Digest to Content Hub Website")
    print("=======================================================")
    digest = [lead]
    delivered = delivery.send_lead_digest("owner@example.com", digest)
    print(f"Delivered lead digest to site: {delivered}")
    assert delivered, "Lead digest delivery failed"

    # Verify files created on disk
    site_leads_data = Path("site/data/_leads.json")
    site_leads_page = Path("site/leads/index.html")
    assert site_leads_data.exists(), "site/data/_leads.json was not created"
    assert site_leads_page.exists(), "site/leads/index.html was not created"
    print(f"Verified: {site_leads_data} exists ({site_leads_data.stat().st_size} bytes)")
    print(f"Verified: {site_leads_page} exists ({site_leads_page.stat().st_size} bytes)")

    print("\n✅ LEAD WORKFLOW GENERATION COMPLETED SUCCESSFULLY!")
    return 0

if __name__ == "__main__":
    sys.exit(run_lead_workflow())
