"""Playwright test suite for the Content Hub static website.
Tests the homepage, business pages, media assets, search filtering,
category chips, posted tracker, copy buttons, and responsive views.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCREENSHOTS_DIR = Path("logs/playwright_screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://127.0.0.1:8089"

def run_tests() -> int:
    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            permissions=["clipboard-read", "clipboard-write"],
        )
        page = context.new_page()

        # Listen for errors and failed requests
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on("requestfailed", lambda req: failed_requests.append(f"{req.method} {req.url} - {req.failure}"))

        print("\n--- 1. Testing Homepage ---")
        page.goto(f"{BASE_URL}/index.html", wait_until="networkidle")
        title = page.title()
        print(f"Page title: {title}")
        assert "Content Hub" in title, f"Unexpected title: {title}"

        hero = page.locator("section.hero h1").text_content()
        print(f"Hero heading: {hero}")
        assert hero and "Your content, ready to post" in hero

        tiles = page.locator(".tiles a.tile")
        tile_count = tiles.count()
        print(f"Found {tile_count} business tiles")
        assert tile_count >= 1, "Expected at least 1 business tile"

        page.screenshot(path=str(SCREENSHOTS_DIR / "homepage_desktop.png"))
        print(f"Saved screenshot: {SCREENSHOTS_DIR / 'homepage_desktop.png'}")

        print("\n--- 2. Testing Warning State on Aimagix Studio ---")
        aimagix_tile = page.locator(".tiles a.tile", has_text="Aimagix Studio")
        if aimagix_tile.count() > 0:
            aimagix_tile.click()
            page.wait_for_load_state("networkidle")
            aimagix_title = page.title()
            print(f"Aimagix Studio title: {aimagix_title}")
            assert "Aimagix Studio" in aimagix_title
            banner = page.locator(".banner.warn").first
            assert banner.count() > 0, "Expected warning banner"
            print(f"Warning banner verified: {(banner.text_content() or '')[:60]}...")
            # Go back to home
            page.locator(".brand").click()
            page.wait_for_load_state("networkidle")

        print("\n--- 3. Testing Business Page (E2E Bakery) ---")
        bakery_tile = page.locator(".tiles a.tile", has_text="E2E Bakery")
        assert bakery_tile.count() > 0, "Expected E2E Bakery tile on homepage"
        bakery_tile.click()
        page.wait_for_load_state("networkidle")
        biz_title = page.title()
        print(f"Business page title: {biz_title}")
        assert "E2E Bakery" in biz_title

        cards = page.locator(".card")
        card_count = cards.count()
        print(f"Total cards rendered: {card_count}")
        assert card_count == 3, f"Expected 3 cards, got {card_count}"

        # Check images on cards
        images = page.locator(".card img")
        img_count = images.count()
        print(f"Images in cards: {img_count}")
        assert img_count == 3

        for i in range(img_count):
            img = images.nth(i)
            src = img.get_attribute("src")
            # Natural width > 0 means the image loaded properly
            natural_width = img.evaluate("el => el.naturalWidth")
            print(f"Image {i+1} src={src} naturalWidth={natural_width}")
            assert natural_width > 0, f"Image {i+1} failed to load: {src}"

        page.screenshot(path=str(SCREENSHOTS_DIR / "biz_page_desktop.png"))

        print("\n--- 3. Testing Search Filtering ---")
        search_input = page.locator("#q")
        search_input.fill("sourdough")
        visible_cards = page.locator(".card:not(.hidden)").count()
        print(f"Search 'sourdough' visible cards: {visible_cards}")
        assert visible_cards >= 1

        search_input.fill("nonexistentquerythatmatchesnothing12345")
        visible_cards_zero = page.locator(".card:not(.hidden)").count()
        print(f"Search nonsense query visible cards: {visible_cards_zero}")
        assert visible_cards_zero == 0

        # Clear search
        search_input.fill("")
        visible_cards_all = page.locator(".card:not(.hidden)").count()
        print(f"Search cleared visible cards: {visible_cards_all}")
        assert visible_cards_all == 3

        print("\n--- 4. Testing Category Chips Filtering ---")
        edu_chip = page.locator(".chip[data-type='educational']")
        edu_chip.click()
        edu_cards = page.locator(".card:not(.hidden)").count()
        print(f"Educational chip visible cards: {edu_cards}")
        assert edu_cards >= 1

        all_chip = page.locator(".chip[data-type='']")
        all_chip.click()
        all_cards = page.locator(".card:not(.hidden)").count()
        print(f"All chip visible cards: {all_cards}")
        assert all_cards == 3

        print("\n--- 5. Testing Posted Tracker & Hide Posted Toggle ---")
        first_box = page.locator(".posted-box").first
        first_box.check()
        first_card = page.locator(".card").first
        assert "is-posted" in (first_card.get_attribute("class") or "")
        print("Checked first card as posted: .is-posted class applied")

        hide_toggle = page.locator("#hide-posted")
        hide_toggle.check()
        hidden_posted_cards = page.locator(".card:not(.hidden)").count()
        print(f"After 'Hide posted' checked, visible cards: {hidden_posted_cards}")
        assert hidden_posted_cards == 2, f"Expected 2 visible cards, got {hidden_posted_cards}"

        hide_toggle.uncheck()
        first_box.uncheck()
        print("Unchecked 'Hide posted' and reset posted box")

        print("\n--- 6. Testing Copy Buttons ---")
        copy_btn = page.locator(".copy").first
        copy_btn.click()
        page.wait_for_timeout(300)
        btn_text = copy_btn.text_content()
        print(f"Copy button after click: {btn_text}")
        assert "Copied" in (btn_text or "")

        toast = page.locator("#toast")
        toast_text = toast.text_content()
        print(f"Toast notification text: {toast_text}")
        assert "Copied" in (toast_text or "")

        print("\n--- 7. Testing Discovered Leads Page ---")
        page.goto(f"{BASE_URL}/leads/index.html", wait_until="networkidle")
        leads_title = page.title()
        print(f"Leads page title: {leads_title}")
        assert "Business leads" in leads_title

        leads_heading = page.locator("section.head h1").text_content()
        print(f"Leads heading: {leads_heading}")
        assert "Discovered business leads" in (leads_heading or "")

        lead_cards = page.locator(".card")
        print(f"Discovered lead cards: {lead_cards.count()}")
        assert lead_cards.count() >= 1

        lead_title = page.locator(".card h3").first.text_content()
        print(f"First lead business: {lead_title}")
        assert "Artisan Bread House" in (lead_title or "")

        # Open details to check creative prompt & image
        details = page.locator(".card details").first
        details.click()
        page.wait_for_timeout(300)
        lead_img = page.locator(".card details img").first
        img_src = lead_img.get_attribute("src")
        img_natural_width = lead_img.evaluate("el => el.naturalWidth")
        print(f"Lead creative image src: {img_src} naturalWidth: {img_natural_width}")
        assert img_natural_width > 0, f"Lead image failed to render: {img_src}"

        page.screenshot(path=str(SCREENSHOTS_DIR / "leads_page_desktop.png"))
        print(f"Saved leads screenshot: {SCREENSHOTS_DIR / 'leads_page_desktop.png'}")

        print("\n--- 8. Testing Weekly Report Page ---")
        page.goto(f"{BASE_URL}/b/aimagix-studio/weekly-2026-10-24.html", wait_until="networkidle")
        weekly_title = page.title()
        print(f"Weekly report title: {weekly_title}")
        assert "Weekly Deep Dive" in weekly_title or "Aimagix Studio" in weekly_title
        report_text = page.locator("main").text_content()
        assert len(report_text or "") > 200, "Weekly report content is too short"
        print("Weekly report content verified!")
        page.screenshot(path=str(SCREENSHOTS_DIR / "weekly_report_desktop.png"))

        print("\n--- 9. Testing Mobile Viewport ---")
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto(f"{BASE_URL}/leads/index.html", wait_until="networkidle")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "leads_page_mobile.png"))
        print(f"Saved mobile screenshot: {SCREENSHOTS_DIR / 'leads_page_mobile.png'}")

        print("\n--- 10. Checking Console & Request Errors ---")
        print(f"Console errors: {len(console_errors)}")
        for ce in console_errors:
            print(f"  [Console Error] {ce}")
        print(f"Page errors: {len(page_errors)}")
        for pe in page_errors:
            print(f"  [Page Error] {pe}")
        print(f"Failed requests: {len(failed_requests)}")
        for fr in failed_requests:
            print(f"  [Failed Request] {fr}")

        assert len(page_errors) == 0, f"Page errors found: {page_errors}"
        # Only fail on critical 404/500 asset requests
        critical_failed = [r for r in failed_requests if not r.endswith(".ico")]
        assert len(critical_failed) == 0, f"Critical request failures: {critical_failed}"

        browser.close()

    print("\n✅ ALL PLAYWRIGHT TESTS PASSED!")
    return 0

if __name__ == "__main__":
    sys.exit(run_tests())
