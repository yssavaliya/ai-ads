"""The static website that replaces e-mail delivery."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.delivery import site_publisher as sp
from src.delivery.channels import Delivery, get_delivery
from src.delivery.site_publisher import SitePublisher
from src.shared.config import settings

BIZ = {"name": "Sweet Crumbs Bakery", "instagram_handle": "sweetcrumbs", "website_url": "https://sweet.example",
       "email": "owner-private@sweet.example"}


def _image(tmp_path: Path, name="a.png", size=(540, 675), color=(200, 120, 60)) -> str:
    path = tmp_path / name
    Image.new("RGB", size, color).save(path)
    return str(path)


def _prompt(tmp_path, pid=1, **over):
    p = {
        "prompt_id": pid, "title": f"Idea {pid}", "prompt_type": "educational", "virality_score": 88,
        "virality_label": "HIGH", "hook": "Stop proofing wrong", "angle": "myth busting",
        "prompt_text": f"Detailed brief {pid}", "caption": "Caption #sourdough", "hashtags": ["#sourdough"],
        "first_comment": "Follow us", "recommended_post_time": "Today at 07:00 PM",
        "format": "single image", "platform_focus": "instagram", "audience_segment": "Home bakers",
        "reasoning": "Strong hook", "key_factors": ["hook"], "risks": ["long"],
        "feedback_links": {"good": "https://github.com/o/r/issues/new?x=1", "bad": "#", "regenerate": "", "posted": "https://github.com/o/r/issues/new?y"},
        "asset_path": _image(tmp_path, f"a{pid}.png"), "asset_type": "image",
    }
    p.update(over)
    return p


@pytest.fixture
def pub(tmp_path):
    return SitePublisher(tmp_path / "site")


def _manifest(pub, key="sweet-crumbs-bakery"):
    return json.loads((pub.site_dir / "data" / f"{key}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- publishing
def test_daily_prompts_produce_a_complete_page_with_optimised_media(pub, tmp_path):
    assert pub.send_daily_prompts(BIZ, [_prompt(tmp_path, 1), _prompt(tmp_path, 2, prompt_type="storytelling")]) is True
    site = pub.site_dir
    for rel in ("index.html", ".nojekyll", "robots.txt", "static/site.css", "static/site.js", "b/sweet-crumbs-bakery/index.html"):
        assert (site / rel).exists(), rel
    page = (site / "b/sweet-crumbs-bakery/index.html").read_text(encoding="utf-8")
    assert page.count('class="card"') == 2
    for needle in ("Sweet Crumbs Bakery", "Caption #sourdough", "Detailed brief 1", "Today at 07:00 PM", "Follow us",
                   "Stop proofing wrong", "@sweetcrumbs", "I posted this"):
        assert needle in page, needle
    entry = _manifest(pub)["entries"][0]
    full, thumb = site / entry["asset"]["image"], site / entry["asset"]["thumb"]
    assert full.suffix == ".jpg" and Image.open(full).format == "JPEG"
    assert Image.open(thumb).width <= sp.THUMB_WIDTH and thumb.stat().st_size < full.stat().st_size * 2
    assert (site / ".dirty").exists()


def test_site_never_publishes_the_contact_email(pub, tmp_path):
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path)])
    pub.send_onboarding(BIZ, {"title": "t"})
    pub.send_weekly_report(BIZ, {"audience_insights": ["x"]})
    for f in pub.site_dir.rglob("*"):
        if f.is_file() and f.suffix in {".html", ".json", ".txt"}:
            assert "owner-private@sweet.example" not in f.read_text(encoding="utf-8", errors="ignore"), f


def test_placeholder_feedback_links_are_dropped_real_ones_kept(pub, tmp_path):
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path)])
    links = _manifest(pub)["entries"][0]["feedback_links"]
    assert set(links) == {"good", "posted"}  # "#" (GitHub not configured) and "" are not links


def test_user_content_is_html_escaped(pub, tmp_path):
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path, caption="<script>alert(1)</script> hi", title="<b>x</b>")])
    page = (pub.site_dir / "b/sweet-crumbs-bakery/index.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_rerunning_the_same_prompt_replaces_instead_of_duplicating(pub, tmp_path):
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path, 7, title="first")])
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path, 7, title="second")])
    entries = _manifest(pub)["entries"]
    assert [e["title"] for e in entries] == ["second"]


def test_reel_is_published_with_poster(pub, tmp_path):
    video = tmp_path / "r.mp4"
    video.write_bytes(b"\x00" * 4096)
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path, asset_type="reel", asset_path=str(video), thumbnail=_image(tmp_path, "t.png"), format="reel")])
    asset = _manifest(pub)["entries"][0]["asset"]
    assert (pub.site_dir / asset["video"]).read_bytes() == video.read_bytes()
    assert (pub.site_dir / asset["poster"]).exists()
    page = (pub.site_dir / "b/sweet-crumbs-bakery/index.html").read_text(encoding="utf-8")
    assert "<video" in page and "Download Reel" in page


def test_oversized_reel_falls_back_to_its_poster(pub, tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "MAX_VIDEO_BYTES", 1000)
    video = tmp_path / "big.mp4"
    video.write_bytes(b"\x00" * 5000)
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path, asset_type="reel", asset_path=str(video), thumbnail=_image(tmp_path, "t.png"))])
    asset = _manifest(pub)["entries"][0]["asset"]
    assert asset["video"] is None and asset["poster"]
    assert not list(pub.site_dir.rglob("*.mp4"))
    assert "too large" in (pub.site_dir / "b/sweet-crumbs-bakery/index.html").read_text(encoding="utf-8")


def test_missing_or_corrupt_asset_never_loses_the_prompt(pub, tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path, 1, asset_path=str(tmp_path / "gone.png")), _prompt(tmp_path, 2, asset_path=str(broken))])
    entries = _manifest(pub)["entries"]
    assert len(entries) == 2 and all(e["asset"] is None and e["prompt_text"] for e in entries)


def test_failed_generation_is_shown_as_a_notice_not_silence(pub):
    pub.send_daily_prompts(BIZ, [])
    page = (pub.site_dir / "b/sweet-crumbs-bakery/index.html").read_text(encoding="utf-8")
    assert "No prompts could be generated today" in page and "No prompts yet" in page


def test_weekly_report_and_onboarding_pages(pub):
    pub.send_onboarding(BIZ, {"title": "Sweet Crumbs", "meta_description": "bread", "page_count": 7, "headings": ["Bread", "Cakes"]})
    pub.send_weekly_report(BIZ, {"top_performing": [{"prompt": "P", "score": 90, "feedback": "good", "why": "w"}],
                                 "risk_alerts": ["watch out"], "calendar": ["Mon: x"]})
    day = date.today().isoformat()
    weekly = (pub.site_dir / f"b/sweet-crumbs-bakery/weekly-{day}.html").read_text(encoding="utf-8")
    assert "Weekly report" in weekly and "watch out" in weekly and "Mon: x" in weekly
    page = (pub.site_dir / "b/sweet-crumbs-bakery/index.html").read_text(encoding="utf-8")
    assert f"weekly-{day}.html" in page and "About this setup" in page and "7 pages analysed" in page


# ------------------------------------------------------------------ retention
def test_old_entries_and_their_media_are_pruned(pub, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "site_retention_days", 10)
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path, 1)])
    manifest = _manifest(pub)
    old_day = (date.today() - timedelta(days=40)).isoformat()
    stale_dir = pub.site_dir / "assets/sweet-crumbs-bakery" / old_day
    stale_dir.mkdir(parents=True)
    (stale_dir / "old.jpg").write_bytes(b"x")
    manifest["entries"].append({**manifest["entries"][0], "id": f"{old_day}-9", "date": old_day, "created_at": old_day})
    (pub.site_dir / "data/sweet-crumbs-bakery.json").write_text(json.dumps(manifest), encoding="utf-8")

    pub.send_daily_prompts(BIZ, [_prompt(tmp_path, 2)])
    assert {e["date"] for e in _manifest(pub)["entries"]} == {date.today().isoformat()}
    assert not stale_dir.exists()
    assert (pub.site_dir / "assets/sweet-crumbs-bakery" / date.today().isoformat()).exists()


# ---------------------------------------------------------------------- leads
def _lead(tmp_path):
    return {"name": "Green Bakery", "instagram_handle": "green_bakery", "website_url": "https://g.example", "niche": "bakery",
            "followers": 1055, "opportunity_score": 72, "opportunity_reasons": ["sweet spot"], "contact_email": "hello@g.example",
            "prompts": [{"title": "Cake", "prompt_type": "social_proof", "virality_score": 80, "prompt_text": "Reveal the cake",
                         "asset_path": _image(tmp_path, "l.png"), "asset_type": "image"}]}


def test_lead_digest_page_hides_contact_emails_unless_private(pub, tmp_path):
    pub.send_lead_digest("ignored@x.io", [_lead(tmp_path)])
    page = (pub.site_dir / "leads/index.html").read_text(encoding="utf-8")
    assert "Green Bakery" in page and "Reveal the cake" in page and "hello@g.example" not in page
    assert "hello@g.example" not in (pub.site_dir / "data/_leads.json").read_text(encoding="utf-8")
    assert 'href="leads/' in (pub.site_dir / "index.html").read_text(encoding="utf-8")


def test_lead_digest_never_shows_emails_even_when_the_site_is_secret(pub, tmp_path, monkeypatch):
    """Audit C-02: SITE_SECRET used to switch lead e-mail publication ON. A secret URL is not
    access control, so the e-mail is never published at any setting."""
    monkeypatch.setattr(settings, "site_secret", "s3cret")
    pub.send_lead_digest("x", [_lead(tmp_path)])
    key = pub.leads_key()
    assert key.startswith("l-")
    assert "hello@g.example" not in (pub.site_dir / key / "index.html").read_text(encoding="utf-8")
    assert "hello@g.example" not in (pub.site_dir / "data/_leads.json").read_text(encoding="utf-8")
    assert not (pub.site_dir / "leads").exists()


def test_a_business_called_leads_cannot_clobber_the_leads_manifest(pub, tmp_path):
    pub.send_lead_digest("x", [_lead(tmp_path)])
    pub.send_daily_prompts({**BIZ, "name": "Leads"}, [_prompt(tmp_path)])
    assert (pub.site_dir / "data/_leads.json").exists() and (pub.site_dir / "data/leads.json").exists()
    assert json.loads((pub.site_dir / "data/_leads.json").read_text())["digests"][0]["leads"][0]["name"] == "Green Bakery"


# -------------------------------------------------------------------- privacy
def test_secret_urls_are_keyed_stable_and_unlisted(pub, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "site_secret", "alpha")
    key_a = SitePublisher.path_key("Sweet Crumbs Bakery")
    assert key_a.startswith("sweet-crumbs-bakery-") and key_a == SitePublisher.path_key("Sweet Crumbs Bakery")
    monkeypatch.setattr(settings, "site_secret", "beta")
    assert SitePublisher.path_key("Sweet Crumbs Bakery") != key_a
    monkeypatch.setattr(settings, "site_secret", "alpha")

    pub.send_daily_prompts(BIZ, [_prompt(tmp_path)])
    assert (pub.site_dir / "b" / key_a / "index.html").exists()
    index = (pub.site_dir / "index.html").read_text(encoding="utf-8")
    assert "Sweet Crumbs" not in index and "This hub is unlisted" in index  # nothing listed
    assert key_a in "\n".join(pub.summary())  # but the owner is told where to look


def test_pages_are_marked_noindex(pub, tmp_path):
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path)])
    assert 'content="noindex, nofollow"' in (pub.site_dir / "index.html").read_text(encoding="utf-8")
    assert "Disallow: /" in (pub.site_dir / "robots.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("repo,override,expected", [
    ("yssavaliya/ai-ads", None, "https://yssavaliya.github.io/ai-ads"),
    ("yssavaliya/yssavaliya.github.io", None, "https://yssavaliya.github.io"),
    ("o/r", "https://hub.example.com/", "https://hub.example.com"),
    (None, None, ""),
])
def test_public_base_url(pub, monkeypatch, repo, override, expected):
    monkeypatch.setattr(settings, "github_repo", repo)
    monkeypatch.setattr(settings, "site_base_url", override)
    assert pub.base_url() == expected


# -------------------------------------------------------------------- robustness
def test_rebuild_is_idempotent_and_survives_a_corrupt_manifest(pub, tmp_path):
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path)])
    first = (pub.site_dir / "b/sweet-crumbs-bakery/index.html").read_text(encoding="utf-8")
    pub.rebuild()
    assert (pub.site_dir / "b/sweet-crumbs-bakery/index.html").read_text(encoding="utf-8") == first

    (pub.site_dir / "data/sweet-crumbs-bakery.json").write_text("{ not json", encoding="utf-8")
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path, 5)])  # starts that business fresh instead of crashing
    assert len(_manifest(pub)["entries"]) == 1


def test_pretty_dates():
    assert sp._pretty_date(date.today().isoformat()) == "Today"
    assert sp._pretty_date((date.today() - timedelta(days=1)).isoformat()) == "Yesterday"
    assert sp._pretty_date("2026-01-05").endswith("2026") and sp._pretty_date("garbage") == "garbage"


# ---------------------------------------------------------- delivery facade
def test_default_delivery_is_the_site_not_email():
    d = get_delivery()
    assert d.names == ["site"]


@pytest.mark.parametrize("mode,expected", [("site", ["site"]), ("email", ["EmailSender"]), ("both", ["site", "EmailSender"]),
                                          ("SITE", ["site"]), (" Both ", ["site", "EmailSender"])])
def test_delivery_modes(mode, expected, monkeypatch):
    monkeypatch.setattr(settings, "delivery_mode", mode)
    assert get_delivery().names == expected


@pytest.mark.parametrize("typo", ["sit", "emial", "all", "none", "site,email"])
def test_unknown_delivery_mode_is_an_error_not_a_silent_fallback(typo):
    """Audit M-09: a typo used to quietly mean 'site'."""
    with pytest.raises(ValueError, match="DELIVERY_MODE"):
        get_delivery(typo)


@pytest.mark.parametrize("typo", ["sit", "emial", "all"])
def test_settings_reject_an_unknown_delivery_mode_at_startup(typo, monkeypatch):
    from pydantic import ValidationError

    from src.shared.config import Settings

    monkeypatch.setenv("DELIVERY_MODE", typo)
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.setenv("DELIVERY_MODE", "")  # unset GitHub variable -> default
    assert Settings().delivery_mode == "site"


def test_one_failing_channel_does_not_block_the_other():
    ok, bad = MagicMock(), MagicMock()
    ok.name, bad.name = "site", "email"
    ok.send_daily_prompts.return_value = True
    bad.send_daily_prompts.side_effect = RuntimeError("smtp down")
    result = Delivery([bad, ok]).send_daily_prompts({}, [])
    assert bool(result) is True and result.channels == {"email": False, "site": True}
    assert result.partial and result.failed == ["email"] and "smtp down" in result.errors["email"]  # visible, not collapsed
    assert bool(Delivery([bad]).send_daily_prompts({}, [])) is False


def test_delivery_reports_failure_when_no_channel_succeeds():
    a = MagicMock()
    a.name = "site"
    a.send_weekly_report.return_value = False
    result = Delivery([a]).send_weekly_report({}, {})
    assert bool(result) is False and not result.partial and result.failed == ["site"]
