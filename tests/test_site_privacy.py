"""Audit C-02 / C-01: the public site must never carry contact details, and "private" URLs must
actually be revocable. Rotating SITE_SECRET, removing a business or changing mode must leave no
stale page, manifest or media reachable."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest
from PIL import Image

from src.delivery.site_publisher import DIGEST_KEEP, SitePublisher
from src.shared.config import settings

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
BIZ = {"name": "Sweet Crumbs Bakery", "instagram_handle": "sweetcrumbs", "website_url": "https://sweet.example", "email": "owner@sweet.example"}


def _img(tmp_path: Path, name="a.png") -> str:
    path = tmp_path / name
    Image.new("RGB", (200, 250), (10, 120, 60)).save(path)
    return str(path)


def _prompt(tmp_path, pid=1, **over):
    p = {"prompt_id": pid, "title": f"Idea {pid}", "prompt_type": "educational", "virality_score": 88, "hook": "Hook",
         "prompt_text": f"Brief {pid}", "caption": "Caption #x", "hashtags": ["#x"], "format": "single image",
         "asset_path": _img(tmp_path, f"a{pid}.png"), "asset_type": "image"}
    p.update(over)
    return p


def _lead(tmp_path, n=0):
    return {"name": f"Lead {n}", "instagram_handle": f"lead{n}", "website_url": "https://l.example", "niche": "bakery",
            "followers": 900, "opportunity_score": 70, "opportunity_reasons": ["ok"], "contact_email": f"private{n}@lead.example",
            "prompts": [{"title": "T", "prompt_type": "educational", "virality_score": 80, "prompt_text": "x",
                         "asset_path": _img(tmp_path, f"l{n}.png"), "asset_type": "image"}]}


def _site_emails(site: Path) -> set[str]:
    found: set[str] = set()
    for f in site.rglob("*"):
        if f.is_file() and f.suffix in {".html", ".json", ".txt", ".js", ".css"}:
            found |= set(EMAIL.findall(f.read_text(encoding="utf-8", errors="ignore")))
    return found


@pytest.fixture
def pub(tmp_path):
    return SitePublisher(tmp_path / "site")


@pytest.mark.parametrize("secret", [None, "s3cret"])
def test_no_email_address_reaches_any_generated_file_in_any_mode(pub, tmp_path, monkeypatch, secret):
    monkeypatch.setattr(settings, "site_secret", secret)
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path, caption="Mail us at hello@sweet.example or +91 98765 43210",
                                         prompt_text="Contact owner@sweet.example", first_comment="dm: who.is@gmail.com")])
    pub.send_weekly_report(BIZ, {"audience_insights": ["write to boss@sweet.example"], "calendar": []})
    pub.send_onboarding(BIZ, {"title": "t", "meta_description": "call +91 98765 43210 or info@sweet.example", "headings": ["mail a@b.co"]})
    pub.send_lead_digest("op@x.io", [_lead(tmp_path)])
    assert _site_emails(pub.site_dir) == set()
    blob = (pub.site_dir / "data/sweet-crumbs-bakery.json").read_text(encoding="utf-8")
    assert "98765" not in blob and "[email removed]" in blob


# ---------------------------------------------------------------- rotation / revocation
def test_rotating_the_secret_revokes_old_urls_and_rehomes_content(pub, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "site_secret", "alpha")
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path)])
    old_key = SitePublisher.path_key(BIZ["name"])
    old_lead_key = pub.leads_key()
    pub.send_lead_digest("x", [_lead(tmp_path)])
    assert (pub.site_dir / "b" / old_key / "index.html").exists() and (pub.site_dir / old_lead_key / "index.html").exists()

    monkeypatch.setattr(settings, "site_secret", "beta")
    new_key, new_lead_key = SitePublisher.path_key(BIZ["name"]), pub.leads_key()
    assert new_key != old_key and new_lead_key != old_lead_key
    pub.rebuild()

    assert not (pub.site_dir / "b" / old_key).exists()  # old URL is gone (404 on Pages)
    assert not (pub.site_dir / old_lead_key).exists()
    assert not (pub.site_dir / "assets" / old_key).exists()
    assert (pub.site_dir / "b" / new_key / "index.html").exists() and (pub.site_dir / new_lead_key / "index.html").exists()
    # the content survived: media moved and the manifest points at the new location
    entry = json.loads((pub.site_dir / "data/sweet-crumbs-bakery.json").read_text())["entries"][0]
    assert entry["asset"]["image"].startswith(f"assets/{new_key}/") and (pub.site_dir / entry["asset"]["image"]).exists()
    assert old_key not in "\n".join(pub.summary()) and new_key in "\n".join(pub.summary())
    assert (pub.site_dir / ".dirty").exists()  # a sweep alone must trigger a deploy
    assert old_key not in (pub.site_dir / "b" / new_key / "index.html").read_text(encoding="utf-8")


def test_manifests_are_named_by_the_stable_slug_not_the_secret(pub, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "site_secret", "alpha")
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path)])
    assert (pub.site_dir / "data/sweet-crumbs-bakery.json").exists()
    assert not list((pub.site_dir / "data").glob("sweet-crumbs-bakery-*.json"))


def test_legacy_manifest_named_by_its_old_key_is_normalised_not_duplicated(pub, tmp_path, monkeypatch):
    """Sites published before this fix have data/<slug>-<hash>.json."""
    monkeypatch.setattr(settings, "site_secret", "alpha")
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path)])
    key = SitePublisher.path_key(BIZ["name"])
    (pub.site_dir / "data/sweet-crumbs-bakery.json").rename(pub.site_dir / "data" / f"{key}.json")
    pub.rebuild()
    assert (pub.site_dir / "data/sweet-crumbs-bakery.json").exists() and not (pub.site_dir / "data" / f"{key}.json").exists()
    assert len(list((pub.site_dir / "b").iterdir())) == 1


def test_removing_a_business_sweeps_its_pages_and_media(pub, tmp_path):
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path)])
    pub.send_daily_prompts({**BIZ, "name": "Gone Co"}, [_prompt(tmp_path, 2)])
    assert (pub.site_dir / "b/gone-co/index.html").exists()
    (pub.site_dir / "data/gone-co.json").unlink()  # offboarded
    (pub.site_dir / ".dirty").unlink()
    pub.rebuild()
    assert not (pub.site_dir / "b/gone-co").exists() and not (pub.site_dir / "assets/gone-co").exists()
    assert (pub.site_dir / "b/sweet-crumbs-bakery/index.html").exists()
    assert (pub.site_dir / ".dirty").exists()


def test_toggling_the_secret_leaves_no_stale_lead_page(pub, tmp_path, monkeypatch):
    pub.send_lead_digest("x", [_lead(tmp_path)])
    assert (pub.site_dir / "leads/index.html").exists()
    monkeypatch.setattr(settings, "site_secret", "now-secret")
    pub.rebuild()
    assert not (pub.site_dir / "leads").exists() and (pub.site_dir / pub.leads_key() / "index.html").exists()


def test_lead_media_beyond_the_digest_window_is_deleted(pub, tmp_path):
    stale = pub.site_dir / "assets/leads/2020-01-01"
    stale.mkdir(parents=True)
    (stale / "old.jpg").write_bytes(b"x")
    pub.send_lead_digest("x", [_lead(tmp_path)])
    assert not stale.exists() and (pub.site_dir / "assets/leads" / date.today().isoformat()).exists()
    assert DIGEST_KEEP >= 1


def test_unrelated_files_in_the_site_root_are_not_touched(pub, tmp_path):
    (pub.site_dir).mkdir(parents=True)
    (pub.site_dir / "CNAME").write_text("hub.example.com", encoding="utf-8")  # custom-domain file
    pub.send_daily_prompts(BIZ, [_prompt(tmp_path)])
    assert (pub.site_dir / "CNAME").read_text(encoding="utf-8") == "hub.example.com"


# ---------------------------------------------------------------- the committed data (C-01)
SAFE_DOMAINS = ("example.com", "example.org", "example.net", "leads.invalid", "users.noreply.github.com")


def _tracked_data_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return sorted(p for p in (root / "data").rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"})


def test_committed_state_files_contain_no_real_email_addresses():
    """Real addresses in git are exposed to everyone with repo access and survive in history."""
    offenders = {}
    for path in _tracked_data_files():
        for found in EMAIL.findall(path.read_text(encoding="utf-8", errors="ignore")):
            if not found.lower().endswith(SAFE_DOMAINS):
                offenders.setdefault(path.name, set()).add(found)
    assert offenders == {}, f"real e-mail addresses are committed: {offenders}"


def test_committed_config_uses_env_references_for_contacts():
    root = Path(__file__).resolve().parents[1]
    businesses = json.loads((root / "data/businesses.json").read_text(encoding="utf-8"))
    assert all(b["email"].startswith("env:") or b["email"].endswith(SAFE_DOMAINS) for b in businesses)
    lead_cfg = json.loads((root / "data/lead_config.json").read_text(encoding="utf-8"))
    assert lead_cfg["digest_email"].startswith("env:") or not lead_cfg["digest_email"]


def test_lead_state_file_never_persists_contact_emails():
    root = Path(__file__).resolve().parents[1]
    blob = (root / "data/leads.json").read_text(encoding="utf-8")
    assert "contact_email" not in blob


def test_the_pipeline_does_not_write_contact_emails_into_data_leads_json(tmp_path, monkeypatch):
    from src.scheduler.lead_job import LeadPipeline

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    pipe = LeadPipeline.__new__(LeadPipeline)
    pipe._save_leads([{"name": "L", "contact_email": "private@lead.example", "prompts": [], "email": "x@y.co"}])
    saved = (tmp_path / "leads.json").read_text(encoding="utf-8")
    assert "private@lead.example" not in saved and "x@y.co" not in saved and '"name": "L"' in saved
