"""Regression tests for the bugs found in the full-project audit.

Each test documents a real failure that previously shipped (and passed the
older, heavily-mocked suite)."""
from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.shared.config import Settings, settings
from src.storage.models import (
    Base,
    Business,
    Engagement,
    FeedbackType,
    Platform,
    Prompt,
    PromptType,
)


def _session(tmp_path: Path):
    eng = create_engine(f"sqlite:///{tmp_path / 'reg.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    return sessionmaker(bind=eng, autoflush=False, autocommit=False)


def _seed(db, *, n_prompts: int = 1, ptype=PromptType.EDUCATIONAL):
    biz = Business(name="RegBiz", email="r@x.io")
    db.add(biz)
    db.commit()
    prompts = []
    for i in range(n_prompts):
        p = Prompt(
            business_id=biz.id,
            prompt_text=f"unique brief number {i} alpha{i} beta{i} gamma{i}",
            caption=f"caption {i} delta{i} epsilon{i} zeta{i}",
            prompt_type=ptype,
            format="reel",
            sent_at=datetime.utcnow() - timedelta(days=1),
        )
        db.add(p)
        prompts.append(p)
    db.commit()
    return biz, prompts


# --------------------------------------------------------------------------
# config: unset GitHub secrets arrive as "" and must not override defaults
# --------------------------------------------------------------------------
def test_empty_string_secrets_fall_back_for_string_fields(monkeypatch):
    for var in ("GROQ_MODEL", "GEMINI_MODEL", "SMTP_HOST"):
        monkeypatch.setenv(var, "")
    s = Settings()
    assert s.groq_model == "openai/gpt-oss-120b"
    assert s.gemini_model
    assert s.smtp_host == "smtp.gmail.com"


# --------------------------------------------------------------------------
# process_feedback
# --------------------------------------------------------------------------
@pytest.fixture
def fb_env(tmp_path, monkeypatch):
    import src.learning.process_feedback as pf

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    Session = _session(tmp_path)
    monkeypatch.setattr(pf, "SessionLocal", Session)
    closed: list[int] = []
    monkeypatch.setattr(pf, "close_issue", lambda n: closed.append(n))
    return pf, Session, closed


def _issues(pf, monkeypatch, items):
    monkeypatch.setattr(pf, "parse_feedback_issues", lambda: items)


def test_good_bad_feedback_does_not_crash_with_unbound_datetime(fb_env, monkeypatch):
    """Previously an inner `from datetime import datetime` in the POSTED branch
    made `datetime` a local -> UnboundLocalError on every GOOD/BAD/REGENERATE."""
    pf, Session, closed = fb_env
    db = Session()
    biz, (p1, p2) = _seed(db, n_prompts=2)
    ids = (p1.id, p2.id)
    db.close()
    _issues(pf, monkeypatch, [
        {"issue_number": 1, "prompt_id": ids[0], "feedback": "good", "body": "", "user": "a"},
        {"issue_number": 2, "prompt_id": ids[1], "feedback": "bad", "body": "", "user": "a"},
    ])
    assert pf.process_feedback() == 2
    db = Session()
    assert db.get(Prompt, ids[0]).feedback == FeedbackType.GOOD
    assert db.get(Prompt, ids[1]).feedback == FeedbackType.BAD
    assert closed == [1, 2]


def test_regenerate_feedback_does_not_crash(fb_env, monkeypatch):
    pf, Session, closed = fb_env
    db = Session()
    _, (p,) = _seed(db)
    pid = p.id
    db.close()
    monkeypatch.setattr(pf, "regenerate_prompt", lambda prompt, db: True)
    _issues(pf, monkeypatch, [{"issue_number": 5, "prompt_id": pid, "feedback": "regenerate", "body": "", "user": ""}])
    assert pf.process_feedback() == 1
    assert Session().get(Prompt, pid).feedback == FeedbackType.REGENERATE


def test_posted_link_records_engagement_but_no_weight_and_no_feedback(fb_env, monkeypatch):
    """Posting is not a hit: weights come from the post's real engagement."""
    pf, Session, closed = fb_env
    db = Session()
    biz, (p,) = _seed(db)
    pid, bid = p.id, biz.weights_key
    db.close()
    _issues(pf, monkeypatch, [{
        "issue_number": 9, "prompt_id": pid, "feedback": "posted", "user": "",
        "body": "https://www.instagram.com/p/ABC123/",
    }])
    assert pf.process_feedback() == 1
    db = Session()
    assert db.get(Prompt, pid).feedback is None  # still matchable by the learner
    rows = db.query(Engagement).filter_by(prompt_id=pid).all()
    assert [r.post_id for r in rows] == ["ABC123"]
    from src.learning.bayesian_updater import BayesianUpdater

    assert BayesianUpdater().get_weights(bid)["topics"] == {}


def test_repeated_clicks_do_not_inflate_weights(fb_env, monkeypatch):
    pf, Session, closed = fb_env
    db = Session()
    biz, (p,) = _seed(db)
    pid, bid = p.id, biz.weights_key
    db.close()
    _issues(pf, monkeypatch, [
        {"issue_number": n, "prompt_id": pid, "feedback": "good", "body": "", "user": ""} for n in (1, 2, 3)
    ])
    assert pf.process_feedback() == 1
    assert closed == [1, 2, 3]  # duplicates closed, counted once
    from src.learning.bayesian_updater import BayesianUpdater

    assert BayesianUpdater().get_weights(bid)["topics"]["educational"]["trials"] == 1


def test_feedback_allowlist_ignores_other_authors(fb_env, monkeypatch):
    pf, Session, closed = fb_env
    db = Session()
    _, (p,) = _seed(db)
    pid = p.id
    db.close()
    monkeypatch.setattr(settings, "feedback_allowed_users", "owner, Teammate")
    _issues(pf, monkeypatch, [
        {"issue_number": 1, "prompt_id": pid, "feedback": "bad", "body": "", "user": "troll"},
        {"issue_number": 2, "prompt_id": pid, "feedback": "good", "body": "", "user": "TEAMMATE"},
    ])
    assert pf.process_feedback() == 1
    assert Session().get(Prompt, pid).feedback == FeedbackType.GOOD


# --------------------------------------------------------------------------
# auto-learner: exact-ID ("I posted this") matching
# --------------------------------------------------------------------------
def test_explicit_link_matches_the_linked_prompt_and_does_not_crash(tmp_path, monkeypatch):
    """Previously pass 1 emitted 5-tuples (ValueError on unpack) and attributed
    the linked post to the FIRST prompt in the list instead of the linked one."""
    from src.learning.auto_learner import AutoLearner

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    db = _session(tmp_path)()
    biz, (first, linked) = _seed(db, n_prompts=2)
    linked.prompt_type = PromptType.STORYTELLING
    db.add(Engagement(prompt_id=linked.id, platform=Platform.INSTAGRAM,
                      post_id="ABC123", post_url="https://www.instagram.com/p/ABC123/"))
    db.commit()

    post = {
        "post_id": "apify-internal-999",  # Apify id != shortcode
        "url": "https://www.instagram.com/p/ABC123/",
        "caption": "nothing in common with either prompt",
        "likes": 40, "comments": 3,
        "posted_at": datetime.utcnow().isoformat(),
    }
    res = AutoLearner().run_for_business(biz.id, {"posts": [post], "follower_count": 1000}, db)
    assert res["matched"] == 1 and res["learned"] == 1

    db.expire_all()
    assert db.get(Prompt, linked.id).feedback is not None
    assert db.get(Prompt, first.id).feedback is None  # not mis-attributed
    from src.learning.bayesian_updater import BayesianUpdater

    topics = BayesianUpdater().get_weights(biz.weights_key)["topics"]
    assert "storytelling" in topics and "educational" not in topics
    assert len(db.query(Engagement).filter_by(prompt_id=linked.id).all()) == 1
    db.close()


def test_prompt_matched_once_across_passes(tmp_path):
    from src.learning.auto_learner import AutoLearner

    db = _session(tmp_path)()
    biz, (p,) = _seed(db)
    db.add(Engagement(prompt_id=p.id, platform=Platform.INSTAGRAM, post_id="LINK1",
                      post_url="https://instagram.com/p/LINK1"))
    db.commit()
    learner = AutoLearner()
    learner._db = db
    prompts = learner._recent_prompts(db, biz.id)
    now = datetime.utcnow().isoformat()
    posts = [
        {"post_id": "x", "url": "https://instagram.com/p/LINK1/", "caption": "", "posted_at": now},
        {"post_id": "y", "url": "https://instagram.com/p/OTHER/", "caption": prompts[0]["caption"], "posted_at": now},
    ]
    matched = learner._match(posts, prompts)
    assert len(matched) == 1 and matched[0][3]["post_id"] == "x"
    db.close()


# --------------------------------------------------------------------------
# weights / virality gate
# --------------------------------------------------------------------------
def test_proven_combo_gets_positive_boost_and_flop_negative(tmp_path):
    """Bayesian weights are in (0,1); the old neutral point of 1.0 meant a
    combo that succeeded 20/20 times was *penalised* versus an untried one."""
    from src.generation.virality_gate import ViralityGate
    from src.learning.bayesian_updater import BayesianUpdater

    u = BayesianUpdater(str(tmp_path / "w.json"))
    for _ in range(20):
        u.record(1, "good", prompt_type="educational")
    for _ in range(20):
        u.record(1, "bad", prompt_type="promotional")
    w = u.to_forge_weights(1)
    assert ViralityGate._weight_boost(w, {"prompt_type": "educational"}) > 0
    assert ViralityGate._weight_boost(w, {"prompt_type": "promotional"}) < 0
    assert ViralityGate._weight_boost(w, {"prompt_type": "never_tried"}) == 0


class _Forge:
    def __init__(self, n_types=1):
        self.n_types = n_types
        self.generate_calls = 0

    def generate_variants(self, **kw):
        self.generate_calls += 1
        return [
            {"prompt_type": kw["prompt_types"][i % len(kw["prompt_types"])],
             "prompt_text": f"v{self.generate_calls}-{i}", "format": "reel"}
            for i in range(3)
        ]

    def refine_with_hints(self, best, hints, *a):
        return {**best, "prompt_text": best["prompt_text"] + "-refined"}


def _pred(score, fail_on_refined=False):
    class P:
        def score(self, text, *a, **k):
            if fail_on_refined and text.endswith("-refined"):
                raise RuntimeError("predictor down")
            return {"virality_score": score, "virality_label": "HIGH", "virality_probability": 0.9,
                    "key_factors": [], "risks": [], "reasoning": "", "improvement_hints": ["h"]}
    return P()


def test_failed_refinement_scoring_does_not_discard_scored_variants():
    from src.generation.virality_gate import ViralityGate

    gate = ViralityGate(threshold=75, max_attempts=2, forge=_Forge(), predictor=_pred(40, fail_on_refined=True))
    out = gate.run({}, {}, {}, ["educational", "storytelling", "social_proof"])
    assert out  # previously the exception escaped and the whole run was lost


def test_gate_stops_early_when_business_has_fewer_than_three_types():
    from src.generation.virality_gate import ViralityGate

    forge = _Forge()
    gate = ViralityGate(threshold=75, max_attempts=5, forge=forge, predictor=_pred(90))
    gate.run({}, {}, {}, ["educational"])
    assert forge.generate_calls == 1  # one type satisfied => done, no 5x LLM burn


# --------------------------------------------------------------------------
# creative forge
# --------------------------------------------------------------------------
def test_segment_filter_is_valid_ffmpeg_syntax():
    from src.generation.creative_forge import CreativeForge

    f = CreativeForge._segment_filter(3.2)
    assert f.count("(") == f.count(")")
    assert "min(zoom+" in f and ",1.08)" in f  # capped zoom, two-arg min()
    assert "d=80" in f and "fps=25" in f


def test_reel_failure_of_any_type_falls_back_to_image(tmp_path, monkeypatch):
    from src.generation import creative_forge as cf

    monkeypatch.setattr(settings, "logs_dir", tmp_path)
    forge = cf.CreativeForge()
    monkeypatch.setattr(forge, "_render_reel", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("ffmpeg")))
    monkeypatch.setattr(forge, "generate_image", lambda *a, **k: str(tmp_path / "raw.png"))
    monkeypatch.setattr(forge, "overlay_text", lambda *a, **k: str(tmp_path / "final.png"))
    res = forge.render_prompt({"format": "reel", "reel_beats": ["a"], "title": "t", "hook": "h"}, {"name": "b"})
    assert res["asset_type"] == "image"
    assert any("falling back" in w for w in res["warnings"])


def test_missing_piper_raises_creative_forge_error(monkeypatch):
    from src.generation import creative_forge as cf

    monkeypatch.setattr(cf.shutil, "which", lambda name: None)
    with pytest.raises(cf.CreativeForgeError):
        cf.CreativeForge()._tts_beats(["hi"], "x")


def test_overlay_never_overwrites_source_image(tmp_path):
    from PIL import Image

    from src.generation.creative_forge import CreativeForge

    src = tmp_path / "raw.jpg"
    Image.new("RGB", (200, 300), (10, 20, 30)).save(src, "JPEG")
    out = CreativeForge().overlay_text(str(src), "hello world")
    assert out != str(src) and out.endswith("-texted.png")


# --------------------------------------------------------------------------
# email size cap
# --------------------------------------------------------------------------
def test_attachment_budget_accounts_for_base64_overhead(tmp_path):
    from src.delivery.email_sender import MAX_RAW_ATTACHMENT_BYTES, EmailSender

    assert MAX_RAW_ATTACHMENT_BYTES * 4 / 3 < 25_000_000
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"0" * 10_000_000)
    b.write_bytes(b"0" * 10_000_000)
    kept, dropped = EmailSender._size_safe_attachments([str(a), str(b)])
    assert len(kept) == 1 and len(dropped) == 1  # 20MB raw would be ~27MB encoded


def test_smtp_starttls_uses_a_verifying_context(monkeypatch):
    from src.delivery import email_sender as es

    server = MagicMock()
    server.__enter__.return_value = server
    monkeypatch.setattr(es.smtplib, "SMTP", lambda *a, **k: server)
    monkeypatch.setattr(settings, "dry_run", False)
    sender = es.EmailSender()
    sender.user, sender.password, sender.from_addr = "u", "p", "u"
    assert sender.send("to@x.io", "s", "<p>hi</p>") is True
    ctx = server.starttls.call_args.kwargs["context"]
    import ssl

    assert ctx.verify_mode == ssl.CERT_REQUIRED


# --------------------------------------------------------------------------
# lead pipeline
# --------------------------------------------------------------------------
def test_lead_config_keeps_enforce_location(tmp_path, monkeypatch):
    import json

    from src.scheduler.lead_job import LeadPipeline

    (tmp_path / "lead_config.json").write_text(json.dumps({"enforce_location": True, "niches": ["x"]}))
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    cfg = LeadPipeline.__new__(LeadPipeline).load_config()
    assert cfg["enforce_location"] is True


def test_min_opportunity_score_is_enforced():
    """A 55-point lead used to be delivered despite min_opportunity_score=60."""
    from src.scheduler.lead_job import LeadPipeline

    leads = [{"opportunity_score": 55}, {"opportunity_score": 60}, {"opportunity_score": 70}]
    kept = LeadPipeline._filter_min_score(leads, {"min_opportunity_score": 60})
    assert [lead["opportunity_score"] for lead in kept] == [60, 70]


def test_known_leads_exclude_real_clients(tmp_path, monkeypatch):
    from src.scheduler import lead_job
    from src.scheduler.lead_job import LeadPipeline

    monkeypatch.setattr(lead_job, "load_businesses", lambda: [{"name": "Client Co"}])
    assert LeadPipeline._client_names() == {"client co"}

    Session = _session(tmp_path)
    db = Session()
    db.add_all([
        Business(name="Client Co", email="c@x.io", instagram_handle="clientco"),
        Business(name="Lead Co", email="unverified+lead@leads.invalid", instagram_handle="leadco"),
    ])
    db.commit()
    scraped: list[str] = []
    pipe = LeadPipeline.__new__(LeadPipeline)

    class FakeSocial:
        async def fetch_instagram(self, handle):
            scraped.append(handle)
            return {"follower_count": 1000, "posts": []}

    monkeypatch.setattr(lead_job, "SocialScraper", FakeSocial)
    pipe.scorer = MagicMock()
    pipe.scorer.score.return_value = {"eligible": True, "score": 80, "reasons": [], "followers": 1000}
    out = pipe._merge_known_leads(db, [], {"min_followers": 1, "max_followers": 10**6,
                                            "target_count": 5, "min_opportunity_score": 60})
    assert [x["name"] for x in out] == ["Lead Co"]
    assert scraped == ["leadco"]  # the client was never scraped


# --------------------------------------------------------------------------
# website scraper / lead finder
# --------------------------------------------------------------------------
def test_www_and_bare_domain_are_the_same_site():
    from src.fetch.website_scraper import WebsiteScraper

    s = WebsiteScraper()
    assert s._same_domain("https://example.com", "https://www.example.com/about")
    assert not s._same_domain("https://example.com", "https://other.com/")


def test_crawl_ignores_fragments_and_binaries(monkeypatch):
    from src.fetch.website_scraper import WebsiteScraper

    pages = {
        "https://shop.com/": '<a href="#about">a</a><a href="/#contact">c</a><a href="/menu">m</a>'
                             '<a href="/brochure.pdf">p</a><a href="https://instagram.com/shop_official">ig</a>',
        "https://shop.com/menu": "<p>menu</p>",
    }
    fetched: list[str] = []

    def fake_fetch(self, url):
        fetched.append(url)
        return pages[url]

    monkeypatch.setattr(WebsiteScraper, "_fetch", fake_fetch)
    s = WebsiteScraper()
    results = s._crawl("https://shop.com/")
    assert fetched == ["https://shop.com/", "https://shop.com/menu"]  # no refetch of "#about", no pdf
    assert len(results) == 2
    assert s._extract_social_links(results) == ["https://instagram.com/shop_official"]


def test_social_handles_come_from_scraped_outbound_links():
    from src.fetch.lead_finder import LeadFinder

    handles = LeadFinder._extract_social_handles(
        [{"url": "https://shop.com/"}],
        "https://shop.com",
        ["https://instagram.com/explore", "https://www.instagram.com/shop_official/",
         "https://linkedin.com/company/shop-co"],
    )
    assert handles == {"instagram": "shop_official", "linkedin": "shop-co"}


# --------------------------------------------------------------------------
# daily job
# --------------------------------------------------------------------------
def test_failed_scrape_stub_is_not_a_usable_profile():
    from src.scheduler.daily_job import DailyPipeline

    assert not DailyPipeline._profile_has_content({"name": "X", "source": "website"})
    assert not DailyPipeline._profile_has_content(None)
    assert DailyPipeline._profile_has_content({"name": "X", "title": "Home", "headings": ["h"]})


# --------------------------------------------------------------------------
# llm client / social scraper
# --------------------------------------------------------------------------
def test_non_retryable_provider_errors_are_not_retried(monkeypatch):
    from src.shared import llm_client as lc

    class AuthError(Exception):
        status_code = 401

    client = MagicMock()
    client.chat.completions.create.side_effect = AuthError("bad key")
    monkeypatch.setattr(lc.time, "sleep", lambda s: pytest.fail("must not sleep/retry on 401"))
    with pytest.raises(AuthError):
        lc.LLMClient._chat(client, "m", "p", 0.5, 3)
    assert client.chat.completions.create.call_count == 1


def test_apify_token_travels_in_header_not_url():
    from src.fetch.social_scraper import SocialScraper

    assert SocialScraper(apify_token="secret")._auth_headers() == {"Authorization": "Bearer secret"}
