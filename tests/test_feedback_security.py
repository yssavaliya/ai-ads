"""Audit C-03: feedback mutates learned weights and can trigger regeneration, and anyone on
GitHub can open an issue. It must fail closed, authenticate the author AND the token, and be
bound to a durable prompt identity (the integer id is recycled when agent.db is rebuilt)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import src.learning.process_feedback as pf
from src.delivery.github_feedback import GithubFeedback
from src.learning import feedback_tokens as ft
from src.learning.bayesian_updater import BayesianUpdater
from src.shared.config import settings
from src.storage.models import Base, Business, FeedbackType, Prompt, PromptType, _migrate


def _env(tmp_path, monkeypatch, *, key="k" * 24, allowed="owner"):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "feedback_signing_key", key)
    monkeypatch.setattr(settings, "feedback_allowed_users", allowed)
    eng = create_engine(f"sqlite:///{tmp_path / 'fs.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    Session = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    monkeypatch.setattr(pf, "SessionLocal", Session)
    closed: list[int] = []
    monkeypatch.setattr(pf, "close_issue", lambda n: closed.append(n))
    return Session, closed


def _prompts(Session, n=2):
    db = Session()
    biz = Business(name="Sec Biz", email="")
    db.add(biz)
    db.commit()
    rows = [Prompt(business_id=biz.id, prompt_text=f"p{i}", prompt_type=PromptType.EDUCATIONAL, sent_at=datetime.utcnow()) for i in range(n)]
    db.add_all(rows)
    db.commit()
    out = [(r.id, r.uid) for r in rows]
    wkey = biz.weights_key
    db.close()
    return wkey, out


def _issue(number, uid, kind, user="owner", title_token=None):
    token = title_token if title_token is not None else ft.sign(uid, kind)
    return {"issue_number": number, "prompt_id": 0, "feedback": kind, "body": "", "user": user,
            "title": f"[content-intel-feedback] prompt #0 [{kind.upper()}] {token}"}


def _feed(monkeypatch, issues):
    calls = {"n": 0}

    def fake():
        calls["n"] += 1
        return issues

    monkeypatch.setattr(pf, "parse_feedback_issues", fake)
    return calls


# ------------------------------------------------------------------ fail closed
@pytest.mark.parametrize("key,allowed", [("", "owner"), ("k" * 24, ""), ("", ""), (None, None)])
def test_missing_allowlist_or_signing_key_means_nothing_is_read_or_changed(tmp_path, monkeypatch, key, allowed):
    Session, closed = _env(tmp_path, monkeypatch, key=key, allowed=allowed)
    _, [(pid, uid)] = _prompts(Session, 1)
    calls = _feed(monkeypatch, [_issue(1, uid, "bad")])
    monkeypatch.setattr(settings, "feedback_signing_key", key)  # _env's sign() needs the key only at issue-build time
    assert pf.process_feedback() == 0
    assert calls["n"] == 0 and closed == []  # didn't even fetch issues; nothing closed
    assert Session().get(Prompt, pid).feedback is None


def test_defaults_are_closed():
    from src.shared.config import Settings

    s = Settings()
    assert s.feedback_allowed_users is None and s.feedback_signing_key is None
    assert s.llm_keyless_fallback is False  # H-01: keyless public endpoint is opt-in


# ------------------------------------------------------------------ forged / misbound
def test_tampered_signature_is_rejected_and_closed(tmp_path, monkeypatch):
    Session, closed = _env(tmp_path, monkeypatch)
    wkey, [(pid, uid)] = _prompts(Session, 1)
    good = ft.sign(uid, "good")
    forged = good[:-3] + ("000" if not good.endswith("000") else "111")
    _feed(monkeypatch, [_issue(7, uid, "good", title_token=forged)])
    assert pf.process_feedback() == 0
    assert closed == [7] and Session().get(Prompt, pid).feedback is None
    assert BayesianUpdater().get_weights(wkey)["topics"] == {}


def test_a_good_token_cannot_be_replayed_as_bad_or_regenerate(tmp_path, monkeypatch):
    Session, closed = _env(tmp_path, monkeypatch)
    _, [(pid, uid)] = _prompts(Session, 1)
    good_token = ft.sign(uid, "good")
    _feed(monkeypatch, [_issue(1, uid, "bad", title_token=good_token), _issue(2, uid, "regenerate", title_token=good_token)])
    assert pf.process_feedback() == 0
    assert Session().get(Prompt, pid).feedback is None


def test_a_token_for_prompt_a_cannot_touch_prompt_b(tmp_path, monkeypatch):
    Session, closed = _env(tmp_path, monkeypatch)
    _, [(id_a, uid_a), (id_b, uid_b)] = _prompts(Session, 2)
    _feed(monkeypatch, [_issue(1, uid_a, "bad")])
    assert pf.process_feedback() == 1
    db = Session()
    assert db.get(Prompt, id_a).feedback == FeedbackType.BAD and db.get(Prompt, id_b).feedback is None


def test_tokenless_legacy_issues_are_rejected(tmp_path, monkeypatch):
    Session, closed = _env(tmp_path, monkeypatch)
    _, [(pid, _uid)] = _prompts(Session, 1)
    legacy = {"issue_number": 3, "prompt_id": pid, "feedback": "good", "body": "", "user": "owner",
              "title": f"[content-intel-feedback] prompt #{pid} [GOOD 👍]"}
    _feed(monkeypatch, [legacy])
    assert pf.process_feedback() == 0 and closed == [3]
    assert Session().get(Prompt, pid).feedback is None


def test_recycled_integer_ids_cannot_hit_the_wrong_prompt(tmp_path, monkeypatch):
    """agent.db was rebuilt: a NEW prompt now owns id 1. An old link for the vanished prompt
    (valid signature, old uid) must not land on it, whatever number it quotes."""
    Session, closed = _env(tmp_path, monkeypatch)
    _, [(new_id, _new_uid)] = _prompts(Session, 1)
    vanished_uid = "ab" * 16
    issue = _issue(4, vanished_uid, "bad")
    issue["title"] = issue["title"].replace("prompt #0", f"prompt #{new_id}")
    _feed(monkeypatch, [issue])
    assert pf.process_feedback() == 0 and closed == [4]
    assert Session().get(Prompt, new_id).feedback is None


def test_non_allowlisted_author_with_a_valid_token_is_still_rejected(tmp_path, monkeypatch):
    Session, closed = _env(tmp_path, monkeypatch, allowed="owner")
    _, [(pid, uid)] = _prompts(Session, 1)
    _feed(monkeypatch, [_issue(1, uid, "good", user="stranger")])
    assert pf.process_feedback() == 0 and Session().get(Prompt, pid).feedback is None


# ------------------------------------------------------------------ tokens + links
def test_token_roundtrip_and_malformed_input():
    ft_key = "x" * 16
    settings.feedback_signing_key = ft_key
    try:
        uid = "0123456789abcdef0123456789abcdef"
        token = ft.sign(uid, "posted")
        assert token and token.startswith(f"t:{uid}.")
        assert ft.verify(f"title {token}", "posted") == uid
        assert ft.verify(f"title {token}", "good") is None
        assert ft.sign("not-a-uid", "good") is None and ft.sign(uid, "nonsense") is None
        for junk in ("", "t:", "t:zz.yy", None):
            assert ft.verify(junk, "good") is None  # type: ignore[arg-type]
        settings.feedback_signing_key = "different-key-entirely"
        assert ft.verify(f"title {token}", "posted") is None  # a key rotation invalidates old links
    finally:
        settings.feedback_signing_key = None


def test_links_are_only_issued_signed_and_never_expose_a_verifiable_id(monkeypatch):
    monkeypatch.setattr(settings, "github_token", "t")
    monkeypatch.setattr(settings, "github_repo", "o/r")
    uid = "fedcba9876543210fedcba9876543210"
    monkeypatch.setattr(settings, "feedback_signing_key", None)
    assert set(GithubFeedback().create_feedback_links(1, "B", "T", uid).values()) == {"#"}  # no key: no links
    monkeypatch.setattr(settings, "feedback_signing_key", "k" * 20)
    assert set(GithubFeedback().create_feedback_links(1, "B", "T", None).values()) == {"#"}  # no durable id: no links
    links = GithubFeedback().create_feedback_links(1, "B", "T", uid)
    assert all(v.startswith("https://github.com/o/r/issues/new?") for v in links.values())
    tokens = {kind: url.split("%20t%3A")[1].split("&")[0] for kind, url in links.items()}
    assert len(set(tokens.values())) == 4  # a different signature per button


def test_parse_exposes_the_title_for_token_verification(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr(settings, "github_token", "t")
    monkeypatch.setattr(settings, "github_repo", "o/r")
    resp = MagicMock()
    resp.json.return_value = [{"number": 5, "title": "[content-intel-feedback] prompt #2 [BAD 👎] t:" + "a" * 32 + "." + "b" * 20,
                               "body": "", "user": {"login": "Owner"}}]
    monkeypatch.setattr(pf.httpx, "get", lambda *a, **k: resp)
    [issue] = pf.parse_feedback_issues()
    assert issue["title"].endswith("b" * 20) and issue["user"] == "Owner" and issue["feedback"] == "bad"


# ------------------------------------------------------------------ durable identity migration
def test_migration_gives_every_legacy_prompt_a_unique_random_uid(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE prompts (id INTEGER PRIMARY KEY, business_id INTEGER, prompt_text TEXT)"))
        conn.execute(text("INSERT INTO prompts (business_id, prompt_text) VALUES (1,'a'),(1,'b'),(2,'c')"))
    _migrate(eng)
    _migrate(eng)  # idempotent
    with eng.connect() as conn:
        uids = [r[0] for r in conn.execute(text("SELECT uid FROM prompts ORDER BY id"))]
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(prompts)"))}
    assert len(set(uids)) == 3 and all(len(u) == 32 for u in uids)
    assert {"uid", "auto_learned_at"} <= cols
    assert not any(u == f"{i:032x}" for i, u in enumerate(uids, 1))  # random, not derived from the id


def test_new_prompts_get_distinct_uids(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'u.db'}")
    Base.metadata.create_all(bind=eng)
    db = sessionmaker(bind=eng)()
    biz = Business(name="U", email="")
    db.add(biz)
    db.commit()
    rows = [Prompt(business_id=biz.id, prompt_text="x", prompt_type=PromptType.EDUCATIONAL) for _ in range(5)]
    db.add_all(rows)
    db.commit()
    assert len({r.uid for r in rows}) == 5 and Path(".").exists()
