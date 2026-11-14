"""GitHub feedback links, feedback issue creation and the degraded-run tracker (was 44% covered)."""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import httpx

from src.delivery import github_feedback as gf
from src.learning import feedback_tokens
from src.shared.config import settings

UID = "a" * 32


class _Resp:
    def __init__(self, payload=None):
        self._payload = payload if payload is not None else {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_links_are_inert_placeholders_without_credentials_key_or_uid(monkeypatch):
    monkeypatch.setattr(settings, "feedback_signing_key", "k")
    placeholder = {"good": "#", "bad": "#", "regenerate": "#", "posted": "#"}
    assert gf.GithubFeedback(token="", repo="").create_feedback_links(1, "B", "T", UID) == placeholder
    assert gf.GithubFeedback(token="t", repo="o/r").create_feedback_links(1, "B", "T", None) == placeholder
    monkeypatch.setattr(settings, "feedback_signing_key", None)
    assert gf.GithubFeedback(token="t", repo="o/r").create_feedback_links(1, "B", "T", UID) == placeholder


def test_signed_links_carry_a_token_that_verifies_only_for_its_own_button(monkeypatch):
    monkeypatch.setattr(settings, "feedback_signing_key", "secret-key")
    links = gf.GithubFeedback(token="t", repo="o/r").create_feedback_links(7, "Biz & Co", "A title", UID)
    assert set(links) == {"good", "bad", "regenerate", "posted"}
    for kind, url in links.items():
        parsed = urlparse(url)
        assert parsed.netloc == "github.com" and parsed.path == "/o/r/issues/new"
        title = parse_qs(parsed.query)["title"][0]
        assert title.startswith(gf.FEEDBACK_PREFIX) and "prompt #7" in title
        assert feedback_tokens.verify(title, kind) == UID
        other = "bad" if kind == "good" else "good"
        assert feedback_tokens.verify(title, other) is None  # a GOOD token cannot be turned into BAD
    assert "Instagram post URL" in parse_qs(urlparse(links["posted"]).query)["body"][0]


def test_create_feedback_issue_returns_the_number_and_never_raises(monkeypatch):
    sent = {}
    monkeypatch.setattr(httpx, "post", lambda url, **k: sent.update(url=url, json=k["json"]) or _Resp({"number": 42}))
    fb = gf.GithubFeedback(token="t", repo="o/r")
    assert fb.create_feedback_issue(3, "B", "T", "good") == 42
    assert sent["url"].endswith("/repos/o/r/issues") and "GOOD" in sent["json"]["title"]

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", boom)
    assert fb.create_feedback_issue(3, "B", "T", "bad") is None
    assert gf.GithubFeedback(token="", repo="").create_feedback_issue(3, "B", "T", "bad") is None


def test_degraded_run_creates_a_new_issue_when_none_is_open(monkeypatch):
    created = {}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp([{"number": 5, "title": "unrelated"}]))
    monkeypatch.setattr(httpx, "post", lambda url, **k: created.update(json=k["json"]) or _Resp({"number": 11}))
    number = gf.GithubFeedback(token="t", repo="o/r").report_degraded_run(
        [{"business": "Biz", "stage": "scoring", "error": "x" * 500}]
    )
    assert number == 11
    assert created["json"]["title"].startswith(gf.DEGRADED_TITLE)
    assert "**Biz**" in created["json"]["body"] and "x" * 301 not in created["json"]["body"]  # error is truncated


def test_degraded_run_appends_to_the_open_report_instead_of_spamming(monkeypatch):
    patched = {}
    open_issue = {"number": 9, "title": f"{gf.DEGRADED_TITLE} — 2026-11-01", "body": "earlier report"}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp([open_issue]))
    monkeypatch.setattr(httpx, "patch", lambda url, **k: patched.update(url=url, json=k["json"]) or _Resp())
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must update in place")))
    assert gf.GithubFeedback(token="t", repo="o/r").report_degraded_run([{"lead": "L", "stage": "enrich", "error": "e"}]) == 9
    assert patched["json"]["body"].startswith("earlier report") and "**L**" in patched["json"]["body"]


def test_degraded_run_is_silent_when_disabled_empty_or_the_api_fails(monkeypatch):
    assert gf.GithubFeedback(token="", repo="").report_degraded_run([{"stage": "x"}]) is None
    assert gf.GithubFeedback(token="t", repo="o/r").report_degraded_run([]) is None

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", boom)
    assert gf.GithubFeedback(token="t", repo="o/r").report_degraded_run([{"stage": "x"}]) is None
