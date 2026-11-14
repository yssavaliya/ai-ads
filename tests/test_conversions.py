"""Conversion tracking: signed lead links, idempotent ledger, credit to the right creative, weekly results."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import pytest

import src.learning.process_feedback as pf
from src.delivery.github_feedback import GithubFeedback
from src.learning import conversions as cv
from src.learning import feedback_tokens as ft
from src.learning.bayesian_updater import BayesianUpdater
from src.shared.config import settings
from tests.test_feedback_security import _env, _feed, _issue, _prompts

NOW = datetime(2026, 11, 14, 12, tzinfo=UTC)
PROMPT = {"uid": "a" * 32, "prompt_id": 7, "business": "B", "angle": "Prompt Blueprint", "hook_type": "framework",
          "format": "reel", "audience_segment": "Founder", "prompt_type": "educational"}


# ------------------------------------------------------------------ ledger
def test_details_are_optional_and_parsed_leniently():
    assert cv.parse_details("") == {"type": "lead", "value": None}
    assert cv.parse_details("Type: booking\nValue: ₹12,500") == {"type": "booking", "value": 12500.0}
    assert cv.parse_details("type: enquiry\nvalue: $40.5")["type"] == "inquiry"
    assert cv.parse_details("Type: lunch\nValue: lots") == {"type": "lead", "value": None}


def test_record_is_idempotent_per_event_and_marks_the_first_lead_per_creative(tmp_path):
    path = tmp_path / "conversions.jsonl"
    first = cv.record(path, "issue-1", PROMPT, "Type: lead\nValue: 100", now=NOW)
    assert first["first_for_prompt"] is True and first["angle"] == "Prompt Blueprint" and first["value"] == 100.0
    assert cv.record(path, "issue-1", PROMPT, now=NOW) is None  # replay: nothing appended
    second = cv.record(path, "issue-2", PROMPT, now=NOW)
    assert second["first_for_prompt"] is False
    assert [e["id"] for e in cv.read_events(path)] == ["issue-1", "issue-2"]


def test_read_events_skips_corrupt_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"id": "a"}\nnope\n{"no_id": 1}\n[]\n{"id": "b"}\n')
    assert [e["id"] for e in cv.read_events(path)] == ["a", "b"] and cv.read_events(tmp_path / "x") == []


def test_summary_and_report_lines():
    events = [
        {"id": "1", "type": "lead", "value": 100, "prompt_uid": "u1", "business": "B", "angle": "Blueprint", "hook_type": "framework", "format": "reel"},
        {"id": "2", "type": "lead", "value": None, "prompt_uid": "u1", "business": "B", "angle": "Blueprint", "hook_type": "framework", "format": "reel"},
        {"id": "3", "type": "booking", "value": 400, "prompt_uid": "u2", "business": "B", "angle": "Cost", "hook_type": "contrarian", "format": "single image"},
        {"id": "4", "type": "lead", "prompt_uid": "u3", "business": "Other", "angle": "X"},
    ]
    summary = cv.summarize(events, "B")
    assert summary["total"] == 3 and summary["creatives"] == 2 and summary["value"] == 500.0
    assert summary["by_type"] == {"lead": 2, "booking": 1} and summary["by_angle"][0] == ("Blueprint", 2)
    text = "\n".join(cv.lines_for_report(summary))
    assert "3 conversion(s) from 2 creative(s), worth 500 in total" in text and "Most conversions by angle: Blueprint (2)" in text
    assert "No leads have been logged" in cv.lines_for_report(cv.summarize([]))[0]


def test_angle_is_read_from_the_published_log(tmp_path):
    (tmp_path / "published_log.json").write_text(json.dumps({"posted": [
        {"prompt_id": 7, "angle": "Old"}, {"prompt_id": 7, "angle": "Prompt Blueprint"}, {"prompt_id": 8, "angle": "Other"}]}))
    assert cv.angle_for_prompt(tmp_path, 7) == "Prompt Blueprint" and cv.angle_for_prompt(tmp_path, 99) == ""
    assert cv.angle_for_prompt(tmp_path / "missing", 7) == ""


# ------------------------------------------------------------------ signed link
def test_the_lead_link_is_signed_for_its_own_button_only(monkeypatch):
    monkeypatch.setattr(settings, "feedback_signing_key", "secret-key")
    links = GithubFeedback(token="t", repo="o/r").create_feedback_links(7, "Biz", "Title", "a" * 32)
    assert set(links) == {"good", "bad", "regenerate", "posted", "lead"}
    query = parse_qs(urlparse(links["lead"]).query)
    title, body = query["title"][0], query["body"][0]
    assert "[LEAD" in title and ft.verify(title, "lead") == "a" * 32 and ft.verify(title, "good") is None
    assert "Type: lead" in body and "Value:" in body


# ------------------------------------------------------------------ feedback job
def _lead_issue(number, uid, body=""):
    issue = _issue(number, uid, "lead")
    issue["body"] = body
    return issue


def test_owner_reported_leads_are_logged_credit_the_creative_once_and_close_the_issue(tmp_path, monkeypatch):
    Session, closed = _env(tmp_path, monkeypatch)
    wkey, [(pid, uid)] = _prompts(Session, 1)
    (tmp_path / "published_log.json").write_text(json.dumps({"posted": [{"prompt_id": pid, "angle": "Prompt Blueprint"}]}))
    _feed(monkeypatch, [_lead_issue(11, uid, "Type: lead\nValue: 250"), _lead_issue(12, uid)])  # two distinct leads, one batch
    assert pf.process_feedback() == 2 and closed == [11, 12]
    events = cv.read_events(cv.ledger_path(tmp_path))
    assert [e["id"] for e in events] == ["issue-11", "issue-12"] and events[0]["value"] == 250.0
    assert events[0]["angle"] == "Prompt Blueprint" and events[0]["business"] == "Sec Biz"
    topics = BayesianUpdater().get_weights(wkey)["topics"]["educational"]
    assert topics["trials"] == 1 and topics["hits"] == 1  # the repeat lead did not inflate the trial count


def test_a_replayed_lead_issue_is_not_double_counted(tmp_path, monkeypatch):
    Session, closed = _env(tmp_path, monkeypatch)
    _, [(pid, uid)] = _prompts(Session, 1)
    _feed(monkeypatch, [_lead_issue(21, uid)])
    assert pf.process_feedback() == 1
    assert pf.process_feedback() == 0  # same issue number seen again (close failed earlier)
    assert len(cv.read_events(cv.ledger_path(tmp_path))) == 1


@pytest.mark.parametrize("user,token", [("stranger", None), ("owner", "t:" + "b" * 32 + "." + "c" * 20)])
def test_lead_issues_from_strangers_or_with_bad_tokens_change_nothing(tmp_path, monkeypatch, user, token):
    Session, closed = _env(tmp_path, monkeypatch)
    _, [(pid, uid)] = _prompts(Session, 1)
    issue = _issue(31, uid, "lead", user=user, title_token=token)
    _feed(monkeypatch, [issue])
    assert pf.process_feedback() == 0 and closed == [31]
    assert cv.read_events(cv.ledger_path(tmp_path)) == []


# ------------------------------------------------------------------ weekly results
def test_weekly_results_block_lists_conversions_and_experiment_verdicts(tmp_path, monkeypatch):
    from src.scheduler.weekly_job import WeeklyPipeline

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    cv.record(cv.ledger_path(tmp_path), "issue-1", PROMPT, "Value: 90", now=NOW)
    (tmp_path / "experiments.json").write_text(json.dumps({"results": {"2026-W46-hook_type": {
        "variable": "hook_type", "status": "winner", "value": "framework", "lift": 0.4, "n_a": 5, "n_b": 5}}}))
    block = WeeklyPipeline._results_block("B")
    assert any("1 conversion(s)" in line for line in block["conversions"])
    assert block["experiments"] == ["hook_type: winner (matured posts 5/5), 'framework' wins (0.4)"]
    assert "No leads" in WeeklyPipeline._results_block("Nobody")["conversions"][0]


def test_weekly_and_business_pages_render_the_new_blocks(tmp_path, monkeypatch):
    from src.delivery.site_publisher import SitePublisher

    monkeypatch.setattr(settings, "site_dir", tmp_path / "site")
    publisher = SitePublisher()
    biz = {"name": "Page Biz"}
    publisher.send_weekly_report(biz, {"results": {"conversions": ["2 conversion(s) from 1 creative(s)"], "experiments": ["hook_type: insufficient (matured posts 1/0)"]}})
    publisher.send_daily_prompts(biz, [{"prompt_id": 1, "title": "T", "prompt_text": "p", "feedback_links": {"lead": "https://github.com/o/r/issues/new?x=1"}}])
    html = "\n".join(p.read_text(encoding="utf-8") for p in (tmp_path / "site").rglob("*.html"))
    assert "Results that matter" in html and "2 conversion(s) from 1 creative(s)" in html
    assert "Got a lead?" in html
