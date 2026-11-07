"""Drafts, duplicate and queue pre-flight, posted log, owner-supplied facts, Cloudflare text provider."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from src.delivery.buffer_state import QueueState, fetch_queue_state, find_duplicate
from src.shared import brand_facts
from src.shared.config import settings
from src.shared.llm_providers import configured_providers, get_provider, provider_order

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _publish_module():
    spec = importlib.util.spec_from_file_location("publish_buffer_safeguards", SCRIPTS / "publish_buffer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- brand facts
def test_brand_facts_ignore_instructions_and_empty_fields(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps({"_instructions": "x", "name": "Studio", "offers": [], "voice": "", "proof_points": ["Founded 2024"]}))
    facts = brand_facts.load_brand_facts(path)
    assert facts == {"name": "Studio", "proof_points": ["Founded 2024"]}
    assert brand_facts.has_content(facts)


def test_brand_facts_missing_or_broken_file_is_empty_and_warns(tmp_path, caplog):
    assert brand_facts.load_brand_facts(tmp_path / "nope.json") == {}
    (tmp_path / "bad.json").write_text("{not json")
    assert brand_facts.load_brand_facts(tmp_path / "bad.json") == {}
    with caplog.at_level("WARNING"):
        assert brand_facts.warn_if_empty({"name": "Studio"}) is False
    assert "brand_facts.json" in caplog.text


def test_facts_reach_the_profile_and_ground_the_claims_guard():
    merged = brand_facts.merge_into_profile({"name": "Studio"}, {"proof_points": ["Reviews arrive 40% faster"]})
    assert merged["verified_facts"]["proof_points"] == ["Reviews arrive 40% faster"]
    assert "40% faster" in brand_facts.grounding_text(merged["verified_facts"])
    assert brand_facts.grounding_text({}) == ""
    assert brand_facts.merge_into_profile(None, {}) == {}


def test_a_claim_in_the_owners_facts_is_allowed_at_publish_but_others_are_not():
    pb = _publish_module()
    entry = {"id": "e1", "title": "t", "score": 80, "warnings": [], "hook": "Reviews arrive 40% faster", "caption": "x"}
    assert pb.pick_prompt([entry]) is None
    assert pb.pick_prompt([entry], grounding="Reviews arrive 40% faster")["id"] == "e1"


# ---------------------------------------------------------------- duplicates and queue
def test_duplicate_detection_uses_the_opening_and_overall_similarity():
    existing = ["The world only celebrates the awakening. They never see the silent hours behind it."]
    assert find_duplicate("The world only celebrates the awakening. They never see the silent hours behind it #anime", existing)
    assert find_duplicate("Stop wasting weeks on brand assets—use this 3-step engine.", existing) is None
    assert find_duplicate("", existing) is None


def test_queue_counts_only_waiting_posts_not_drafts_or_sent():
    state = QueueState([{"status": "scheduled"}, {"status": "draft"}, {"status": "sent"}, {"status": "needs_approval"}])
    assert state.scheduled == 2


def test_fetch_queue_state_reads_posts_without_mutating_anything():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["query"])
        if "GetOrganizations" in body["query"]:
            return httpx.Response(200, json={"data": {"account": {"organizations": [{"id": "org1", "name": "o"}]}}})
        return httpx.Response(200, json={"data": {"posts": {"edges": [{"node": {"id": "p1", "status": "scheduled", "text": "hello"}}]}}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    state = fetch_queue_state("tok", "chan", client=client)
    assert state.scheduled == 1 and state.texts() == ["hello"]
    assert all("mutation" not in q.lower() for q in seen)


# ---------------------------------------------------------------- posted log
def test_posted_log_stops_a_creative_being_posted_twice(tmp_path, monkeypatch):
    pb = _publish_module()
    monkeypatch.setattr(pb, "PUBLISHED_LOG", tmp_path / "published_log.json")
    assert pb.posted_ids() == set()
    entry = {"id": "2026-11-01-51", "title": "A", "score": 90, "warnings": [], "hook": "h", "caption": "c"}
    other = {"id": "2026-11-01-52", "title": "B", "score": 70, "warnings": [], "hook": "h", "caption": "c"}
    pb.record_posted(entry)
    assert pb.posted_ids() == {"2026-11-01-51"}
    assert pb.pick_prompt([entry, other], pb.posted_ids())["id"] == "2026-11-01-52"
    assert pb.pick_prompt([entry], pb.posted_ids()) is None


def test_posted_log_is_bounded(tmp_path, monkeypatch):
    pb = _publish_module()
    monkeypatch.setattr(pb, "PUBLISHED_LOG", tmp_path / "published_log.json")
    monkeypatch.setattr(pb, "MAX_LOG_ENTRIES", 3)
    for i in range(5):
        pb.record_posted({"id": f"e{i}", "title": "t"})
    assert pb.posted_ids() == {"e2", "e3", "e4"}


# ---------------------------------------------------------------- Cloudflare text provider
def test_cloudflare_is_a_text_provider_only_when_both_secrets_exist(monkeypatch):
    monkeypatch.setattr(settings, "cloudflare_account_id", None)
    monkeypatch.setattr(settings, "cloudflare_api_token", None)
    assert get_provider("cloudflare") is None
    monkeypatch.setattr(settings, "cloudflare_account_id", "acct")
    monkeypatch.setattr(settings, "cloudflare_api_token", "tok")
    provider = get_provider("cloudflare")
    assert provider.base_url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1"
    assert provider.models == ("@cf/meta/llama-3.3-70b-instruct-fp8-fast",)


def test_default_provider_order_uses_open_source_huggingface_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider_order", "")
    order = provider_order()
    assert order == ["huggingface"]


@pytest.mark.parametrize("name", ["cloudflare"])
def test_configured_providers_lists_cloudflare_when_set(monkeypatch, name):
    monkeypatch.setattr(settings, "llm_provider_order", "cloudflare")
    monkeypatch.setattr(settings, "cloudflare_account_id", "acct")
    monkeypatch.setattr(settings, "cloudflare_api_token", "tok")
    assert [p.name for p in configured_providers()] == [name]
