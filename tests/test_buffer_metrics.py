"""Buffer metrics sync: the only path that learns from exact per-post performance (was 13% covered)."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.learning import buffer_metrics as bm
from src.shared.config import settings
from src.shared.utils import load_weights
from src.storage.models import Business, Prompt, PromptType, SessionLocal, init_db

NOW = datetime.now(UTC)


def iso(delta_hours: float) -> str:
    return (NOW - timedelta(hours=delta_hours)).isoformat()


# ------------------------------------------------------------------ pure helpers
def test_parse_date_handles_z_naive_and_garbage():
    assert bm._parse_date("2026-11-01T10:00:00Z") == datetime(2026, 11, 1, 10, tzinfo=UTC)
    assert bm._parse_date("2026-11-01T10:00:00") == datetime(2026, 11, 1, 10, tzinfo=UTC)
    assert bm._parse_date("nope") is None and bm._parse_date(None) is None and bm._parse_date(5) is None


def test_metrics_map_ignores_nulls_bools_and_unnamed():
    raw = [
        {"type": "reach", "value": 100}, {"type": "reactions", "value": 4.5},
        {"type": "saves", "value": None}, {"type": "flag", "value": True}, {"value": 3}, {"type": "x", "value": "7"},
    ]
    assert bm._metrics_map(raw) == {"reach": 100.0, "reactions": 4.5}


def test_score_weights_interactions_by_reach_and_never_coerces_missing_to_zero():
    rate, components, watch = bm._score({"reach": 200.0, "reactions": 10.0, "comments": 5.0})
    assert rate == round((10 * 1 + 5 * 2) / 200, 6)
    assert components == ("comments", "reactions") and watch is None
    assert bm._score({"reactions": 10.0})[0] is None  # no reach
    assert bm._score({"reach": 0.0, "reactions": 1.0})[0] is None
    assert bm._score({"reach": 100.0})[0] is None  # no interaction metrics reported


def test_score_prefers_average_watch_seconds_and_converts_total_minutes():
    assert bm._score({"reach": 10.0, "reactions": 1.0, "averageTimeWatched": 12.5})[2] == 12.5
    assert bm._score({"reach": 10.0, "reactions": 1.0, "totalTimeWatched": 2.0, "views": 4.0})[2] == 30.0


def test_latest_by_post_keeps_newest_snapshot_and_learned_ids():
    events = [
        {"type": "snapshot", "buffer_post_id": "a", "metrics_updated_at": iso(30), "checked_at": iso(30), "q": 1},
        {"type": "snapshot", "buffer_post_id": "a", "metrics_updated_at": iso(5), "checked_at": iso(5), "q": 2},
        {"type": "snapshot", "buffer_post_id": "a", "metrics_updated_at": iso(40), "checked_at": iso(2), "q": 0},
        {"type": "learned", "buffer_post_id": "b", "checked_at": iso(1)},
        {"type": "check", "buffer_post_id": ""},
    ]
    snapshots, learned, checked = bm._latest_by_post(events)
    assert snapshots["a"]["q"] == 2 and learned == {"b"} and set(checked) == {"a", "b"}


def test_read_events_skips_corrupt_lines(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"type": "check"}\nnot json\n\n[1, 2]\n{"type": "error"}\n')
    assert [e["type"] for e in bm._read_events(path)] == ["check", "error"]
    assert bm._read_events(tmp_path / "missing.jsonl") == []


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
        self.is_error = status >= 400
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self, resp):
        self.resp = resp
        self.query = None

    def post(self, url, headers, json):
        self.query = json["query"]
        return self.resp


def test_fetch_post_embeds_the_id_as_an_escaped_string_literal():
    client = _Client(_Resp(200, {"data": {"post": {"id": "x"}}}))
    assert bm._fetch_post(client, "tok", 'ab"cd')["id"] == "x"
    assert '"ab\\"cd"' in client.query


def test_fetch_post_raises_on_graphql_errors_and_http_errors():
    with pytest.raises(RuntimeError, match="Syntax"):
        bm._fetch_post(_Client(_Resp(200, {"errors": [{"message": "Syntax Error"}]})), "t", "1")
    with pytest.raises(RuntimeError, match="HTTP 400"):
        bm._fetch_post(_Client(_Resp(400, {"x": 1})), "t", "1")
    assert bm._fetch_post(_Client(_Resp(200, {"data": {"post": None}})), "t", "1") is None


# ------------------------------------------------------------------ sync flow
@pytest.fixture()
def paths(tmp_path, monkeypatch):
    log, metrics = tmp_path / "published_log.json", tmp_path / "buffer_metrics.jsonl"
    monkeypatch.setattr(bm, "PUBLISHED_LOG", log)
    monkeypatch.setattr(bm, "METRICS_LOG", metrics)
    monkeypatch.setattr(settings, "buffer_access_token", "token")
    return log, metrics


def test_sync_without_token_is_a_noop(monkeypatch, paths):
    monkeypatch.setattr(settings, "buffer_access_token", None)
    monkeypatch.delenv("BUFFER_ACCESS_TOKEN", raising=False)
    assert bm.sync_buffer_metrics() == {"mapped": 0, "refreshed": 0, "learned": 0, "skipped": 0}


def test_sync_without_linked_posts_maps_nothing(paths):
    log, _ = paths
    log.write_text(json.dumps({"posted": [{"id": "x", "buffer_posts": []}]}))
    assert bm.sync_buffer_metrics()["mapped"] == 0


def test_sync_records_errors_and_unsent_checks_without_learning(paths, monkeypatch):
    log, metrics = paths
    log.write_text(json.dumps({"posted": [
        {"id": "c1", "prompt_id": 1, "buffer_posts": [{"id": "p-err"}]},
        {"id": "c2", "prompt_id": 2, "buffer_posts": [{"id": "p-sched"}]},
        {"id": "c3", "prompt_id": 3, "buffer_posts": [{"id": "p-missing"}]},
    ]}))

    def fake_fetch(client, token, post_id):
        if post_id == "p-err":
            raise RuntimeError("boom")
        if post_id == "p-missing":
            return None
        return {"id": post_id, "status": "scheduled", "dueAt": "2026-12-01T00:00:00Z", "sentAt": None,
                "metricsUpdatedAt": None, "metrics": []}

    monkeypatch.setattr(bm, "_fetch_post", fake_fetch)
    result = bm.sync_buffer_metrics()
    assert result == {"mapped": 3, "refreshed": 0, "learned": 0, "skipped": 3}
    kinds = sorted((e["type"], e.get("status") or "") for e in bm._read_events(metrics))
    assert kinds == [("check", "not_found"), ("check", "scheduled"), ("error", "")]


def test_sent_post_with_metrics_is_stored_as_a_snapshot(paths, monkeypatch):
    log, metrics = paths
    log.write_text(json.dumps({"posted": [{"id": "c", "prompt_id": 9, "business_name": "B", "buffer_posts": [{"id": "p"}]}]}))
    monkeypatch.setattr(bm, "_fetch_post", lambda *a: {
        "id": "p", "status": "sent", "sentAt": iso(100), "metricsUpdatedAt": iso(2),
        "metrics": [{"type": "reach", "value": 200}, {"type": "reactions", "value": 10}],
    })
    result = bm.sync_buffer_metrics()
    assert result["refreshed"] == 1 and result["learned"] == 0  # no baseline yet
    snapshot = [e for e in bm._read_events(metrics) if e["type"] == "snapshot"][0]
    assert snapshot["quality_rate"] == 0.05 and snapshot["interaction_components"] == ["reactions"]
    assert snapshot["business_name"] == "B"


def test_recently_checked_posts_are_not_refetched(paths, monkeypatch):
    log, metrics = paths
    log.write_text(json.dumps({"posted": [{"id": "c", "prompt_id": 1, "buffer_posts": [{"id": "p"}]}]}))
    metrics.write_text(json.dumps({"type": "check", "buffer_post_id": "p", "checked_at": iso(1)}) + "\n")
    monkeypatch.setattr(bm, "_fetch_post", lambda *a: pytest.fail("must not refetch within the refresh window"))
    assert bm.sync_buffer_metrics()["skipped"] == 1


def _seed_posts(db, count):
    init_db()
    biz = Business(name="Metrics Biz", email="x@example.com")
    db.add(biz)
    db.flush()
    prompts = []
    for i in range(count):
        p = Prompt(business_id=biz.id, prompt_text=f"prompt {i}", prompt_type=PromptType.STORYTELLING,
                   format="single image", hook_type="before_after", audience_segment="Founder")
        db.add(p)
        prompts.append(p)
    db.commit()
    return biz, prompts


def _write_snapshots(log, metrics, biz, prompts, sent_hours, reactions, updated_hours=10):
    posted, events = [], []
    for i, prompt in enumerate(prompts):
        pid = f"post-{i}"
        posted.append({"id": f"c{i}", "prompt_id": prompt.id, "business_name": biz.name, "buffer_posts": [{"id": pid}]})
        events.append({
            "type": "snapshot", "buffer_post_id": pid, "prompt_id": prompt.id, "business_name": biz.name,
            "sent_at": iso(sent_hours[i]), "metrics_updated_at": iso(updated_hours), "checked_at": iso(updated_hours),
            "metrics": {"reach": 200, "reactions": reactions[i]}, "interaction_components": ["reactions"],
            "quality_rate": reactions[i] / 200, "avg_watch_seconds": None,
        })
    log.write_text(json.dumps({"posted": posted}))
    metrics.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_mature_post_with_enough_baseline_updates_weights_exactly_once(paths, monkeypatch):
    log, metrics = paths
    db = SessionLocal()
    try:
        biz, prompts = _seed_posts(db, 7)
        # six older mature posts (rate 0.05) form the baseline; the newest (rate 0.5) beats it
        _write_snapshots(log, metrics, biz, prompts, [400, 376, 352, 328, 304, 280, 256], [10, 10, 10, 10, 10, 10, 100])
        monkeypatch.setattr(bm, "_fetch_post", lambda *a: pytest.fail("snapshots were checked recently"))

        assert bm.sync_buffer_metrics()["learned"] >= 1
        learned = [e for e in bm._read_events(metrics) if e["type"] == "learned"]
        assert any(e["buffer_post_id"] == "post-6" and e["outcome"] == "good" for e in learned)
        topics = load_weights()[biz.weights_key]["topics"]["storytelling"]
        assert topics["trials"] >= 1

        before = topics["trials"]
        assert bm.sync_buffer_metrics()["learned"] == 0  # replay is idempotent
        assert load_weights()[biz.weights_key]["topics"]["storytelling"]["trials"] == before
    finally:
        db.close()


def test_young_posts_are_never_learned(paths, monkeypatch):
    log, metrics = paths
    db = SessionLocal()
    try:
        biz, prompts = _seed_posts(db, 7)
        _write_snapshots(log, metrics, biz, prompts, [10 + i for i in range(7)], [10] * 7, updated_hours=5)
        monkeypatch.setattr(bm, "_fetch_post", lambda *a: pytest.fail("recent"))
        assert bm.sync_buffer_metrics()["learned"] == 0
    finally:
        db.close()


def test_stale_metrics_are_never_learned(paths, monkeypatch):
    log, metrics = paths
    db = SessionLocal()
    try:
        biz, prompts = _seed_posts(db, 7)
        _write_snapshots(log, metrics, biz, prompts, [400 - 24 * i for i in range(7)], [10] * 7, updated_hours=200)
        # checked 200h ago -> refetch is attempted; make it fail so the stale snapshot stays the only data
        monkeypatch.setattr(bm, "_fetch_post", lambda *a: (_ for _ in ()).throw(RuntimeError("offline")))
        assert bm.sync_buffer_metrics()["learned"] == 0
    finally:
        db.close()


def test_buffer_managed_prompt_ids_only_with_token_and_linked_posts(paths, monkeypatch):
    log, _ = paths
    log.write_text(json.dumps({"posted": [
        {"id": "a", "prompt_id": 5, "buffer_posts": [{"id": "x"}]},
        {"id": "b", "prompt_id": 6, "buffer_posts": []},
        {"id": "c", "prompt_id": "bad", "buffer_posts": [{"id": "y"}]},
    ]}))
    assert bm._buffer_managed_prompt_ids() == {5}
    monkeypatch.setattr(settings, "buffer_access_token", None)
    monkeypatch.delenv("BUFFER_ACCESS_TOKEN", raising=False)
    assert bm._buffer_managed_prompt_ids() == set()
