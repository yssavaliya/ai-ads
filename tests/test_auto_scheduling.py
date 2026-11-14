"""Slot planning, unattended draft review, custom scheduling with fallback."""
from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from src.delivery import buffer_state as bs
from src.delivery import draft_review as dr
from src.delivery import slot_planner as sp

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 14, 6, 0, tzinfo=UTC)  # 11:30 IST


# ------------------------------------------------------------------ slot planner
def test_slot_is_the_next_occurrence_of_the_wanted_local_hour():
    slot = sp.plan_slot(NOW, 19, "Asia/Kolkata", [])
    assert slot == datetime(2026, 11, 14, 13, 30, tzinfo=UTC)  # 19:00 IST today


def test_slot_moves_to_tomorrow_when_the_hour_has_passed_or_is_too_close():
    late = datetime(2026, 11, 14, 14, 0, tzinfo=UTC)  # 19:30 IST: 19:00 already gone
    assert sp.plan_slot(late, 19, "Asia/Kolkata", []) == datetime(2026, 11, 15, 13, 30, tzinfo=UTC)
    close = datetime(2026, 11, 14, 12, 30, tzinfo=UTC)  # 18:00 IST: only 60 minutes ahead (< 90 lead)
    assert sp.plan_slot(close, 19, "Asia/Kolkata", []) == datetime(2026, 11, 15, 13, 30, tzinfo=UTC)


def test_slot_keeps_the_minimum_gap_from_queued_posts():
    queued = [datetime(2026, 11, 14, 13, 30, tzinfo=UTC)]
    slot = sp.plan_slot(NOW, 19, "Asia/Kolkata", queued)
    assert slot == datetime(2026, 11, 15, 13, 30, tzinfo=UTC)
    assert slot - queued[0] >= timedelta(hours=sp.MIN_GAP_HOURS)
    # a post at 07:30 IST next day is only 12h away from 19:30 today -> blocked, so it is pushed further
    dense = [datetime(2026, 11, 14, 13, 30, tzinfo=UTC), datetime(2026, 11, 15, 13, 30, tzinfo=UTC)]
    assert sp.plan_slot(NOW, 19, "Asia/Kolkata", dense) == datetime(2026, 11, 16, 13, 30, tzinfo=UTC)


def test_slot_is_none_when_nothing_fits_the_horizon():
    full = [NOW + timedelta(days=d, hours=7) for d in range(0, 12)]
    assert sp.plan_slot(NOW, 19, "Asia/Kolkata", full, horizon_days=3) is None


def test_slot_defaults_and_bad_inputs_never_crash():
    assert sp.plan_slot(NOW, None, "Mars/Phobos", None) is not None
    assert sp.plan_slot(NOW, 99, None, None).astimezone(sp._zone(None)).hour == 23
    naive = sp.plan_slot(datetime(2026, 11, 14, 6, 0), 19, "UTC", [])
    assert naive.tzinfo is not None and naive.hour == 19


def test_slot_keeps_the_wall_clock_hour_across_a_dst_change():
    before = datetime(2026, 10, 30, 6, 0, tzinfo=UTC)  # Europe/London leaves BST on 2026-10-25; New York on 11-01
    slot = sp.plan_slot(before, 18, "America/New_York", [], horizon_days=5)
    assert slot.astimezone(sp._zone("America/New_York")).hour == 18


def test_parse_due_handles_z_naive_and_garbage():
    assert sp.parse_due("2026-11-19T04:51:00.000Z") == datetime(2026, 11, 19, 4, 51, tzinfo=UTC)
    assert sp.parse_due("2026-11-19T04:51:00") == datetime(2026, 11, 19, 4, 51, tzinfo=UTC)
    assert sp.parse_due("nope") is None and sp.parse_due(None) is None
    assert "Nov 2026" in sp.describe(datetime(2026, 11, 15, 13, 30, tzinfo=UTC), "Asia/Kolkata")


# ------------------------------------------------------------------ draft review
def record(**kw):
    base = {"id": "c1", "title": "A title", "score": 82, "threshold": 70, "below_threshold": False,
            "window": {"hour": 19, "source": "scraped post performance"}, "timezone": "Asia/Kolkata",
            "caption": "Save this breakdown for your next product ad.", "buffer_posts": [{"id": "p1"}]}
    base.update(kw)
    return base


def draft(text="Save this breakdown for your next product ad."):
    return {"id": "p1", "text": text}


def test_index_records_maps_buffer_ids_to_creatives():
    idx = dr.index_records({"posted": [record(), {"id": "x", "buffer_posts": [{"id": "p2"}]}, "junk", {"buffer_posts": None}]})
    assert idx["p1"]["id"] == "c1" and set(idx) == {"p1", "p2"}


def test_a_good_draft_is_scheduled_at_its_window_with_the_basis_stated():
    d = dr.review_draft(draft(), record(), [], [], NOW)
    assert d.action == "schedule" and d.hour == 19 and d.due_at == datetime(2026, 11, 14, 13, 30, tzinfo=UTC)
    assert "scraped post performance" in d.reason


def test_experiment_arm_hour_overrides_the_measured_window():
    d = dr.review_draft(draft(), record(schedule_hour=8, experiment={"variable": "post_block", "arm": "B"}), [], [], NOW)
    assert d.action == "schedule" and d.hour == 8 and "experiment arm B" in d.reason


@pytest.mark.parametrize("rec,needle", [
    (None, "not created by this pipeline"),
    (record(below_threshold=True, score=68), "below its quality bar"),
    (record(score=None), "never scored"),
    (record(score=60), "under the quality bar"),
])
def test_drafts_that_fail_the_quality_gates_are_held_for_a_human(rec, needle):
    d = dr.review_draft(draft(), rec, [], [], NOW)
    assert d.action == "hold" and needle in d.reason and d.due_at is None


def test_unverified_claims_hold_the_draft():
    d = dr.review_draft(draft("We grew sales 250% in 48 hours"), record(), [], [], NOW, grounding="")
    assert d.action == "hold" and "unverified claims" in d.reason
    ok = dr.review_draft(draft("We grew sales 250% in 48 hours"), record(), [], [], NOW, grounding="250% 48 hours")
    assert ok.action == "schedule"


def test_duplicates_of_queued_posts_are_held():
    d = dr.review_draft(draft(), record(), ["Save this breakdown for your next product ad."], [], NOW)
    assert d.action == "hold" and "near-duplicate" in d.reason


def test_no_free_slot_holds_instead_of_publishing_at_a_bad_time():
    taken = [NOW + timedelta(days=x, hours=7.5) for x in range(0, 12)]
    d = dr.review_draft(draft(), record(), [], taken, NOW)
    assert d.action == "hold" and "no free publish slot" in d.reason


def test_min_score_override_applies():
    assert dr.review_draft(draft(), record(score=75), [], [], NOW, min_score=80).action == "hold"
    assert dr.review_draft(draft(), record(score=75), [], [], NOW, min_score=70).action == "schedule"


# ------------------------------------------------------------------ buffer schema probe / custom schedule
def _fake_graphql(monkeypatch, responder):
    monkeypatch.setattr(bs, "_graphql", lambda client, token, query, variables=None: responder(query, variables or {}))


def test_probe_reads_the_mode_enum(monkeypatch):
    def responder(query, variables):
        if "EditPostMode" in query:
            return {"__type": {"inputFields": [{"name": "mode", "type": {"name": None, "kind": "NON_NULL", "ofType": {"name": "ShareMode"}}}]}}
        return {"__type": {"enumValues": [{"name": "addToQueue"}, {"name": "customScheduled"}]}}

    _fake_graphql(monkeypatch, responder)
    assert bs.supports_custom_schedule("t", client=httpx.Client()) is True

    _fake_graphql(monkeypatch, lambda q, v: {"__type": {"inputFields": [{"name": "mode", "type": {"name": "ShareMode"}}]}}
                  if "EditPostMode" in q else {"__type": {"enumValues": [{"name": "addToQueue"}]}})
    assert bs.supports_custom_schedule("t", client=httpx.Client()) is False


def test_probe_is_unknown_not_false_when_introspection_fails(monkeypatch):
    def boom(q, v):
        raise RuntimeError("introspection disabled")

    _fake_graphql(monkeypatch, boom)
    assert bs.supports_custom_schedule("t", client=httpx.Client()) is None
    _fake_graphql(monkeypatch, lambda q, v: {"__type": None})
    assert bs.supports_custom_schedule("t", client=httpx.Client()) is None


def test_schedule_draft_sends_custom_mode_and_utc_due_time(monkeypatch):
    seen = {}

    def responder(query, variables):
        seen.update(variables)
        return {"editPost": {"__typename": "PostActionSuccess", "post": {"id": "p1", "status": "scheduled", "dueAt": "x"}}}

    _fake_graphql(monkeypatch, responder)
    post = bs.schedule_draft("t", "p1", datetime(2026, 11, 15, 13, 30, tzinfo=UTC), client=httpx.Client())
    assert post["status"] == "scheduled"
    assert seen["input"] == {"postId": "p1", "saveToDraft": False, "mode": "customScheduled",
                             "schedulingType": "automatic", "dueAt": "2026-11-15T13:30:00.000Z"}
    _fake_graphql(monkeypatch, lambda q, v: {"editPost": {"__typename": "InvalidInputError", "message": "bad mode"}})
    with pytest.raises(RuntimeError, match="bad mode"):
        bs.schedule_draft("t", "p1", NOW, client=httpx.Client())


def test_queue_state_exposes_waiting_due_times():
    queue = bs.QueueState([
        {"status": "scheduled", "dueAt": "2026-11-19T04:51:00.000Z"},
        {"status": "draft", "dueAt": "2026-11-20T04:51:00.000Z"},
        {"status": "sent", "dueAt": "2026-11-01T04:51:00.000Z"},
        {"status": "scheduled", "dueAt": None},
    ])
    assert queue.due_times() == [datetime(2026, 11, 19, 4, 51, tzinfo=UTC)]


# ------------------------------------------------------------------ publisher
def test_publisher_uses_custom_mode_only_for_real_scheduling(monkeypatch):
    from src.delivery.buffer_publisher import BufferPublisher

    sent = []

    class Resp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"createPost": {"__typename": "PostActionSuccess", "post": {"id": "n", "status": "scheduled"}}}}

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers, json):
            sent.append(json["variables"]["input"])
            return Resp()

    monkeypatch.setattr(httpx, "Client", Client)
    pub = BufferPublisher(access_token="t", profile_ids=["c"], draft_mode=False, now=False)
    due = datetime(2026, 11, 15, 13, 30, tzinfo=UTC)
    pub.schedule_post("hello", image_url="https://x/y.jpg", due_at=due)
    pub.schedule_post("hello", image_url="https://x/y.jpg", draft=True, due_at=due)
    pub.schedule_post("hello", image_url="https://x/y.jpg", now=True, due_at=due)
    pub.schedule_post("hello", image_url="https://x/y.jpg")
    assert sent[0]["mode"] == "customScheduled" and sent[0]["dueAt"] == "2026-11-15T13:30:00.000Z"
    assert sent[1]["mode"] == "addToQueue" and "dueAt" not in sent[1]
    assert sent[2]["mode"] == "shareNow" and "dueAt" not in sent[2]
    assert sent[3]["mode"] == "addToQueue"


# ------------------------------------------------------------------ publish_buffer enrichment + promote script
def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_t", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_log_keeps_what_unattended_scheduling_needs(tmp_path):
    pb = _script("publish_buffer")
    pb.PUBLISHED_LOG = tmp_path / "published_log.json"
    entry = {"id": "e1", "title": "T", "score": 81, "threshold": 70, "below_threshold": False,
             "post_time_evidence": {"hour": 19, "source": "scraped post performance"}, "_timezone": "Asia/Kolkata",
             "experiment": {"variable": "post_block", "arm": "A"}, "strategy": {"angle": "Blueprint"}}
    pb.record_posted(entry, [{"id": "b1", "status": "draft"}], "2026-11-15T13:30:00+00:00")
    rec = json.loads(pb.PUBLISHED_LOG.read_text())["posted"][0]
    assert rec["score"] == 81 and rec["window"]["hour"] == 19 and rec["timezone"] == "Asia/Kolkata"
    assert rec["experiment"]["arm"] == "A" and rec["angle"] == "Blueprint" and rec["scheduled_for"].startswith("2026-11-15")


def test_choose_due_prefers_the_experiment_hour_then_the_measured_window():
    pb = _script("publish_buffer")
    base = {"post_time_evidence": {"hour": 19}, "_timezone": "Asia/Kolkata"}
    assert pb.choose_due(base, [], NOW) == datetime(2026, 11, 14, 13, 30, tzinfo=UTC)
    assert pb.choose_due({**base, "schedule_hour": 9}, [], NOW) == datetime(2026, 11, 15, 3, 30, tzinfo=UTC)
    assert pb.choose_due({}, [], NOW) is not None  # default evening slot


def _run_promote(monkeypatch, tmp_path, drafts, record_map, queue_posts=(), custom=True, dry=False, schedule_fails=False):
    promote = _script("promote_drafts")
    log = tmp_path / "data"
    log.mkdir(exist_ok=True)
    (log / "published_log.json").write_text(json.dumps({"posted": record_map}))
    monkeypatch.setattr(promote, "ROOT", tmp_path)
    monkeypatch.setenv("BUFFER_ACCESS_TOKEN", "t")
    monkeypatch.setenv("BUFFER_PROFILE_IDS", "chan")
    monkeypatch.setenv("DRY_RUN", "true" if dry else "false")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    calls = {"schedule": [], "promote": [], "delete": []}
    monkeypatch.setattr(promote, "supports_custom_schedule", lambda token: custom)
    monkeypatch.setattr(promote, "fetch_queue_state", lambda token, ch: bs.QueueState(list(queue_posts)))
    monkeypatch.setattr(promote, "fetch_drafts", lambda token, ch: drafts)
    monkeypatch.setattr(promote, "check_asset_url", lambda url: True)
    monkeypatch.setattr(promote, "delete_post", lambda token, pid: calls["delete"].append(pid))

    def fake_schedule(token, pid, due):
        if schedule_fails:
            raise RuntimeError("Buffer rejected customScheduled")
        calls["schedule"].append((pid, due))
        return {"dueAt": due.isoformat()}

    monkeypatch.setattr(promote, "schedule_draft", fake_schedule)
    monkeypatch.setattr(promote, "promote_draft", lambda token, pid: calls["promote"].append(pid) or {"dueAt": "auto"})
    promote.main()
    return calls, (tmp_path / "summary.md").read_text(encoding="utf-8")


def old_draft(pid, text="Save this breakdown for your next product ad.", hours=30):
    created = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    return {"id": pid, "text": text, "createdAt": created, "assets": []}


def test_promote_schedules_reviewed_drafts_holds_the_rest_and_respects_the_window(monkeypatch, tmp_path):
    drafts = [old_draft("p1"), old_draft("p2", "Another idea entirely about lighting"), old_draft("young", hours=3),
              old_draft("ancient", hours=400), old_draft("stranger", "Not ours")]
    records = [record(), record(id="c2", score=55, caption="Another idea entirely about lighting", buffer_posts=[{"id": "p2"}])]
    calls, summary = _run_promote(monkeypatch, tmp_path, drafts, records)
    assert [pid for pid, _ in calls["schedule"]] == ["p1"]
    assert calls["delete"] == ["ancient"] and calls["promote"] == []
    assert "scheduled" in summary and "held: score 55 is under the quality bar 70" in summary
    assert "waiting for human review" in summary and "not created by this pipeline" in summary


def test_promote_falls_back_to_the_automatic_queue_when_custom_scheduling_fails(monkeypatch, tmp_path):
    calls, summary = _run_promote(monkeypatch, tmp_path, [old_draft("p1")], [record()], schedule_fails=True)
    assert calls["promote"] == ["p1"] and "next automatic slot" in summary


def test_promote_spaces_multiple_approved_drafts_and_honours_queue_capacity(monkeypatch, tmp_path):
    drafts = [old_draft("p1"), old_draft("p2", "Second distinct caption about lenses and light"),
              old_draft("p3", "Third distinct caption about tripods and rigs")]
    records = [record(), record(id="c2", caption="x", buffer_posts=[{"id": "p2"}]), record(id="c3", caption="y", buffer_posts=[{"id": "p3"}])]
    calls, summary = _run_promote(monkeypatch, tmp_path, drafts, records)
    times = [due for _, due in calls["schedule"]]
    assert len(times) == 3 and all(b - a >= timedelta(hours=sp.MIN_GAP_HOURS) for a, b in zip(times, times[1:]))
    drafts.append(old_draft("p4", "Fourth distinct caption about gels and flags"))
    records.append(record(id="c4", caption="z", buffer_posts=[{"id": "p4"}]))
    calls, summary = _run_promote(monkeypatch, tmp_path, drafts, records)
    assert len(calls["schedule"]) == 3 and "queue is full" in summary  # BUFFER_QUEUE_LIMIT default 3


def test_promote_dry_run_changes_nothing(monkeypatch, tmp_path):
    calls, summary = _run_promote(monkeypatch, tmp_path, [old_draft("p1"), old_draft("ancient", hours=400)], [record()], dry=True)
    assert calls == {"schedule": [], "promote": [], "delete": []}
    assert "would schedule" in summary


def test_promote_does_not_double_book_an_occupied_slot(monkeypatch, tmp_path):
    first = sp.plan_slot(datetime.now(UTC), 19, "Asia/Kolkata", [])
    queue = [{"status": "scheduled", "dueAt": first.isoformat(), "text": "something else"}]
    calls, _ = _run_promote(monkeypatch, tmp_path, [old_draft("p1")], [record()], queue_posts=queue)
    (_, due), = calls["schedule"]
    assert abs(due - first) >= timedelta(hours=sp.MIN_GAP_HOURS)


# ------------------------------------------------------------------ schema-adaptive reads (Asset.url vanished)
def test_asset_selection_uses_only_fields_the_live_schema_has(monkeypatch):
    bs._ASSET_SELECTION_CACHE.clear()
    _fake_graphql(monkeypatch, lambda q, v: {"__type": {"fields": [{"name": "id"}, {"name": "source"}, {"name": "mimeType"}]}})
    assert bs.asset_selection(httpx.Client(), "tok1") == "source mimeType"
    # introspection failing -> the always-valid selection, and it is cached per token
    bs._ASSET_SELECTION_CACHE.clear()
    _fake_graphql(monkeypatch, lambda q, v: (_ for _ in ()).throw(RuntimeError("off")))
    assert bs.asset_selection(httpx.Client(), "tok2") == "__typename"
    _fake_graphql(monkeypatch, lambda q, v: {"__type": None})
    bs._ASSET_SELECTION_CACHE.clear()
    assert bs.asset_selection(httpx.Client(), "tok3") == "__typename"


def test_posts_query_has_no_asset_url_unless_selected():
    q = bs.posts_query("source")
    assert "assets { source }" in q and "url" not in q and "dueAt sentAt" in q
    assert "dueAt" not in bs.posts_query("__typename", include_times=False)


def test_asset_urls_accepts_url_or_source_and_ignores_junk():
    post = {"assets": [{"url": "https://a/x.jpg"}, {"source": "https://b/y.mp4"}, {"source": "not-a-url"}, "junk", {}]}
    assert bs.asset_urls(post) == ["https://a/x.jpg", "https://b/y.mp4"] and bs.asset_urls({}) == []


def test_fetch_drafts_raises_instead_of_reading_a_schema_error_as_no_drafts(monkeypatch):
    bs._ASSET_SELECTION_CACHE.clear()

    def responder(query, variables):
        if "GetOrganizations" in query:
            return {"account": {"organizations": [{"id": "o1"}]}}
        if "AssetFields" in query:
            return {"__type": {"fields": [{"name": "url"}]}}
        raise RuntimeError('Cannot query field "url" on type "Asset"')

    _fake_graphql(monkeypatch, responder)
    with pytest.raises(RuntimeError, match="Could not read Buffer drafts"):
        bs.fetch_drafts("tok", "chan", client=httpx.Client())
    _fake_graphql(monkeypatch, lambda q, v: {"account": {"organizations": [{"id": "o1"}]}} if "GetOrganizations" in q
                  else {"__type": {"fields": []}} if "AssetFields" in q else {"posts": {"edges": [{"node": {"id": "d1"}}]}})
    assert bs.fetch_drafts("tok2", "chan", client=httpx.Client()) == [{"id": "d1"}]


def test_generated_posts_query_is_syntactically_balanced():
    """Real run: 'Syntax Error: Expected Name, found <EOF>' (one closing brace was missing)."""
    for selection, times in (("source thumbnail mimeType type", True), ("__typename", False)):
        q = bs.posts_query(selection, include_times=times)
        assert q.count("{") == q.count("}"), q
        assert q.rstrip().endswith("}") and "assets { " + selection + " }" in q


def test_a_draft_is_never_compared_with_itself_when_reviewed():
    """Real dry run: every fresh draft was held as a 'near-duplicate of a post already in Buffer' (itself)."""
    queue = bs.QueueState([
        {"id": "d1", "status": "draft", "text": "Save this breakdown for your next product ad."},
        {"id": "s1", "status": "scheduled", "text": "Something else entirely", "dueAt": "2026-11-19T04:51:00.000Z"},
    ])
    assert queue.texts() == ["Save this breakdown for your next product ad.", "Something else entirely"]
    assert queue.texts(include_drafts=False) == ["Something else entirely"]
    d = dr.review_draft(draft(), record(), queue.texts(include_drafts=False), [], NOW)
    assert d.action == "schedule"


# ------------------------------------------------------------------ delete (live run found it always failed)
def test_delete_mutation_is_valid_for_buffers_payload_type():
    """Real run: 'Fragment cannot be spread here as objects of type DeletePostPayload can never be PostActionSuccess'."""
    q = bs.DELETE_POST_MUTATION
    assert "PostActionSuccess" not in q and "__typename" in q and q.count("{") == q.count("}")


def test_delete_post_reads_success_from_the_result_type_name(monkeypatch):
    for typename, expected in (("DeletePostSuccess", True), ("VoidMutationSuccess", True), ("UnauthorizedError", False),
                               ("InvalidInputError", False), ("", False)):
        _fake_graphql(monkeypatch, lambda q, v, t=typename: {"deletePost": {"__typename": t}})
        assert bs.delete_post("tok", "p1", client=httpx.Client()) is expected, typename
    _fake_graphql(monkeypatch, lambda q, v: {})
    assert bs.delete_post("tok", "p1", client=httpx.Client()) is False


def test_promote_does_not_report_a_refused_delete_as_deleted(monkeypatch, tmp_path):
    promote = _script("promote_drafts")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "published_log.json").write_text(json.dumps({"posted": []}))
    monkeypatch.setattr(promote, "ROOT", tmp_path)
    for k, v in (("BUFFER_ACCESS_TOKEN", "t"), ("BUFFER_PROFILE_IDS", "c"), ("DRY_RUN", "false"),
                 ("DRAFT_MAX_AGE_HOURS", "1"), ("GITHUB_STEP_SUMMARY", str(tmp_path / "s.md"))):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(promote, "supports_custom_schedule", lambda t: True)
    monkeypatch.setattr(promote, "fetch_queue_state", lambda t, c: bs.QueueState([]))
    monkeypatch.setattr(promote, "fetch_drafts", lambda t, c: [old_draft("d1", hours=30)])
    monkeypatch.setattr(promote, "check_asset_url", lambda u: True)
    calls = []
    monkeypatch.setattr(promote, "delete_post", lambda t, pid: calls.append(pid) or False)
    promote.main()
    summary = (tmp_path / "s.md").read_text(encoding="utf-8")
    assert calls == ["d1"] and "delete refused by Buffer" in summary and "🗑️ deleted" not in summary
