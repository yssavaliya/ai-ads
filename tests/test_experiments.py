"""Designed experiments: weekly rotation, balanced assignment, honest conclusions."""
from __future__ import annotations

import json
from datetime import date

from src.learning import experiments as ex

MON = date(2026, 11, 9)   # ISO week 46
NEXT = date(2026, 11, 16)  # ISO week 47


def prompt(hook, fmt="single image", score=80, **kw):
    return {"title": f"{hook}-{fmt}", "hook_type": hook, "format": fmt, "virality_score": score, **kw}


def rec(exp_id=None, arm=None, post="p", **kw):
    r = {"business_name": "B", "buffer_posts": [{"id": post}], **kw}
    if exp_id:
        r["experiment"] = {"id": exp_id, "arm": arm}
    return r


def test_variable_rotates_weekly_and_a_new_week_archives_the_old_experiment():
    assert ex.variable_for_week(MON) != ex.variable_for_week(NEXT)
    state: dict = {}
    first = ex.ensure_active(state, MON, [], 19, "framework")
    assert ex.ensure_active(state, date(2026, 11, 11), [], 19, "framework") is first  # same week: same experiment
    second = ex.ensure_active(state, NEXT, [], 19, "framework")
    assert second["id"] != first["id"] and state["history"][-1]["id"] == first["id"]


def test_hook_arms_are_the_proven_hook_versus_the_least_tried_one():
    records = [{"hook_type": "contrarian"}] * 3 + [{"hook_type": "framework"}] * 2
    arms = ex.arms_for("hook_type", records, 19, "framework")
    assert arms["A"]["value"] == "framework" and arms["B"]["value"] in {"roi_proof", "curiosity_gap", "before_after"}
    assert ex.arms_for("hook_type", [], 19, None)["A"]["value"] == "framework"  # no evidence -> default control


def test_timing_arms_never_share_a_three_hour_block():
    arms = ex.arms_for("post_block", [], 19, None)
    assert arms["A"]["value"] == 19 and arms["B"]["value"] // 3 != 19 // 3
    for control in range(24):
        assert ex.arms_for("post_block", [], control, None)["B"]["value"] // 3 != control // 3


def test_format_arms_differ():
    arms = ex.arms_for("format", [{"format": "single image"}] * 3, None, None)
    assert arms["A"]["value"] == "single image" and arms["B"]["value"] == "reel"


def exp(variable="hook_type", a="framework", b="contrarian"):
    return {"id": "2026-W46-" + variable, "week": "2026-W46", "variable": variable,
            "arms": {"A": {"value": a}, "B": {"value": b}}}


def test_assignment_balances_arms_using_what_was_really_published():
    e = exp()
    prompts = [prompt("framework", score=75), prompt("contrarian", score=90), prompt("framework", score=82)]
    tag = ex.assign(e, prompts, [], 70)
    assert tag["arm"] == "A" and tag["value"] == "framework"
    assert [p for p in prompts if p.get("experiment")][0]["virality_score"] == 82  # best matching creative
    records = [rec(e["id"], "A", "p1")]
    assert ex.assign(e, [prompt("framework"), prompt("contrarian")], records, 70)["arm"] == "B"
    assert ex.assign(e, [prompt("framework"), prompt("contrarian")], records + [rec(e["id"], "B", "p2")], 70)["arm"] == "A"


def test_an_experiment_never_promotes_a_weak_or_flagged_creative():
    e = exp()
    assert ex.assign(e, [prompt("framework", score=60)], [], 70) is None
    assert ex.assign(e, [prompt("framework", score=90, below_threshold=True)], [], 70) is None
    assert ex.assign(e, [prompt("framework", score=None)], [], 70) is None
    assert ex.assign(e, [], [], 70) is None


def test_if_the_preferred_arm_has_no_matching_creative_the_other_arm_is_used():
    e = exp()
    assert ex.assign(e, [prompt("contrarian")], [], 70)["arm"] == "B"
    assert ex.assign(e, [prompt("before_after")], [], 70) is None  # neither arm available today


def test_timing_experiment_sets_the_schedule_hour_only_for_the_challenger():
    e = exp("post_block", 19, 9)
    prompts = [prompt("framework", score=85), prompt("contrarian", score=75)]
    assert ex.assign(e, prompts, [], 70)["arm"] == "A" and "schedule_hour" not in prompts[0]
    prompts = [prompt("framework", score=85)]
    tag = ex.assign(e, prompts, [rec(e["id"], "A")], 70)
    assert tag["arm"] == "B" and prompts[0]["schedule_hour"] == 9


def test_welch_and_conclusion_rules():
    assert ex.welch_t([1.0], [2.0, 3.0]) is None
    assert ex.welch_t([1, 1, 1], [1, 1, 1]) is None
    e = exp()
    assert ex.conclude(e, {"A": [0.1] * 3, "B": [0.2] * 3})["status"] == "insufficient"
    noisy = ex.conclude(e, {"A": [0.10, 0.30, 0.05, 0.25], "B": [0.12, 0.28, 0.07, 0.22]})
    assert noisy["status"] == "no_difference"
    win = ex.conclude(e, {"A": [0.20, 0.22, 0.21, 0.23, 0.2], "B": [0.10, 0.11, 0.09, 0.12, 0.1]})
    assert win["status"] == "winner" and win["winner"] == "A" and win["value"] == "framework" and win["lift"] > 0.9
    lose = ex.conclude(e, {"B": [0.20, 0.22, 0.21, 0.23, 0.2], "A": [0.10, 0.11, 0.09, 0.12, 0.1]})
    assert lose["winner"] == "B" and lose["value"] == "contrarian"


def test_evaluate_joins_exact_post_ids_and_ignores_unmatured_posts():
    e = exp()
    state = {"active": e}
    records = [rec(e["id"], "A", f"a{i}") for i in range(4)] + [rec(e["id"], "B", f"b{i}") for i in range(4)]
    events = [{"type": "learned", "buffer_post_id": f"a{i}", "quality_rate": 0.2 + i * 0.01} for i in range(4)]
    events += [{"type": "learned", "buffer_post_id": f"b{i}", "quality_rate": 0.1 + i * 0.01} for i in range(3)]
    events += [{"type": "snapshot", "buffer_post_id": "b3", "quality_rate": 9.9}]  # immature: not a verdict
    changed = ex.evaluate(state, records, events)
    assert changed[0]["status"] == "insufficient" and changed[0]["n_a"] == 4 and changed[0]["n_b"] == 3
    assert ex.evaluate(state, records, events) == []  # idempotent: nothing changed on replay
    events.append({"type": "learned", "buffer_post_id": "b3", "quality_rate": 0.13})
    assert ex.evaluate(state, records, events)[0]["status"] == "winner"


def test_consumers_only_see_conclusive_results():
    state = {"results": {
        "2026-W45-hook_type": {"status": "winner", "variable": "hook_type", "winner": "A", "value": "framework",
                               "arms": {"A": {"value": "framework"}, "B": {"value": "contrarian"}},
                               "lift": 0.4, "n_a": 5, "n_b": 5},
        "2026-W46-post_block": {"status": "insufficient", "variable": "post_block", "value": 9},
        "2026-W47-post_block": {"status": "winner", "variable": "post_block", "winner": "B", "value": 9,
                                "arms": {"A": {"value": 19}, "B": {"value": 9}}, "lift": 0.3, "n_a": 4, "n_b": 4},
    }}
    hints = ex.strategy_hints(state)
    assert len(hints) == 2 and "'framework' beat 'contrarian'" in hints[1] and "+40%" in hints[1]
    assert ex.preferred_hour(state) == 9
    assert ex.preferred_hour({}) is None and ex.strategy_hints({}) == []


def test_state_roundtrip_and_corrupt_files(tmp_path):
    path = tmp_path / "experiments.json"
    assert ex.load_state(path) == {}
    ex.save_state(path, {"active": {"id": "x"}})
    assert ex.load_state(path) == {"active": {"id": "x"}}
    path.write_text("[1, 2]")
    assert ex.load_state(path) == {}
    path.write_text("not json")
    assert ex.load_state(path) == {}


def test_main_writes_verdicts_and_a_summary(tmp_path, monkeypatch, capsys):
    from src.shared.config import settings

    e = exp()
    (tmp_path / "experiments.json").write_text(json.dumps({"active": e}))
    (tmp_path / "published_log.json").write_text(json.dumps({"posted": [rec(e["id"], "A", "a1")]}))
    (tmp_path / "buffer_metrics.jsonl").write_text(
        json.dumps({"type": "learned", "buffer_post_id": "a1", "quality_rate": 0.2}) + "\n"
    )
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    ex.main()
    assert "insufficient" in capsys.readouterr().out
    assert json.loads((tmp_path / "experiments.json").read_text())["results"][e["id"]]["status"] == "insufficient"
    assert "Experiments" in (tmp_path / "summary.md").read_text()


def test_daily_pipeline_assigns_tags_and_survives_failures(tmp_path, monkeypatch):
    from src.scheduler.daily_job import DailyPipeline
    from src.shared.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    pipe = DailyPipeline.__new__(DailyPipeline)
    prompts = [prompt("framework", score=85), prompt("contrarian", score=80), prompt("before_after", score=78)]
    weights = {"hook_types::framework": 0.7, "hook_types::contrarian": 0.5}
    pipe._assign_experiment("B", prompts, weights, 70)
    state = ex.load_state(tmp_path / "experiments.json")
    assert state["active"]["id"].startswith("20") and any(p.get("experiment") for p in prompts)
    monkeypatch.setattr(ex, "ensure_active", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    pipe._assign_experiment("B", prompts, weights, 70)  # must not raise
    monkeypatch.setattr(settings, "experiments_enabled", False)
    assert pipe._assign_experiment("B", [], weights, 70) is None


def test_publish_prefers_the_creative_carrying_the_experiment():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "publish_buffer.py"
    spec = importlib.util.spec_from_file_location("pb_exp", path)
    pb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pb)
    low = {"id": "1", "score": 72, "title": "t1", "experiment": {"id": "e", "arm": "B"}}
    high = {"id": "2", "score": 95, "title": "t2"}
    assert pb.pick_prompt([high, low])["id"] == "1"
    assert pb.pick_prompt([high])["id"] == "2"
