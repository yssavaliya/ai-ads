"""Does the product actually LEARN?  A repeatable experiment on the real code path:

    scraped engagement -> AutoLearner (outcome vs. the business's own benchmark)
        -> BayesianUpdater (weights.json + ledger) -> ViralityGate selection / forge prompt

Classification under test: online, count-based incremental learning (Beta-Bernoulli posterior per
business over prompt type / format / audience segment), applied as a BOUNDED nudge at generation.
It is not model training, and 'history in the prompt' is retrieval, not learning - the tests below
therefore check that BEHAVIOUR CHANGES, PERSISTS and can be UNDONE, against a no-learning baseline.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.generation.prompt_forge import PromptForge
from src.generation.virality_gate import ViralityGate
from src.learning.auto_learner import MATURITY_HOURS, AutoLearner
from src.learning.bayesian_updater import BayesianUpdater, read_ledger, replay_ledger
from src.learning.evaluation import evaluate
from src.shared.config import settings
from src.storage.models import Base, Business, Engagement, Prompt, PromptType
from tests.test_prompt_integrity import VARIANT, CapturingLLM


@pytest.fixture
def lab(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'lab.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    db = sessionmaker(bind=eng, autoflush=False, autocommit=False)()
    biz = Business(name="Learning Lab Bakery", email="")
    db.add(biz)
    db.commit()
    return db, biz


def _prompt(db, biz, kind, i, *, sent_days_ago=6, ptype=PromptType.EDUCATIONAL, score=80):
    caption = f"{kind} idea number{i} zorb{i}{kind} gleam{i}{kind} quill{i}"
    p = Prompt(business_id=biz.id, prompt_text=f"brief {caption}", caption=caption, prompt_type=ptype, format=kind,
               virality_score=score, audience_segment="home bakers", sent_at=datetime.utcnow() - timedelta(days=sent_days_ago))
    db.add(p)
    db.commit()
    return p, caption


def _post(caption, likes, i, *, age_hours=120, hidden=False):
    return {"post_id": f"post-{i}", "caption": caption, "likes": likes, "comments": 0, "likes_hidden": hidden,
            "url": f"https://www.instagram.com/p/POST{i}/", "posted_at": (datetime.utcnow() - timedelta(hours=age_hours)).isoformat()}


def _experiment(db, biz, n=8):
    """n reel prompts that flopped, n carousel prompts that did well (identical predicted virality)."""
    posts = []
    for i in range(n):
        _, cap = _prompt(db, biz, "reel", i)
        posts.append(_post(cap, 5, i))
        _, cap = _prompt(db, biz, "carousel", i)
        posts.append(_post(cap, 100, 100 + i))
    return posts


class _TwoFormatForge:
    """Same angle, same type, same predicted virality - only the FORMAT differs (reel is offered first)."""

    def generate_variants(self, **kw):
        return [{"prompt_type": "educational", "prompt_text": f"idea as a {fmt}", "format": fmt, "audience_segment": "home bakers"}
                for fmt in ("reel", "carousel")]

    def refine_with_hints(self, best, *a):
        return best


class _EqualScore:
    def score(self, *a, **k):
        return {"virality_score": 80, "virality_label": "HIGH", "virality_probability": 0.8, "key_factors": [], "risks": [],
                "reasoning": "", "improvement_hints": []}


def _chosen_format(weights):
    gate = ViralityGate(threshold=75, max_attempts=1, forge=_TwoFormatForge(), predictor=_EqualScore())
    [prompt] = gate.run({}, {}, {}, ["educational"], weights=weights)
    return prompt["format"]


def _learn(db, biz, posts):
    return AutoLearner().run_for_business(biz.id, {"posts": posts, "follower_count": 1000}, db)


# ================================================================== the experiment
def test_feedback_changes_subsequent_behaviour_persists_and_can_be_rolled_back(lab, capsys):
    db, biz = lab
    key = biz.weights_key

    # 1. BASELINE - nothing learned yet: a tie, so the first variant (reel) wins
    baseline_weights = BayesianUpdater().to_forge_weights(key)
    baseline_choice = _chosen_format(baseline_weights)
    assert baseline_weights.get("formats::reel") is None and baseline_choice == "reel"

    # 2. LEARN from real (mock-scraped) engagement through the production path
    result = _learn(db, biz, _experiment(db, biz))
    assert result["matched"] == 16 and result["learned"] == 16 and result["deferred"] == 0

    # 3. BEHAVIOUR CHANGED: the system now prefers the format that actually performed
    learned_weights = BayesianUpdater().to_forge_weights(key)
    assert learned_weights["formats::carousel"] == pytest.approx(0.9)   # (8 hits + 1) / (8 trials + 2)
    assert learned_weights["formats::reel"] == pytest.approx(0.1)       # (0 hits + 1) / (8 trials + 2)
    learned_choice = _chosen_format(learned_weights)
    assert learned_choice == "carousel" and learned_choice != baseline_choice

    # 4. PERSISTENCE: a brand-new process (fresh objects, state only from disk) behaves the same
    weights_file = json.loads((Path(settings.data_dir) / "weights.json").read_text(encoding="utf-8"))
    assert weights_file[key]["formats"]["carousel"]["hits"] == 8 and weights_file[key]["formats"]["reel"]["hits"] == 0
    restarted = BayesianUpdater()
    assert restarted.to_forge_weights(key) == learned_weights and _chosen_format(restarted.to_forge_weights(key)) == "carousel"
    assert restarted.version() == 16

    # 5. The forge PROMPT carries what was learned too (the second, LLM-side channel of influence)
    llm = CapturingLLM({"prompt-forge": [VARIANT]})
    forge = PromptForge.__new__(PromptForge)
    forge.llm = llm
    forge.generate_variants({}, {}, {}, ["educational"], weights=learned_weights)
    assert "formats::carousel" in llm.prompts["prompt-forge"] and "0.9" in llm.prompts["prompt-forge"]

    # 6. EVALUATION: the monitoring report sees the experiment
    report = evaluate(Path(settings.data_dir) / "engagement_log.jsonl", Path(settings.data_dir) / "learning_ledger.jsonl")
    assert report["judged_posts"] == 16 and report["hit_rate"] == 0.5 and report["sources"] == {"auto": 16}
    assert report["ledger_version"] == 16 and report["trend"] is not None

    # 7. ROLLBACK restores the baseline behaviour exactly (append-only: the rollback is recorded)
    restarted.rollback(0)
    assert BayesianUpdater().to_forge_weights(key) == baseline_weights
    assert _chosen_format(BayesianUpdater().to_forge_weights(key)) == baseline_choice == "reel"
    assert read_ledger(restarted.ledger_path())[-1]["type"] == "rollback" and restarted.version() == 0

    with capsys.disabled():
        print(
            f"\n[self-learning probe] baseline choice={baseline_choice} | after 16 judged posts: choice={learned_choice} "
            f"(carousel weight {learned_weights['formats::carousel']:.2f} vs reel {learned_weights['formats::reel']:.2f}) "
            "| persisted across restart: yes | rolled back to baseline: yes"
        )


def test_control_group_with_no_signal_does_not_change_behaviour(lab):
    """Equal engagement for both formats must leave the choice at the tie-break (no spurious learning)."""
    db, biz = lab
    posts = []
    for i in range(6):
        _, cap = _prompt(db, biz, "reel", i)
        posts.append(_post(cap, 50, i))
        _, cap = _prompt(db, biz, "carousel", i)
        posts.append(_post(cap, 50, 100 + i))
    _learn(db, biz, posts)
    weights = BayesianUpdater().to_forge_weights(biz.weights_key)
    assert abs(weights["formats::reel"] - weights["formats::carousel"]) < 0.01
    assert _chosen_format(weights) == "reel"  # same as the baseline


# ================================================================== quality guards
def test_rescraping_refreshes_engagement_but_never_recounts_the_weights(lab):
    """Audit H-03: counts used to freeze at the first scrape AND be one-shot; now they refresh, once-counted."""
    db, biz = lab
    posts = _experiment(db, biz, n=4)
    _learn(db, biz, posts)
    key = biz.weights_key
    trials_before = BayesianUpdater().get_weights(key)["formats"]["carousel"]["trials"]

    grown = [{**p, "likes": p["likes"] + 40, "comments": 3} for p in posts]  # engagement kept growing
    result = _learn(db, biz, grown)
    assert result["learned"] == 0 and result["refreshed"] == 8
    assert BayesianUpdater().get_weights(key)["formats"]["carousel"]["trials"] == trials_before  # not re-counted
    assert BayesianUpdater().version() == 8
    db.expire_all()
    carousel_likes = sorted(e.likes for e in db.query(Engagement).all())[-1]
    assert carousel_likes == 140  # the STORED engagement did refresh (it used to stay at 100)


def test_a_post_is_not_judged_until_it_has_matured(lab):
    db, biz = lab
    _, cap = _prompt(db, biz, "reel", 1)
    fresh = _post(cap, 2, 1, age_hours=MATURITY_HOURS - 20)
    first = _learn(db, biz, [fresh, _post("filler zzz yyy", 90, 9), _post("filler aaa bbb", 80, 8), _post("filler ccc ddd", 70, 7)])
    assert first["learned"] == 0 and first["deferred"] == 1
    assert BayesianUpdater().version() == 0
    db.expire_all()
    assert db.query(Prompt).one().feedback is None  # stays eligible
    assert db.query(Engagement).count() == 1  # but its engagement is already being tracked

    aged = _post(cap, 2, 1, age_hours=MATURITY_HOURS + 5)
    second = _learn(db, biz, [aged, _post("filler zzz yyy", 90, 9), _post("filler aaa bbb", 80, 8), _post("filler ccc ddd", 70, 7)])
    assert second["learned"] == 1 and BayesianUpdater().version() == 1


def test_hidden_like_counts_are_missing_data_not_a_flop(lab):
    db, biz = lab
    _, cap = _prompt(db, biz, "reel", 1)
    result = _learn(db, biz, [_post(cap, 0, 1, hidden=True), _post("other a b c", 60, 2), _post("other d e f", 70, 3), _post("other g h i", 80, 4)])
    assert result["learned"] == 0 and result["unusable"] == 1 and BayesianUpdater().version() == 0


def test_duplicate_or_unmatched_posts_teach_nothing(lab):
    db, biz = lab
    p, cap = _prompt(db, biz, "reel", 1)
    posts = [_post(cap, 100, 1), _post(cap, 100, 1), _post("a totally unrelated caption about sports", 500, 2)]  # duplicate + unmatched
    assert _learn(db, biz, posts)["learned"] == 1
    assert BayesianUpdater().version() == 1


def test_prompts_judged_by_a_human_are_not_double_counted_by_the_learner(lab):
    from src.storage.models import FeedbackType

    db, biz = lab
    p, cap = _prompt(db, biz, "reel", 1)
    p.feedback = FeedbackType.BAD  # a human already judged it (and the weights moved once, elsewhere)
    db.commit()
    result = _learn(db, biz, [_post(cap, 100, 1), _post("x y z q", 10, 2), _post("w v u t", 20, 3)])
    assert result["matched"] == 0 and BayesianUpdater().version() == 0


def test_learned_influence_is_bounded_so_it_cannot_overrule_the_predictor():
    """Even a perfect 1000-0 record moves a candidate by at most 6 points, never past a real quality gap."""
    updater_weights = {"formats::reel": (1000 + 1) / (1000 + 2), "topics::educational": 1000 / 1002}
    boost = ViralityGate._weight_boost(updater_weights, {"format": "reel", "prompt_type": "educational"})
    assert 0 < boost <= 6.0
    worst = ViralityGate._weight_boost({"formats::reel": 1 / 1002}, {"format": "reel"})
    assert -6.0 <= worst < 0


def test_ledger_replay_reproduces_the_live_weights_exactly(lab):
    db, biz = lab
    _learn(db, biz, _experiment(db, biz, n=5))
    updater = BayesianUpdater()
    live = json.loads((Path(settings.data_dir) / "weights.json").read_text(encoding="utf-8"))
    assert replay_ledger(read_ledger(updater.ledger_path())) == live
    partial = replay_ledger(read_ledger(updater.ledger_path()), upto=4)
    assert sum(e["trials"] for e in partial[biz.weights_key]["formats"].values()) == 4


def test_a_pre_existing_weights_file_becomes_the_ledger_baseline(tmp_path):
    """Weights that predate the ledger must not be lost by a replay/rollback."""
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"old-biz": {"topics": {"educational": {"weight": 0.75, "trials": 2, "hits": 2}}, "formats": {}, "audience_segments": {}}}))
    updater = BayesianUpdater(str(path))
    updater.record("old-biz", "bad", prompt_type="educational", source="manual")
    assert updater.version() == 1
    assert replay_ledger(read_ledger(updater.ledger_path()), upto=0)["old-biz"]["topics"]["educational"]["trials"] == 2  # baseline kept
    updater.rollback(0)
    assert json.loads(path.read_text())["old-biz"]["topics"]["educational"]["trials"] == 2


def test_rollback_rejects_impossible_versions(tmp_path):
    updater = BayesianUpdater(str(tmp_path / "w.json"))
    updater.record("b", "good", prompt_type="educational")
    for bad in (-1, 2):
        with pytest.raises(ValueError):
            updater.rollback(bad)


def test_manual_and_auto_updates_are_labelled_in_the_ledger(tmp_path):
    updater = BayesianUpdater(str(tmp_path / "w.json"))
    updater.record("b", "good", prompt_type="educational", source="manual")
    updater.record("b", "bad", prompt_type="educational", source="auto")
    assert [e["source"] for e in read_ledger(updater.ledger_path())] == ["manual", "auto"]


def test_evaluation_refuses_to_claim_a_trend_from_too_little_data(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps({"post_id": str(i), "outcome": "good", "prompt_type": "educational", "virality_score": 80}) for i in range(3)))
    report = evaluate(log, tmp_path / "none.jsonl")
    assert report["trend"] is None and "no trend can be claimed" in report["caveat"] and report["judged_posts"] == 3


def test_evaluation_measures_whether_the_predictor_ranks_winners_higher(tmp_path):
    rows = [{"post_id": f"g{i}", "outcome": "good", "prompt_type": "educational", "virality_score": 90} for i in range(5)]
    rows += [{"post_id": f"b{i}", "outcome": "bad", "prompt_type": "promotional", "virality_score": 70} for i in range(5)]
    rows.append(rows[0])  # a duplicated record must not count twice
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows))
    report = evaluate(log, tmp_path / "none.jsonl")
    assert report["judged_posts"] == 10 and report["score_separation"] == 20.0
    assert report["by_type"]["educational"]["hit_rate"] == 1.0 and report["by_type"]["promotional"]["hit_rate"] == 0.0
