from src.learning.bayesian_updater import BayesianUpdater


def test_bayesian_update_good(tmp_path):
    updater = BayesianUpdater(str(tmp_path / "weights.json"))
    updater.record(1, "good", prompt_type="educational", format="reel")

    weights = updater.get_weights(1)
    topic = weights["topics"]["educational"]
    assert topic["trials"] == 1
    assert topic["hits"] == 1
    assert abs(topic["weight"] - (1 + 1) / (1 + 2)) < 1e-6


def test_bayesian_update_mixed(tmp_path):
    updater = BayesianUpdater(str(tmp_path / "weights.json"))
    updater.record(1, "good", prompt_type="educational")
    updater.record(1, "bad", prompt_type="educational")
    updater.record(1, "bad", prompt_type="storytelling")

    weights = updater.get_weights(1)
    edu = weights["topics"]["educational"]
    story = weights["topics"]["storytelling"]

    assert edu["trials"] == 2
    assert edu["hits"] == 1
    assert story["trials"] == 1
    assert story["hits"] == 0
    # bad (with prior) should score lower than good
    assert edu["weight"] > story["weight"]


def test_to_forge_weights(tmp_path):
    updater = BayesianUpdater(str(tmp_path / "weights.json"))
    updater.record(1, "good", prompt_type="educational", format="reel")
    flat = updater.to_forge_weights(1)
    assert "topics::educational" in flat
    assert "formats::reel" in flat


def test_invalid_feedback_ignored(tmp_path):
    updater = BayesianUpdater(str(tmp_path / "weights.json"))
    updater.record(1, "regenerate", prompt_type="educational")
    weights = updater.get_weights(1)
    assert "topics" not in weights or not weights["topics"]


def test_cold_start_seeds_global_weights(tmp_path):
    path = str(tmp_path / "weights.json")
    # A different business proves the "reel" format wins.
    BayesianUpdater(path).record(2, "good", format="reel")
    BayesianUpdater(path).record(2, "good", format="reel")
    # New business 1 has zero trials -> should inherit the global reel weight.
    flat = BayesianUpdater(path).to_forge_weights(1)
    reel_keys = [k for k in flat if k.endswith("::reel")]
    assert reel_keys, flat
    assert flat[reel_keys[0]] < 1.0  # blended below proven strength


def test_cold_start_disabled_after_enough_trials(tmp_path):
    path = str(tmp_path / "weights.json")
    BayesianUpdater(path).record(99, "good", format="reel")
    biz = BayesianUpdater(path)
    for _ in range(6):
        biz.record(1, "good", format="carousel")
    flat = biz.to_forge_weights(1)
    # Once it has its own trials, global reel seed should NOT appear.
    assert all(not k.endswith("::reel") for k in flat), flat
