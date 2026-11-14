"""Designed experiments: one variable per week, control vs challenger, concluded only on evidence.

Random exploration (the old 25% "explore" variant) cannot tell you WHY something worked. An agency tests one thing
at a time. Each ISO week this module activates one experiment (rotating hook type -> posting block -> format), assigns
the week's published creative to the arm that has fewer posts so far (balanced), and later compares the arms'
measured performance (Buffer quality rate of matured posts).

Rules that keep it honest:
- arm sizes come from the published log (what was really published), not from counters;
- a winner needs >= MIN_PER_ARM matured posts per arm AND a Welch t statistic >= T_MIN; otherwise the answer is
  "insufficient" or "no difference", never a guess;
- results only inform (strategy hints, a preferred posting hour); they never change code, secrets or publishing rules.
"""
from __future__ import annotations

import json
import math
import statistics
from datetime import date
from pathlib import Path
from typing import Any

HOOKS = ("contrarian", "roi_proof", "curiosity_gap", "before_after", "framework")
FORMATS = ("single image", "reel")
BLOCK_CANDIDATES = (9, 12, 17, 20)
VARIABLES = ("hook_type", "post_block", "format")
DEFAULT_CONTROL_HOUR = 19
MIN_PER_ARM = 4
T_MIN = 2.0


# ------------------------------------------------------------------ state
def load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return data if isinstance(data, dict) else {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def week_key(day: date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def variable_for_week(day: date) -> str:
    return VARIABLES[day.isocalendar()[1] % len(VARIABLES)]


def _used(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _block(hour: int) -> int:
    return hour // 3


def arms_for(
    variable: str, records: list[dict[str, Any]], control_hour: int | None, top_hook: str | None
) -> dict[str, dict[str, Any]]:
    """Control (A) = what we believe is best today; challenger (B) = the least-tried alternative."""
    if variable == "hook_type":
        control = top_hook if top_hook in HOOKS else "framework"
        used = _used(records, "hook_type")
        challenger = min((h for h in HOOKS if h != control), key=lambda h: (used.get(h, 0), HOOKS.index(h)))
        return {"A": {"value": control}, "B": {"value": challenger}}
    if variable == "format":
        used = _used(records, "format")
        control = max(FORMATS, key=lambda f: (used.get(f, 0), -FORMATS.index(f)))  # what we already do most
        challenger = next(f for f in FORMATS if f != control)
        return {"A": {"value": control}, "B": {"value": challenger}}
    hour = DEFAULT_CONTROL_HOUR if control_hour is None else int(control_hour)
    hours_used: dict[int, int] = {}
    for record in records:
        scheduled = record.get("schedule_hour")
        if isinstance(scheduled, int):
            hours_used[_block(scheduled)] = hours_used.get(_block(scheduled), 0) + 1
    options = [h for h in BLOCK_CANDIDATES if _block(h) != _block(hour)]
    challenger_hour = min(options, key=lambda h: (hours_used.get(_block(h), 0), BLOCK_CANDIDATES.index(h)))
    return {"A": {"value": hour}, "B": {"value": challenger_hour}}


def ensure_active(
    state: dict[str, Any], today: date, records: list[dict[str, Any]], control_hour: int | None, top_hook: str | None
) -> dict[str, Any]:
    """This week's experiment (created on first use each week; last week's moves to history)."""
    key = week_key(today)
    active = state.get("active")
    if isinstance(active, dict) and active.get("week") == key:
        return active
    if isinstance(active, dict):
        state.setdefault("history", []).append(active)
        state["history"] = state["history"][-52:]
    variable = variable_for_week(today)
    state["active"] = {
        "id": f"{key}-{variable}", "week": key, "variable": variable,
        "arms": arms_for(variable, records, control_hour, top_hook),
        "started": today.isoformat(),
    }
    return state["active"]  # type: ignore[no-any-return]


# ------------------------------------------------------------------ assignment
def _arm_counts(experiment_id: str, records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"A": 0, "B": 0}
    for record in records:
        tag = record.get("experiment") or {}
        if tag.get("id") == experiment_id and tag.get("arm") in counts:
            counts[tag["arm"]] += 1
    return counts


def _score(prompt: dict[str, Any]) -> float:
    value = prompt.get("virality_score")
    return float(value) if isinstance(value, (int, float)) else -1.0


def _attribute(prompt: dict[str, Any], variable: str) -> Any:
    if variable == "hook_type":
        return prompt.get("hook_type")
    if variable == "format":
        return prompt.get("format")
    return None


def assign(
    experiment: dict[str, Any], prompts: list[dict[str, Any]], records: list[dict[str, Any]], threshold: int
) -> dict[str, Any] | None:
    """Tag the creative that should carry this week's test. Only publishable creatives (scored, over the bar) are
    eligible, so an experiment can never push a weak creative out. Returns the tag, or None."""
    eligible = [p for p in prompts if not p.get("below_threshold") and _score(p) >= threshold]
    if not eligible:
        return None
    counts = _arm_counts(experiment["id"], records)
    order = sorted(("A", "B"), key=lambda a: (counts[a], a))  # fewer posts first; ties -> control
    variable = experiment["variable"]
    for arm in order:
        value = experiment["arms"][arm]["value"]
        if variable == "post_block":
            chosen = max(eligible, key=_score)
        else:
            matches = [p for p in eligible if _attribute(p, variable) == value]
            if not matches:
                continue
            chosen = max(matches, key=_score)
        tag = {"id": experiment["id"], "variable": variable, "arm": arm, "value": value}
        chosen["experiment"] = tag
        if variable == "post_block" and arm == "B":
            chosen["schedule_hour"] = int(value)
        return tag
    return None


# ------------------------------------------------------------------ evaluation
def welch_t(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return None if statistics.mean(a) == statistics.mean(b) else math.inf
    return (statistics.mean(a) - statistics.mean(b)) / se


def conclude(experiment: dict[str, Any], rates: dict[str, list[float]]) -> dict[str, Any]:
    a, b = rates.get("A", []), rates.get("B", [])
    base: dict[str, Any] = {
        "id": experiment["id"], "variable": experiment["variable"], "arms": experiment["arms"],
        "n_a": len(a), "n_b": len(b),
    }
    if len(a) < MIN_PER_ARM or len(b) < MIN_PER_ARM:
        return {**base, "status": "insufficient", "note": f"needs {MIN_PER_ARM} matured posts per arm"}
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    t = welch_t(a, b)
    base.update({"mean_a": round(mean_a, 5), "mean_b": round(mean_b, 5), "t": None if t is None else round(t, 2)})
    if t is None or abs(t) < T_MIN:
        return {**base, "status": "no_difference", "note": "difference is within noise"}
    winner = "A" if t > 0 else "B"
    loser_mean = mean_b if winner == "A" else mean_a
    lift = (max(mean_a, mean_b) / loser_mean - 1.0) if loser_mean > 0 else None
    return {
        **base, "status": "winner", "winner": winner, "value": experiment["arms"][winner]["value"],
        "lift": None if lift is None else round(lift, 3),
    }


def collect_rates(
    experiment: dict[str, Any], records: list[dict[str, Any]], learned: dict[str, float]
) -> dict[str, list[float]]:
    """Matured quality rate per arm, joined through the exact Buffer post ids."""
    rates: dict[str, list[float]] = {"A": [], "B": []}
    for record in records:
        tag = record.get("experiment") or {}
        if tag.get("id") != experiment["id"] or tag.get("arm") not in rates:
            continue
        for post in record.get("buffer_posts") or []:
            rate = learned.get(str(post.get("id")))
            if isinstance(rate, (int, float)):
                rates[tag["arm"]].append(float(rate))
    return rates


def learned_rates(metric_events: list[dict[str, Any]]) -> dict[str, float]:
    """buffer post id -> quality rate for every post the metrics sync judged mature."""
    out: dict[str, float] = {}
    for event in metric_events:
        if event.get("type") == "learned" and isinstance(event.get("quality_rate"), (int, float)):
            out[str(event.get("buffer_post_id"))] = float(event["quality_rate"])
    return out


def evaluate(
    state: dict[str, Any], records: list[dict[str, Any]], metric_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Re-judge every experiment (active and past); store conclusions; return the ones that are new or changed."""
    learned = learned_rates(metric_events)
    results: dict[str, Any] = state.setdefault("results", {})
    changed: list[dict[str, Any]] = []
    experiments = [e for e in [*(state.get("history") or []), state.get("active")] if isinstance(e, dict) and e.get("id")]
    for experiment in experiments:
        verdict = conclude(experiment, collect_rates(experiment, records, learned))
        if results.get(experiment["id"]) != verdict:
            results[experiment["id"]] = verdict
            changed.append(verdict)
    return changed


# ------------------------------------------------------------------ consumers
def strategy_hints(state: dict[str, Any], limit: int = 3) -> list[str]:
    hints = []
    for verdict in [v for _, v in sorted((state.get("results") or {}).items(), reverse=True)]:
        if verdict.get("status") != "winner":
            continue
        loser = "B" if verdict["winner"] == "A" else "A"
        lift = f"{verdict['lift']:+.0%}" if verdict.get("lift") is not None else "higher"
        hints.append(
            f"Experiment {verdict['variable']}: '{verdict['value']}' beat '{verdict['arms'][loser]['value']}' "
            f"({lift} quality rate, {verdict['n_a']}/{verdict['n_b']} matured posts)"
        )
        if len(hints) >= limit:
            break
    return hints


def preferred_hour(state: dict[str, Any]) -> int | None:
    """The posting hour from the latest conclusive timing experiment, if any."""
    for verdict in [v for _, v in sorted((state.get("results") or {}).items(), reverse=True)]:
        if verdict.get("variable") == "post_block" and verdict.get("status") == "winner":
            try:
                return int(verdict["value"])
            except (TypeError, ValueError):
                return None
    return None


# ------------------------------------------------------------------ files and CLI
def read_records(data_dir: Path, business_name: str | None = None) -> list[dict[str, Any]]:
    try:
        posted = json.loads((data_dir / "published_log.json").read_text(encoding="utf-8")).get("posted") or []
    except (OSError, ValueError):
        return []
    return [
        r for r in posted
        if isinstance(r, dict) and (business_name is None or not r.get("business_name") or r.get("business_name") == business_name)
    ]


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
    except OSError:
        pass
    return events


def main() -> None:
    """6-hourly job: judge every experiment from matured, exactly-attributed posts and store the verdicts."""
    import logging
    import os

    from src.shared.config import settings

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data_dir = Path(settings.data_dir)
    state_path = data_dir / "experiments.json"
    state = load_state(state_path)
    changed = evaluate(state, read_records(data_dir), read_events(data_dir / "buffer_metrics.jsonl"))
    save_state(state_path, state)
    lines = ["### Experiments", ""]
    for verdict in (state.get("results") or {}).values():
        lines.append(
            f"- `{verdict['id']}`: **{verdict['status']}**"
            + (f", winner {verdict['value']} ({verdict.get('lift')})" if verdict.get("status") == "winner" else "")
            + f" (matured posts A={verdict['n_a']} B={verdict['n_b']})"
        )
    if len(lines) == 2:
        lines.append("No experiments have run yet.")
    print("\n".join(lines))
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
    print(f"Changed verdicts: {len(changed)}")


if __name__ == "__main__":
    main()
