from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from src.shared.config import settings
from src.shared.utils import load_json, load_weights, save_json, save_weights, utc_now_naive

logger = logging.getLogger(__name__)

WEIGHT_KEY_SEPARATOR = "::"


class BayesianUpdater:
    """Updates per-business topic/format/audience weights using feedback.
    Implements a simple beta-Bernoulli update: weight = (hits + prior) / (trials + 2*prior)."""

    PRIOR = 1.0
    COLD_START_TRIALS = 6          # below this many local trials, seed from global winners
    COLD_START_BLEND = 0.85        # how strongly to trust the cross-business prior
    GLOBAL_TOP_N = 15              # how many global winning combos to seed

    def __init__(self, weights_path: str | None = None) -> None:
        self.path = weights_path

    def _data(self) -> dict[str, Any]:
        if self.path:
            return cast(dict[str, Any], load_json(Path(self.path), {}))
        return cast(dict[str, Any], load_weights())

    def _business(self, business_id: int | str) -> dict[str, Any]:
        data = self._data()
        key = str(business_id)
        if key not in data:
            data[key] = {"topics": {}, "formats": {}, "audience_segments": {}, "hook_types": {}}
            self._save(data)
        return cast(dict[str, Any], data[key])

    def record(
        self,
        business_id: int | str,
        feedback: str,
        prompt_type: str | None = None,
        format: str | None = None,
        audience_segment: str | None = None,
        hook_type: str | None = None,
        source: str = "unknown",
    ) -> None:
        """feedback: 'good' (hit) or 'bad' (miss). Regenerate is treated as a miss.

        Every accepted update is also appended to an immutable LEDGER. The weights are pure counts
        (hits/trials), so replaying the ledger reproduces them exactly: that is the version history,
        the audit trail ("why is this weight 0.3?") and the rollback mechanism."""
        if feedback not in ("good", "bad"):
            return
        data = self._data()
        key = str(business_id)
        self._ledger_snapshot_if_new(data)
        biz = data.setdefault(key, {"topics": {}, "formats": {}, "audience_segments": {}, "hook_types": {}})
        self.apply_update(biz, feedback, prompt_type, format, audience_segment, hook_type)
        self._save(data)
        self._ledger_append({
            "type": "update", "ts": utc_now_naive().isoformat(timespec="seconds"), "business": key,
            "feedback": feedback, "prompt_type": prompt_type, "format": format,
            "audience_segment": audience_segment, "hook_type": hook_type, "source": source,
        })
        logger.info(
            "Bayesian update for business %s: type=%s hook=%s feedback=%s source=%s",
            business_id, prompt_type, hook_type, feedback, source,
        )

    @classmethod
    def apply_update(
        cls,
        biz: dict[str, Any],
        feedback: str,
        prompt_type: str | None,
        format: str | None,
        audience_segment: str | None,
        hook_type: str | None = None,
    ) -> None:
        """The one place the update rule lives (used live AND by ledger replay)."""

        def update(dim: str, key: str | None) -> None:
            if not key:
                return
            store = biz.setdefault(dim, {})
            entry = store.setdefault(key, {"weight": cls.PRIOR, "trials": 0, "hits": 0})
            entry["trials"] += 1
            entry["hits"] += 1 if feedback == "good" else 0
            entry["weight"] = (entry["hits"] + cls.PRIOR) / (entry["trials"] + 2 * cls.PRIOR)

        update("topics", prompt_type)
        update("formats", format)
        update("audience_segments", audience_segment)
        update("hook_types", hook_type)

    # ------------------------------------------------------------------ ledger
    def ledger_path(self) -> Path:
        if self.path:
            return Path(self.path).with_suffix(".ledger.jsonl")
        return Path(settings.data_dir) / "learning_ledger.jsonl"

    def _ledger_append(self, event: dict[str, Any]) -> None:
        path = self.ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _ledger_snapshot_if_new(self, data: dict[str, Any]) -> None:
        """Weights that predate the ledger become its baseline, so a replay is still exact."""
        path = self.ledger_path()
        if data and not path.exists():
            self._ledger_append({"type": "snapshot", "ts": utc_now_naive().isoformat(timespec="seconds"), "weights": data})

    def version(self) -> int:
        """Number of learning updates applied so far."""
        return len(effective_updates(read_ledger(self.ledger_path()))[1])

    def rollback(self, to_version: int) -> dict[str, Any]:
        """Restore the weights as they were after `to_version` updates. Append-only: the rollback
        itself is recorded, nothing in the ledger is erased."""
        events = read_ledger(self.ledger_path())
        current = len(effective_updates(events)[1])
        if not 0 <= to_version <= current:
            raise ValueError(f"version must be between 0 and {current}")
        self._ledger_append({"type": "rollback", "ts": utc_now_naive().isoformat(timespec="seconds"), "to": to_version, "from": current})
        weights = replay_ledger(read_ledger(self.ledger_path()))
        self._save(weights)
        return weights

    def _save(self, data: dict[str, Any]) -> None:
        if self.path:
            save_json(Path(self.path), data)
        else:
            save_weights(data)

    def get_weights(self, business_id: int | str) -> dict[str, Any]:
        return self._business(business_id)

    def to_forge_weights(self, business_id: int | str) -> dict[str, Any]:
        """Flatten into {topic::format::segment: weight} for the forge prompt.

        Cold-start: a brand-new business has no trials, so its weights are all
        flat PRIOR (=1.0) and the forge gets no useful signal. To bootstrap
        learning we blend in the top cross-business winning combos (scaled
        slightly below their proven strength) until the business has enough
        of its own data."""
        biz = self._business(business_id)
        flat: dict[str, float] = {}
        dims = ("topics", "formats", "audience_segments", "hook_types")
        for dim in dims:
            for key, entry in biz.get(dim, {}).items():
                flat[f"{dim}::{key}"] = round(entry["weight"], 3)

        total_trials = sum(
            entry.get("trials", 0)
            for dim in dims
            for entry in biz.get(dim, {}).values()
        )
        if total_trials < self.COLD_START_TRIALS:
            for key, weight in self._global_top_weights():
                if key not in flat:
                    flat[key] = round(weight * self.COLD_START_BLEND, 3)
        return flat

    def _global_top_weights(self) -> list[tuple[str, float]]:
        """Aggregate proven combos across ALL businesses (averaged per key),
        returning the strongest N to seed cold-start accounts."""
        data = self._data()
        agg: dict[str, list[float]] = {}
        for biz in data.values():
            for dim in ("topics", "formats", "audience_segments", "hook_types"):
                for key, entry in (biz.get(dim) or {}).items():
                    k = f"{dim}::{key}"
                    agg.setdefault(k, []).append(float(entry.get("weight", 0.0)))
        scored = [(k, sum(v) / len(v)) for k, v in agg.items() if v]
        scored.sort(key=lambda x: -x[1])
        return scored[: self.GLOBAL_TOP_N]


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except ValueError:
                logger.warning("Skipping corrupt ledger line")
    return events


def effective_updates(events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(baseline weights, the updates that currently count). A snapshot resets the baseline; a
    rollback event drops every update after its first `to`."""
    baseline: dict[str, Any] = {}
    updates: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("type")
        if kind == "snapshot":
            baseline, updates = json.loads(json.dumps(event.get("weights") or {})), []
        elif kind == "update":
            updates.append(event)
        elif kind == "rollback":
            updates = updates[: max(int(event.get("to", 0)), 0)]
    return baseline, updates


def replay_ledger(events: list[dict[str, Any]], upto: int | None = None) -> dict[str, Any]:
    """Rebuild the weights from scratch: baseline + the first `upto` effective updates."""
    weights, updates = effective_updates(events)
    for event in updates if upto is None else updates[:upto]:
        biz = weights.setdefault(event["business"], {"topics": {}, "formats": {}, "audience_segments": {}, "hook_types": {}})
        BayesianUpdater.apply_update(
            biz,
            event["feedback"],
            event.get("prompt_type"),
            event.get("format"),
            event.get("audience_segment"),
            event.get("hook_type"),
        )
    return weights
