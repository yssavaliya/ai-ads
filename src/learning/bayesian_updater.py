from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.shared.utils import load_json, load_weights, save_json, save_weights

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
            return load_json(Path(self.path), {})
        return load_weights()

    def _business(self, business_id: int) -> dict[str, Any]:
        data = self._data()
        key = str(business_id)
        if key not in data:
            data[key] = {"topics": {}, "formats": {}, "audience_segments": {}}
            self._save(data)
        return data[key]

    def record(
        self,
        business_id: int,
        feedback: str,
        prompt_type: str | None = None,
        format: str | None = None,
        audience_segment: str | None = None,
    ) -> None:
        """feedback: 'good' (hit) or 'bad' (miss). Regenerate is treated as a miss."""
        if feedback not in ("good", "bad"):
            return
        data = self._data()
        biz = data.setdefault(str(business_id), {"topics": {}, "formats": {}, "audience_segments": {}})

        def update(dim: str, key: str | None) -> None:
            if not key:
                return
            store = biz.setdefault(dim, {})
            entry = store.setdefault(key, {"weight": self.PRIOR, "trials": 0, "hits": 0})
            entry["trials"] += 1
            entry["hits"] += 1 if feedback == "good" else 0
            entry["weight"] = (entry["hits"] + self.PRIOR) / (entry["trials"] + 2 * self.PRIOR)

        update("topics", prompt_type)
        update("formats", format)
        update("audience_segments", audience_segment)
        self._save(data)
        logger.info(
            "Bayesian update for business %s: type=%s feedback=%s",
            business_id, prompt_type, feedback,
        )

    def _save(self, data: dict[str, Any]) -> None:
        if self.path:
            save_json(Path(self.path), data)
        else:
            save_weights(data)

    def get_weights(self, business_id: int) -> dict[str, Any]:
        return self._business(business_id)

    def to_forge_weights(self, business_id: int) -> dict[str, Any]:
        """Flatten into {topic::format::segment: weight} for the forge prompt.

        Cold-start: a brand-new business has no trials, so its weights are all
        flat PRIOR (=1.0) and the forge gets no useful signal. To bootstrap
        learning we blend in the top cross-business winning combos (scaled
        slightly below their proven strength) until the business has enough
        of its own data."""
        biz = self._business(business_id)
        flat: dict[str, float] = {}
        dims = ("topics", "formats", "audience_segments")
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
            for dim in ("topics", "formats", "audience_segments"):
                for key, entry in (biz.get(dim) or {}).items():
                    k = f"{dim}::{key}"
                    agg.setdefault(k, []).append(float(entry.get("weight", 0.0)))
        scored = [(k, sum(v) / len(v)) for k, v in agg.items() if v]
        scored.sort(key=lambda x: -x[1])
        return scored[: self.GLOBAL_TOP_N]
