from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, cast

from src.shared.config import settings


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_naive() -> datetime:
    """Naive UTC "now" — the DB stores naive UTC datetimes. Replaces the
    deprecated datetime.utcnow() with identical semantics."""
    return datetime.now(UTC).replace(tzinfo=None)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logging.getLogger(__name__).warning("Failed to load %s: %s", path, exc)
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def resolve_env_ref(value: Any) -> Any:
    """Strings of the form "env:NAME" are replaced by the environment variable
    NAME (a GitHub secret/variable), so e-mail addresses and other personal data
    never need to be committed to data/*.json."""
    if isinstance(value, str) and value.startswith("env:"):
        name = value[4:].strip()
        resolved = os.environ.get(name, "")
        if not resolved:
            logging.getLogger(__name__).warning("Environment variable %s (referenced as %r) is not set", name, value)
        return resolved
    return value


def load_businesses() -> list[dict[str, Any]]:
    raw = cast(list[dict[str, Any]], load_json(settings.data_dir / "businesses.json", []))
    return [{k: resolve_env_ref(v) for k, v in b.items()} for b in raw]


def save_businesses(businesses: list[dict[str, Any]]) -> None:
    save_json(settings.data_dir / "businesses.json", businesses)


def load_weights() -> dict[str, Any]:
    return cast(dict[str, Any], load_json(settings.data_dir / "weights.json", {}))


def save_weights(weights: dict[str, Any]) -> None:
    save_json(settings.data_dir / "weights.json", weights)
