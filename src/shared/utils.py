from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.shared.config import settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def load_businesses() -> list[dict[str, Any]]:
    return load_json(settings.data_dir / "businesses.json", [])


def save_businesses(businesses: list[dict[str, Any]]) -> None:
    save_json(settings.data_dir / "businesses.json", businesses)


def load_weights() -> dict[str, Any]:
    return load_json(settings.data_dir / "weights.json", {})


def save_weights(weights: dict[str, Any]) -> None:
    save_json(settings.data_dir / "weights.json", weights)