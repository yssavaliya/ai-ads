"""requirements.txt is the pinned lock the pipelines install; it must stay in step with pyproject.toml."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _name(spec: str) -> str:
    return re.split(r"[<>=!~;\[ ]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")


def test_every_runtime_dependency_is_pinned_in_requirements():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    pinned = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", " ")) and "==" in line:
            pinned[_name(line)] = line.split("==")[1].split(";")[0].strip()
    missing = sorted(_name(d) for d in declared if _name(d) not in pinned)
    assert not missing, f"not pinned in requirements.txt (re-run the uv pip compile command in its header): {missing}"
    assert "huggingface-hub" in pinned  # image_providers imports it; the old unpinned file omitted it


def test_requirements_pins_are_exact():
    lines = [ln for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if ln and not ln.startswith(("#", " "))]
    assert lines and all("==" in ln for ln in lines), [ln for ln in lines if "==" not in ln]
