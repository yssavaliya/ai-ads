"""Print learning-loop monitoring metrics (see src/learning/evaluation.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning.evaluation import evaluate  # noqa: E402
from src.shared.config import settings  # noqa: E402


def main() -> int:
    report = evaluate(Path(settings.data_dir) / "engagement_log.jsonl", Path(settings.data_dir) / "learning_ledger.jsonl")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
