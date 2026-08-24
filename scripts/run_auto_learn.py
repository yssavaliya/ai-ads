from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.config import settings


def main() -> None:
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from src.learning.auto_learner import run_auto_learning

    logger = logging.getLogger(__name__)
    results = run_auto_learning()
    for name, res in results.items():
        logger.info("AUTO-LEARN %s: %s", name, res)
    total = sum(r.get("learned", 0) for r in results.values())
    logger.info("Total auto-learned signals: %d", total)


if __name__ == "__main__":
    main()
