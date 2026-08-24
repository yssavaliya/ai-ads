from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.config import settings
from src.storage.models import init_db

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Initializing database at %s", settings.database_path)
    init_db()
    logger.info("Database ready")


if __name__ == "__main__":
    main()
