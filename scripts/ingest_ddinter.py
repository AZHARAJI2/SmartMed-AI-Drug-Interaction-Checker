"""One-off CLI: ingest the 8 DDInter CSVs into the SQLite reference database.

Usage: python scripts/ingest_ddinter.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG  # noqa: E402
from core.ddinter_loader import DDInterIngestor  # noqa: E402
from db.database import Database  # noqa: E402
from db.repositories import (  # noqa: E402
    DrugRepository,
    IngredientRepository,
    InteractionRepository,
)
from nlp.cleaner import TextCleaner  # noqa: E402
from nlp.matcher import build_matcher  # noqa: E402
from nlp.normalizer import IngredientNormalizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    database = Database(CONFIG.database)
    database.create_schema()
    with database.session() as session:
        ingredient_repo = IngredientRepository(session)
        interaction_repo = InteractionRepository(session)
        drug_repo = DrugRepository(session)
        ingestor = DDInterIngestor(CONFIG, ingredient_repo, interaction_repo, drug_repo)
        stats = ingestor.ingest()
        session.commit()
    print(f"Ingestion complete: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())