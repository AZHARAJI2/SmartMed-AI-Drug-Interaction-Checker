"""Ingests the 8 DDInter CSVs into Ingredients + Interactions (idempotent)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import AppConfig
from db.repositories import DrugRepository, IngredientRepository, InteractionRepository

logger = logging.getLogger(__name__)


class DDInterIngestor:
    """Builds the interaction reference database from ddinter_downloads_code_*.csv."""

    def __init__(self, config: AppConfig, ingredient_repo: IngredientRepository,
                 interaction_repo: InteractionRepository, drug_repo: DrugRepository) -> None:
        self.config = config
        self.ingredients = ingredient_repo
        self.interactions = interaction_repo
        self.drugs = drug_repo

    def ingest(self) -> dict[str, int]:
        csv_files = sorted(Path(self.config.data.ddinter_dir).glob("ddinter_downloads_code_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No DDInter CSVs found in {self.config.data.ddinter_dir}")
        n_new_interactions = 0
        n_ingredients_before = len(self.ingredients.all())
        for csv_path in csv_files:
            frame = pd.read_csv(csv_path)
            for _, row in frame.iterrows():
                ing_a = self._ensure_ingredient(str(row["Drug_A"]), str(row.get("DDInterID_A", "")))
                ing_b = self._ensure_ingredient(str(row["Drug_B"]), str(row.get("DDInterID_B", "")))
                created = self.interactions.add(
                    ing_a.id, ing_b.id,
                    severity=str(row["Level"]).strip(),
                    description="",
                    source="DDInter",
                )
                if created is not None:
                    n_new_interactions += 1
        stats = {
            "files": len(csv_files),
            "ingredients_added": len(self.ingredients.all()) - n_ingredients_before,
            "interactions_added": n_new_interactions,
            "interactions_total": len(self.interactions.all()) if hasattr(self.interactions, "all") else 0,
        }
        logger.info("DDInter ingestion: %s", stats)
        return stats

    def _ensure_ingredient(self, drug_name: str, ddinter_id: str):
        from nlp.cleaner import TextCleaner

        canonical = TextCleaner.canonical(drug_name)
        ingredient = self.ingredients.get_by_name(canonical)
        if ingredient is None:
            # drug_class = DDInter ATC letter from the DDInterID (e.g. DDInter582 -> unknown; code letter in file)
            ingredient = self.ingredients.add(canonical, aliases=[drug_name.lower()], drug_class="")
        # also expose the trade-level name as a Drug for text lookups
        self.drugs.add(canonical, ingredient_ids=[ingredient.id])
        return ingredient