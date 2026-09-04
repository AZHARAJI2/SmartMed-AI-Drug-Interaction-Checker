"""Logs pharmacist corrections; periodically updates the OCR→drug matching dictionary.

The dictionary is a JSON map {raw_or_ocr_text: canonical_drug_name} persisted to
data/matching_dictionary.json; ScanPipeline consults it before fuzzy matching.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from config import DictionaryConfig, SplitConfig
from db.models import Drug
from db.repositories import CorrectionLogRepository, DrugRepository
from nlp.cleaner import TextCleaner

logger = logging.getLogger(__name__)


def load_matching_dictionary(config: DictionaryConfig) -> dict[str, str]:
    """Read the learned {raw_or_ocr_text: canonical_drug_name} map (empty when absent)."""
    path = Path(config.matching_dictionary_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Matching dictionary unreadable — starting fresh")
        return {}


class CorrectionLogger:
    def __init__(self, config: DictionaryConfig, correction_repo: CorrectionLogRepository,
                 drug_repo: DrugRepository) -> None:
        self.config = config
        self.corrections = correction_repo
        self.drugs = drug_repo
        self.cleaner = TextCleaner(SplitConfig())

    def log_correction(self, raw_ocr_text: str, corrected_drug_id: int, corrected_by: str = "pharmacist") -> None:
        """Record the correction and immediately teach the dictionary."""
        self.corrections.add(raw_ocr_text, corrected_drug_id, corrected_by)
        drug = self.drugs.get(corrected_drug_id)
        if drug is not None:
            self._append_to_dictionary(raw_ocr_text, drug.trade_name)
        logger.info("Correction logged: %r -> drug#%d by %s", raw_ocr_text[:60], corrected_drug_id, corrected_by)

    def _append_to_dictionary(self, raw_text: str, canonical_drug_name: str) -> None:
        dictionary = self.load_dictionary()
        key = TextCleaner.canonical(raw_text)
        if key:
            dictionary[key] = TextCleaner.canonical(canonical_drug_name)
        self._save_dictionary(dictionary)

    def load_dictionary(self) -> dict[str, str]:
        return load_matching_dictionary(self.config)

    def _save_dictionary(self, dictionary: dict[str, str]) -> None:
        path = Path(self.config.matching_dictionary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dictionary, ensure_ascii=False, indent=2), encoding="utf-8")

    def rebuild_from_history(self) -> dict[str, str]:
        """Periodic dictionary rebuild over the whole Correction_Log."""
        dictionary: dict[str, str] = {}
        for entry in self.corrections.all():
            drug = self.drugs.get(entry.corrected_drug_id)
            if drug is None:
                continue
            key = TextCleaner.canonical(entry.raw_ocr_text)
            if key:
                dictionary[key] = TextCleaner.canonical(drug.trade_name)
        self._save_dictionary(dictionary)
        logger.info("Matching dictionary rebuilt: %d entries", len(dictionary))
        return dictionary