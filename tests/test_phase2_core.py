"""Phase 2 tests: nlp normalization, interaction engine, repositories, DDInter ingestion."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from config import AppConfig, DatabaseConfig, SplitConfig
from core.ddinter_loader import DDInterIngestor
from core.engine import InteractionChecker
from db.database import Database
from db.repositories import (
    CorrectionLogRepository,
    DrugRepository,
    IngredientRepository,
    InteractionRepository,
    MedicationRepository,
    ScanLogRepository,
    UserRepository,
)
from nlp.cleaner import TextCleaner
from nlp.matcher import DifflibMatcher
from nlp.normalizer import CompoundDrugSplitter, IngredientNormalizer


@pytest.fixture()
def db(tmp_path):
    database = Database(DatabaseConfig(url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}"))
    database.create_schema()
    return database


@pytest.fixture()
def seeded(db):
    """Ingredients: paracetamol, ibuprofen (Major pair), warfarin (Moderate with ibuprofen)."""
    with db.session() as session:
        ingredients = IngredientRepository(session)
        interactions = InteractionRepository(session)
        p = ingredients.add("paracetamol")
        i = ingredients.add("ibuprofen")
        w = ingredients.add("warfarin")
        interactions.add(p.id, i.id, "Major")
        interactions.add(i.id, w.id, "Moderate")
        session.commit()
    return db


@pytest.fixture()
def checker(seeded):
    with seeded.session() as session:
        ingredients = IngredientRepository(session)
        matcher = DifflibMatcher([ing.name for ing in ingredients.all()])
        cleaner = TextCleaner(SplitConfig())
        yield InteractionChecker(
            ingredient_repo=ingredients,
            interaction_repo=InteractionRepository(session),
            drug_repo=DrugRepository(session),
            normalizer=IngredientNormalizer(cleaner, matcher, ingredients.aliases_map()),
            splitter=CompoundDrugSplitter(cleaner),
        )


class TestCompoundSplitter:
    def test_plus_separator(self):
        splitter = CompoundDrugSplitter(TextCleaner(SplitConfig()))
        assert splitter.split("paracetamol + ibuprofen") == ["paracetamol", "ibuprofen"]

    def test_slash_and_ampersand(self):
        splitter = CompoundDrugSplitter(TextCleaner(SplitConfig()))
        assert splitter.split("amoxicillin/clavulanic acid") == ["amoxicillin", "clavulanic acid"]
        assert len(splitter.split("paracetamol & caffeine")) == 2

    def test_single_name_unchanged(self):
        splitter = CompoundDrugSplitter(TextCleaner(SplitConfig()))
        assert splitter.split("Paracetamol 500mg") == ["paracetamol 500mg"]


class TestIngredientNormalizer:
    def test_exact_and_alias(self):
        cleaner = TextCleaner(SplitConfig())
        normalizer = IngredientNormalizer(cleaner, DifflibMatcher(["paracetamol"]), {"paracetamol": ["acetaminophen"]})
        assert normalizer.normalize("paracetamol") == "paracetamol"
        assert normalizer.normalize("Acetaminophen") == "paracetamol"

    def test_fuzzy_fallback(self):
        cleaner = TextCleaner(SplitConfig())
        normalizer = IngredientNormalizer(cleaner, DifflibMatcher(["paracetamol", "ibuprofen"]))
        assert normalizer.normalize("paracetmol") == "paracetamol"

    def test_unmatched_returns_none(self):
        normalizer = IngredientNormalizer(TextCleaner(SplitConfig()), DifflibMatcher(["paracetamol"]))
        assert normalizer.normalize("zzzqqq") is None

    def test_learned_alias(self):
        normalizer = IngredientNormalizer(TextCleaner(SplitConfig()), DifflibMatcher(["paracetamol"]), {"paracetamol": []})
        normalizer.extend_aliases("paracetamol", "para 500")
        assert normalizer.normalize("para 500") == "paracetamol"


class TestInteractionChecker:
    def test_major_pair_is_danger(self, checker):
        report = checker.check_drug_names(["paracetamol", "ibuprofen"])
        assert report.status == "danger"
        assert len(report.findings) == 1
        assert report.findings[0].severity == "Major"

    def test_moderate_pair_is_caution(self, checker):
        report = checker.check_drug_names(["ibuprofen", "warfarin"])
        assert report.status == "caution"

    def test_single_known_drug_is_safe(self, checker):
        report = checker.check_drug_names(["paracetamol"])
        assert report.status == "safe"

    def test_unknown_drug_never_reports_safe(self, checker):
        report = checker.check_drug_names(["zzzunknownzzz"])
        assert report.status == "caution"
        assert report.unmatched_names

    def test_compound_string_split_and_checked(self, checker):
        report = checker.check_drug_names(["paracetamol + ibuprofen"])
        assert report.status == "danger"

    def test_fuzzy_typo_resolves(self, checker):
        report = checker.check_drug_names(["paracetmol", "ibuprofen"])
        assert report.status == "danger"


class TestRepositories:
    def test_user_medication_roundtrip(self, seeded):
        with seeded.session() as session:
            users = UserRepository(session)
            drugs = DrugRepository(session)
            meds = MedicationRepository(session)
            user = users.add("Azhar", "azhar@example.com", "hash")
            drug = drugs.add("paracetamol", ingredient_ids=[1])
            meds.add(user.id, drug.id)
            meds.add(user.id, drug.id)  # idempotent
            assert len(meds.list_for_patient(user.id)) == 1
            assert meds.remove(user.id, drug.id) is True
            assert meds.list_for_patient(user.id) == []

    def test_scan_log_and_correction_log(self, seeded):
        with seeded.session() as session:
            drugs = DrugRepository(session)
            scans = ScanLogRepository(session)
            corrections = CorrectionLogRepository(session)
            drug = drugs.add("paracetamol")
            entry = scans.add(patient_id=None, image_path="x.jpg", ocr_raw_text="para",
                              classifier_prediction="paracetamol", fused_confidence=0.9,
                              matched_drug_id=drug.id)
            assert entry.id > 0
            corrections.add("para", drug.id, "pharmacist")
            assert len(corrections.all()) == 1
            assert scans.list_for_patient(999) == []


class TestScanPipelineDictionary:
    """Pharmacist corrections (matching dictionary) must beat fuzzy matching at scan time."""

    def test_dictionary_maps_ocr_label_to_drug(self, seeded, tmp_path):
        import cv2
        import numpy as np

        from config import AppConfig, DictionaryConfig, SplitConfig
        from core.scan_pipeline import ScanPipeline
        from ocr.reader import FakeOCRReader, OCRResult

        dict_path = tmp_path / "dict.json"
        dict_path.write_text('{"para 500": "paracetamol"}', encoding="utf-8")
        config = replace(AppConfig(), dictionary=DictionaryConfig(matching_dictionary_path=dict_path))

        with seeded.session() as session:
            ingredients = IngredientRepository(session)
            drugs = DrugRepository(session)
            drugs.add("paracetamol", ingredient_ids=[1])
            session.commit()
            matcher = DifflibMatcher([ing.name for ing in ingredients.all()])
            cleaner = TextCleaner(SplitConfig())
            checker = InteractionChecker(
                ingredient_repo=ingredients,
                interaction_repo=InteractionRepository(session),
                drug_repo=drugs,
                normalizer=IngredientNormalizer(cleaner, matcher, ingredients.aliases_map()),
                splitter=CompoundDrugSplitter(cleaner),
            )
            pipeline = ScanPipeline(
                config=config,
                ocr_reader=FakeOCRReader(OCRResult(texts=["para 500"], boxes=[], confidences=[0.9])),
                predictor=None,
                matcher=matcher,
                ingredient_repo=ingredients,
                drug_repo=drugs,
                interaction_repo=InteractionRepository(session),
                checker=checker,
            )
            image = np.full((480, 640, 3), 200, dtype=np.uint8)
            cv2.rectangle(image, (200, 150), (440, 330), (0, 0, 0), 8)
            outcome = pipeline.process(image, image_path="t.jpg")

        # "para" is below the fuzzy threshold (ratio ~53 < 70) — only the learned
        # dictionary entry can resolve it to the paracetamol drug row.
        assert outcome.matched_drug_name == "paracetamol"
        assert outcome.ocr_label == "paracetamol"
        assert outcome.report is not None and outcome.report.status == "safe"


class TestDDInterIngestor:
    def _write_csv(self, tmp_path):
        ddinter_dir = tmp_path / "ddinter"
        ddinter_dir.mkdir(exist_ok=True)
        frame = pd.DataFrame([
            {"DDInterID_A": "DDInter1", "Drug_A": "Abacavir",
             "DDInterID_B": "DDInter1263", "Drug_B": "Naltrexone", "Level": "Moderate"},
            {"DDInterID_A": "DDInter582", "Drug_A": "Dolutegravir",
             "DDInterID_B": "DDInter582", "Drug_B": "Calcium carbonate", "Level": "Major"},
        ])
        frame.to_csv(ddinter_dir / "ddinter_downloads_code_A.csv", index=False)
        return ddinter_dir

    def test_ingestion_and_idempotency(self, tmp_path):
        ddinter_dir = self._write_csv(tmp_path)
        config = replace(AppConfig(), data=replace(AppConfig().data, ddinter_dir=ddinter_dir))
        database = Database(DatabaseConfig(url=f"sqlite:///{(tmp_path / 'dd.db').as_posix()}"))
        database.create_schema()
        with database.session() as session:
            ingestor = DDInterIngestor(
                config,
                IngredientRepository(session),
                InteractionRepository(session),
                DrugRepository(session),
            )
            stats = ingestor.ingest()
            assert stats["interactions_added"] == 2
            assert stats["ingredients_added"] == 4
            stats2 = ingestor.ingest()  # idempotent
            assert stats2["interactions_added"] == 0
            session.commit()

    def test_missing_dir_raises(self, tmp_path):
        config = replace(AppConfig(), data=replace(AppConfig().data, ddinter_dir=tmp_path / "nope"))
        database = Database(DatabaseConfig(url=f"sqlite:///{(tmp_path / 'x.db').as_posix()}"))
        database.create_schema()
        with database.session() as session:
            ingestor = DDInterIngestor(
                config,
                IngredientRepository(session),
                InteractionRepository(session),
                DrugRepository(session),
            )
            with pytest.raises(FileNotFoundError):
                ingestor.ingest()

