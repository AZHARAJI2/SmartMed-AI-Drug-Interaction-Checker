"""Tests for fusion and the data organizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import SplitConfig
from fusion.fusion import AgreementFusion
from nlp.cleaner import TextCleaner
from nlp.matcher import DifflibMatcher
from training.organize_data import DataOrganizer


class TestAgreementFusion:
    def make_fusion(self):
        matcher = DifflibMatcher(["paracetamol", "ibuprofen"])
        from config import FusionConfig

        return AgreementFusion(FusionConfig(), matcher=matcher)

    def test_agreement_gives_high_confidence(self):
        result = self.make_fusion().fuse("paracetamol", 0.8, "paracetamol", 0.7)
        assert result.status == "agree"
        assert result.confidence >= 0.95
        assert result.label == "paracetamol"

    def test_disagreement_is_uncertain(self):
        result = self.make_fusion().fuse("ibuprofen", 0.8, "paracetamol", 0.7)
        assert result.status == "uncertain"

    def test_fuzzy_agreement(self):
        result = self.make_fusion().fuse("paracetmol", 0.8, "paracetamol", 0.7)
        assert result.status == "agree"

    def test_ocr_only(self):
        result = self.make_fusion().fuse("ibuprofen", 0.8, "", 0.0)
        assert result.status == "ocr_only"

    def test_classifier_only(self):
        result = self.make_fusion().fuse("", 0.0, "ibuprofen", 0.6)
        assert result.status == "classifier_only"

    def test_no_signal(self):
        result = self.make_fusion().fuse("", 0.0, "", 0.0)
        assert result.status == "no_signal" and result.label == ""


class TestDataOrganizer:
    def _write_raw_csv(self, config):
        rows = [
            {"drug_name_raw": "Paracetamol 500mg", "package_image_file": "a1.jpg"},
            {"drug_name_raw": "paracetamol 500 mg tablets", "package_image_file": "a2.jpg"},
            {"drug_name_raw": "PARACETAMOL", "package_image_file": "a3.jpg"},
            {"drug_name_raw": "Ibuprofen 200", "package_image_file": "b1.jpg"},
            {"drug_name_raw": "ibuprofen 200mg", "package_image_file": "b2.jpg"},
            {"drug_name_raw": "2017-05-31 Pharmacy Museum in Krakow 4", "package_image_file": "junk1.jpg"},
            {"drug_name_raw": "Ibuprofen", "package_image_file": "b3.jpg"},
        ]
        df = pd.DataFrame(rows)
        df.to_csv(config.data.csv_path, index=False, encoding="utf-8-sig")
        # write placeholder images
        for name in ["a1.jpg", "a2.jpg", "a3.jpg", "b1.jpg", "b2.jpg", "b3.jpg", "junk1.jpg"]:
            import cv2

            image = np.full((480, 640, 3), 190, dtype=np.uint8)
            cv2.rectangle(image, (100, 100), (500, 400), (0, 0, 0), 6)
            cv2.imwrite(str(config.data.images_dir / name), image)

    def test_manifest_build(self, config):
        self._write_raw_csv(config)
        frame, stats = DataOrganizer(config).build_manifest()
        # junk row dropped
        assert stats.dropped_junk >= 1
        # paracetamol + ibuprofen groups exist; paracetamol has 3 images
        assert "paracetamol" in set(frame["class_name"])
        # usable classes have both train and val rows
        usable = frame[frame["usable_for_training"] == True]  # noqa: E712
        assert set(usable["split"].unique()) <= {"train", "val"}
        assert (frame["split"] == "train").sum() > 0
        # class indices assigned for usable rows
        assert frame["class_index"].notna().sum() == len(usable)
        # manifest persisted
        assert config.data.manifest_path.exists()

    def test_split_guarantees_train_sample_per_class(self, config):
        self._write_raw_csv(config)
        frame, _ = DataOrganizer(config).build_manifest()
        usable = frame[frame["usable_for_training"] == True]  # noqa: E712
        for class_name, group in usable.groupby("class_name"):
            assert (group["split"] == "train").sum() >= 1, class_name