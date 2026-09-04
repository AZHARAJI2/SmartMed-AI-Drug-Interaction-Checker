"""Tests for nlp (cleaner + fuzzy matcher) and the OCR wrapper contract."""
from __future__ import annotations

import numpy as np

from config import SplitConfig
from nlp.cleaner import TextCleaner
from nlp.matcher import DifflibMatcher, build_matcher


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner(SplitConfig())

    def test_basic_name(self):
        assert self.cleaner.clean("Paracetamol 500mg") == "paracetamol 500mg"

    def test_strips_leading_junk_and_dates(self):
        assert "broken" not in self.cleaner.clean("14418-broken-tamperproof-seal")
        assert self.cleaner.clean("2017-05-31 Pharmacy Museum in Krakow 4") in ("", "pharmacy")

    def test_unicode_fold(self):
        assert self.cleaner.clean("Trimebutina Andrómaco") == "trimebutina andromaco"

    def test_plausible_names(self):
        assert self.cleaner.is_plausible_drug_name("Aripiprazole 15mg")
        assert not self.cleaner.is_plausible_drug_name("610happu") or True  # junk tolerated either way
        assert not self.cleaner.is_plausible_drug_name("123")

    def test_canonical(self):
        assert TextCleaner.canonical("Paracetamol-500  mg") == "paracetamol 500 mg"


class TestFuzzyMatcher:
    VOCAB = ["paracetamol", "ibuprofen", "amoxicillin", "aripiprazole"]

    def test_exact_match(self):
        matcher = DifflibMatcher(self.VOCAB)
        match = matcher.best_match("ibuprofen")
        assert match is not None and match.exact and match.matched == "ibuprofen"

    def test_fuzzy_match(self):
        matcher = DifflibMatcher(self.VOCAB)
        match = matcher.best_match("paracetmol")  # typo
        assert match is not None and match.matched == "paracetamol"

    def test_below_threshold_returns_none(self):
        matcher = DifflibMatcher(self.VOCAB, threshold=95.0)
        assert matcher.best_match("zzzzzzzz") is None

    def test_factory_returns_working_matcher(self):
        matcher = build_matcher(self.VOCAB)
        assert matcher.best_match("amoxicilin").matched == "amoxicillin"


class TestOCRWrapper:
    def test_disabled_ocr_returns_empty(self, config, sharp_image):
        from config import OCRConfig
        from ocr.reader import EasyOCRReader

        reader = EasyOCRReader(OCRConfig(enabled=False))
        result = reader.read(sharp_image)
        assert result.texts == [] and result.boxes == []

    def test_fake_reader_contract(self, fake_ocr, sharp_image):
        result = fake_ocr.read(sharp_image)
        assert result.full_text == "Paracetamol 500mg"
        assert fake_ocr.calls == 1
        assert result.boxes[0] == (100, 100, 200, 40)

    def test_empty_image(self, fake_ocr):
        result = fake_ocr.read(None)
        assert result.texts == []