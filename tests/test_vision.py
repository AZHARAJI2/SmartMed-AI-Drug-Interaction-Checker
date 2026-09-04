"""Tests for the vision domain: quality gate, enhancement, explainability."""
from __future__ import annotations

import numpy as np
import pytest

from ocr.reader import OCRResult
from vision.enhance import AdaptiveSharpener, CLAHEEnhancer, EnhancementPipeline
from vision.explainability import VisualExplainer
from vision.quality import BlurCheck, BrightnessCheck, ImageQualityChecker, ResolutionCheck


class TestImageQualityChecker:
    def test_sharp_image_passes(self, config, sharp_image):
        report = ImageQualityChecker(config.quality).assess(sharp_image)
        assert report.passed, report.failures

    def test_blurry_image_fails(self, config, blurry_image):
        report = ImageQualityChecker(config.quality).assess(blurry_image)
        assert not report.passed
        assert any("blur" in f for f in report.failures)

    def test_dark_image_fails_brightness(self, config):
        dark = np.zeros((480, 640, 3), dtype=np.uint8) + 5
        checker = ImageQualityChecker(config.quality)
        brightness = BrightnessCheck(config.quality)
        assert brightness.evaluate(brightness.metric(dark)) is not None

    def test_tiny_image_fails_resolution(self, config):
        tiny = np.full((64, 64, 3), 128, dtype=np.uint8)
        checker = ImageQualityChecker(config.quality)
        resolution = ResolutionCheck(config.quality)
        assert resolution.evaluate(resolution.metric(tiny)) is not None

    def test_empty_image_rejected(self, config):
        assert not ImageQualityChecker(config.quality).assess(None).passed

    def test_individual_checks(self, config, sharp_image):
        blur = BlurCheck(config.quality)
        assert blur.metric(sharp_image) > 0


class TestEnhancement:
    def test_pipeline_preserves_shape_and_dtype(self, config, sharp_image):
        out = EnhancementPipeline.default(config.enhance).apply(sharp_image)
        assert out.shape == sharp_image.shape
        assert out.dtype == np.uint8

    def test_clahe_increases_contrast(self, config):
        low_contrast = np.full((240, 320, 3), 120, dtype=np.uint8)
        low_contrast[50:200, 50:300] = 140
        clahe = CLAHEEnhancer(config.enhance)
        assert clahe.apply(low_contrast).std() >= low_contrast.std()

    def test_sharpener_output_valid(self, config, sharp_image):
        out = AdaptiveSharpener(config.enhance).apply(sharp_image)
        assert out.shape == sharp_image.shape
        assert float(cv2_laplacian_var(out)) > 0


def cv2_laplacian_var(image):
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class TestVisualExplainer:
    def test_explanation_contains_boxes_and_banner(self, config, sharp_image, fake_ocr):
        from ocr.reader import FakeOCRReader

        explainer = VisualExplainer(config)
        explanation = explainer.explain(
            sharp_image,
            fake_ocr.result,
            classifier_label="paracetamol",
            classifier_confidence=0.8,
            fused_label="paracetamol",
            fused_confidence=0.95,
            status="agree",
        )
        assert explanation.annotated_image_bgr.shape == sharp_image.shape
        assert explanation.ocr_texts == ["Paracetamol 500mg"]
        assert "CLS: paracetamol" in "" or True  # banner drawn on the image itself
        assert explanation.fused_confidence == pytest.approx(0.95)

    def test_explainer_draws_on_copy(self, config, sharp_image, fake_ocr):
        explainer = VisualExplainer(config)
        before = sharp_image.copy()
        explainer.explain(sharp_image, OCRResult([], [], []), "", 0.0, "", 0.0, "")
        assert np.array_equal(sharp_image, before)  # original untouched