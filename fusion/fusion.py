"""Confidence fusion of the OCR path and the classifier path.

Rule (per Master plan): both paths always run. Agreement on the same drug → very high
confidence; disagreement → "uncertain" flag for pharmacist review / manual entry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import FusionConfig
from nlp.matcher import FuzzyMatcher


@dataclass
class FusionResult:
    label: str
    confidence: float
    status: str  # "agree" | "ocr_only" | "classifier_only" | "uncertain"


class FusionStrategy(ABC):
    """Strategy interface so fusion rules stay interchangeable."""

    @abstractmethod
    def fuse(
        self,
        ocr_label: str,
        ocr_confidence: float,
        classifier_label: str,
        classifier_confidence: float,
    ) -> FusionResult: ...


class AgreementFusion(FusionStrategy):
    """Fuses by agreement, using a fuzzy matcher to bridge OCR text to class names."""

    def __init__(
        self,
        config: FusionConfig,
        matcher: FuzzyMatcher | None = None,
        match_threshold: float = 70.0,
    ) -> None:
        self.config = config
        self._matcher = matcher
        self._match_threshold = match_threshold

    def _ocr_matches(self, ocr_label: str, classifier_label: str) -> tuple[bool, float]:
        ocr_label = (ocr_label or "").strip().lower()
        classifier_label = (classifier_label or "").strip().lower()
        if not ocr_label or not classifier_label:
            return False, 0.0
        if ocr_label == classifier_label:
            return True, 100.0
        if self._matcher is None:
            return False, 0.0
        score = self._matcher._score(ocr_label, classifier_label)
        return score >= self._match_threshold, score

    def fuse(
        self,
        ocr_label: str,
        ocr_confidence: float,
        classifier_label: str,
        classifier_confidence: float,
    ) -> FusionResult:
        has_ocr = bool((ocr_label or "").strip())
        has_cls = bool((classifier_label or "").strip())
        if has_ocr and has_cls:
            matched, _ = self._ocr_matches(ocr_label, classifier_label)
            if matched:
                # both paths agree → very high confidence
                return FusionResult(
                    label=classifier_label,
                    confidence=max(self.config.agree_confidence, classifier_confidence),
                    status="agree",
                )
            # disagreement → uncertain, needs pharmacist review
            return FusionResult(label=classifier_label, confidence=self.config.uncertain_confidence, status="uncertain")
        if has_ocr:
            return FusionResult(label=ocr_label, confidence=self.config.single_path_confidence, status="ocr_only")
        if has_cls:
            return FusionResult(label=classifier_label, confidence=self.config.single_path_confidence, status="classifier_only")
        return FusionResult(label="", confidence=0.0, status="no_signal")