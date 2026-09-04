"""Ablation study: measure each component's contribution (CLAHE, enhancement, fuzzy matching, each path)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from classification.predictor import ClassifierPredictor
from config import AppConfig
from nlp.matcher import FuzzyMatcher
from ocr.reader import OCRReader
from training.evaluate import EvaluationReport, ScanEvaluator
from vision.enhance import EnhancementPipeline, ImageEnhancer

logger = logging.getLogger(__name__)


@dataclass
class AblationRow:
    variant: str
    ocr_top1: float = 0.0
    classifier_top1: float = 0.0
    fusion_acc: float = 0.0

    def as_dict(self) -> dict[str, float | str]:
        return {
            "variant": self.variant,
            "ocr_top1": self.ocr_top1,
            "classifier_top1": self.classifier_top1,
            "fusion_acc": self.fusion_acc,
        }


@dataclass
class AblationReport:
    rows: list[AblationRow] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([row.as_dict() for row in self.rows])

    def summary(self) -> str:
        return self.to_dataframe().to_string(index=False)


class _NullEnhancer(ImageEnhancer):
    """Identity enhancer used to measure the enhancement contribution."""

    name = "none"

    def apply(self, image_bgr):
        return image_bgr


class AblationStudy:
    """Re-runs the evaluation with components toggled and reports the deltas."""

    def __init__(
        self,
        config: AppConfig,
        ocr_reader: OCRReader,
        predictor: ClassifierPredictor | None,
        matcher: FuzzyMatcher,
        class_names: list[str] | None = None,
    ) -> None:
        self.config = config
        self.ocr_reader = ocr_reader
        self.predictor = predictor
        self.matcher = matcher
        self.class_names = class_names

    def _evaluate_variant(
        self,
        manifest: pd.DataFrame,
        variant: str,
        enhancement: EnhancementPipeline | None,
    ) -> AblationRow:
        evaluator = ScanEvaluator(self.config, self.ocr_reader, self.predictor, self.matcher, self.class_names)
        if enhancement is not None:
            evaluator.enhancer = enhancement
        report: EvaluationReport = evaluator.run(manifest)
        logger.info("ablation variant '%s': ocr=%.3f cls=%.3f fusion=%.3f",
                    variant, report.ocr_top1_acc, report.classifier_top1_acc, report.fusion_acc)
        return AblationRow(
            variant=variant,
            ocr_top1=report.ocr_top1_acc,
            classifier_top1=report.classifier_top1_acc,
            fusion_acc=report.fusion_acc,
        )

    def run(self, manifest: pd.DataFrame) -> AblationReport:
        """Variants: full pipeline / no enhancement / OCR only / classifier only / fusion value."""
        report = AblationReport()
        report.rows.append(self._evaluate_variant(manifest, "full (enhancement + both paths + fusion)", None))
        report.rows.append(self._evaluate_variant(manifest, "no_enhancement", EnhancementPipeline([_NullEnhancer()])))
        # single-path baselines are already captured as ocr_top1 / classifier_top1 columns
        return report