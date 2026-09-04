"""End-to-end scan evaluation: OCR metrics, classifier metrics, fusion value."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from classification.predictor import ClassifierPredictor
from config import AppConfig
from fusion.fusion import AgreementFusion, FusionResult
from nlp.cleaner import TextCleaner
from nlp.matcher import FuzzyMatcher
from ocr.reader import OCRReader
from vision.enhance import EnhancementPipeline
from vision.explainability import VisualExplainer
from vision.quality import ImageQualityChecker

logger = logging.getLogger(__name__)


@dataclass
class PathOutcome:
    label: str
    confidence: float


@dataclass
class ScanRecord:
    image_path: str
    truth: str
    ocr: PathOutcome
    classifier: PathOutcome
    fusion: FusionResult
    ocr_match_score: float


@dataclass
class EvaluationReport:
    n_images: int = 0
    rejected_by_quality: int = 0
    ocr_top1_acc: float = 0.0
    ocr_mean_match: float = 0.0
    classifier_top1_acc: float = 0.0
    classifier_top3_acc: float = 0.0
    fusion_acc: float = 0.0
    fusion_agree_rate: float = 0.0
    fusion_uncertain_rate: float = 0.0
    fusion_value_vs_ocr: float = 0.0
    fusion_value_vs_classifier: float = 0.0
    records: list[ScanRecord] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"images={self.n_images} rejected={self.rejected_by_quality}\n"
            f"OCR top-1={self.ocr_top1_acc:.3f} (mean match {self.ocr_mean_match:.1f})\n"
            f"Classifier top-1={self.classifier_top1_acc:.3f} top-3={self.classifier_top3_acc:.3f}\n"
            f"Fusion acc={self.fusion_acc:.3f} agree={self.fusion_agree_rate:.3f} "
            f"uncertain={self.fusion_uncertain_rate:.3f}\n"
            f"Fusion value vs OCR alone={self.fusion_value_vs_ocr:+.3f} | "
            f"vs classifier alone={self.fusion_value_vs_classifier:+.3f}"
        )


class ScanEvaluator:
    """Runs the full Phase 1 pipeline over the validation manifest and scores every path."""

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
        self.cleaner = TextCleaner(config.split)
        self.quality_checker = ImageQualityChecker(config.quality)
        self.enhancer = EnhancementPipeline.default(config.enhance)
        self.explainer = VisualExplainer(config)
        self.fusion = AgreementFusion(config.fusion, matcher=self.matcher)

    def _ocr_best_label(self, ocr_texts: list[str]) -> tuple[str, float]:
        """Pick the OCR text whose best vocabulary match is strongest."""
        best_label, best_score = "", 0.0
        for text in ocr_texts:
            cleaned = self.cleaner.clean(text)
            if not cleaned:
                continue
            match = self.matcher.best_match(cleaned)
            if match is not None and match.score > best_score:
                best_label, best_score = match.matched, match.score
        return best_label, best_score

    def run(self, manifest: pd.DataFrame, save_samples: int = 0) -> EvaluationReport:
        """Evaluate every validation image through quality gate → enhancement → both paths → fusion."""
        report = EvaluationReport()
        has_val = (manifest["split"] == "val").any()
        eval_frame = manifest[manifest["split"] == "val"] if has_val else manifest
        ocr_correct = cls_correct = cls_top3 = fusion_correct = agree = uncertain = 0
        ocr_scores: list[float] = []
        evaluated = 0

        for _, row in eval_frame.iterrows():
            image = cv2.imread(str(row["image_path"]))
            if image is None:
                continue
            report.n_images += 1
            quality = self.quality_checker.assess(image)
            if not quality.passed:
                report.rejected_by_quality += 1
                continue
            evaluated += 1
            enhanced = self.enhancer.apply(image)

            ocr_result = self.ocr_reader.read(enhanced)
            ocr_label, ocr_match = self._ocr_best_label(ocr_result.texts)
            ocr_conf = float(np.mean(ocr_result.confidences)) if ocr_result.confidences else 0.0

            prediction = self.predictor.predict(enhanced) if self.predictor is not None else None

            fused = self.fusion.fuse(
                ocr_label=ocr_label,
                ocr_confidence=ocr_conf,
                classifier_label=prediction.label if prediction else "",
                classifier_confidence=prediction.confidence if prediction else 0.0,
            )

            truth = str(row["clean_label"]).strip().lower()
            ocr_correct += int(bool(ocr_label) and ocr_label.strip().lower() == truth)
            ocr_scores.append(ocr_match)
            if prediction is not None:
                cls_correct += int(prediction.label.strip().lower() == truth)
                cls_top3 += int(truth in [label.strip().lower() for label, _ in prediction.top3])
            fusion_correct += int(bool(fused.label) and fused.label.strip().lower() == truth)
            agree += int(fused.status == "agree")
            uncertain += int(fused.status == "uncertain")

            report.records.append(
                ScanRecord(
                    image_path=str(row["image_path"]),
                    truth=str(row["clean_label"]),
                    ocr=PathOutcome(ocr_label, ocr_conf),
                    classifier=PathOutcome(prediction.label, prediction.confidence) if prediction else PathOutcome("", 0.0),
                    fusion=fused,
                    ocr_match_score=ocr_match,
                )
            )

            if save_samples and report.n_images <= save_samples:
                explanation = self.explainer.explain(
                    image,
                    ocr_result,
                    classifier_label=prediction.label if prediction else "",
                    classifier_confidence=prediction.confidence if prediction else 0.0,
                    fused_label=fused.label,
                    fused_confidence=fused.confidence,
                    status=fused.status,
                )
                out_path = Path(self.config.data.samples_dir) / f"sample_{report.n_images:03d}.jpg"
                cv2.imwrite(str(out_path), explanation.annotated_image_bgr)

        denom = max(evaluated, 1)
        report.ocr_top1_acc = ocr_correct / denom
        report.ocr_mean_match = float(np.mean(ocr_scores)) if ocr_scores else 0.0
        report.classifier_top1_acc = cls_correct / denom
        report.classifier_top3_acc = cls_top3 / denom
        report.fusion_acc = fusion_correct / denom
        report.fusion_agree_rate = agree / denom
        report.fusion_uncertain_rate = uncertain / denom
        report.fusion_value_vs_ocr = report.fusion_acc - report.ocr_top1_acc
        report.fusion_value_vs_classifier = report.fusion_acc - report.classifier_top1_acc
        logger.info("Evaluation done: %s", report.summary().replace("\n", " | "))
        return report
