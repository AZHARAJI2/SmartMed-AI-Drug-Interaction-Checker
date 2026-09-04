"""Visual explainability: annotated image showing OCR boxes + classifier label."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from config import AppConfig
from ocr.reader import OCRResult


@dataclass
class Explanation:
    """Everything needed to explain a scan decision visually."""

    annotated_image_bgr: np.ndarray
    ocr_texts: list[str] = field(default_factory=list)
    ocr_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    classifier_label: str = ""
    classifier_confidence: float = 0.0
    fused_label: str = ""
    fused_confidence: float = 0.0
    status: str = ""


class VisualExplainer:
    """Draws OCR bounding boxes and the classifier verdict onto the image."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def explain(
        self,
        image_bgr: np.ndarray,
        ocr_result: OCRResult,
        classifier_label: str,
        classifier_confidence: float,
        fused_label: str,
        fused_confidence: float,
        status: str,
    ) -> Explanation:
        annotated = image_bgr.copy()
        for text, box, confidence in zip(ocr_result.texts, ocr_result.boxes, ocr_result.confidences):
            x, y, w, h = box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                annotated,
                f"{text} ({confidence:.2f})",
                (max(0, x), max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )
        banner = f"CLS: {classifier_label or 'n/a'} | FUSED: {fused_label or 'n/a'} ({fused_confidence:.2f}) {status}"
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (30, 30, 30), -1)
        cv2.putText(
            annotated,
            banner,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return Explanation(
            annotated_image_bgr=annotated,
            ocr_texts=list(ocr_result.texts),
            ocr_boxes=list(ocr_result.boxes),
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            fused_label=fused_label,
            fused_confidence=fused_confidence,
            status=status,
        )