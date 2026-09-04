"""EasyOCR wrapper — Path 1 of the two complementary recognition paths."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import OCRConfig

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Raw OCR output for one image."""

    texts: list[str]
    boxes: list[tuple[int, int, int, int]]  # (x, y, w, h)
    confidences: list[float]

    @property
    def full_text(self) -> str:
        return " ".join(self.texts).strip()


class OCRReader(ABC):
    """Strategy interface so tests/ablations can swap the OCR engine."""

    @abstractmethod
    def read(self, image_bgr: np.ndarray) -> OCRResult:
        """Read text from a BGR image."""


class EasyOCRReader(OCRReader):
    """Wraps EasyOCR (text detection + CRNN reading). Weights load lazily."""

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._reader = None

    def _ensure_reader(self) -> None:
        if self._reader is None:
            import easyocr  # deferred so offline tests never import heavy deps

            logger.info("Initializing EasyOCR (languages=%s)...", self.config.languages)
            self._reader = easyocr.Reader(
                list(self.config.languages),
                gpu=self.config.gpu,
                verbose=False,
            )

    def read(self, image_bgr: np.ndarray) -> OCRResult:
        if not self.config.enabled:
            return OCRResult(texts=[], boxes=[], confidences=[])
        if image_bgr is None or image_bgr.size == 0:
            return OCRResult(texts=[], boxes=[], confidences=[])
        self._ensure_reader()
        results = self._reader.readtext(image_bgr)
        texts: list[str] = []
        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        for box, text, confidence in results:
            if float(confidence) < self.config.confidence_threshold:
                continue
            # EasyOCR gives 4 corner points; convert to axis-aligned (x, y, w, h).
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x, y = max(0, min(xs)), max(0, min(ys))
            w, h = max(xs) - x, max(ys) - y
            texts.append(str(text).strip())
            boxes.append((x, y, w, h))
            confidences.append(float(confidence))
        return OCRResult(texts=texts, boxes=boxes, confidences=confidences)


class FakeOCRReader(OCRReader):
    """Deterministic stub used in tests and offline ablations."""

    def __init__(self, result: OCRResult | None = None) -> None:
        self.result = result or OCRResult(texts=[], boxes=[], confidences=[])
        self.calls = 0

    def read(self, image_bgr: np.ndarray) -> OCRResult:
        self.calls += 1
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return OCRResult(texts=[], boxes=[], confidences=[])
        return self.result


__all__ = ["OCRReader", "EasyOCRReader", "OCRResult", "FakeOCRReader"]
# (cv2 imported to keep parity with image pipelines; numpy typing used above.)
_ = cv2