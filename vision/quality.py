"""Image quality gate — instant rejection of poor images before any processing."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Final

import cv2
import numpy as np

from config import QualityConfig


@dataclass
class QualityReport:
    """Result of all quality checks for one image."""

    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - trivial
        status = "PASS" if self.passed else f"FAIL({', '.join(self.failures)})"
        metrics = ", ".join(f"{k}={v:.1f}" for k, v in self.metrics.items())
        return f"[{status}] {metrics}"


class QualityCheck(ABC):
    """Strategy interface for a single quality check."""

    name: Final[str] = "check"

    def __init__(self, config: QualityConfig) -> None:
        self.config = config

    @abstractmethod
    def metric(self, image_bgr: np.ndarray) -> float:
        """Compute the check's scalar metric."""

    @abstractmethod
    def evaluate(self, metric: float) -> str | None:
        """Return a failure reason, or None when the check passes."""


class ResolutionCheck(QualityCheck):
    """Shortest side must be at least ``min_resolution`` pixels."""

    name = "resolution"

    def metric(self, image_bgr: np.ndarray) -> float:
        height, width = image_bgr.shape[:2]
        return float(min(height, width))

    def evaluate(self, metric: float) -> str | None:
        if metric < self.config.min_resolution:
            return f"resolution {metric:.0f}px < {self.config.min_resolution}px"
        return None


class BlurCheck(QualityCheck):
    """Variance of Laplacian — low values indicate a blurry photo."""

    name = "blur"

    def metric(self, image_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def evaluate(self, metric: float) -> str | None:
        if metric < self.config.blur_threshold:
            return f"blur score {metric:.1f} < {self.config.blur_threshold}"
        return None


class BrightnessCheck(QualityCheck):
    """Mean V-channel must stay inside a usable exposure window."""

    name = "brightness"

    def metric(self, image_bgr: np.ndarray) -> float:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 2].mean())

    def evaluate(self, metric: float) -> str | None:
        if metric < self.config.brightness_min:
            return f"too dark (mean V={metric:.1f})"
        if metric > self.config.brightness_max:
            return f"too bright (mean V={metric:.1f})"
        return None


class ImageQualityChecker:
    """Runs every registered quality check and aggregates a QualityReport."""

    def __init__(
        self,
        config: QualityConfig,
        checks: list[QualityCheck] | None = None,
    ) -> None:
        self.config = config
        self.checks: list[QualityCheck] = checks or [
            ResolutionCheck(config),
            BlurCheck(config),
            BrightnessCheck(config),
        ]

    def assess(self, image_bgr: np.ndarray) -> QualityReport:
        if image_bgr is None or image_bgr.size == 0:
            return QualityReport(passed=False, failures=["empty image"])
        report = QualityReport(passed=True)
        for check in self.checks:
            value = check.metric(image_bgr)
            report.metrics[check.name] = value
            failure = check.evaluate(value)
            if failure is not None:
                report.passed = False
                report.failures.append(failure)
        return report