"""Adaptive image enhancement applied before OCR (its effect is measured in evaluation)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from config import EnhanceConfig


class ImageEnhancer(ABC):
    """Strategy interface for one enhancement step."""

    name: str = "enhancer"

    @abstractmethod
    def apply(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return the enhanced BGR image."""


class CLAHEEnhancer(ImageEnhancer):
    """Contrast Limited Adaptive Histogram Equalization on the L channel."""

    name = "clahe"

    def __init__(self, config: EnhanceConfig) -> None:
        self.config = config
        self._clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
        )

    def apply(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness = self._clahe.apply(lightness)
        merged = cv2.merge((lightness, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


class AdaptiveSharpener(ImageEnhancer):
    """Unsharp-mask sharpening to make printed text crisper for OCR."""

    name = "sharpen"

    def __init__(self, config: EnhanceConfig) -> None:
        self.config = config

    def apply(self, image_bgr: np.ndarray) -> np.ndarray:
        radius = max(1, self.config.unsharp_radius)
        blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=radius)
        # adaptive: stronger sharpening for blurrier inputs
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scale = min(2.0, max(0.5, 60.0 / max(sharpness_score, 1.0)))
        amount = self.config.unsharp_amount * scale
        return cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)


class EnhancementPipeline:
    """Ordered chain of enhancement strategies (composite)."""

    def __init__(self, enhancers: list[ImageEnhancer] | None = None) -> None:
        self.enhancers: list[ImageEnhancer] = enhancers if enhancers is not None else []

    @classmethod
    def default(cls, config: EnhanceConfig) -> "EnhancementPipeline":
        return cls([CLAHEEnhancer(config), AdaptiveSharpener(config)])

    def apply(self, image_bgr: np.ndarray) -> np.ndarray:
        out = image_bgr
        for enhancer in self.enhancers:
            out = enhancer.apply(out)
        return out