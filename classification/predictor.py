"""Inference wrapper for the trained classifier."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from classification.dataset import build_transform
from classification.model import TransferLearningClassifier
from config import ClassifierConfig

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: str
    confidence: float
    top3: list[tuple[str, float]]


class ClassifierPredictor:
    """Loads a trained checkpoint and predicts the package class (Path 2)."""

    def __init__(self, model: TransferLearningClassifier, class_names: list[str], config: ClassifierConfig) -> None:
        self.model = model
        self.model.eval()
        self.class_names = class_names
        self.config = config
        self.transform = build_transform(config, train=False)

    @classmethod
    def from_checkpoint(cls, path: Path, config: ClassifierConfig) -> "ClassifierPredictor":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        class_names = checkpoint["class_names"]
        model = TransferLearningClassifier(num_classes=len(class_names), config=config)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Loaded classifier checkpoint %s (%d classes)", path.name, len(class_names))
        return cls(model, class_names, config)

    def save_checkpoint(self, path: Path, history: dict[str, list[float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "class_names": self.class_names,
                "history": history,
                "backbone": self.config.backbone,
            },
            path,
        )

    @torch.no_grad()
    def predict(self, image_bgr) -> Prediction:  # numpy BGR array
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = self.transform(pil_image).unsqueeze(0)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        k = min(3, len(self.class_names))
        top3_indices = torch.topk(probabilities, k=k).indices.tolist()
        top3 = [(self.class_names[i], float(probabilities[i])) for i in top3_indices]
        best_label, best_confidence = top3[0]
        return Prediction(label=best_label, confidence=best_confidence, top3=top3)