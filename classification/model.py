"""Transfer-learning classifier: frozen pretrained backbone + newly trained head only."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from config import ClassifierConfig

logger = logging.getLogger(__name__)

SUPPORTED_BACKBONES = ("mobilenet_v3_small", "resnet18")


class TransferLearningClassifier(nn.Module):
    """Lightweight transfer learning — no full fine-tuning (Simplicity First).

    The pretrained backbone is fully frozen; only the new classification head
    receives gradients, so training runs in minutes on free Colab / CPU.
    """

    def __init__(self, num_classes: int, config: ClassifierConfig) -> None:
        super().__init__()
        if config.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"backbone must be one of {SUPPORTED_BACKBONES}, got '{config.backbone}'")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        try:
            self.backbone, feature_dim = self._build_backbone(config.backbone, pretrained=config.pretrained)
        except (RuntimeError, OSError, ConnectionError) as exc:
            if not config.pretrained:
                raise
            # Offline environment: pretrained weights could not be downloaded.
            logger.warning(
                "Pretrained weights unavailable (%s) — falling back to random-init frozen backbone. "
                "Run on Colab/online for ImageNet-initialized transfer learning.",
                exc,
            )
            self.backbone, feature_dim = self._build_backbone(config.backbone, pretrained=False)
        for param in self.backbone.parameters():  # frozen base layers
            param.requires_grad = False
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes),
        )

    @staticmethod
    def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
        """Return (backbone_without_classifier, feature_dim); weights=None never downloads."""
        if name == "mobilenet_v3_small":
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            feature_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()  # head replaced by TransferLearningClassifier
            return backbone, feature_dim
        # resnet18
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return self.head(features)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]