"""Dataset and transforms for the transfer-learning classifier."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import ClassifierConfig

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(config: ClassifierConfig, train: bool) -> transforms.Compose:
    """Lightweight transforms — the backbone is frozen so heavy augmentation is not needed."""
    ops: list = [transforms.Resize((config.image_size, config.image_size))]
    if train:
        ops += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(5),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class DrugImageDataset(Dataset):
    """Reads (image_path, class_index) pairs from the curated manifest dataframe."""

    def __init__(self, records: list[dict], class_to_index: dict[str, int], config: ClassifierConfig, train: bool) -> None:
        self.records = records
        self.class_to_index = class_to_index
        self.config = config
        self.transform = build_transform(config, train=train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        path = Path(record["image_path"])
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except (OSError, ValueError):
            logger.warning("Unreadable image %s — substituting black frame", path.name)
            image = Image.fromarray(np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8))
        tensor = self.transform(image)
        label = self.class_to_index[record["class_name"]]
        return tensor, label