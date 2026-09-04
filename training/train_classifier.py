"""Trainer for the transfer-learning classifier (frozen backbone + new head)."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classification.dataset import DrugImageDataset
from classification.model import TransferLearningClassifier
from config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(default_factory=lambda: {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []})
    best_val_acc: float = 0.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    """Trains only the classification head; backbone stays frozen."""

    def __init__(self, config: AppConfig, manifest: pd.DataFrame) -> None:
        self.config = config
        self.manifest = manifest
        seed_everything(config.classifier.seed)
        usable = manifest[manifest["usable_for_training"] == True]  # noqa: E712
        self.class_names = sorted(usable["class_name"].unique())
        self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.model: TransferLearningClassifier | None = None
        self.result = TrainResult()

    def _make_loaders(self) -> tuple[DataLoader, DataLoader]:
        train_records = self.manifest[self.manifest["split"] == "train"].to_dict("records")
        val_records = self.manifest[self.manifest["split"] == "val"].to_dict("records")
        cfg = self.config.classifier
        train_ds = DrugImageDataset(train_records, self.class_to_index, cfg, train=True)
        val_ds = DrugImageDataset(val_records, self.class_to_index, cfg, train=False)
        return (
            DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
            DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
        )

    def train(self) -> TrainResult:
        if self.num_classes < 2:
            raise RuntimeError(f"Need >= 2 classes to train, found {self.num_classes}")
        cfg = self.config.classifier
        train_loader, val_loader = self._make_loaders()
        logger.info("Training head: %d classes | train=%d val=%d | backbone=%s (frozen)",
                    self.num_classes, len(train_loader.dataset), len(val_loader.dataset), cfg.backbone)
        self.model = TransferLearningClassifier(num_classes=self.num_classes, config=cfg)
        optimizer = torch.optim.Adam(self.model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, optimizer, criterion, train_mode=True)
            val_loss, val_acc = self._run_epoch(val_loader, optimizer, criterion, train_mode=False)
            history = self.result.history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            best_val_acc = max(best_val_acc, val_acc)
            logger.info("epoch %02d | train loss %.4f acc %.3f | val loss %.4f acc %.3f",
                        epoch, train_loss, train_acc, val_loss, val_acc)
        self.result.best_val_acc = best_val_acc
        return self.result

    def _run_epoch(self, loader: DataLoader, optimizer, criterion, train_mode: bool) -> tuple[float, float]:
        if train_mode:
            self.model.train()
        else:
            self.model.eval()
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in loader:
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)
        return total_loss / max(seen, 1), correct / max(seen, 1)