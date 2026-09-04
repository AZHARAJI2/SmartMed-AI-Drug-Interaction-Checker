"""Tests for the transfer-learning classifier, the trainer, and the end-to-end evaluator.

These tests construct the network with pretrained=False so no weight download is needed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from classification.dataset import DrugImageDataset, build_transform
from classification.model import TransferLearningClassifier
from config import ClassifierConfig
from nlp.matcher import DifflibMatcher
from training.evaluate import ScanEvaluator
from training.train_classifier import Trainer


@pytest.fixture(scope="module")
def classifier_config_untrained():
    # tiny images + no pretrained weights → fast, offline
    return ClassifierConfig(
        backbone="resnet18",
        image_size=64,
        batch_size=4,
        epochs=1,
        pretrained=False,
        num_workers=0,
    )


class TestModel:
    def test_forward_shape(self, classifier_config_untrained):
        model = TransferLearningClassifier(num_classes=5, config=classifier_config_untrained)
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 5)

    def test_backbone_frozen_head_trainable(self, classifier_config_untrained):
        model = TransferLearningClassifier(num_classes=3, config=classifier_config_untrained)
        assert all(not p.requires_grad for p in model.backbone.parameters())
        assert all(p.requires_grad for p in model.head.parameters())

    def test_invalid_backbone_rejected(self, classifier_config_untrained):
        with pytest.raises(ValueError):
            TransferLearningClassifier(num_classes=3, config=ClassifierConfig(backbone="vgg99"))

    def test_invalid_num_classes_rejected(self, classifier_config_untrained):
        with pytest.raises(ValueError):
            TransferLearningClassifier(num_classes=1, config=classifier_config_untrained)


class TestDataset:
    def test_dataset_returns_tensor_and_label(self, tiny_manifest, classifier_config_untrained):
        class_to_index = {name: int(i) for i, name in enumerate(sorted(tiny_manifest["class_name"].unique()))}
        records = tiny_manifest.to_dict("records")
        ds = DrugImageDataset(records, class_to_index, classifier_config_untrained, train=True)
        tensor, label = ds[0]
        assert tensor.shape == (3, 64, 64)
        assert isinstance(label, int)

    def test_transform_output_range(self, classifier_config_untrained):
        import PIL.Image

        image = PIL.Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
        tensor = build_transform(classifier_config_untrained, train=False)(image)
        assert tensor.shape == (3, 64, 64)


class TestTrainer:
    def test_training_improves_loss_and_saves(self, config, tiny_manifest, classifier_config_untrained):
        cfg = config
        from dataclasses import replace

        cfg = replace(config, classifier=classifier_config_untrained)
        trainer = Trainer(cfg, tiny_manifest)
        result = trainer.train()
        assert result.best_val_acc >= 0.0
        assert len(result.history["train_loss"]) == 1
        assert trainer.num_classes == 3


class TestScanEvaluator:
    def test_evaluator_runs_end_to_end(self, config, tiny_manifest, classifier_config_untrained, fake_ocr):
        from dataclasses import replace

        cfg = replace(config, classifier=classifier_config_untrained)
        trainer = Trainer(cfg, tiny_manifest)
        trainer.train()
        from classification.predictor import ClassifierPredictor

        predictor = ClassifierPredictor(trainer.model, trainer.class_names, classifier_config_untrained)
        matcher = DifflibMatcher(trainer.class_names)
        evaluator = ScanEvaluator(cfg, fake_ocr, predictor, matcher, trainer.class_names)
        report = evaluator.run(tiny_manifest)
        assert report.n_images == 3  # one val image per class
        assert 0.0 <= report.classifier_top1_acc <= 1.0
        assert 0.0 <= report.fusion_acc <= 1.0
        assert 0.0 <= report.ocr_top1_acc <= 1.0
        assert len(report.records) == 3