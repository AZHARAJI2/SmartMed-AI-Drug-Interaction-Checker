"""Phase 1 driver: organize data, train classifier, evaluate all paths, run ablation.

Usage (from project root):
    python run_phase1.py organize            # build manifest.csv only
    python run_phase1.py train               # organize + train classifier
    python run_phase1.py evaluate            # organize + train (if no checkpoint) + evaluate + ablation
    python run_phase1.py all                 # full pipeline with sample visualizations
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import CONFIG, AppConfig
from training.ablation import AblationStudy
from training.evaluate import ScanEvaluator
from training.organize_data import DataOrganizer
from training.train_classifier import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase1")

CHECKPOINT = None  # resolved at runtime


def organize(config: AppConfig) -> pd.DataFrame:
    frame, stats = DataOrganizer(config).build_manifest()
    logger.info("Manifest stats: %s", stats)
    return frame


def train(config: AppConfig, manifest: pd.DataFrame, force: bool = False) -> Path:
    checkpoint = config.data.models_dir / "classifier.pt"
    if checkpoint.exists() and not force:
        logger.info("Checkpoint %s exists — skipping training (use force to retrain)", checkpoint)
        return checkpoint
    trainer = Trainer(config, manifest)
    result = trainer.train()
    from classification.predictor import ClassifierPredictor

    predictor = ClassifierPredictor(trainer.model, trainer.class_names, config.classifier)
    predictor.save_checkpoint(checkpoint, result.history)
    logger.info("Best val acc %.3f — checkpoint saved to %s", result.best_val_acc, checkpoint)
    return checkpoint


def build_evaluators(config: AppConfig, manifest: pd.DataFrame, checkpoint: Path):
    from classification.predictor import ClassifierPredictor
    from nlp.cleaner import TextCleaner
    from nlp.matcher import build_matcher
    from ocr.reader import EasyOCRReader

    class_names = sorted(manifest.loc[manifest["usable_for_training"] == True, "class_name"].unique())  # noqa: E712
    matcher = build_matcher(class_names)
    predictor: ClassifierPredictor | None = None
    if checkpoint is not None and Path(checkpoint).exists():
        predictor = ClassifierPredictor.from_checkpoint(checkpoint, config.classifier)
    reader = EasyOCRReader(config.ocr)
    evaluator = ScanEvaluator(config, reader, predictor, matcher, class_names)
    ablation = AblationStudy(config, reader, predictor, matcher, class_names)
    return evaluator, ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 — Vision & Training driver")
    parser.add_argument("step", choices=["organize", "train", "evaluate", "all"])
    parser.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    parser.add_argument("--samples", type=int, default=5, help="annotated sample images to save on 'all'")
    args = parser.parse_args()

    config = CONFIG
    config.data.ensure_dirs()

    if args.step == "organize":
        organize(config)
        return 0

    manifest = organize(config)
    checkpoint = train(config, manifest, force=args.force)

    if args.step == "train":
        return 0

    evaluator, ablation = build_evaluators(config, manifest, checkpoint)
    report = evaluator.run(manifest, save_samples=args.samples if args.step == "all" else 0)
    logger.info("\n=== Evaluation ===\n%s", report.summary())
    (config.data.reports_dir / "evaluation.txt").write_text(report.summary(), encoding="utf-8")

    ablation_report = ablation.run(manifest)
    logger.info("\n=== Ablation ===\n%s", ablation_report.summary())
    ablation_report.to_dataframe().to_csv(config.data.reports_dir / "ablation.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())