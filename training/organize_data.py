"""Data organization: clean noisy raw labels, build the curated manifest, split train/val."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import AppConfig
from nlp.cleaner import TextCleaner

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "image_path", "raw_label", "clean_label", "class_name", "class_index", "split",
    "usable_for_training", "quality_pass",
]


@dataclass
class ManifestStats:
    total_records: int
    usable_records: int
    num_classes: int
    dropped_junk: int
    below_min_images: int
    train_size: int
    val_size: int


class DataOrganizer:
    """Turns the raw Wikimedia CSV + images folder into a curated training manifest."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cleaner = TextCleaner(config.split)

    def build_manifest(self) -> tuple[pd.DataFrame, ManifestStats]:
        raw = pd.read_csv(self.config.data.csv_path, encoding="utf-8-sig")
        raw = raw.dropna(subset=["package_image_file", "drug_name_raw"]).reset_index(drop=True)
        records: list[dict] = []
        dropped_junk = 0

        # --- Step 1: clean labels, drop junk rows -------------------------------
        for _, row in raw.iterrows():
            clean_label = self.cleaner.clean(str(row["drug_name_raw"]))
            if not clean_label or not self.cleaner.is_plausible_drug_name(str(row["drug_name_raw"])):
                dropped_junk += 1
                continue
            records.append(
                {
                    # CSV paths are relative ("images/xxx.jpg"); all images live in images_dir
                    "image_path": str(self.config.data.images_dir / Path(str(row["package_image_file"]).strip()).name),
                    "raw_label": str(row["drug_name_raw"]),
                    "clean_label": clean_label,
                }
            )
        frame = pd.DataFrame(records, columns=["image_path", "raw_label", "clean_label"])
        if frame.empty:
            raise RuntimeError("No usable records after label cleaning — check the CSV/images")

        # --- Step 2: group into canonical classes -------------------------------
        frame["class_name"] = frame["clean_label"].map(TextCleaner.canonical)
        counts = frame["class_name"].value_counts()
        frame["usable_for_training"] = frame["class_name"].map(counts) >= self.config.split.min_class_images
        below_min = int((~frame["usable_for_training"]).sum())

        # --- Step 3: split over usable classes (one holdout per class when possible) ---
        frame["split"] = ""
        train_mask = pd.Series(False, index=frame.index)
        rng = __import__("random").Random(self.config.classifier.seed)
        for class_name, group in frame[frame["usable_for_training"]].groupby("class_name"):
            indices = list(group.index)
            rng.shuffle(indices)
            n_val = max(1, int(round(len(indices) * self.config.classifier.val_fraction)))
            # guarantee at least one training sample per class
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            if not train_indices:
                train_indices, val_indices = val_indices[:1], val_indices[1:]
            train_mask.loc[train_indices] = True
        frame.loc[frame["usable_for_training"] & train_mask, "split"] = "train"
        frame.loc[frame["usable_for_training"] & ~train_mask, "split"] = "val"
        frame.loc[~frame["usable_for_training"], "split"] = "excluded"

        # --- Step 4: quality pre-screen (fast blur/brightness gate) --------------
        from vision.quality import ImageQualityChecker

        checker = ImageQualityChecker(self.config.quality)
        import cv2

        quality_flags: list[bool] = []
        for path in frame["image_path"]:
            try:
                image = cv2.imread(str(path))
                quality_pass = bool(image is not None and checker.assess(image).passed)
            except Exception:  # noqa: BLE001 — manifest must survive any bad file
                quality_pass = False
            quality_flags.append(quality_pass)
        frame["quality_pass"] = quality_flags

        # --- Step 5: class indices over usable classes ---------------------------
        usable_classes = sorted(frame.loc[frame["usable_for_training"], "class_name"].unique())
        class_to_index = {name: idx for idx, name in enumerate(usable_classes)}
        frame["class_index"] = frame["class_name"].map(class_to_index).astype("Int64")

        self.config.data.ensure_dirs()
        frame.to_csv(self.config.data.manifest_path, index=False, encoding="utf-8-sig")
        stats = ManifestStats(
            total_records=len(raw),
            usable_records=int(frame["usable_for_training"].sum()),
            num_classes=len(usable_classes),
            dropped_junk=dropped_junk,
            below_min_images=below_min,
            train_size=int((frame["split"] == "train").sum()),
            val_size=int((frame["split"] == "val").sum()),
        )
        logger.info("Manifest saved: %s | %s", self.config.data.manifest_path, stats)
        return frame, stats