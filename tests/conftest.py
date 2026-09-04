"""Shared fixtures — synthetic images and offline-safe components (no network, no model downloads)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from config import AppConfig, OCRConfig
from ocr.reader import FakeOCRReader, OCRResult


@pytest.fixture()
def config(tmp_path) -> AppConfig:
    """AppConfig redirected to a temp outputs dir."""
    from dataclasses import replace
    from config import DataConfig

    cfg = AppConfig()
    data = DataConfig(
        yemen_root=tmp_path / "yemen",
        images_dir=tmp_path / "yemen" / "images",
        csv_path=tmp_path / "yemen" / "drugs.csv",
        manifest_path=tmp_path / "manifest.csv",
        outputs_dir=tmp_path / "outputs",
        models_dir=tmp_path / "outputs" / "models",
        reports_dir=tmp_path / "outputs" / "reports",
        samples_dir=tmp_path / "outputs" / "samples",
    )
    data.ensure_dirs()
    data.images_dir.mkdir(parents=True, exist_ok=True)  # test fixtures write images here
    return replace(cfg, data=data)


@pytest.fixture()
def sharp_image() -> np.ndarray:
    """A clean 480x640 synthetic image with strong edges (passes the quality gate)."""
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    for i in range(0, 640, 20):
        cv2.line(image, (i, 0), (i, 480), (30, 30, 30), 2)
    cv2.rectangle(image, (200, 150), (440, 330), (0, 0, 0), 8)
    return image


@pytest.fixture()
def blurry_image() -> np.ndarray:
    """A heavily blurred version (fails the blur check)."""
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=25)
    return image


@pytest.fixture()
def fake_ocr() -> FakeOCRReader:
    return FakeOCRReader(
        OCRResult(
            texts=["Paracetamol 500mg"],
            boxes=[(100, 100, 200, 40)],
            confidences=[0.92],
        )
    )


@pytest.fixture()
def tiny_manifest(config, tmp_path) -> pd.DataFrame:
    """A minimal manifest built from synthetic images on disk."""
    import os

    classes = {
        "paracetamol": 3,
        "ibuprofen": 3,
        "amoxicillin": 2,
    }
    records = []
    idx = 0
    for class_name, count in classes.items():
        for j in range(count):
            path = config.data.images_dir / f"img_{idx:03d}.jpg"
            image = np.full((480, 640, 3), 180 + idx, dtype=np.uint8)
            cv2.putText(image, class_name.upper(), (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            cv2.imwrite(str(path), image)
            split = "train" if j < count - 1 else "val"
            records.append(
                {
                    "image_path": str(path),
                    "raw_label": class_name,
                    "clean_label": class_name,
                    "class_name": class_name,
                    "split": split,
                    "usable_for_training": True,
                    "class_index": list(classes).index(class_name),
                    "quality_pass": True,
                }
            )
            idx += 1
    return pd.DataFrame(records)


@pytest.fixture()
def ocr_config_offline() -> OCRConfig:
    return OCRConfig(enabled=False)