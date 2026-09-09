"""Central configuration for the Drug Package Image Interaction Checker.

All paths and hyperparameters live here so that every domain module stays configurable
and testable. Phase 1 uses the vision/ocr/nlp/classification/fusion/training sections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DataConfig:
    """Paths to the raw datasets and generated artifacts."""

    yemen_root: Path = PROJECT_ROOT / "yemen_drug_database"
    images_dir: Path = yemen_root / "images"
    csv_path: Path = yemen_root / "yemen_global_drugs_500.csv"
    ddinter_dir: Path = PROJECT_ROOT / "ddinter_database"
    manifest_path: Path = PROJECT_ROOT / "data" / "manifest.csv"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    models_dir: Path = outputs_dir / "models"
    reports_dir: Path = outputs_dir / "reports"
    samples_dir: Path = outputs_dir / "samples"

    def ensure_dirs(self) -> None:
        for d in (self.manifest_path.parent, self.outputs_dir, self.models_dir, self.reports_dir, self.samples_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class QualityConfig:
    """Image quality gate thresholds (a failing check rejects the image instantly)."""

    min_resolution: int = 80           # shortest side, pixels (relaxed for camera/cropped images)
    blur_threshold: float = 12.0       # variance of Laplacian (relaxed from 45.0 for real-world cameras)
    brightness_min: float = 12.0       # mean V-channel 0..255 (relaxed for indoor webcam captures)
    brightness_max: float = 245.0      # relaxed to accommodate packaging reflections


@dataclass(frozen=True)
class EnhanceConfig:
    """Adaptive enhancement applied before OCR (its effect is measured in evaluation)."""

    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    unsharp_amount: float = 1.5        # adaptive sharpening strength
    unsharp_radius: int = 3


@dataclass(frozen=True)
class OCRConfig:
    """EasyOCR wrapper settings."""

    languages: tuple[str, ...] = ("en",)
    gpu: bool = False
    confidence_threshold: float = 0.35  # per-text-block confidence floor
    # EasyOCR downloads weights on first use; allow disabling for offline runs/tests.
    enabled: bool = True


@dataclass(frozen=True)
class ClassifierConfig:
    """Transfer-learning classifier: frozen backbone + newly trained head only."""

    backbone: str = "mobilenet_v3_small"  # or "resnet18"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    seed: int = 42
    num_workers: int = 0                  # Windows-safe default
    pretrained: bool = True               # frozen base layers


@dataclass(frozen=True)
class FusionConfig:
    """Agreement fusion between the two paths (both always run)."""

    agree_confidence: float = 0.95
    single_path_confidence: float = 0.55
    uncertain_confidence: float = 0.40


@dataclass(frozen=True)
class SplitConfig:
    """Data organization / label cleaning rules."""

    min_class_images: int = 2        # classes below this are excluded from classifier training
    max_name_words: int = 8          # cleaned names longer than this are treated as junk
    junk_keywords: tuple[str, ...] = (
        "museum", "flickr", "seal", "pharmacy counter", "shelf", "carton pile",
        "sign", "poster", "shop", "store interior", "close-up", "photograph",
        "paolo", "monti", "servizio", "fotografico", "beic", "wikimedia",
        "collection", "globoid", "machine", "chemist", "fainting",
    )


@dataclass(frozen=True)
class DatabaseConfig:
    """SQLite via SQLAlchemy (Phase 2)."""

    url: str = f"sqlite:///{(PROJECT_ROOT / 'app.db').as_posix()}"
    echo: bool = False


@dataclass(frozen=True)
class AuthConfig:
    """JWT + password hashing for patients (python-jose/passlib when installed,
    stdlib HS256/PBKDF2 fallbacks otherwise — see auth/security.py)."""

    secret_key: str = "dev-only-secret-change-me-9f3ac2e17b5d4c08a6e1"
    algorithm: str = "HS256"
    token_expiry_minutes: int = 60 * 24  # 24h


@dataclass(frozen=True)
class DictionaryConfig:
    """Pharmacist-correction matching dictionary (correction_learning)."""

    matching_dictionary_path: Path = PROJECT_ROOT / "data" / "matching_dictionary.json"


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)


CONFIG = AppConfig()