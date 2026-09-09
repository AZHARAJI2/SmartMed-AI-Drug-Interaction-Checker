# SmartMed — AI-Powered Drug Interaction Checker

## Project Documentation — Part 2: Technical Deep-Dive

> **Project Title**: Drug Package Image Interaction Checker  
> **Date**: September 2026  
> **Platform**: Windows 10/11 · Python 3.12+  
> **Architecture**: Domain-Driven Design · Strategy Pattern · Repository Pattern

---

## Table of Contents

1. [Project Idea & Objectives](#1-project-idea--objectives)
2. [About Dataset & Data Analysis](#2-about-dataset--data-analysis)
3. [Data Preprocessing & Algorithms](#3-data-preprocessing--algorithms)
4. [Model Training & Results](#4-model-training--results)
5. [Application Development & Deployment](#5-application-development--deployment)

---

## 1. Project Idea & Objectives

### 1.1 Technical Problem Definition

The project solves a **multi-modal recognition + knowledge-graph lookup** problem. Given an input image of a drug package, the system must:

1. **Extract** the drug identity from visual features (two parallel paths).
2. **Resolve** the identified drug to its active pharmaceutical ingredients.
3. **Query** a large interaction database (160K+ pairs) for all pairwise combinations.
4. **Classify** the overall risk and present actionable results to the end user.

This requires bridging computer vision, natural language processing, and database engineering into a single production-ready pipeline.

### 1.2 Architectural Objectives & Design Principles

The system was designed around five core engineering principles:

| Principle | Implementation | Example |
|---|---|---|
| **Domain-Driven Design** | Each module has a single responsibility in its own directory | `vision/`, `ocr/`, `classification/`, `fusion/`, `nlp/`, `core/`, `db/`, `api/`, `auth/`, `ui/` |
| **Abstract Base Classes (ABC)** | Interchangeable strategies defined by interfaces | `FusionStrategy(ABC)`, `ImageEnhancer(ABC)`, `OCRReader(ABC)`, `FuzzyMatcher(ABC)`, `QualityCheck(ABC)` |
| **Strategy Pattern** | Runtime-swappable implementations | `RapidFuzzMatcher` vs. `DifflibMatcher`; `CLAHEEnhancer` vs. `AdaptiveSharpener` |
| **Repository Pattern** | All database access through repository classes | 7 repositories in `db/repositories.py` for Drugs, Ingredients, Interactions, Users, etc. |
| **Offline-First / Fallback Design** | Every external library has a stdlib alternative | `bcrypt` → `PBKDF2-SHA256`; `python-jose` → manual HS256; `RapidFuzz` → `difflib` |

### 1.3 Complete Module Map

```
DL_IM_Project/
│
├── vision/                      # Computer Vision
│   ├── quality.py               # ImageQualityChecker — quality gate (resolution, blur, brightness)
│   ├── enhance.py               # ImageEnhancer(ABC) → CLAHEEnhancer, AdaptiveSharpener
│   └── explainability.py        # VisualExplainer — annotated overlay for scan results
│
├── ocr/                         # Optical Character Recognition
│   └── reader.py                # OCRReader(ABC) → EasyOCRReader (lazy loading)
│
├── classification/              # Deep Learning Image Classification
│   ├── dataset.py               # DrugImageDataset — PyTorch Dataset with augmentation
│   ├── model.py                 # TransferLearningClassifier (MobileNetV3/ResNet18 + head)
│   └── predictor.py             # Predictor — inference from a trained checkpoint
│
├── fusion/                      # Result Fusion
│   └── fusion.py                # FusionStrategy(ABC) → AgreementFusion
│
├── training/                    # Training & Evaluation
│   ├── organize_data.py         # DataOrganizer — label cleaning, manifest building, train/val split
│   ├── train_classifier.py      # Trainer — frozen-backbone training loop
│   ├── evaluate.py              # ScanEvaluator — full pipeline evaluation with metrics
│   └── ablation.py              # AblationStudy — component contribution measurement
│
├── nlp/                         # Natural Language Processing
│   ├── cleaner.py               # TextCleaner — OCR text normalization
│   ├── matcher.py               # FuzzyMatcher(ABC) → RapidFuzzMatcher / DifflibMatcher
│   └── normalizer.py            # IngredientNormalizer, CompoundDrugSplitter
│
├── core/                        # Core Business Logic
│   ├── engine.py                # InteractionChecker — safe/caution/danger classification
│   ├── ddinter_loader.py        # DDInterIngestor — CSV → SQLite ingestion
│   └── scan_pipeline.py         # ScanPipeline — end-to-end image → result pipeline
│
├── db/                          # Data Access Layer
│   ├── database.py              # SQLAlchemy engine + session management
│   ├── models.py                # ORM models: Ingredient, Drug, Interaction, User, etc.
│   └── repositories.py          # 7 Repository classes (Drug, Ingredient, Interaction, etc.)
│
├── api/                         # REST API Layer
│   ├── app.py                   # FastAPI application factory + Swagger
│   ├── deps.py                  # Dependency injection (config, DB session, auth)
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── patient.py               # Patient endpoints (register, login, medications, scan)
│   ├── doctor_tool.py           # Pharmacist endpoints (check, scan, search, review)
│   └── scan.py                  # Shared image scan endpoint
│
├── auth/                        # Authentication & Security
│   ├── security.py              # PasswordHasher + TokenCodec (ABC with fallbacks)
│   └── service.py               # AuthService — register/login/JWT token management
│
├── ui/                          # User Interface
│   ├── app.py                   # Streamlit entry point + role switcher
│   ├── patient_page.py          # Full patient interface
│   ├── doctor_page.py           # Full pharmacist interface
│   ├── widgets.py               # Shared UI components (reports, status badges, tables)
│   ├── api_client.py            # HTTP client for UI → API communication
│   └── pdf_report.py            # Arabic medical PDF report generator
│
├── correction_learning/         # Adaptive Learning
│   └── logger.py                # CorrectionLogger + matching_dictionary.json
│
├── scripts/                     # Utility Scripts
│   ├── ingest_ddinter.py        # Ingest DDInter CSVs into SQLite
│   └── seed_aliases.py          # Seed common ingredient aliases
│
├── tests/                       # Test Suite
│   ├── conftest.py              # Shared fixtures (temp DB, synthetic images)
│   ├── test_phase1_vision.py    # Vision & training pipeline tests
│   ├── test_phase2_backend.py   # Backend, DB, API, engine tests
│   └── test_phase3_ui.py        # UI integration tests via ASGI transport
│
├── config.py                    # Centralized configuration (all dataclasses)
├── main.py                      # Entry point (uvicorn)
└── requirements.txt             # Pinned dependencies
```

---

## 2. About Dataset & Data Analysis

### 2.1 Yemen Drug Database — Detailed Structure

**Source File**: `yemen_drug_database/yemen_global_drugs_500.csv`

This CSV file contains drug metadata with the following columns used by the system:

| Column | Type | Description |
|---|---|---|
| `drug_name_raw` | string | Raw drug name as scraped (often noisy) |
| `package_image_file` | string | Relative path to the corresponding package image |

**Data Quality Issues Discovered:**

1. **Noisy Labels**: Many raw names contained non-drug text such as photographer credits (e.g., "Paolo Monti servizio fotografico"), Wikimedia metadata, museum tags, and generic descriptions.

2. **Inconsistent Naming**: The same drug appeared under different naming conventions (e.g., variations in capitalization, extra words, pharmaceutical form embedded in the name).

3. **Missing or Unreadable Images**: Some image paths pointed to corrupted or missing files.

4. **Class Imbalance**: Some drug classes had only 1 image, making them unsuitable for supervised classification training.

### 2.2 DDInter Database — Detailed Structure

**Source**: [DDInter (Drug-Drug Interaction Database)](http://ddinter.scbdd.com/)  
**Location**: `ddinter_database/` — 8 CSV files

Each CSV file contains interaction records with fields including:
- Active ingredient names (both sides of the interaction pair)
- Severity classification: **Major**, **Moderate**, or **Minor**
- Clinical description of the interaction effect
- Source attribution

**Database Schema After Ingestion:**

```sql
-- Active pharmaceutical ingredients
CREATE TABLE ingredients (
    id          INTEGER PRIMARY KEY,
    name        VARCHAR(255) UNIQUE NOT NULL,  -- Scientific name
    aliases     TEXT DEFAULT '',                -- JSON list of alternative names
    drug_class  VARCHAR(255) DEFAULT ''        -- Pharmacological classification
);

-- Commercial drug products
CREATE TABLE drugs (
    id                    INTEGER PRIMARY KEY,
    trade_name            VARCHAR(255) UNIQUE NOT NULL,
    active_ingredient_ids TEXT DEFAULT '',      -- JSON list of ingredient IDs
    strength              VARCHAR(255) DEFAULT '',
    form                  VARCHAR(255) DEFAULT '',
    image_ref             VARCHAR(512) DEFAULT ''
);

-- Drug-drug interactions (160,000+ records)
CREATE TABLE interactions (
    id               INTEGER PRIMARY KEY,
    ingredient_a_id  INTEGER REFERENCES ingredients(id),
    ingredient_b_id  INTEGER REFERENCES ingredients(id),
    severity         VARCHAR(32) NOT NULL,      -- Major | Moderate | Minor
    description      TEXT DEFAULT '',
    source           VARCHAR(255) DEFAULT 'DDInter'
);

-- Patient accounts
CREATE TABLE users (
    id            INTEGER PRIMARY KEY,
    name          VARCHAR(255) NOT NULL,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(512) NOT NULL
);

-- Patient medication lists
CREATE TABLE patient_medications (
    id         INTEGER PRIMARY KEY,
    patient_id INTEGER REFERENCES users(id),
    drug_id    INTEGER REFERENCES drugs(id)
);

-- Image scan audit trail
CREATE TABLE scan_log (
    id                     INTEGER PRIMARY KEY,
    patient_id             INTEGER REFERENCES users(id),
    image_path             VARCHAR(512),
    ocr_raw_text           TEXT,
    classifier_prediction  VARCHAR(255),
    fused_confidence       FLOAT,
    matched_drug_id        INTEGER REFERENCES drugs(id),
    review_status          VARCHAR(32) DEFAULT 'auto',
    reviewed_by            VARCHAR(255) DEFAULT '',
    timestamp              DATETIME
);

-- Pharmacist correction history
CREATE TABLE correction_log (
    id                INTEGER PRIMARY KEY,
    raw_ocr_text      TEXT NOT NULL,
    corrected_drug_id INTEGER REFERENCES drugs(id),
    corrected_by      VARCHAR(255) DEFAULT 'pharmacist',
    timestamp         DATETIME
);
```

### 2.3 Data Organization Pipeline

The `DataOrganizer` class (`training/organize_data.py`) performs a 5-step data curation process:

**Step 1 — Label Cleaning & Junk Filtering:**
```python
# Configurable junk keywords for filtering non-drug entries
junk_keywords = (
    "museum", "flickr", "seal", "pharmacy counter", "shelf",
    "carton pile", "sign", "poster", "shop", "store interior",
    "close-up", "photograph", "paolo", "monti", "servizio",
    "fotografico", "beic", "wikimedia", "collection", "globoid",
    "machine", "chemist", "fainting",
)
```
Each raw label is cleaned via `TextCleaner.clean()`, then tested with `is_plausible_drug_name()`. Labels matching any junk keyword or exceeding 8 words are discarded.

**Step 2 — Canonical Class Grouping:**
Cleaned labels are normalized to canonical forms using `TextCleaner.canonical()` (lowercase, stripped, deterministic). This ensures that "Amoxicillin 500mg" and "amoxicillin" map to the same class.

**Step 3 — Stratified Train/Validation Split:**
- Each class is independently shuffled (seeded for reproducibility).
- 20% of samples per class go to validation, 80% to training.
- A minimum of 1 training sample is guaranteed per class.
- Classes with fewer than `min_class_images` (default: 2) samples are excluded entirely.

**Step 4 — Quality Pre-Screening:**
Every image is run through the quality gate (resolution, blur, brightness checks) and flagged in the manifest. This catches unreadable images before training.

**Step 5 — Manifest Generation:**
The final manifest (`data/manifest.csv`) contains:

| Column | Description |
|---|---|
| `image_path` | Absolute path to the image file |
| `raw_label` | Original drug name from the CSV |
| `clean_label` | Cleaned drug name after normalization |
| `class_name` | Canonical class name for grouping |
| `class_index` | Integer index for the classifier |
| `split` | `train`, `val`, or `excluded` |
| `usable_for_training` | Boolean flag |
| `quality_pass` | Boolean flag from quality pre-screen |

---

## 3. Data Preprocessing & Algorithms

### 3.1 Image Quality Gate — Implementation Details

The quality gate is implemented as a **composite of strategy objects** (`vision/quality.py`):

```python
class QualityCheck(ABC):
    """Strategy interface for a single quality check."""
    
    @abstractmethod
    def metric(self, image_bgr: np.ndarray) -> float:
        """Compute the check's scalar metric."""

    @abstractmethod
    def evaluate(self, metric: float) -> str | None:
        """Return a failure reason, or None when the check passes."""
```

**Three concrete checks are registered by default:**

| Check Class | Algorithm | Pass Condition |
|---|---|---|
| `ResolutionCheck` | `min(height, width)` | ≥ 128 pixels |
| `BlurCheck` | `cv2.Laplacian(gray, CV_64F).var()` | ≥ 45.0 (higher = sharper) |
| `BrightnessCheck` | `hsv[:,:,2].mean()` (V-channel) | Between 35.0 and 235.0 |

The `ImageQualityChecker` runs all checks and produces a `QualityReport` with metrics and failure reasons. An image is rejected if **any** check fails.

### 3.2 Image Enhancement — Implementation Details

Enhancement follows the **Strategy Pattern** with an ordered composite pipeline:

**CLAHE Enhancer** (`CLAHEEnhancer`):
```
BGR → LAB → Apply CLAHE to L-channel → LAB → BGR
```
- Clip limit: 2.0 (prevents over-amplification of noise)
- Tile grid: 8×8 (controls local region size)

**Adaptive Sharpener** (`AdaptiveSharpener`):
```
1. Compute sharpness score: Laplacian variance of the grayscale image
2. Scale factor = min(2.0, max(0.5, 60.0 / max(sharpness_score, 1.0)))
3. Amount = base_amount × scale factor
4. Result = image × (1 + amount) - blurred × amount
```
This means blurrier images receive stronger sharpening — the system adapts to input quality.

### 3.3 OCR Path — Implementation Details

**EasyOCR Reader** (`ocr/reader.py`):
- Languages: English (`en`)
- GPU: Disabled by default (CPU-only environment)
- Confidence threshold: 0.35 per text block (low blocks are discarded)
- **Lazy loading**: The OCR model is only loaded on first use, avoiding startup delays.
- **Disable flag**: OCR can be completely disabled via `OCRConfig.enabled = False` (used in tests).

**OCR Post-Processing:**
1. Raw OCR texts are cleaned via `TextCleaner`.
2. Each cleaned text is matched against the drug vocabulary using `FuzzyMatcher.best_match()`.
3. The text with the **highest matching score** is selected as the OCR label.
4. If a **learned correction** exists in the matching dictionary, it takes priority over fuzzy matching.

### 3.4 Classification Path — Implementation Details

**Model Architecture** (`classification/model.py`):

```python
class TransferLearningClassifier(nn.Module):
    def __init__(self, num_classes, config):
        # Load pretrained backbone (MobileNetV3-Small or ResNet18)
        self.backbone = mobilenet_v3_small(weights=IMAGENET1K_V1)
        
        # Freeze ALL backbone parameters (no gradients)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace classifier with new head
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes),
        )
```

**Offline Fallback**: If pretrained weights cannot be downloaded (network blocked), the model falls back to random-initialized frozen features. This ensures the pipeline never crashes due to missing weights — it simply produces lower-quality features.

**Dataset** (`classification/dataset.py`):
- Extends `torch.utils.data.Dataset`.
- Applies `torchvision.transforms` with ImageNet normalization.
- Handles corrupt images by substituting a black frame.
- Training transforms include color jitter, horizontal flip, and rotation.

### 3.5 Fusion Algorithm — Implementation Details

**AgreementFusion** (`fusion/fusion.py`) implements a three-tier decision process:

```
Input: (ocr_label, ocr_confidence, classifier_label, classifier_confidence)

IF both labels are non-empty:
    Compute fuzzy similarity score between ocr_label and classifier_label
    IF score ≥ 70.0 (match threshold):
        → AGREE: label = classifier_label, confidence = max(0.95, classifier_confidence)
    ELSE:
        → UNCERTAIN: label = classifier_label, confidence = 0.40
ELIF only ocr_label is non-empty:
    → OCR_ONLY: label = ocr_label, confidence = 0.55
ELIF only classifier_label is non-empty:
    → CLASSIFIER_ONLY: label = classifier_label, confidence = 0.55
ELSE:
    → NO_SIGNAL: label = "", confidence = 0.00
```

The fuzzy similarity uses the same `FuzzyMatcher` infrastructure, so it benefits from the same RapidFuzz/difflib fallback mechanism.

### 3.6 Drug Name Resolution — Full Pipeline

The NLP pipeline resolves free-text drug names to database entries through multiple stages:

```
Raw Name (e.g., "Augmentin 625mg")
    │
    ├── CompoundDrugSplitter
    │   → ["amoxicillin", "clavulanic acid"]
    │
    ├── For each component:
    │   ├── IngredientNormalizer.normalize()
    │   │   └── TextCleaner → FuzzyMatcher → alias lookup
    │   │
    │   ├── IngredientRepository.get_by_name()
    │   │   └── Exact match in the ingredients table
    │   │
    │   └── DrugRepository.get_by_name() (fallback)
    │       └── Match by trade name → extract ingredient IDs
    │
    └── Result: list of resolved ingredient IDs
```

### 3.7 Interaction Checking Engine — Implementation Details

**InteractionChecker** (`core/engine.py`) implements a severity-ranked risk assessment:

```python
SEVERITY_RANK = {"minor": 1, "moderate": 2, "major": 3}
STATUS_BY_TOP_SEVERITY = {0: "safe", 1: "caution", 2: "caution", 3: "danger"}
```

**Algorithm:**
1. Receive a list of drug names.
2. Split compound drugs → resolve each to ingredient IDs.
3. Query the `interactions` table for all pairwise combinations of resolved ingredients.
4. Rank each interaction's severity (Minor=1, Moderate=2, Major=3).
5. Overall status = highest severity found:
   - Rank 0 → **Safe** 🟢
   - Rank 1–2 → **Caution** 🟡
   - Rank 3 → **Danger** 🔴
6. If unmatched names exist and 2+ drugs are being compared → escalate to **Caution**.

**Important Design Decision**: A single unmatched drug does NOT escalate to Caution — this prevents misleading warnings when a user simply enters a drug not in the database. Only when comparing multiple drugs does an unresolved name trigger the caution flag.

---

## 4. Model Training & Results

### 4.1 Training Loop — Implementation Details

The `Trainer` class (`training/train_classifier.py`) implements a standard PyTorch training loop with the following specifics:

```python
class Trainer:
    def train(self) -> TrainResult:
        model = TransferLearningClassifier(num_classes, config)
        optimizer = Adam(model.trainable_parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = CrossEntropyLoss()
        
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            for images, labels in train_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()         # Only head parameters receive gradients
                optimizer.step()
            
            # Validation phase
            model.eval()
            # ... compute val_loss and val_acc
            
        return TrainResult(history, best_val_acc)
```

**Key Implementation Details:**

| Aspect | Detail |
|---|---|
| **Reproducibility** | `seed_everything(42)` seeds Python random, NumPy, and PyTorch |
| **Trainable params** | Only `model.head` (Dropout + Linear) receives gradients |
| **Backbone params** | All frozen — `param.requires_grad = False` |
| **DataLoader** | `num_workers=0` for Windows compatibility |
| **Metric tracking** | Per-epoch: train_loss, train_acc, val_loss, val_acc |

### 4.2 Evaluation System — Implementation Details

The `ScanEvaluator` class (`training/evaluate.py`) runs the complete Phase 1 pipeline over validation data:

**Evaluation Flow Per Image:**
```
1. Read image from disk (cv2.imread)
2. Quality gate check (skip if rejected)
3. Enhancement (CLAHE + sharpening)
4. OCR path → best fuzzy match → ocr_label, ocr_confidence
5. Classifier path → prediction.label, prediction.confidence
6. Agreement fusion → fused_label, fused_confidence, status
7. Compare all results against ground truth label
8. Optionally save annotated sample images
```

**Metrics Computed:**

```python
@dataclass
class EvaluationReport:
    n_images: int              # Total images evaluated
    rejected_by_quality: int   # Failed quality gate
    ocr_top1_acc: float        # OCR path accuracy
    ocr_mean_match: float      # Average fuzzy match score
    classifier_top1_acc: float # Classifier top-1 accuracy
    classifier_top3_acc: float # Classifier top-3 accuracy
    fusion_acc: float          # Fusion accuracy
    fusion_agree_rate: float   # Both paths agreed
    fusion_uncertain_rate: float  # Paths disagreed
    fusion_value_vs_ocr: float    # fusion_acc - ocr_top1_acc
    fusion_value_vs_classifier: float  # fusion_acc - classifier_top1_acc
```

The `fusion_value_vs_*` metrics directly quantify the **added value** of running two paths instead of one.

### 4.3 Ablation Study — Implementation Details

The `AblationStudy` class (`training/ablation.py`) uses a **component-removal methodology**:

| Experiment | What Changes | What Stays |
|---|---|---|
| **Full Pipeline** | Nothing removed | Enhancement + OCR + Classifier + Fusion |
| **No Enhancement** | `_NullEnhancer` replaces CLAHE + Sharpening | OCR + Classifier + Fusion |
| **OCR Only** | (Measured from evaluation report `ocr_top1`) | OCR path only |
| **Classifier Only** | (Measured from evaluation report `classifier_top1`) | Classifier path only |

The `_NullEnhancer` is an identity function that returns the input image unchanged:
```python
class _NullEnhancer(ImageEnhancer):
    name = "none"
    def apply(self, image_bgr):
        return image_bgr
```

This enables measuring the exact contribution of the enhancement stage by comparing the "full" and "no_enhancement" variants.

### 4.4 Saved Artifacts

After training and evaluation, the following artifacts are produced:

| Artifact | Location | Description |
|---|---|---|
| `classifier.pt` | `outputs/models/` | Trained model checkpoint (state dict + metadata) |
| `manifest.csv` | `data/` | Complete data manifest with splits and quality flags |
| `sample_*.jpg` | `outputs/samples/` | Annotated sample images from evaluation |
| Evaluation report | `outputs/reports/` | Metrics summary (text) |
| Ablation report | `outputs/reports/` | Component contribution table |

---

## 5. Application Development & Deployment

### 5.1 Backend Architecture — FastAPI Application Factory

The backend uses a **factory pattern** (`api/app.py`) to construct the FastAPI application:

```python
def create_app(config: AppConfig | None = None) -> FastAPI:
    app = FastAPI(
        title="Drug Package Image Interaction Checker",
        description="Two-path drug recognition with DDInter interaction checking.",
        version="0.2.0",
    )
    
    # Wire up all components
    app.state.database = Database(cfg.database)
    app.state.config = cfg
    app.state.password_hasher = build_password_hasher()      # bcrypt or PBKDF2
    app.state.token_codec = build_token_codec(cfg.auth)      # jose or manual HS256
    app.state.scan_pipeline_factory = _make_pipeline_factory(cfg)
    
    # Register routers
    app.include_router(patient_router)    # /patient/*
    app.include_router(doctor_router)     # /doctor/*
    app.include_router(scan_router)       # /scan/*
    
    return app
```

**Dependency Injection** (`api/deps.py`):
- Database sessions are injected per-request via FastAPI's `Depends()`.
- Configuration is accessible via `app.state.config`.
- The scan pipeline is created per-request from a factory that keeps the classifier loaded in memory.

### 5.2 Authentication System — Implementation Details

The authentication system (`auth/`) is designed with **progressive enhancement**:

**Password Hashing:**
```
Priority 1: passlib + bcrypt (industry standard)
Priority 2: PBKDF2-HMAC-SHA256 (stdlib fallback)
```

Both produce a hash string that is stored in the `users.password_hash` column. The `build_password_hasher()` function auto-detects which library is available.

**JWT Token Encoding:**
```
Priority 1: python-jose (JOSE standard library)
Priority 2: Manual HS256 implementation (stdlib: hmac + hashlib + base64)
```

Both produce the same wire format (standard JWT with HS256). Token expiry is set to 24 hours by default.

**API Authentication Flow:**
```
1. POST /patient/register → Create user + hash password → Return success
2. POST /patient/login → Verify password hash → Generate JWT → Return token
3. Protected endpoints → Extract JWT from Authorization header → Decode + verify → Inject user_id
```

### 5.3 Scan Pipeline — End-to-End Implementation

The `ScanPipeline` class (`core/scan_pipeline.py`) orchestrates the complete scanning process:

```
Step 1: Quality Gate
    → Reject poor images with descriptive error message

Step 2: Enhancement
    → CLAHE (contrast) + Adaptive Sharpening (text crispness)

Step 3: OCR Path
    → EasyOCR read → TextCleaner → Dictionary lookup → Fuzzy matching

Step 4: Classifier Path
    → MobileNetV3 predict → Junk filtering → label + confidence

Step 5: Fusion
    → AgreementFusion → fused_label + confidence + status

Step 6: Drug Matching
    → Dictionary lookup → Exact canonical match → Fuzzy match against DB

Step 7: Interaction Check (if patient has 2+ drugs)
    → InteractionChecker → safe/caution/danger + findings

Step 8: Visual Explanation
    → Annotated image with OCR regions + labels + fusion status

Step 9: Audit Logging
    → ScanLogRepository → persist scan record to scan_log table

Step 10: Return ScanOutcome
    → accepted, ocr_text, classifier_label, fusion, matched_drug, report, annotated_image
```

**Pharmacist Correction Learning:**
When a pharmacist reviews and corrects a scan result via the `/doctor/review` endpoint:
1. The correction is logged in the `correction_log` table.
2. The raw OCR text → correct drug name mapping is saved to `matching_dictionary.json`.
3. On subsequent scans, the dictionary is checked **before** fuzzy matching — learned corrections take priority.
4. This creates a self-improving feedback loop where the system gets more accurate with use.

### 5.4 Frontend Implementation — Streamlit

**Application Entry** (`ui/app.py`):
- Configures RTL layout for Arabic text support.
- Provides a role switcher in the sidebar (Patient / Pharmacist).
- The API base URL is configurable via the sidebar.

**Patient Page** (`ui/patient_page.py`):
- JWT-based session management stored in `st.session_state`.
- Camera input and file upload for drug package scanning.
- Automatic interaction check triggered when 2+ medications are in the list.
- PDF report generation via `pdf_report.py`.

**Pharmacist Page** (`ui/doctor_page.py`):
- Session-state-based drug basket (`st.session_state.doctor_basket`).
- Unique element keys for each widget to prevent `StreamlitDuplicateElementKey` errors.
- Visual explainability overlay preview for scanned images.
- Correction submission that feeds back into the matching dictionary.

**API Client** (`ui/api_client.py`):
- Synchronous `httpx` client with configurable base URL.
- JWT token management for authenticated requests.
- `ScanView` dataclass for structured scan result handling.

### 5.5 PDF Report Generation — Arabic RTL Support

The PDF medical report (`ui/pdf_report.py`) handles Arabic text through a three-library pipeline:

```
Arabic Text → arabic-reshaper (glyph reshaping) → python-bidi (RTL reordering) → ReportLab (PDF rendering)
```

**Features:**
- Full Arabic RTL layout.
- Arial Unicode MS font for Unicode coverage.
- Color-coded severity tables (green/yellow/red for safe/caution/danger).
- Medical disclaimer at the bottom of every report.

### 5.6 Deployment Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    User's Browser                             │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP (port 8501)
┌──────────────────────────▼───────────────────────────────────┐
│                 Streamlit Frontend                            │
│     ui/app.py → patient_page.py | doctor_page.py             │
│                 ui/api_client.py                             │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP (port 8000)
┌──────────────────────────▼───────────────────────────────────┐
│                 FastAPI Backend                               │
│     main.py → api/app.py → patient.py | doctor_tool.py       │
│                 auth/service.py (JWT)                         │
│                 core/scan_pipeline.py                         │
│                 core/engine.py (InteractionChecker)           │
└──────────────────────────┬───────────────────────────────────┘
                           │ SQLAlchemy ORM
┌──────────────────────────▼───────────────────────────────────┐
│                 SQLite Database (app.db)                      │
│     7 tables: ingredients, drugs, interactions,               │
│               users, patient_medications, scan_log,           │
│               correction_log                                  │
└──────────────────────────────────────────────────────────────┘
```

### 5.7 Production Deployment Checklist

| Item | Current State | Production Recommendation |
|---|---|---|
| Secret Key | Hardcoded dev-only key in `config.py` | Use environment variable `SECRET_KEY` |
| Database | SQLite file (`app.db`) | Migrate to PostgreSQL for concurrent access |
| HTTPS | Not configured | Add TLS termination via reverse proxy (nginx) |
| OCR Model | Downloaded on first use | Pre-download weights during Docker build |
| Classifier | Loaded from checkpoint file | Include checkpoint in deployment artifact |
| Rate Limiting | Not implemented | Add via FastAPI middleware |
| Logging | Console output | Structured logging to a centralized service |
| CORS | Not configured | Add CORS middleware for cross-origin access |

### 5.8 Testing Architecture

**Test Infrastructure:**

```python
# conftest.py — shared fixtures
@pytest.fixture
def temp_db():
    """Creates an isolated, temporary SQLite database for each test."""
    # Uses a separate temp file, not the production app.db

@pytest.fixture
def synthetic_image():
    """Generates a test image without requiring real drug photos."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
```

**Test Coverage by Phase:**

| Phase | File | Tests | Key Coverage Areas |
|---|---|---|---|
| Phase 1 | `test_phase1_vision.py` | Quality gate, enhancement, OCR wrapper, classifier architecture, fusion logic, data organization |
| Phase 2 | `test_phase2_backend.py` | SQLAlchemy models, all 7 repositories, API endpoints, interaction engine, auth service |
| Phase 3 | `test_phase3_ui.py` | Streamlit pages via ASGI transport bridge, API client, end-to-end UI workflows |

**Test Design Principles:**
- **Offline**: No internet access required.
- **Isolated**: Temp SQLite databases, no shared state between tests.
- **Fast**: OCR disabled, synthetic images, ASGI transport (no real HTTP).
- **Total**: 85/85 tests passing across all phases.

### 5.9 Known Issues & Engineering Solutions

| # | Issue | Root Cause | Solution |
|---|---|---|---|
| 1 | RapidFuzz install failure | Network blocked (`WinError 10013`) | `FuzzyMatcher(ABC)` with `DifflibMatcher` fallback |
| 2 | Missing passlib/python-jose | Network blocked | `PasswordHasher`/`TokenCodec` ABCs with stdlib fallbacks |
| 3 | 404 when adding "Aspirin" | Only registered as ingredient, not as trade name | Multi-strategy lookup: drugs → ingredients → aliases → create temp record |
| 4 | Duplicate Streamlit keys | Non-unique widget keys in loops | Index + name + timestamp-based unique keys |
| 5 | Misleading warnings for single drug | Caution escalation for 1 unmatched drug | Only escalate when comparing 2+ drugs |
| 6 | OCR reading junk text | Photographer credits, metadata in images | Configurable junk keyword filtering at multiple pipeline stages |
| 7 | ASGI attribute error | `app` not defined at module level | Added `app = create_app()` at module scope |

---

> **Document**: DOCUMENTATION_PART2.md — Technical Deep-Dive  
> **Companion**: See DOCUMENTATION_PART1.md for high-level overview
