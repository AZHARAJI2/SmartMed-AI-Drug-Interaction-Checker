# SmartMed — AI-Powered Drug Interaction Checker

## Project Documentation — Part 1: Overview & Summary

> **Project Title**: Drug Package Image Interaction Checker  
> **Date**: September 2026  
> **Platform**: Windows 10/11 · Python 3.12+  
> **Status**: Phases 1–3 Complete · Phase 4 (Hardening) In Progress

---

## Table of Contents

1. [Project Idea & Objectives](#1-project-idea--objectives)
2. [About Dataset & Data Analysis](#2-about-dataset--data-analysis)
3. [Data Preprocessing & Algorithms](#3-data-preprocessing--algorithms)
4. [Model Training & Results](#4-model-training--results)
5. [Application Development & Deployment](#5-application-development--deployment)

---

## 1. Project Idea & Objectives

### 1.1 Problem Statement

Drug-drug interactions (DDIs) are a leading cause of adverse drug events worldwide. Patients who take multiple medications simultaneously — a practice known as **polypharmacy** — often lack the knowledge to identify dangerous combinations. Pharmacists, while trained to spot these interactions, still need rapid, reliable tools to verify safety across hundreds of thousands of known interaction pairs.

**Key challenges addressed:**

| Challenge | Impact |
|---|---|
| Patients unaware of dangerous drug combinations | Hospitalization, organ damage, or death |
| Manual pharmacist verification is time-consuming | Delayed dispensing, human error under pressure |
| No existing system links *image recognition* to *interaction checking* | Gap between visual identification and safety verification |

### 1.2 Proposed Solution

An **end-to-end intelligent system** that:

1. **Recognizes drugs from package images** using two parallel AI paths (OCR + Deep Learning classifier), then fuses the results for higher accuracy.
2. **Checks drug-drug interactions** against the DDInter global database (160,000+ recorded interactions).
3. **Generates medical reports** in Arabic PDF format, classifying results as Safe 🟢, Caution 🟡, or Danger 🔴.
4. **Serves two user roles** — a Patient interface and a Pharmacist interface — each tailored to their needs.

### 1.3 Project Objectives

| # | Objective | Status |
|---|---|---|
| O1 | Build a computer vision pipeline that identifies drugs from package photos | ✅ Achieved |
| O2 | Implement a dual-path recognition system (OCR + image classification) with result fusion | ✅ Achieved |
| O3 | Integrate the DDInter drug interaction database (160,000+ interactions) | ✅ Achieved |
| O4 | Develop a REST API backend with user authentication | ✅ Achieved |
| O5 | Create interactive user interfaces for patients and pharmacists | ✅ Achieved |
| O6 | Enable pharmacist correction learning — the system improves over time | ✅ Achieved |
| O7 | Generate Arabic medical reports in PDF format | ✅ Achieved |
| O8 | Ensure offline-first design with graceful fallbacks | ✅ Achieved |

### 1.4 Target Users

- **Patients**: Individuals taking multiple medications who want to verify their safety.
- **Pharmacists**: Healthcare professionals who need a fast, reliable interaction-checking tool with image scanning capabilities.

---

## 2. About Dataset & Data Analysis

### 2.1 Data Sources

The system relies on **two distinct datasets**, each serving a different purpose:

#### Dataset 1: Yemen Drug Database (Drug Recognition Training)

| Attribute | Details |
|---|---|
| **Source** | Curated collection of drug package images |
| **File** | `yemen_drug_database/yemen_global_drugs_500.csv` |
| **Content** | Drug names, package image file references, metadata |
| **Images** | Drug package photographs stored in `yemen_drug_database/images/` |
| **Purpose** | Training the image classification model and building the OCR vocabulary |

#### Dataset 2: DDInter Database (Drug Interaction Reference)

| Attribute | Details |
|---|---|
| **Source** | [DDInter — Drug-Drug Interaction Database](http://ddinter.scbdd.com/) |
| **Files** | 8 CSV files in `ddinter_database/` |
| **Size** | 160,000+ recorded drug-drug interactions |
| **Content** | Active ingredient pairs, severity levels (Major/Moderate/Minor), clinical descriptions |
| **Purpose** | Reference database for real-time interaction checking |

### 2.2 Data Analysis Summary

**Yemen Drug Database Analysis:**

- Raw records were analyzed for label quality — many raw drug names contained noise (photographer credits, source metadata, junk keywords).
- A configurable list of **junk keywords** was established (e.g., "museum", "flickr", "wikimedia", "photograph") to filter non-drug entries.
- Classes with fewer than 2 samples were excluded from classifier training (configurable via `MIN_CLASS_IMAGES`).
- Drug names longer than 8 words were treated as noise and discarded.
- After cleaning, usable records were grouped into canonical drug classes using a text normalization pipeline.

**DDInter Database Analysis:**

- The 8 CSV files were parsed, deduplicated, and ingested into a relational SQLite database.
- Each interaction record maps two active ingredients to a severity level and a clinical description.
- Severity distribution covers Major, Moderate, and Minor interactions.
- The ingestion process is idempotent — re-running it does not create duplicates.

### 2.3 Data Split Strategy

| Split | Purpose | Selection Method |
|---|---|---|
| **Training** | Model parameter learning | Stratified random split (80%) per class |
| **Validation** | Performance monitoring & early stopping | Stratified random split (20%) per class |
| **Excluded** | Classes with insufficient samples | Automatic exclusion (< 2 images per class) |

- At least **one training sample** is guaranteed per class.
- A fixed random seed (`42`) ensures reproducible splits.
- Quality pre-screening (blur, brightness, resolution) flags images that may cause issues.

---

## 3. Data Preprocessing & Algorithms

### 3.1 Image Preprocessing Pipeline

The system applies a multi-stage preprocessing pipeline before any recognition attempt:

```
Input Image → Quality Gate → Enhancement → Recognition Paths
```

#### Stage 1: Quality Gate (Instant Rejection)

Images that fail any quality check are immediately rejected with a descriptive error message:

| Check | Metric | Threshold | Method |
|---|---|---|---|
| **Resolution** | Shortest side (pixels) | ≥ 128 px | Direct measurement |
| **Blur** | Laplacian variance | ≥ 45.0 | `cv2.Laplacian` variance |
| **Brightness** | Mean V-channel (HSV) | 35 – 235 | HSV conversion, V-channel mean |

#### Stage 2: Image Enhancement

Two enhancement techniques are applied sequentially:

| Technique | Purpose | Implementation |
|---|---|---|
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization on the L-channel | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=8×8)` |
| **Adaptive Sharpening** | Unsharp-mask sharpening calibrated to input blur level | Gaussian blur + weighted addition, strength scales inversely with sharpness |

### 3.2 Text Preprocessing (NLP Pipeline)

Drug names extracted by OCR undergo several cleaning steps:

1. **Text Cleaning** — Strip punctuation, normalize whitespace, lowercase.
2. **Junk Filtering** — Remove non-drug keywords (photographer names, source tags).
3. **Fuzzy Matching** — Match cleaned text against a known drug vocabulary using sequence similarity scoring.
4. **Ingredient Normalization** — Map trade names to active ingredients using aliases and known mappings.
5. **Compound Drug Splitting** — Split combined drugs (e.g., "Paracetamol + Ibuprofen") into individual components.

### 3.3 Algorithms Used

| Algorithm | Purpose | Component |
|---|---|---|
| **Transfer Learning** (MobileNetV3) | Drug package image classification | `classification/model.py` |
| **EasyOCR** (CRNN-based) | Text detection and recognition from images | `ocr/reader.py` |
| **CLAHE** | Adaptive contrast enhancement | `vision/enhance.py` |
| **Laplacian Variance** | Blur detection for quality gating | `vision/quality.py` |
| **Fuzzy String Matching** | Approximate drug name matching (RapidFuzz / difflib) | `nlp/matcher.py` |
| **Agreement Fusion** | Combining OCR + classifier results | `fusion/fusion.py` |
| **Severity Ranking** | Mapping interaction severity to risk levels | `core/engine.py` |

### 3.4 Dual-Path Recognition & Fusion Strategy

The system runs **two independent recognition paths** on every image:

```
                    ┌─────────────────────┐
                    │   Drug Package Image │
                    └──────────┬──────────┘
                               │
                       ┌───────┴───────┐
              ┌────────▼────┐   ┌──────▼───────┐
              │  OCR Path   │   │ Classifier   │
              │  (EasyOCR)  │   │ (MobileNetV3)│
              └────────┬────┘   └──────┬───────┘
                       └───────┬───────┘
                    ┌──────────▼──────────┐
                    │   Agreement Fusion   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
          "agree"         "uncertain"      "single_path"
        (high conf.)    (needs review)     (one path only)
```

**Fusion Rules:**

| Scenario | Result | Confidence |
|---|---|---|
| Both paths agree on the same drug | Use the agreed label | 0.95 (very high) |
| Only OCR produced a result | Use OCR label | 0.55 (moderate) |
| Only classifier produced a result | Use classifier label | 0.55 (moderate) |
| Both paths disagree | Use classifier label, flag as "uncertain" | 0.40 (low — pharmacist review needed) |
| Neither path produced a result | No signal | 0.00 |

---

## 4. Model Training & Results

### 4.1 Model Architecture

**Backbone**: MobileNetV3-Small (pretrained on ImageNet)

| Component | Details |
|---|---|
| **Base Model** | MobileNetV3-Small with ImageNet weights |
| **Training Strategy** | Transfer Learning — backbone fully **frozen**, only the classification head trains |
| **Classification Head** | Dropout (p=0.2) → Linear (features → num_classes) |
| **Optimizer** | Adam (lr=1e-3, weight_decay=1e-4) |
| **Loss Function** | CrossEntropyLoss |
| **Input Size** | 224 × 224 pixels |
| **Normalization** | ImageNet mean/std (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225) |

**Why MobileNetV3?**
- Lightweight architecture suitable for CPU inference.
- Strong ImageNet features transfer well to product/package recognition.
- Fast training (minutes, not hours) since only the head is trained.
- Supports both MobileNetV3-Small and ResNet18 backbones (configurable).

### 4.2 Data Augmentation

Applied only during training to reduce overfitting:

| Augmentation | Parameters |
|---|---|
| Color Jitter | brightness=0.2, contrast=0.2, saturation=0.2 |
| Random Horizontal Flip | probability=0.2 |
| Random Rotation | ±5 degrees |

Augmentation is intentionally lightweight because the backbone is frozen — heavy augmentation provides diminishing returns when only training a small head.

### 4.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Batch Size | 16 |
| Epochs | 15 |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Validation Fraction | 20% |
| Random Seed | 42 |
| GPU | Not required (CPU training) |

### 4.4 Evaluation Metrics

The system measures performance across all three paths:

| Metric | Description |
|---|---|
| **OCR Top-1 Accuracy** | Fraction of images where the best OCR text match equals the ground truth |
| **OCR Mean Match Score** | Average fuzzy matching score across all OCR attempts |
| **Classifier Top-1 Accuracy** | Fraction where the classifier's top prediction matches ground truth |
| **Classifier Top-3 Accuracy** | Fraction where ground truth appears in the classifier's top 3 predictions |
| **Fusion Accuracy** | Accuracy after combining both paths |
| **Fusion Agreement Rate** | Fraction of images where both paths agree |
| **Fusion Uncertainty Rate** | Fraction of images flagged as "uncertain" |
| **Fusion Value vs. OCR** | Accuracy improvement of fusion over OCR alone |
| **Fusion Value vs. Classifier** | Accuracy improvement of fusion over classifier alone |

### 4.5 Ablation Study

The system includes a built-in ablation study (`training/ablation.py`) that measures each component's contribution:

| Variant | Description |
|---|---|
| **Full Pipeline** | Enhancement + OCR + Classifier + Fusion |
| **No Enhancement** | Raw images (no CLAHE, no sharpening) → OCR + Classifier + Fusion |
| **OCR Only** | Captured as `ocr_top1` in the evaluation report |
| **Classifier Only** | Captured as `classifier_top1` in the evaluation report |

This systematically demonstrates the value added by each component.

---

## 5. Application Development & Deployment

### 5.1 System Architecture

The application follows a **three-tier architecture**:

```
┌─────────────────────────────────────────────┐
│              Presentation Layer              │
│         Streamlit (Patient + Pharmacist)     │
└──────────────────────┬──────────────────────┘
                       │ HTTP (httpx)
┌──────────────────────▼──────────────────────┐
│              Application Layer               │
│    FastAPI REST API + Swagger Documentation  │
│    Authentication (JWT) · Scan Pipeline      │
└──────────────────────┬──────────────────────┘
                       │ SQLAlchemy ORM
┌──────────────────────▼──────────────────────┐
│                Data Layer                    │
│   SQLite (drugs, ingredients, interactions,  │
│    users, medications, scan logs, corrections)│
└─────────────────────────────────────────────┘
```

### 5.2 Technology Stack

| Layer | Technology | Version |
|---|---|---|
| **Deep Learning** | PyTorch (Transfer Learning) | 2.11.0 |
| **Image Classification** | MobileNetV3-Small (torchvision) | 0.26.0 |
| **OCR** | EasyOCR | 1.7.2 |
| **Image Processing** | OpenCV (headless) | 4.8.1 |
| **Backend API** | FastAPI | 0.141.1 |
| **Database** | SQLAlchemy + SQLite | 2.0.36 |
| **Authentication** | JWT (HS256) + bcrypt/PBKDF2 | — |
| **Frontend UI** | Streamlit | 1.50.0 |
| **PDF Reports** | ReportLab + arabic-reshaper + python-bidi | 5.0.1 |
| **Data Validation** | Pydantic | 2.10.4 |
| **Testing** | pytest | 9.0.2 |
| **Text Matching** | RapidFuzz (with difflib fallback) | 3.13.0 |

### 5.3 User Interfaces

#### Patient Interface

| Feature | Description |
|---|---|
| Account Management | Register and login with JWT authentication |
| Add Medications | By name (free text), search, or camera/upload scan |
| Image Scanning | Capture or upload drug package photo → automatic recognition |
| Interaction Check | Automatic check when 2+ medications are in the patient's list |
| PDF Report | Download Arabic medical report with risk assessment |
| History | View past scan records |

#### Pharmacist Interface

| Feature | Description |
|---|---|
| Quick Check | Instant interaction check for multiple drugs (no login required) |
| Batch Input | Add drugs by name, search, or paste a list |
| Image Scan | Scan with visual explainability overlay |
| Correction Learning | Review and correct AI results → system learns from corrections |
| Recent Scans | View latest scan history |

### 5.4 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/patient/register` | Register a new patient account |
| `POST` | `/patient/login` | Patient login (returns JWT) |
| `POST` | `/patient/medications` | Add a medication to patient's list |
| `GET` | `/patient/medications` | List patient's medications |
| `DELETE` | `/patient/medications/{id}` | Remove a medication |
| `POST` | `/doctor/check` | Instant interaction check (drug names) |
| `POST` | `/doctor/scan` | Scan a drug package image (anonymous) |
| `GET` | `/doctor/drugs?q=...` | Search the drug database |
| `POST` | `/doctor/review` | Submit pharmacist correction |
| `POST` | `/scan/image` | Scan image (authenticated patient) |

### 5.5 Deployment

#### Local Deployment Steps

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the database (ingest DDInter data)
python scripts/ingest_ddinter.py
python scripts/seed_aliases.py

# 4. Start the backend API server
uvicorn main:app --reload
# Swagger docs: http://127.0.0.1:8000/docs

# 5. Start the frontend (separate terminal)
streamlit run ui/app.py
# UI: http://localhost:8501
```

#### Environment Configuration

| Variable | Default | Description |
|---|---|---|
| `DL_IM_API_URL` | `http://127.0.0.1:8000` | Backend API base URL |

### 5.6 Testing

The project includes a comprehensive automated test suite:

| Test File | Scope | Description |
|---|---|---|
| `test_phase1_vision.py` | Computer Vision | Quality gate, enhancement, OCR, classifier, fusion |
| `test_phase2_backend.py` | Backend | Database models, repositories, API endpoints, interaction engine |
| `test_phase3_ui.py` | User Interface | Streamlit pages via ASGI transport bridge |

**Test Characteristics:**
- All tests run **fully offline** (no internet required).
- Database tests use a **temporary SQLite** instance for isolation.
- Image tests use **synthetic images** (no real drug photos needed).
- OCR is disabled in tests for speed.
- UI tests drive the real FastAPI app through a sync ASGI transport.
- **85/85 tests passing** across all three phases.

---

> **Document**: DOCUMENTATION_PART1.md — High-Level Overview  
> **Companion**: See DOCUMENTATION_PART2.md for detailed technical deep-dive
