# PROJECT_MAP

## [TECH_STACK]
| Category | Technology | Version (installed) |
|---|---|---|
| Classical image processing | OpenCV, Pillow | cv2 4.13.0, PIL 10.1.0 |
| Deep Learning | PyTorch (transfer learning, frozen backbone) | torch 2.11.0+cpu, torchvision 0.26.0+cpu |
| OCR | EasyOCR | installed (lazy model load) |
| Text matching | RapidFuzz (+ stdlib difflib fallback — network blocked rapidfuzz install) | fallback active |
| Database | SQLite + SQLAlchemy | Phase 2 |
| Backend | FastAPI (+Swagger) | Phase 2 (0.141.1 present) |
| Validation | Pydantic | Phase 2 |
| Auth | python-jose (JWT) + passlib (bcrypt) — patients only | Phase 2 |
| Testing | pytest | 9.0.2 |
| Logging | logging (stdlib) | used everywhere |
| Initial UI | Streamlit | Phase 3 |
| Data/table tooling | pandas, scikit-learn (metrics/splits) | 2.3.1 / 1.7.0 |

## [SYSTEM_FLOW]
image → quality gate → enhancement → {OCR path, classifier path} → fusion → explainability overlay
→ (Phase 2: normalization → DDInter interaction check → result; corrections loop)

## [ARCHITECTURE] (domain-driven; every component is a class, ABCs for interchangeable strategies)
```
project_root/
├── vision/                ImageQualityChecker, ImageEnhancer(ABC)→CLAHEEnhancer/AdaptiveSharpener, VisualExplainer   [Phase 1 ✅]
├── ocr/                   OCRReader(ABC) → EasyOCRReader (lazy)                                                      [Phase 1 ✅]
├── classification/        DrugImageDataset, TransferLearningClassifier (frozen MobileNetV3/ResNet18 + new head), Predictor [Phase 1 ✅]
├── fusion/                FusionStrategy(ABC) → AgreementFusion (always both paths; disagree → uncertain)            [Phase 1 ✅]
├── training/              organize_data (label cleaning → manifest + split), Trainer, Evaluator, AblationStudy       [Phase 1 ✅]
├── nlp/                   TextCleaner, FuzzyMatcher(ABC) → RapidFuzzMatcher / DifflibMatcher                         [Phase 1 (minimal) ✅; full normalization Phase 2]
├── core/                  Interaction-checking engine (DDInter)                                                      [Phase 2 — Next Up after approval]
├── correction_learning/   pharmacist corrections → dictionary updates                                                [Phase 2]
├── api/                   FastAPI routers (patient, doctor_tool, scan)                                               [Phase 2]
├── db/                    SQLAlchemy models + repositories                                                           [Phase 2]
├── auth/                  JWT + bcrypt, patients only                                                                [Phase 2]
├── tests/                 pytest — mirrors the folders above                                                         [Phase 1 ✅]
├── config.py              all paths + hyperparameters                                                                [Phase 1 ✅]
├── requirements.txt       documented stable versions                                                                 [Phase 1 ✅]
├── agent_history.md / app_explanation.md / PROJECT_MAP.md (this file)                                                [Memory Trinity ✅]
├── data/                  manifest.csv (cleaned labels + train/val split) — generated                                [Phase 1 ✅]
└── outputs/               models, metrics, ablation results, annotated samples — generated                           [Phase 1 ✅]
```

## Phase status
- Phase 1 — Vision & Training: ✅ implemented (data organization, quality gate, enhancement,
  OCR wrapper, transfer-learning classifier, fusion, explainability, evaluation metrics, ablation, pytest).
- **Phase 2 — Backend & Interaction Engine: ✅ implemented** (SQLAlchemy models + repositories,
  DDInter ingestion, interaction-checking engine with safe/caution/danger, active-ingredient
  normalization + compound-drug splitting, JWT auth for patients with offline-safe hashing/token
  fallbacks, FastAPI routers for patient + doctor_tool + scan with Swagger, pharmacist correction
  logging + matching-dictionary learning).
- **Phase 3 — Interfaces (Streamlit): DONE (2026-09-03).**
  - `ui/app.py` role switcher, `ui/patient_page.py` (register/login JWT, medication add via drug search
    selectbox, scan upload + history), `ui/doctor_page.py` (anonymous text check, image scan with
    overlay preview, review/correction that teaches the matching dictionary), `ui/widgets.py` shared
    status/findings/explainer renderers, `ui/api_client.py` sync httpx client (session-scoped, base URL
    overridable via `DL_IM_API_URL`).
  - Tests: `tests/test_phase3_ui.py` — 7 tests driving the real FastAPI app through a sync ASGI
    transport bridge (offline, no server needed). Full suite: **85/85 passing**.
  - Live validation: API on :8000 + Streamlit on :8501 (docs 200, danger check on 160k DDInter
    interactions, medication roundtrip, drug search).
- Phase 4 — Hardening & Final Evaluation: pending.

### Phase 2 component map
```
db/         database.py (engine/sessions), models.py (Drugs, Ingredients, Interactions, Users,
            Patient_Medications, Scan_Log, Correction_Log), repositories.py (7 repository classes)
core/       engine.py (InteractionChecker → safe/caution/danger), ddinter_loader.py (DDInterIngestor),
            scan_pipeline.py (ScanPipeline: quality→enhance→OCR+classifier→fusion→match→interactions→Scan_Log)
nlp/        normalizer.py (IngredientNormalizer, CompoundDrugSplitter)  [extends Phase 1 cleaner/matcher]
auth/       security.py (PasswordHasher/TokenCodec ABCs: passlib/python-jose adapters + PBKDF2/HS256
            stdlib fallbacks), service.py (AuthService: register/login/JWT)
api/        schemas.py (Pydantic), deps.py, patient.py, doctor_tool.py, scan.py, app.py (create_app)
correction_learning/  logger.py (CorrectionLogger + matching_dictionary.json learning)
scripts/    ingest_ddinter.py  (CLI: build the reference DB from the 8 DDInter CSVs)
main.py     uvicorn entry — Swagger at http://127.0.0.1:8000/docs
```

### Running Phase 2
1. `python scripts/ingest_ddinter.py` — builds `app.db` from `ddinter_database/*.csv` (idempotent).
2. `python scripts/seed_aliases.py` — seeds common ingredient aliases (e.g. paracetamol→acetaminophen; idempotent, extend `ALIASES` as needed).
3. `uvicorn main:app --reload` (or `python main.py`) — API + Swagger docs.
4. `python -m pytest tests -q` — full offline test suite (temp SQLite, OCR disabled, synthetic images).

## [ORPHANS & PENDING]
1. `yemen_drug_database/Medicine_data/` — 437 images (`imagesN.jpg`) with **no label file**; cannot be used
   for supervised training. Needs a labels source or manual labeling.
2. `ddinter_database/*.csv` — staged for Phase 2 (core interaction engine + Ingredients/Interactions tables).
3. Label cleaning in `training/organize_data.py` is heuristic; classes below `MIN_CLASS_IMAGES` are excluded
   from classifier training and reported in the manifest stats (they remain usable by the OCR path).
4. `rapidfuzz` not installable offline → DifflibMatcher fallback used; install later to switch automatically.
5. EasyOCR model weights download on first real run requires network; tests never depend on it.