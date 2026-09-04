"""FastAPI application factory — wires routers, DB, auth, engines, and the scan pipeline."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.deps import get_default_config
from config import AppConfig

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("App startup: database ready")
    yield
    logger.info("App shutdown")


def create_app(config: AppConfig | None = None) -> FastAPI:
    from api.doctor_tool import router as doctor_router
    from api.patient import router as patient_router
    from api.scan import router as scan_router

    app = FastAPI(
        title="Drug Package Image Interaction Checker",
        description="Two-path drug recognition (OCR + transfer learning) with DDInter interaction checking.",
        version="0.2.0",
        lifespan=_lifespan,
    )
    cfg = config or get_default_config()

    from auth.security import build_password_hasher, build_token_codec
    from db.database import Database

    database = Database(cfg.database)
    database.create_schema()
    app.state.database = database
    app.state.config = cfg
    app.state.password_hasher = build_password_hasher()
    app.state.token_codec = build_token_codec(cfg.auth)
    app.state.scan_pipeline_factory = _make_pipeline_factory(cfg)

    app.include_router(patient_router)
    app.include_router(doctor_router)
    app.include_router(scan_router)
    return app


def _make_pipeline_factory(cfg: AppConfig):
    """Returns a function session -> ScanPipeline (predictor loaded once if checkpoint exists)."""
    from pathlib import Path

    from core.engine import InteractionChecker
    from db.repositories import DrugRepository, IngredientRepository, InteractionRepository
    from nlp.cleaner import TextCleaner
    from nlp.matcher import build_matcher
    from nlp.normalizer import CompoundDrugSplitter, IngredientNormalizer
    from ocr.reader import EasyOCRReader
    from vision.enhance import EnhancementPipeline
    from vision.quality import ImageQualityChecker

    predictor = None
    checkpoint = Path(cfg.data.models_dir) / "classifier.pt"
    if checkpoint.exists():
        try:
            from classification.predictor import ClassifierPredictor

            predictor = ClassifierPredictor.from_checkpoint(checkpoint, cfg.classifier)
            logger.info("Classifier checkpoint loaded: %s", checkpoint)
        except Exception as exc:  # noqa: BLE001 — scanning must survive a bad checkpoint
            logger.warning("Could not load classifier checkpoint: %s", exc)
    else:
        logger.info("No classifier checkpoint at %s — OCR-only fusion mode", checkpoint)

    def factory(session):
        from core.scan_pipeline import ScanPipeline

        ingredient_repo = IngredientRepository(session)
        drug_repo = DrugRepository(session)
        interaction_repo = InteractionRepository(session)
        matcher = build_matcher([ing.name for ing in ingredient_repo.all()])
        cleaner = TextCleaner(cfg.split)
        checker = InteractionChecker(
            ingredient_repo=ingredient_repo,
            interaction_repo=interaction_repo,
            drug_repo=drug_repo,
            normalizer=IngredientNormalizer(cleaner, matcher, ingredient_repo.aliases_map()),
            splitter=CompoundDrugSplitter(cleaner),
        )
        return ScanPipeline(
            config=cfg,
            ocr_reader=EasyOCRReader(cfg.ocr),
            predictor=predictor,
            matcher=matcher,
            ingredient_repo=ingredient_repo,
            drug_repo=drug_repo,
            interaction_repo=interaction_repo,
            checker=checker,
        )

    return factory


app = create_app()