"""Shared FastAPI dependencies: DB session, auth service, engines, pipeline."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from auth.security import build_password_hasher, build_token_codec
from auth.service import AuthService
from config import CONFIG, AppConfig
from core.engine import InteractionChecker
from core.scan_pipeline import ScanPipeline
from db.database import Database
from db.repositories import (
    CorrectionLogRepository,
    DrugRepository,
    IngredientRepository,
    InteractionRepository,
    MedicationRepository,
    ScanLogRepository,
    UserRepository,
)
from nlp.cleaner import TextCleaner
from nlp.matcher import build_matcher
from nlp.normalizer import CompoundDrugSplitter, IngredientNormalizer


@lru_cache
def get_default_config() -> AppConfig:
    return CONFIG


def get_config(request: Request) -> AppConfig:
    """Prefer the app-bound config (tests inject tmp paths) over the global default."""
    return getattr(request.app.state, "config", None) or CONFIG


@lru_cache
def get_database(request: Request) -> Database:
    """The app stores the Database instance on app.state at startup."""
    return request.app.state.database


def get_db_session(request: Request):
    database: Database = request.app.state.database
    session = database.session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_auth_service(request: Request, session: Session = Depends(get_db_session)) -> AuthService:
    return AuthService(
        UserRepository(session),
        request.app.state.password_hasher,
        request.app.state.token_codec,
        get_config(request).auth,
    )


def _bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    return header.removeprefix("Bearer ").strip()


def get_current_patient(
    request: Request,
    token: str = Depends(_bearer_token),
    auth_service: AuthService = Depends(get_auth_service),
):
    user = auth_service.resolve_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return user


def get_interaction_checker(
    request: Request,
    session: Session = Depends(get_db_session),
    config: AppConfig = Depends(get_config),
) -> InteractionChecker:
    ingredient_repo = IngredientRepository(session)
    matcher = build_matcher([ing.name for ing in ingredient_repo.all()])
    cleaner = TextCleaner(config.split)
    return InteractionChecker(
        ingredient_repo=ingredient_repo,
        interaction_repo=InteractionRepository(session),
        drug_repo=DrugRepository(session),
        normalizer=IngredientNormalizer(cleaner, matcher, ingredient_repo.aliases_map()),
        splitter=CompoundDrugSplitter(cleaner),
    )


def get_scan_pipeline(request: Request, session: Session = Depends(get_db_session)) -> ScanPipeline:
    return request.app.state.scan_pipeline_factory(session)


def get_correction_logger(request: Request, session: Session = Depends(get_db_session)):
    from correction_learning.logger import CorrectionLogger

    return CorrectionLogger(
        get_config(request).dictionary,
        CorrectionLogRepository(session),
        DrugRepository(session),
    )


__all__ = [
    "get_default_config",
    "get_config",
    "get_database",
    "get_db_session",
    "get_auth_service",
    "get_current_patient",
    "get_interaction_checker",
    "get_scan_pipeline",
    "get_correction_logger",
]