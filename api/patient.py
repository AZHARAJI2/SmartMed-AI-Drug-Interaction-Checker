"""Patient router — JWT-authenticated: register, login, medications, scan history."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_auth_service, get_current_patient, get_db_session
from api.schemas import (
    LoginRequest,
    MedicationOut,
    MedicationRequest,
    RegisterRequest,
    ScanHistoryItem,
    TokenResponse,
)
from auth.service import AuthService
from db.models import User
from db.repositories import DrugRepository, IngredientRepository, MedicationRepository, ScanLogRepository

router = APIRouter(prefix="/patient", tags=["patient"])


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, auth_service: AuthService = Depends(get_auth_service)) -> TokenResponse:
    try:
        user = auth_service.register(payload.name, payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TokenResponse(access_token=auth_service.create_token(user))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, auth_service: AuthService = Depends(get_auth_service)) -> TokenResponse:
    try:
        user, token = auth_service.authenticate(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    return TokenResponse(access_token=token)


def _resolve_or_create_drug(name: str, session: Session):
    """Try to resolve a drug by name/alias/ingredient. If not found, create a free-text placeholder Drug.
    Returns (drug, in_db) — in_db=False means it wasn't in the DDInter reference.
    """
    drug_repo = DrugRepository(session)
    ing_repo = IngredientRepository(session)

    # 1. Direct drug name match (case-insensitive)
    drug = drug_repo.get_by_name(name)
    if drug is not None:
        return drug, True

    # 2. Match via ingredient name or alias
    target_ing = ing_repo.get_by_name(name)
    if target_ing is None:
        for ing in ing_repo.all():
            aliases = [a.lower() for a in json.loads(ing.aliases or "[]")]
            if name.lower() == ing.name.lower() or name.lower() in aliases:
                target_ing = ing
                break

    if target_ing is not None:
        for d in drug_repo.all():
            if target_ing.id in drug_repo.ingredient_ids_of(d):
                return d, True
        # Ingredient exists but no drug entry — create one
        drug = drug_repo.add(trade_name=name.capitalize(), ingredient_ids=[target_ing.id])
        return drug, True

    # 3. Not found at all — create a free-text placeholder (no ingredient link)
    drug = drug_repo.add(trade_name=name.capitalize(), ingredient_ids=[])
    return drug, False


@router.post("/medications", response_model=MedicationOut, status_code=201)
def add_medication(
    payload: MedicationRequest,
    patient: User = Depends(get_current_patient),
    session: Session = Depends(get_db_session),
) -> MedicationOut:
    name = payload.drug_name.strip()
    drug, _in_db = _resolve_or_create_drug(name, session)
    MedicationRepository(session).add(patient.id, drug.id)
    return MedicationOut(drug_id=drug.id, drug_name=drug.trade_name)


@router.get("/medications", response_model=list[MedicationOut])
def list_medications(
    patient: User = Depends(get_current_patient),
    session: Session = Depends(get_db_session),
) -> list[MedicationOut]:
    drug_repo = DrugRepository(session)
    out = []
    for record in MedicationRepository(session).list_for_patient(patient.id):
        drug = drug_repo.get(record.drug_id)
        if drug is not None:
            out.append(MedicationOut(drug_id=drug.id, drug_name=drug.trade_name))
    return out


@router.delete("/medications/{drug_id}", status_code=204)
def remove_medication(
    drug_id: int,
    patient: User = Depends(get_current_patient),
    session: Session = Depends(get_db_session),
) -> None:
    removed = MedicationRepository(session).remove(patient.id, drug_id)
    if not removed:
        raise HTTPException(status_code=404, detail="medication not in list")


@router.get("/scans", response_model=list[ScanHistoryItem])
def scan_history(
    patient: User = Depends(get_current_patient),
    session: Session = Depends(get_db_session),
) -> list[ScanHistoryItem]:
    return [
        ScanHistoryItem(
            scan_id=entry.id,
            image_path=entry.image_path,
            ocr_raw_text=entry.ocr_raw_text,
            classifier_prediction=entry.classifier_prediction,
            fused_confidence=entry.fused_confidence,
            matched_drug_id=entry.matched_drug_id,
            review_status=entry.review_status,
            reviewed_by=entry.reviewed_by,
        )
        for entry in ScanLogRepository(session).list_for_patient(patient.id)
    ]