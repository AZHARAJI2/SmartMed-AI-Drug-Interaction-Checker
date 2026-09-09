"""Scan router — patient (JWT) image scan; persists to Scan_Log and returns the verdict."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.deps import get_current_patient, get_db_session, get_scan_pipeline
from api.doctor_tool import _outcome_out, cv2_imdecode
from api.schemas import ScanOutcomeOut
from core.scan_pipeline import ScanPipeline
from db.models import User

router = APIRouter(prefix="/scan", tags=["scan"])


@router.post("/image", response_model=ScanOutcomeOut) 
async def scan_image(
    image: UploadFile,
    patient: User = Depends(get_current_patient),
    pipeline: ScanPipeline = Depends(get_scan_pipeline),
    session: Session = Depends(get_db_session),
) -> ScanOutcomeOut:
    """Patient scans a package photo; result is stored in scan history."""
    data = await image.read()
    decoded = cv2_imdecode(data)
    if decoded is None:
        raise HTTPException(status_code=400, detail="unreadable image file")
    outcome = pipeline.process(decoded, patient_id=patient.id, image_path=image.filename or "")
    if outcome.accepted:
        candidate = outcome.matched_drug_name or (outcome.fusion.label if outcome.fusion else "") or outcome.ocr_label or outcome.classifier_label
        candidate = candidate.strip()
        if candidate:
            from api.patient import _resolve_or_create_drug
            from db.repositories import MedicationRepository

            drug, _ = _resolve_or_create_drug(candidate, session)
            MedicationRepository(session).add(patient.id, drug.id)
            session.commit()
    return _outcome_out(outcome)