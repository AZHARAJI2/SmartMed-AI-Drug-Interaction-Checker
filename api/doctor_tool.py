"""Doctor/pharmacist router — anonymous: text check, image scan + review/correction."""
from __future__ import annotations

import base64

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.deps import get_correction_logger, get_db_session, get_interaction_checker, get_scan_pipeline
from api.schemas import (
    DoctorCheckRequest,
    DoctorReviewRequest,
    DrugOut,
    InteractionFindingOut,
    InteractionReportOut,
    ScanHistoryItem,
    ScanOutcomeOut,
)
from core.engine import InteractionChecker, InteractionReport
from core.scan_pipeline import ScanPipeline
from db.repositories import DrugRepository, ScanLogRepository

router = APIRouter(prefix="/doctor", tags=["doctor_tool"])


def _report_out(report: InteractionReport) -> InteractionReportOut:
    return InteractionReportOut(
        status=report.status,
        findings=[
            InteractionFindingOut(
                drug_a=f.drug_a, drug_b=f.drug_b,
                ingredient_a=f.ingredient_a, ingredient_b=f.ingredient_b,
                severity=f.severity, description=f.description, source=f.source,
            )
            for f in report.findings
        ],
        resolved_ingredients=report.resolved_ingredients,
        unmatched_names=report.unmatched_names,
    )


@router.get("/drugs", response_model=list[DrugOut])
def search_drugs(q: str = "", limit: int = 25,
                 session: Session = Depends(get_db_session)) -> list[DrugOut]:
    """Drug-name search (canonically case/space-insensitive) for UI pickers."""
    from nlp.cleaner import TextCleaner

    needle = TextCleaner.canonical(q)
    drugs = DrugRepository(session).all()
    if needle:
        drugs = [d for d in drugs if needle in TextCleaner.canonical(d.trade_name)]
    return [DrugOut(drug_id=d.id, trade_name=d.trade_name)
            for d in drugs[: max(1, min(limit, 100))]]


@router.get("/recent-scans", response_model=list[ScanHistoryItem])
def get_recent_scans(
    limit: int = 30,
    session: Session = Depends(get_db_session),
) -> list[ScanHistoryItem]:
    """List recent scans for doctor/pharmacist inspection and training."""
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
        for entry in ScanLogRepository(session).list_recent(limit=limit)
    ]


@router.post("/check", response_model=InteractionReportOut)
def check_text(payload: DoctorCheckRequest,
               checker: InteractionChecker = Depends(get_interaction_checker)) -> InteractionReportOut:
    """Anonymous instant check from typed drug names."""
    return _report_out(checker.check_drug_names(payload.drug_names))


@router.post("/scan", response_model=ScanOutcomeOut)
async def scan_image(
    image: UploadFile,
    pipeline: ScanPipeline = Depends(get_scan_pipeline),
    session: Session = Depends(get_db_session),
) -> ScanOutcomeOut:
    """Upload a package photo; runs both recognition paths + fusion + interaction check."""
    data = await image.read()
    decoded = cv2_imdecode(data)
    if decoded is None:
        raise HTTPException(status_code=400, detail="unreadable image file")
    outcome = pipeline.process(decoded, patient_id=None, image_path=image.filename or "")
    return _outcome_out(outcome)


@router.post("/review", response_model=InteractionReportOut)
def review_correction(
    payload: DoctorReviewRequest,
    checker: InteractionChecker = Depends(get_interaction_checker),
    session: Session = Depends(get_db_session),
    correction_logger=Depends(get_correction_logger),
) -> InteractionReportOut:
    """Pharmacist confirms/corrects what the system recognized; learns the dictionary."""
    drug_repo = DrugRepository(session)
    drug = drug_repo.get_by_name(payload.corrected_drug_name.strip().lower())
    if drug is None:
        raise HTTPException(status_code=404, detail=f"unknown drug '{payload.corrected_drug_name}'")
    correction_logger.log_correction(payload.raw_ocr_text, drug.id, payload.corrected_by)
    entry = ScanLogRepository(session).get(payload.scan_id)
    if entry is not None:
        entry.review_status = "corrected"
        entry.reviewed_by = payload.corrected_by
        entry.matched_drug_id = drug.id
        session.flush()
    return _report_out(checker.check_drug_names([drug.trade_name]))


def cv2_imdecode(data: bytes):
    import cv2

    buffer = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _annotated_to_base64(image_bgr) -> str:
    """Encode the explainability overlay as base64 PNG (empty string when absent)."""
    if image_bgr is None:
        return ""
    import cv2

    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _outcome_out(outcome) -> ScanOutcomeOut:
    """Present a ScanOutcome (or None-report) as the API response schema."""
    report_out = None
    if outcome.report is not None:
        report_out = _report_out(outcome.report)
    return ScanOutcomeOut(
        accepted=outcome.accepted,
        rejection_reason=outcome.rejection_reason,
        scan_id=outcome.scan_id,
        ocr_text=outcome.ocr_text,
        ocr_label=outcome.ocr_label,
        classifier_label=outcome.classifier_label,
        fused_label=outcome.fusion.label if outcome.fusion else "",
        fused_confidence=outcome.fusion.confidence if outcome.fusion else 0.0,
        fusion_status=outcome.fusion.status if outcome.fusion else "",
        matched_drug_id=outcome.matched_drug_id,
        matched_drug_name=outcome.matched_drug_name,
        interaction_report=report_out,
        annotated_image_base64=_annotated_to_base64(getattr(outcome, "annotated_image", None)),
    )