"""Pydantic schemas for the API (Phase 2 validation layer)."""
from __future__ import annotations

from pydantic import BaseModel, Field

# simple RFC-ish email check without the optional email-validator dependency
_EMAIL = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    email: str = Field(pattern=_EMAIL, max_length=255)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(pattern=_EMAIL, max_length=255)
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MedicationRequest(BaseModel):
    drug_name: str = Field(min_length=1, max_length=255)


class MedicationOut(BaseModel):
    drug_id: int
    drug_name: str
    in_db: bool = True


class InteractionFindingOut(BaseModel):
    drug_a: str
    drug_b: str
    ingredient_a: str
    ingredient_b: str
    severity: str
    description: str
    source: str


class InteractionReportOut(BaseModel):
    status: str
    findings: list[InteractionFindingOut] = []
    resolved_ingredients: list[str] = []
    unmatched_names: list[str] = []


class DoctorCheckRequest(BaseModel):
    drug_names: list[str] = Field(min_length=1)


class DoctorReviewRequest(BaseModel):
    scan_id: int
    raw_ocr_text: str = Field(min_length=1)
    corrected_drug_name: str = Field(min_length=1)
    corrected_by: str = Field(default="pharmacist", max_length=255)


class DoctorInteractionOverrideRequest(BaseModel):
    drug_a: str = Field(min_length=1)
    drug_b: str = Field(min_length=1)
    has_interaction: bool
    severity: str = Field(default="Moderate")
    description: str = Field(default="")
    reviewed_by: str = Field(default="طبيب / صيدلاني")


class DrugOut(BaseModel):
    drug_id: int
    trade_name: str


class ScanOutcomeOut(BaseModel):
    accepted: bool
    rejection_reason: str = ""
    scan_id: int | None = None
    ocr_text: str = ""
    ocr_label: str = ""
    classifier_label: str = ""
    fused_label: str = ""
    fused_confidence: float = 0.0
    fusion_status: str = ""
    matched_drug_id: int | None = None
    matched_drug_name: str = ""
    interaction_report: InteractionReportOut | None = None
    annotated_image_base64: str = ""  # visual explainability overlay (PNG, base64)


class ScanHistoryItem(BaseModel):
    scan_id: int
    image_path: str
    ocr_raw_text: str
    classifier_prediction: str
    fused_confidence: float
    matched_drug_id: int | None
    review_status: str
    reviewed_by: str