"""Tests for the medical PDF report generator."""
from __future__ import annotations

from ui.pdf_report import generate_patient_pdf_report


def test_pdf_report_empty_medications():
    pdf_bytes = generate_patient_pdf_report(
        patient_email="patient@example.com",
        medications=[],
        interaction_report=None,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-")


def test_pdf_report_safe_regimen():
    medications = [
        {"drug_name": "Paracetamol", "drug_id": 101},
        {"drug_name": "Amoxicillin", "drug_id": 102},
    ]
    report_data = {
        "status": "safe",
        "findings": [],
    }
    pdf_bytes = generate_patient_pdf_report(
        patient_email="patient@example.com",
        medications=medications,
        interaction_report=report_data,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


def test_pdf_report_danger_interaction():
    medications = [
        {"drug_name": "Aspirin", "drug_id": 201},
        {"drug_name": "Warfarin", "drug_id": 202},
    ]
    report_data = {
        "status": "danger",
        "findings": [
            {
                "drug_a": "Aspirin",
                "drug_b": "Warfarin",
                "severity": "Major",
                "source": "DDInter",
                "description": "Increased risk of severe bleeding.",
            }
        ],
    }
    pdf_bytes = generate_patient_pdf_report(
        patient_email="patient@example.com",
        medications=medications,
        interaction_report=report_data,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1500
    assert pdf_bytes.startswith(b"%PDF-")
