"""SQLAlchemy models — the exact tables from the Master plan (surrogate PKs added for SQLite)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Ingredient(Base):
    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    aliases: Mapped[str] = mapped_column(Text, default="")   # JSON list of alias strings
    drug_class: Mapped[str] = mapped_column(String(255), default="")


class Drug(Base):
    __tablename__ = "drugs"

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    active_ingredient_ids: Mapped[str] = mapped_column(Text, default="")  # JSON list of Ingredient ids
    strength: Mapped[str] = mapped_column(String(255), default="")
    form: Mapped[str] = mapped_column(String(255), default="")
    image_ref: Mapped[str] = mapped_column(String(512), default="")


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    ingredient_a_id: Mapped[int] = mapped_column(ForeignKey("ingredients.id"), index=True)
    ingredient_b_id: Mapped[int] = mapped_column(ForeignKey("ingredients.id"), index=True)
    severity: Mapped[str] = mapped_column(String(32))          # Major / Moderate / Minor
    description: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(255), default="DDInter")

    ingredient_a = relationship("Ingredient", foreign_keys=[ingredient_a_id])
    ingredient_b = relationship("Ingredient", foreign_keys=[ingredient_b_id])


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512))


class PatientMedication(Base):
    __tablename__ = "patient_medications"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    drug_id: Mapped[int] = mapped_column(ForeignKey("drugs.id"), index=True)


class ScanLog(Base):
    __tablename__ = "scan_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    image_path: Mapped[str] = mapped_column(String(512), default="")
    ocr_raw_text: Mapped[str] = mapped_column(Text, default="")
    classifier_prediction: Mapped[str] = mapped_column(String(255), default="")
    fused_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    matched_drug_id: Mapped[int | None] = mapped_column(ForeignKey("drugs.id"), nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), default="auto")  # auto|confirmed|corrected
    reviewed_by: Mapped[str] = mapped_column(String(255), default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CorrectionLog(Base):
    __tablename__ = "correction_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    raw_ocr_text: Mapped[str] = mapped_column(Text)
    corrected_drug_id: Mapped[int] = mapped_column(ForeignKey("drugs.id"), index=True)
    corrected_by: Mapped[str] = mapped_column(String(255), default="pharmacist")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)