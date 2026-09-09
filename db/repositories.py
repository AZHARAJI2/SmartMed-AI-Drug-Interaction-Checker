"""Repository classes — all DB access goes through these (no queries in routers)."""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import (
    CorrectionLog,
    Drug,
    Ingredient,
    Interaction,
    PatientMedication,
    ScanLog,
    User,
)


class IngredientRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, name: str, aliases: list[str] | None = None, drug_class: str = "") -> Ingredient:
        existing = self.get_by_name(name)
        if existing is not None:
            return existing
        ingredient = Ingredient(name=name, aliases=json.dumps(aliases or []), drug_class=drug_class)
        self.session.add(ingredient)
        self.session.flush()
        return ingredient

    def get_by_name(self, name: str) -> Ingredient | None:
        from sqlalchemy import func
        clean = name.strip()
        res = self.session.scalar(select(Ingredient).where(Ingredient.name == clean))
        if res is not None:
            return res
        return self.session.scalar(
            select(Ingredient).where(func.lower(Ingredient.name) == clean.lower())
        )

    def all(self) -> list[Ingredient]:
        return list(self.session.scalars(select(Ingredient)).all())

    def aliases_map(self) -> dict[str, list[str]]:
        return {ing.name: json.loads(ing.aliases or "[]") for ing in self.all()}


class DrugRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, trade_name: str, ingredient_ids: list[int] | None = None,
            strength: str = "", form: str = "", image_ref: str = "") -> Drug:
        existing = self.get_by_name(trade_name)
        if existing is not None:
            return existing
        drug = Drug(
            trade_name=trade_name,
            active_ingredient_ids=json.dumps(ingredient_ids or []),
            strength=strength,
            form=form,
            image_ref=image_ref,
        )
        self.session.add(drug)
        self.session.flush()
        return drug

    def get_by_name(self, trade_name: str) -> Drug | None:
        from sqlalchemy import func
        clean = trade_name.strip()
        res = self.session.scalar(select(Drug).where(Drug.trade_name == clean))
        if res is not None:
            return res
        return self.session.scalar(
            select(Drug).where(func.lower(Drug.trade_name) == clean.lower())
        )

    def get(self, drug_id: int) -> Drug | None:
        return self.session.get(Drug, drug_id)

    def all(self) -> list[Drug]:
        return list(self.session.scalars(select(Drug)).all())

    def ingredient_ids_of(self, drug: Drug) -> list[int]:
        return json.loads(drug.active_ingredient_ids or "[]")


class InteractionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, ingredient_a_id: int, ingredient_b_id: int, severity: str,
            description: str = "", source: str = "DDInter") -> Interaction | None:
        a, b = sorted((ingredient_a_id, ingredient_b_id))
        if self.get_pair(a, b) is not None:
            return None  # idempotent ingestion
        interaction = Interaction(
            ingredient_a_id=a, ingredient_b_id=b,
            severity=severity, description=description, source=source,
        )
        self.session.add(interaction)
        self.session.flush()
        return interaction

    def get_pair(self, ingredient_a_id: int, ingredient_b_id: int) -> Interaction | None:
        a, b = sorted((ingredient_a_id, ingredient_b_id))
        return self.session.scalar(
            select(Interaction).where(
                Interaction.ingredient_a_id == a,
                Interaction.ingredient_b_id == b,
            )
        )

    def all(self) -> list[Interaction]:
        return list(self.session.scalars(select(Interaction)).all())

    def for_ingredients(self, ingredient_ids: list[int]) -> list[Interaction]:
        """All interactions among the given ingredients (every unordered pair)."""
        found: list[Interaction] = []
        ids = sorted(set(ingredient_ids))
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                interaction = self.get_pair(a, b)
                if interaction is not None:
                    found.append(interaction)
        return found

    def set_or_update(
        self,
        ingredient_a_id: int,
        ingredient_b_id: int,
        severity: str,
        description: str = "",
        source: str = "مراجعة الطبيب",
    ) -> Interaction:
        a, b = sorted((ingredient_a_id, ingredient_b_id))
        pair = self.get_pair(a, b)
        if pair is not None:
            pair.severity = severity
            pair.description = description
            pair.source = source
            self.session.flush()
            return pair
        interaction = Interaction(
            ingredient_a_id=a,
            ingredient_b_id=b,
            severity=severity,
            description=description,
            source=source,
        )
        self.session.add(interaction)
        self.session.flush()
        return interaction

    def remove_pair(self, ingredient_a_id: int, ingredient_b_id: int) -> bool:
        a, b = sorted((ingredient_a_id, ingredient_b_id))
        pair = self.get_pair(a, b)
        if pair is not None:
            self.session.delete(pair)
            self.session.flush()
            return True
        return False


class UserRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, name: str, email: str, password_hash: str) -> User:
        user = User(name=name, email=email.lower(), password_hash=password_hash)
        self.session.add(user)
        self.session.flush()
        return user

    def get_by_email(self, email: str) -> User | None:
        return self.session.scalar(select(User).where(User.email == email.lower()))

    def get(self, user_id: int) -> User | None:
        return self.session.get(User, user_id)


class MedicationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, patient_id: int, drug_id: int) -> PatientMedication:
        existing = self.get(patient_id, drug_id)
        if existing is not None:
            return existing
        record = PatientMedication(patient_id=patient_id, drug_id=drug_id)
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, patient_id: int, drug_id: int) -> PatientMedication | None:
        return self.session.scalar(
            select(PatientMedication).where(
                PatientMedication.patient_id == patient_id,
                PatientMedication.drug_id == drug_id,
            )
        )

    def remove(self, patient_id: int, drug_id: int) -> bool:
        record = self.get(patient_id, drug_id)
        if record is None:
            return False
        self.session.delete(record)
        self.session.flush()
        return True

    def list_for_patient(self, patient_id: int) -> list[PatientMedication]:
        return list(self.session.scalars(
            select(PatientMedication).where(PatientMedication.patient_id == patient_id)
        ).all())


class ScanLogRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, patient_id: int | None, image_path: str, ocr_raw_text: str,
            classifier_prediction: str, fused_confidence: float,
            matched_drug_id: int | None, review_status: str = "auto",
            reviewed_by: str = "") -> ScanLog:
        entry = ScanLog(
            patient_id=patient_id,
            image_path=image_path,
            ocr_raw_text=ocr_raw_text,
            classifier_prediction=classifier_prediction,
            fused_confidence=fused_confidence,
            matched_drug_id=matched_drug_id,
            review_status=review_status,
            reviewed_by=reviewed_by,
            timestamp=datetime.utcnow(),
        )
        self.session.add(entry)
        self.session.flush()
        return entry

    def get(self, scan_id: int) -> ScanLog | None:
        return self.session.get(ScanLog, scan_id)

    def list_for_patient(self, patient_id: int) -> list[ScanLog]:
        return list(self.session.scalars(
            select(ScanLog).where(ScanLog.patient_id == patient_id).order_by(ScanLog.timestamp.desc())
        ).all())

    def list_recent(self, limit: int = 50) -> list[ScanLog]:
        return list(self.session.scalars(
            select(ScanLog).order_by(ScanLog.timestamp.desc()).limit(limit)
        ).all())


class CorrectionLogRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, raw_ocr_text: str, corrected_drug_id: int, corrected_by: str = "pharmacist") -> CorrectionLog:
        entry = CorrectionLog(
            raw_ocr_text=raw_ocr_text,
            corrected_drug_id=corrected_drug_id,
            corrected_by=corrected_by,
            timestamp=datetime.utcnow(),
        )
        self.session.add(entry)
        self.session.flush()
        return entry

    def all(self) -> list[CorrectionLog]:
        return list(self.session.scalars(select(CorrectionLog)).all())
