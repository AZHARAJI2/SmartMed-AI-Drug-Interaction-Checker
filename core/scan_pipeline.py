"""Scan pipeline: Phase 1 vision paths + drug matching + interaction check + Scan_Log."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import AppConfig
from correction_learning.logger import load_matching_dictionary
from db.repositories import DrugRepository, IngredientRepository, InteractionRepository
from fusion.fusion import AgreementFusion, FusionResult
from nlp.cleaner import TextCleaner
from nlp.matcher import FuzzyMatcher
from nlp.normalizer import IngredientNormalizer
from ocr.reader import OCRReader
from vision.enhance import EnhancementPipeline
from vision.explainability import VisualExplainer
from vision.quality import ImageQualityChecker

logger = logging.getLogger(__name__)


@dataclass
class ScanOutcome:
    accepted: bool
    rejection_reason: str = ""
    scan_id: int | None = None
    ocr_text: str = ""
    ocr_label: str = ""
    classifier_label: str = ""
    fusion: FusionResult | None = None
    matched_drug_id: int | None = None
    matched_drug_name: str = ""
    report: object | None = None          # InteractionReport
    annotated_image: np.ndarray | None = field(default=None, repr=False)


class ScanPipeline:
    """End-to-end scan: quality gate → enhancement → OCR+classifier → fusion → match → interactions."""

    def __init__(
        self,
        config: AppConfig,
        ocr_reader: OCRReader,
        predictor,                     # ClassifierPredictor | None
        matcher: FuzzyMatcher,
        ingredient_repo: IngredientRepository,
        drug_repo: DrugRepository,
        interaction_repo: InteractionRepository,
        checker,                       # InteractionChecker
    ) -> None:
        self.config = config
        self.ocr_reader = ocr_reader
        self.predictor = predictor
        self.matcher = matcher
        self.quality_checker = ImageQualityChecker(config.quality)
        self.enhancer = EnhancementPipeline.default(config.enhance)
        self.explainer = VisualExplainer(config)
        self.fusion = AgreementFusion(config.fusion, matcher=matcher)
        self.cleaner = TextCleaner(config.split)
        self.dictionary = load_matching_dictionary(config.dictionary)
        self.normalizer = IngredientNormalizer.from_vocabulary(
            self.cleaner, matcher, [ing.name for ing in ingredient_repo.all()]
        )
        self.ingredients = ingredient_repo
        self.drugs = drug_repo
        self.checker = checker

    def _dictionary_lookup(self, raw: str, cleaned: str = "") -> str | None:
        """Learned corrections win over fuzzy matching (pharmacist feedback loop)."""
        for key in (TextCleaner.canonical(raw), TextCleaner.canonical(cleaned)):
            if key and key in self.dictionary:
                return self.dictionary[key]
        return None

    def _best_drug_match(self, label: str):
        """Match a recognized label to a Drug row (dictionary, exact canonical, then fuzzy)."""
        key = TextCleaner.canonical(label)
        if not key:
            return None
        mapped = self._dictionary_lookup(label)
        if mapped:
            drug = self.drugs.get_by_name(mapped)
            if drug is not None:
                return drug
        drug = self.drugs.get_by_name(key)
        if drug is not None:
            return drug
        names = [TextCleaner.canonical(d.trade_name) for d in self.drugs.all()]
        if names:
            self.matcher.vocabulary = names
            match = self.matcher.best_match(key)
            if match is not None:
                return self.drugs.get_by_name(match.matched)
        return None

    def process(self, image_bgr: np.ndarray, patient_id: int | None = None,
                image_path: str = "", save_annotated_to: str = "") -> ScanOutcome:
        from db.repositories import ScanLogRepository

        quality = self.quality_checker.assess(image_bgr)
        if not quality.passed:
            return ScanOutcome(accepted=False, rejection_reason="; ".join(quality.failures))
        enhanced = self.enhancer.apply(image_bgr)

        ocr_result = self.ocr_reader.read(enhanced)
        ocr_conf = float(np.mean(ocr_result.confidences)) if ocr_result.confidences else 0.0
        ocr_label, best_score = "", 0.0
        for text in ocr_result.texts:
            cleaned = self.cleaner.clean(text)
            mapped = self._dictionary_lookup(text, cleaned)
            if mapped is not None:
                ocr_label, best_score = mapped, 1.0
                continue
            match = self.matcher.best_match(cleaned) if cleaned else None
            if match is not None and match.score > best_score:
                ocr_label, best_score = match.matched, match.score

        # Fallback: if fuzzy matcher did not match a DB drug, use the cleanest plausible text line
        if not ocr_label and ocr_result.texts:
            for text in ocr_result.texts:
                cleaned = self.cleaner.clean(text)
                if cleaned and not any(j in cleaned.lower() for j in self.config.split.junk_keywords):
                    ocr_label = cleaned
                    break

        prediction = self.predictor.predict(enhanced) if self.predictor is not None else None
        classifier_label = prediction.label if prediction else ""
        classifier_confidence = prediction.confidence if prediction else 0.0

        # Discard non-drug or junk predictions from the classifier
        if classifier_label:
            cleaned_pred = self.cleaner.clean(classifier_label)
            if not cleaned_pred or any(j in classifier_label.lower() for j in self.config.split.junk_keywords):
                classifier_label = ""
                classifier_confidence = 0.0

        fused = self.fusion.fuse(
            ocr_label=ocr_label,
            ocr_confidence=ocr_conf,
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
        )

        label_for_match = fused.label or ocr_label or classifier_label
        if label_for_match and any(j in label_for_match.lower() for j in self.config.split.junk_keywords):
            label_for_match = ""

        matched_drug = self._best_drug_match(label_for_match) if label_for_match else None

        scanned_name = matched_drug.trade_name if matched_drug is not None else label_for_match
        drug_names = [scanned_name] if scanned_name else []

        # If patient_id is provided, include the patient's existing medications
        patient_med_names: list[str] = []
        if patient_id is not None:
            from db.repositories import MedicationRepository
            patient_meds = MedicationRepository(self.ingredients.session).list_for_patient(patient_id)
            for pm in patient_meds:
                d = self.drugs.get(pm.drug_id)
                if d is not None:
                    patient_med_names.append(d.trade_name)

        all_to_check = list(drug_names)
        for p_name in patient_med_names:
            if scanned_name and p_name.lower() != scanned_name.lower() and p_name not in all_to_check:
                all_to_check.append(p_name)

        # Multi-drug interaction check requires at least 2 drugs
        report = self.checker.check_drug_names(all_to_check) if len(all_to_check) >= 2 else None

        annotated = self.explainer.explain(
            image_bgr, ocr_result,
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            fused_label=fused.label, fused_confidence=fused.confidence, status=fused.status,
        )
        if save_annotated_to:
            cv2.imwrite(save_annotated_to, annotated.annotated_image_bgr)

        entry = ScanLogRepository(self.ingredients.session).add(
            patient_id=patient_id,
            image_path=image_path,
            ocr_raw_text=ocr_result.full_text,
            classifier_prediction=classifier_label,
            fused_confidence=fused.confidence,
            matched_drug_id=matched_drug.id if matched_drug is not None else None,
        )
        return ScanOutcome(
            accepted=True,
            scan_id=entry.id,
            ocr_text=ocr_result.full_text,
            ocr_label=ocr_label,
            classifier_label=classifier_label,
            fusion=fused,
            matched_drug_id=matched_drug.id if matched_drug is not None else None,
            matched_drug_name=matched_drug.trade_name if matched_drug is not None else "",
            report=report,
            annotated_image=annotated.annotated_image_bgr,
        )
