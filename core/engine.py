"""Interaction-checking engine — the heart of the system (safe / caution / danger)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from db.repositories import DrugRepository, IngredientRepository, InteractionRepository
from nlp.normalizer import CompoundDrugSplitter, IngredientNormalizer

logger = logging.getLogger(__name__)

SEVERITY_RANK = {"minor": 1, "moderate": 2, "major": 3}
STATUS_BY_TOP_SEVERITY = {0: "safe", 1: "caution", 2: "caution", 3: "danger"}


@dataclass
class InteractionFinding:
    drug_a: str
    drug_b: str
    ingredient_a: str
    ingredient_b: str
    severity: str
    description: str
    source: str


@dataclass
class InteractionReport:
    status: str                      # safe | caution | danger
    findings: list[InteractionFinding] = field(default_factory=list)
    resolved_ingredients: list[str] = field(default_factory=list)
    unmatched_names: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Status: {self.status.upper()}"]
        for f in self.findings:
            lines.append(f"- [{f.severity}] {f.drug_a} + {f.drug_b} ({f.source})")
        if self.unmatched_names:
            lines.append(f"Unrecognized: {', '.join(self.unmatched_names)}")
        return "\n".join(lines)


class InteractionChecker:
    """Resolves drug/ingredient names via normalization, then checks DDInter pairs.

    Supports compound drugs ("paracetamol + ibuprofen") through CompoundDrugSplitter.
    """

    def __init__(
        self,
        ingredient_repo: IngredientRepository,
        interaction_repo: InteractionRepository,
        drug_repo: DrugRepository,
        normalizer: IngredientNormalizer,
        splitter: CompoundDrugSplitter,
    ) -> None:
        self.ingredients = ingredient_repo
        self.interactions = interaction_repo
        self.drugs = drug_repo
        self.normalizer = normalizer
        self.splitter = splitter

    def check_drug_names(self, drug_names: list[str]) -> InteractionReport:
        """Check interactions among a list of drug names (compounds auto-split)."""
        ingredient_ids: list[int] = []
        resolved: list[str] = []
        unmatched: list[str] = []
        for raw_name in drug_names:
            for part in self.splitter.split(raw_name):
                ingredient = self._resolve(part)
                if ingredient is None:
                    unmatched.append(part)
                    continue
                if ingredient.id not in ingredient_ids:
                    ingredient_ids.append(ingredient.id)
                    resolved.append(ingredient.name)
        return self._build_report(ingredient_ids, resolved, unmatched)

    def check_drug_ids(self, drug_ids: list[int]) -> InteractionReport:
        ingredient_ids: list[int] = []
        resolved: list[str] = []
        unmatched: list[str] = []
        for drug_id in drug_ids:
            drug = self.drugs.get(drug_id)
            if drug is None:
                unmatched.append(f"drug#{drug_id}")
                continue
            ingredient_ids.extend(self.drugs.ingredient_ids_of(drug))
            resolved.append(drug.trade_name)
        return self._build_report(ingredient_ids, resolved, unmatched)

    def _resolve(self, part: str):
        """Ingredient by normalized name; falls back to a drug whose trade name matches."""
        from db.models import Ingredient

        canonical = self.normalizer.normalize(part)
        if canonical is not None:
            ingredient = self.ingredients.get_by_name(canonical)
            if ingredient is not None:
                return ingredient
        drug = self.drugs.get_by_name(part)
        if drug is not None:
            for ing_id in self.drugs.ingredient_ids_of(drug):
                ingredient = self.drugs.session.get(Ingredient, ing_id)
                if ingredient is not None:
                    return ingredient
        return None

    def _build_report(self, ingredient_ids: list[int], resolved: list[str],
                      unmatched: list[str]) -> InteractionReport:
        findings: list[InteractionFinding] = []
        top_rank = 0
        name_of = {ing.id: ing.name for ing in self.ingredients.all()}
        for interaction in self.interactions.for_ingredients(ingredient_ids):
            rank = SEVERITY_RANK.get(interaction.severity.lower(), 0)
            top_rank = max(top_rank, rank)
            findings.append(
                InteractionFinding(
                    drug_a=name_of.get(interaction.ingredient_a_id, str(interaction.ingredient_a_id)),
                    drug_b=name_of.get(interaction.ingredient_b_id, str(interaction.ingredient_b_id)),
                    ingredient_a=name_of.get(interaction.ingredient_a_id, "?"),
                    ingredient_b=name_of.get(interaction.ingredient_b_id, "?"),
                    severity=interaction.severity,
                    description=interaction.description,
                    source=interaction.source,
                )
            )
        status = STATUS_BY_TOP_SEVERITY.get(top_rank, "safe")
        # Only escalate to caution when we have MULTIPLE drugs being compared
        # and at least one is not resolvable — a single unmatched drug is just "not in DB"
        total_drugs = len(ingredient_ids) + len(unmatched)
        if unmatched and status == "safe" and total_drugs >= 2:
            status = "caution"
        return InteractionReport(
            status=status,
            findings=findings,
            resolved_ingredients=resolved,
            unmatched_names=unmatched,
        )