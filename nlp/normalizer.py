"""Active-ingredient normalization and compound-drug splitting (Phase 2)."""
from __future__ import annotations

import re

from nlp.cleaner import TextCleaner
from nlp.matcher import FuzzyMatcher

_SPLIT_RE = re.compile(r"\s*(?:\+|&|/|;|,|\band\b)\s*")


class CompoundDrugSplitter:
    """Splits a package string that carries more than one active ingredient."""

    def __init__(self, cleaner: TextCleaner) -> None:
        self.cleaner = cleaner

    def split(self, text: str) -> list[str]:
        """Return cleaned individual ingredient/drug names (may be a single item)."""
        if not text:
            return []
        parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
        cleaned = [c for p in parts if (c := self.cleaner.clean(p))]
        return cleaned or ([self.cleaner.clean(text)] if self.cleaner.clean(text) else [])


class IngredientNormalizer:
    """Normalizes drug/ingredient text to canonical ingredient names.

    Lookup order: exact canonical name → alias → fuzzy match over vocabulary.
    Aliases come from the Ingredients table (correction_learning also feeds this).
    """

    def __init__(self, cleaner: TextCleaner, matcher: FuzzyMatcher, aliases: dict[str, list[str]] | None = None) -> None:
        self.cleaner = cleaner
        self.matcher = matcher
        # canonical -> [alias, ...]
        self._aliases: dict[str, list[str]] = aliases or {}
        self._alias_index: dict[str, str] = {
            self.cleaner.canonical(alias): canon
            for canon, alias_list in self._aliases.items()
            for alias in alias_list
        }

    @classmethod
    def from_vocabulary(cls, cleaner: TextCleaner, matcher: FuzzyMatcher, names: list[str]) -> "IngredientNormalizer":
        return cls(cleaner, matcher, aliases={name: [] for name in names})

    def normalize(self, text: str) -> str | None:
        """Return the canonical ingredient name, or None when unmatched."""
        key = self.cleaner.canonical(text or "")
        if not key:
            return None
        if key in self._aliases:  # already canonical
            return key
        if key in self._alias_index:
            return self._alias_index[key]
        match = self.matcher.best_match(key)
        return match.matched if match is not None else None

    def extend_aliases(self, canonical: str, alias: str) -> None:
        """Learn a new alias (used by correction_learning)."""
        canon = self.cleaner.canonical(canonical)
        alias_key = self.cleaner.canonical(alias)
        if not canon or not alias_key or canon == alias_key:
            return
        self._alias_index[alias_key] = canon
        self._aliases.setdefault(canon, []).append(alias)