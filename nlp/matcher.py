"""Fuzzy text matching strategies (RapidFuzz when available, stdlib difflib otherwise)."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on environment
    from rapidfuzz import fuzz as _rf_fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:  # network-blocked environment fallback
    _rf_fuzz = None
    _HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz unavailable — using stdlib difflib matcher fallback")


@dataclass
class MatchScore:
    """Best match of a query against a reference vocabulary."""

    matched: str
    score: float  # 0..100
    exact: bool


class FuzzyMatcher(ABC):
    """Strategy interface for fuzzy string matching against a vocabulary."""

    engine: str = "abstract"

    def __init__(self, vocabulary: list[str], threshold: float = 70.0) -> None:
        self.vocabulary = list(vocabulary)
        self.threshold = threshold

    @abstractmethod
    def _score(self, a: str, b: str) -> float: ...

    def best_match(self, query: str) -> MatchScore | None:
        query = query.strip().lower()
        if not query or not self.vocabulary:
            return None
        best_name, best_score = "", 0.0
        for candidate in self.vocabulary:
            candidate_lc = candidate.strip().lower()
            if query == candidate_lc:
                return MatchScore(matched=candidate, score=100.0, exact=True)
            score = self._score(query, candidate_lc)
            if score > best_score:
                best_name, best_score = candidate, score
        if best_score >= self.threshold:
            return MatchScore(matched=best_name, score=best_score, exact=False)
        return None


class RapidFuzzMatcher(FuzzyMatcher):
    """Uses rapidfuzz.fuzz.ratio (fast C implementation)."""

    engine = "rapidfuzz"

    def _score(self, a: str, b: str) -> float:
        return float(_rf_fuzz.ratio(a, b))  # type: ignore[misc]


class DifflibMatcher(FuzzyMatcher):
    """Pure-stdlib fallback with ratio semantics close to rapidfuzz."""

    engine = "difflib"

    def _score(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100.0


def build_matcher(vocabulary: list[str], threshold: float = 70.0) -> FuzzyMatcher:
    """Factory: prefer RapidFuzz, fall back to difflib when it is not installable."""
    if _HAS_RAPIDFUZZ:
        return RapidFuzzMatcher(vocabulary, threshold)
    return DifflibMatcher(vocabulary, threshold)