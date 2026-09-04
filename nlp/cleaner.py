"""Text cleanup for OCR output and raw dataset labels."""
from __future__ import annotations

import re
import unicodedata

from config import SplitConfig

_JUNK_CHARS = re.compile(r"[^0-9a-zA-Z\s\-\+\.]")
_MULTI_SPACE = re.compile(r"\s+")
_LEADING_NON_LETTERS = re.compile(r"^[^A-Za-z]+")


class TextCleaner:
    """Normalizes raw drug-name text (dataset labels, OCR lines, user input)."""

    def __init__(self, config: SplitConfig) -> None:
        self.config = config

    def clean(self, text: str) -> str:
        """Return a normalized single-line name, or "" when the text is junk."""
        if not text:
            return ""
        # 1. Unicode fold (é→e etc.) and lowercase
        normalized = unicodedata.normalize("NFKD", text)
        normalized = "".join(c for c in normalized if not unicodedata.combining(c))
        text = normalized.lower()
        # 2. Strip URLs, dates, camera/file junk tokens
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\b(19|20)\d{2}[-_.]?\d{0,2}[-_.]?\d{0,2}\b", " ", text)  # dates
        text = re.sub(r"\b(img|image|photo|dsc|sab|jpg|jpeg|png)\b", " ", text)
        # 3. Drop leading non-letters (e.g. "14418-broken-...")
        text = _LEADING_NON_LETTERS.sub(" ", text)
        # 4. Remove junk characters, collapse whitespace
        text = _JUNK_CHARS.sub(" ", text)
        text = _MULTI_SPACE.sub(" ", text).strip(" -.")
        words = text.split()
        # 5. Any junk-keyword word marks the whole text as junk (museum/shop shots etc.)
        if any(j in w for w in words for j in self.config.junk_keywords):
            return ""
        # 6. Drop standalone pure digits and 1-char tokens
        words = [w for w in words if len(w) > 1 and not w.isdigit()]
        text = " ".join(words[: self.config.max_name_words])
        return text if len(text) >= 3 else ""

    def is_plausible_drug_name(self, text: str) -> bool:
        """A cleaned name must contain at least one alphabetic word >= 3 chars."""
        cleaned = self.clean(text)
        return bool(cleaned) and any(len(w) >= 3 and w.isalpha() for w in cleaned.split())

    @staticmethod
    def canonical(name: str) -> str:
        """Canonical class key: lowercase, alphanumerics only (punctuation → space)."""
        key = re.sub(r"[^a-z0-9]+", " ", name.lower())
        return _MULTI_SPACE.sub(" ", key).strip()