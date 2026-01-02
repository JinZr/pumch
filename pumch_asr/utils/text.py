"""Text normalization helpers."""

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, lowercase: bool = False) -> str:
    text = text.strip()
    if lowercase:
        text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text
