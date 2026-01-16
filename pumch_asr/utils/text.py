"""Text normalization helpers."""

from __future__ import annotations

import re

from whisper_normalization import BasicTextNormalizer


_WHITESPACE_RE = re.compile(r"\s+")
_WHISPER_BASIC_NORMALIZER = BasicTextNormalizer()


def whisper_normalize_text(text: str) -> str:
    return _WHISPER_BASIC_NORMALIZER(text)


def normalize_text(text: str, lowercase: bool = False) -> str:
    text = text.strip()
    if lowercase:
        text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text
