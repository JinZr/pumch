"""Vocabulary construction for CTC char models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from pumch_asr.config import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER
from pumch_asr.utils.text import normalize_text


def load_texts_from_kaldi_text(path: str | Path, lowercase: bool = False) -> List[str]:
    path = Path(path)
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            _, text = parts
            texts.append(normalize_text(text, lowercase=lowercase))
    return texts


def build_vocab_from_texts(
    texts: Iterable[str],
    lowercase: bool = False,
    word_delimiter: str = WORD_DELIMITER,
) -> Dict[str, int]:
    vocab_chars = set()
    for text in texts:
        text = normalize_text(text, lowercase=lowercase)
        text = text.replace(" ", word_delimiter)
        vocab_chars.update(list(text))

    special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    vocab: Dict[str, int] = {}
    for token in special_tokens:
        vocab[token] = len(vocab)

    for ch in sorted(vocab_chars):
        if ch in vocab:
            continue
        vocab[ch] = len(vocab)

    if word_delimiter not in vocab:
        vocab[word_delimiter] = len(vocab)

    return vocab


def save_vocab(vocab: Dict[str, int], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False, indent=2)
