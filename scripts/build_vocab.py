#!/usr/bin/env python3
"""Build a char vocabulary JSON from Kaldi text."""

from __future__ import annotations

import argparse
from pathlib import Path

from pumch_asr.utils.vocab import (
    build_vocab_from_texts,
    load_texts_from_kaldi_text,
    save_vocab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CTC char vocabulary from Kaldi text"
    )
    parser.add_argument("--text", required=True, help="Path to Kaldi text file")
    parser.add_argument("--output", required=True, help="Output vocab.json path")
    parser.add_argument(
        "--lowercase", action="store_true", help="Lowercase transcripts"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = load_texts_from_kaldi_text(args.text, lowercase=args.lowercase)
    vocab = build_vocab_from_texts(texts, lowercase=args.lowercase)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, output)


if __name__ == "__main__":
    main()
