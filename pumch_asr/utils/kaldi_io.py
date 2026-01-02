"""Kaldi feature writer helpers."""

from __future__ import annotations

from typing import Iterable, Tuple

import kaldiio
import numpy as np


def write_kaldi_feats(
    feats: Iterable[Tuple[str, np.ndarray]],
    ark_path: str,
    scp_path: str,
) -> None:
    with kaldiio.WriteHelper(f"ark,scp:{ark_path},{scp_path}") as writer:
        for utt_id, feat in feats:
            writer(utt_id, feat)
