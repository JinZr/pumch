#!/usr/bin/env python3
"""Speaker-level 5-fold partitioning for Kaldi-format data dirs."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SPEAKER2_RE = re.compile(r"Speaker_2(?!\d)")
SPEAKER_RE = re.compile(r"Speaker_\d+")


def read_kaldi_kv(path: Path, require_value: bool = True) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    entries: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            key = parts[0]
            value = parts[1] if len(parts) > 1 else ""
            if require_value and not value:
                raise ValueError(f"Missing value for key '{key}' in {path}:{line_no}")
            if key in entries:
                raise ValueError(f"Duplicate key '{key}' in {path}:{line_no}")
            entries[key] = value
    return entries


def validate_dir(
    data_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    wav_scp = read_kaldi_kv(data_dir / "wav.scp")
    text = read_kaldi_kv(data_dir / "text", require_value=False)
    utt2spk = read_kaldi_kv(data_dir / "utt2spk")

    wav_keys = set(wav_scp)
    text_keys = set(text)
    u2s_keys = set(utt2spk)

    if wav_keys != text_keys or wav_keys != u2s_keys:
        missing_in_wav = sorted((text_keys | u2s_keys) - wav_keys)
        missing_in_text = sorted((wav_keys | u2s_keys) - text_keys)
        missing_in_u2s = sorted((wav_keys | text_keys) - u2s_keys)
        msg = [f"Utterance id mismatch in {data_dir}:"]
        if missing_in_wav:
            msg.append(f"  missing in wav.scp: {', '.join(missing_in_wav[:5])}")
        if missing_in_text:
            msg.append(f"  missing in text: {', '.join(missing_in_text[:5])}")
        if missing_in_u2s:
            msg.append(f"  missing in utt2spk: {', '.join(missing_in_u2s[:5])}")
        raise ValueError("\n".join(msg))

    return wav_scp, text, utt2spk


def write_kaldi_dir(
    output_dir: Path,
    wav_scp: Dict[str, str],
    text: Dict[str, str],
    utt2spk: Dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "wav.scp").open("w", encoding="utf-8") as wav_f, (
        output_dir / "text"
    ).open("w", encoding="utf-8") as text_f, (
        output_dir / "utt2spk"
    ).open("w", encoding="utf-8") as u2s_f:
        for utt_id in sorted(wav_scp):
            wav_f.write(f"{utt_id} {wav_scp[utt_id]}\n")
            text_f.write(f"{utt_id} {text[utt_id]}\n")
            u2s_f.write(f"{utt_id} {utt2spk[utt_id]}\n")

    spk2utt: Dict[str, list[str]] = {}
    for utt_id, spk_id in utt2spk.items():
        spk2utt.setdefault(spk_id, []).append(utt_id)

    with (output_dir / "spk2utt").open("w", encoding="utf-8") as s2u_f:
        for spk_id in sorted(spk2utt):
            s2u_f.write(f"{spk_id} {' '.join(sorted(spk2utt[spk_id]))}\n")


def is_batch_ctrl_dir(wav_scp: Dict[str, str], token: str) -> bool:
    if not wav_scp:
        return False
    return all(token in path for path in wav_scp.values())


def split_folds(items: List[str], n_folds: int, rng: random.Random) -> List[List[str]]:
    items = list(items)
    rng.shuffle(items)
    base = len(items) // n_folds
    rem = len(items) % n_folds
    folds: List[List[str]] = []
    idx = 0
    for i in range(n_folds):
        size = base + (1 if i < rem else 0)
        folds.append(items[idx : idx + size])
        idx += size
    return folds


def subset_for_speakers(
    speakers: Iterable[str],
    spk2utt: Dict[str, List[str]],
    wav_scp: Dict[str, str],
    text: Dict[str, str],
    utt2spk: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    out_wav: Dict[str, str] = {}
    out_text: Dict[str, str] = {}
    out_u2s: Dict[str, str] = {}
    for spk_id in speakers:
        for utt_id in spk2utt.get(spk_id, []):
            out_wav[utt_id] = wav_scp[utt_id]
            out_text[utt_id] = text[utt_id]
            out_u2s[utt_id] = utt2spk[utt_id]
    return out_wav, out_text, out_u2s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speaker-level 5-fold partitioning for Kaldi data dirs"
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="Input Kaldi data dirs to merge and partition",
    )
    parser.add_argument("--output-dir", required=True, help="Output base dir")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--batch-ctrl-token",
        default="batch_ctrl",
        help="Substring used to identify the batch_ctrl data dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dirs = [Path(p) for p in args.data_dirs]
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    batch_ctrl_candidates: List[Path] = []

    merged_wav: Dict[str, str] = {}
    merged_text: Dict[str, str] = {}
    merged_u2s: Dict[str, str] = {}
    spk_dirs: Dict[str, set[Path]] = {}

    for data_dir in data_dirs:
        wav_scp, text, utt2spk = validate_dir(data_dir)
        if is_batch_ctrl_dir(wav_scp, args.batch_ctrl_token):
            batch_ctrl_candidates.append(data_dir)

        for utt_id in wav_scp:
            if utt_id in merged_wav:
                raise ValueError(f"Duplicate utterance id '{utt_id}' across datasets")

        for utt_id, spk_id in utt2spk.items():
            spk_dirs.setdefault(spk_id, set()).add(data_dir)

        merged_wav.update(wav_scp)
        merged_text.update(text)
        merged_u2s.update(utt2spk)

    if len(batch_ctrl_candidates) != 1:
        raise ValueError(
            "Expected exactly one batch_ctrl data dir; found "
            f"{len(batch_ctrl_candidates)}"
        )
    batch_ctrl_dir = batch_ctrl_candidates[0]

    groups: Dict[str, List[str]] = {"CTRL": [], "CLP": [], "INV": []}
    for spk_id, dirs in spk_dirs.items():
        in_batch = batch_ctrl_dir in dirs
        in_other = any(d != batch_ctrl_dir for d in dirs)

        if SPEAKER2_RE.search(spk_id):
            if in_batch and in_other:
                raise ValueError(
                    f"Speaker '{spk_id}' appears in batch_ctrl and non-batch_ctrl "
                    "but has Speaker_2 in id"
                )
            group = "CTRL" if in_batch else "CLP"
        elif SPEAKER_RE.search(spk_id):
            group = "INV"
        else:
            raise ValueError(f"Speaker id '{spk_id}' does not match Speaker_* pattern")

        groups[group].append(spk_id)

    ctrl_folds = split_folds(groups["CTRL"], 5, rng)
    inv_folds = split_folds(groups["INV"], 5, rng)
    clp_folds = split_folds(groups["CLP"], 5, rng)

    spk2utt: Dict[str, List[str]] = {}
    for utt_id, spk_id in merged_u2s.items():
        spk2utt.setdefault(spk_id, []).append(utt_id)

    all_speakers = set(merged_u2s.values())

    for fold_idx in range(5):
        fold_name = f"fold{fold_idx + 1}"
        fold_dir = output_dir / fold_name

        test_ctrl = set(ctrl_folds[fold_idx])
        test_inv = set(inv_folds[fold_idx])
        test_clp = set(clp_folds[fold_idx])
        test_all = test_ctrl | test_inv | test_clp
        train_spk = all_speakers - test_all

        train_wav, train_text, train_u2s = subset_for_speakers(
            train_spk, spk2utt, merged_wav, merged_text, merged_u2s
        )
        test_wav, test_text, test_u2s = subset_for_speakers(
            test_all, spk2utt, merged_wav, merged_text, merged_u2s
        )
        ctrl_wav, ctrl_text, ctrl_u2s = subset_for_speakers(
            test_ctrl, spk2utt, merged_wav, merged_text, merged_u2s
        )
        inv_wav, inv_text, inv_u2s = subset_for_speakers(
            test_inv, spk2utt, merged_wav, merged_text, merged_u2s
        )
        clp_wav, clp_text, clp_u2s = subset_for_speakers(
            test_clp, spk2utt, merged_wav, merged_text, merged_u2s
        )

        write_kaldi_dir(fold_dir / "train", train_wav, train_text, train_u2s)
        write_kaldi_dir(fold_dir / "test", test_wav, test_text, test_u2s)
        write_kaldi_dir(
            fold_dir / "test_ctrl", ctrl_wav, ctrl_text, ctrl_u2s
        )
        write_kaldi_dir(fold_dir / "test_inv", inv_wav, inv_text, inv_u2s)
        write_kaldi_dir(fold_dir / "test_clp", clp_wav, clp_text, clp_u2s)


if __name__ == "__main__":
    main()
