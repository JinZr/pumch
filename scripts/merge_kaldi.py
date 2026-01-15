#!/usr/bin/env python3
"""Merge multiple Kaldi-format data dirs into one."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple


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


def validate_dir(data_dir: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple Kaldi-format data dirs into one."
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="Input Kaldi data dirs to merge",
    )
    parser.add_argument("--output-dir", required=True, help="Output data dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dirs = [Path(p) for p in args.data_dirs]
    output_dir = Path(args.output_dir)

    merged_wav: Dict[str, str] = {}
    merged_text: Dict[str, str] = {}
    merged_u2s: Dict[str, str] = {}

    seen_spk: Dict[str, Path] = {}

    for data_dir in data_dirs:
        wav_scp, text, utt2spk = validate_dir(data_dir)

        for utt_id in wav_scp:
            if utt_id in merged_wav:
                raise ValueError(f"Duplicate utterance id '{utt_id}' across datasets")

        for spk_id in set(utt2spk.values()):
            existing = seen_spk.get(spk_id)
            if existing is not None and existing != data_dir:
                raise ValueError(
                    f"Duplicate speaker id '{spk_id}' across datasets: "
                    f"{existing} and {data_dir}"
                )
            seen_spk[spk_id] = data_dir

        merged_wav.update(wav_scp)
        merged_text.update(text)
        merged_u2s.update(utt2spk)

    write_kaldi_dir(output_dir, merged_wav, merged_text, merged_u2s)


if __name__ == "__main__":
    main()
