#!/usr/bin/env python3
"""Segment m4a audio using diarization JSON and emit Kaldi data dir.

Expected diarization format: JSON files with key `utterances`, where each item
contains `speaker_id`, `start_time`, `end_time`, and optional `transcript`.
One JSON per recording, named with the same base name as the audio file
(e.g., foo.m4a -> foo.json).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from pumch_asr.config import TARGET_SAMPLING_RATE, UNK_TOKEN

AUDIO_EXTS = {".m4a"}


@dataclass
class Segment:
    recording_id: str
    start: float
    duration: float
    speaker: str
    transcript: str | None

    @property
    def end(self) -> float:
        return self.start + self.duration


def parse_diar_json(path: Path) -> List[Segment]:
    segments: List[Segment] = []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    utterances = data.get("utterances") if isinstance(data, dict) else None
    if not isinstance(utterances, list):
        return segments

    recording_id = path.stem
    for item in utterances:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start_time", 0.0))
        end = float(item.get("end_time", 0.0))
        if end <= start:
            continue
        segments.append(
            Segment(
                recording_id=recording_id,
                start=start,
                duration=end - start,
                speaker=str(item.get("speaker_id", "spk")),
                transcript=item.get("transcript"),
            )
        )
    return segments


def normalize_spk_id(speaker: str) -> str:
    speaker = speaker.strip()
    if not speaker:
        speaker = "spk"
    return re.sub(r"[^A-Za-z0-9_\-]", "_", speaker)

def normalize_rec_id(recording_id: str) -> str:
    recording_id = recording_id.strip()
    if not recording_id:
        recording_id = "rec"
    return re.sub(r"[^A-Za-z0-9_\-]", "_", recording_id)

def make_spk_id(recording_id: str, speaker: str) -> str:
    rec_id = normalize_rec_id(recording_id)
    spk = normalize_spk_id(speaker)
    return f"{rec_id}-{spk}"


def safe_utt_id(recording_id: str, start: float, end: float) -> str:
    start_ms = int(round(start * 1000))
    end_ms = int(round(end * 1000))
    rec_id = normalize_rec_id(recording_id)
    base = f"{rec_id}-{start_ms}-{end_ms}"
    return re.sub(r"[^A-Za-z0-9_\-]", "_", base)


def find_audio_for_recording(audio_dir: Path, recording_id: str) -> Path | None:
    for ext in AUDIO_EXTS:
        candidate = audio_dir / f"{recording_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_segment_wav(
    audio_path: Path,
    start: float,
    duration: float,
    output_path: Path,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLING_RATE),
    ]
    cmd.append("-y" if overwrite else "-n")
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)


def write_kaldi_dir(
    output_dir: Path,
    wav_entries: List[Tuple[str, Path]],
    utt2spk: Dict[str, str],
    texts: Dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_scp_path = output_dir / "wav.scp"
    text_path = output_dir / "text"
    utt2spk_path = output_dir / "utt2spk"
    spk2utt_path = output_dir / "spk2utt"

    with wav_scp_path.open("w", encoding="utf-8") as wav_f, text_path.open(
        "w", encoding="utf-8"
    ) as text_f, utt2spk_path.open("w", encoding="utf-8") as u2s_f:
        for utt_id, wav_path in wav_entries:
            wav_f.write(f"{utt_id} {wav_path}\n")
            text_f.write(f"{utt_id} {texts[utt_id]}\n")
            u2s_f.write(f"{utt_id} {utt2spk[utt_id]}\n")

    spk2utt: Dict[str, List[str]] = {}
    for utt_id, spk in utt2spk.items():
        spk2utt.setdefault(spk, []).append(utt_id)

    with spk2utt_path.open("w", encoding="utf-8") as s2u_f:
        for spk, utts in sorted(spk2utt.items()):
            s2u_f.write(f"{spk} {' '.join(sorted(utts))}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment m4a by diarization JSON")
    parser.add_argument("--audio-dir", required=True, help="Directory with m4a files")
    parser.add_argument(
        "--diar-dir", required=True, help="Directory with JSON diarization"
    )
    parser.add_argument("--output-dir", required=True, help="Output Kaldi data dir")
    parser.add_argument(
        "--segments-dir",
        help="Directory to store extracted wav segments (default: <output-dir>/wav)",
    )
    parser.add_argument(
        "--text-placeholder",
        default=UNK_TOKEN,
        help="Placeholder text for missing transcripts (default: <unk>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing wav segments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio_dir = Path(args.audio_dir)
    diar_dir = Path(args.diar_dir)
    output_dir = Path(args.output_dir)
    segments_dir = Path(args.segments_dir) if args.segments_dir else output_dir / "wav"

    json_files = sorted(
        p for p in diar_dir.glob("*.json") if not p.name.startswith("._")
    )
    if not json_files:
        raise FileNotFoundError(f"No JSON diarization files found in {diar_dir}")

    wav_entries: List[Tuple[str, Path]] = []
    utt2spk: Dict[str, str] = {}
    texts: Dict[str, str] = {}

    for json_path in json_files:
        segments = parse_diar_json(json_path)
        if not segments:
            continue

        recording_id = json_path.stem
        audio_path = find_audio_for_recording(audio_dir, recording_id)
        if audio_path is None:
            print(f"[WARN] Missing audio for recording {recording_id}")
            continue

        for seg in segments:
            speaker = seg.speaker
            spk_id = make_spk_id(recording_id, speaker)
            utt_id = safe_utt_id(recording_id, seg.start, seg.end)
            wav_path = segments_dir / f"{utt_id}.wav"
            extract_segment_wav(
                audio_path,
                seg.start,
                seg.duration,
                wav_path,
                overwrite=args.overwrite,
            )
            wav_entries.append((utt_id, wav_path.resolve()))
            utt2spk[utt_id] = spk_id
            transcript = (seg.transcript or "").strip()
            texts[utt_id] = transcript if transcript else args.text_placeholder

    write_kaldi_dir(output_dir, wav_entries, utt2spk, texts)


if __name__ == "__main__":
    main()
