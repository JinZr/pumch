"""Kaldi-style data directory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class Utterance:
    utt_id: str
    wav: str
    text: Optional[str]
    spk_id: Optional[str]


@dataclass
class KaldiData:
    wav_scp: Dict[str, str]
    text: Dict[str, str]
    utt2spk: Dict[str, str]
    spk2utt: Dict[str, str]

    def utt_ids(self) -> List[str]:
        return sorted(self.wav_scp.keys())

    def iter_utterances(self, require_text: bool = False) -> Iterable[Utterance]:
        for utt_id in self.utt_ids():
            wav = self.wav_scp[utt_id]
            text = self.text.get(utt_id)
            spk_id = self.utt2spk.get(utt_id)
            if require_text and text is None:
                raise ValueError(f"Missing text for utterance {utt_id}")
            yield Utterance(utt_id=utt_id, wav=wav, text=text, spk_id=spk_id)


def _read_mapping(path: Path, allow_missing: bool = False) -> Dict[str, str]:
    if not path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Missing required file: {path}")
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            key, value = parts
            mapping[key] = value
    return mapping


def load_kaldi_dir(data_dir: str | Path, require_text: bool = False) -> KaldiData:
    data_dir = Path(data_dir)
    wav_scp = _read_mapping(data_dir / "wav.scp", allow_missing=False)
    text = _read_mapping(data_dir / "text", allow_missing=not require_text)
    utt2spk = _read_mapping(data_dir / "utt2spk", allow_missing=True)
    spk2utt = _read_mapping(data_dir / "spk2utt", allow_missing=True)
    return KaldiData(wav_scp=wav_scp, text=text, utt2spk=utt2spk, spk2utt=spk2utt)
