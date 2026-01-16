"""PyTorch datasets for Kaldi-style data."""

from __future__ import annotations

from typing import List

from torch.utils.data import Dataset

from pumch_asr.config import WORD_DELIMITER
from pumch_asr.data.kaldi import KaldiData, load_kaldi_dir
from pumch_asr.utils.audio import load_audio
from pumch_asr.utils.text import normalize_text, whisper_normalize_text


class KaldiASRDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        processor,
        target_sampling_rate: int,
        lowercase: bool = False,
    ) -> None:
        self.data: KaldiData = load_kaldi_dir(data_dir, require_text=True)
        self.utt_ids: List[str] = self.data.utt_ids()
        self.processor = processor
        self.target_sampling_rate = target_sampling_rate
        self.lowercase = lowercase
        self._normalize_texts()

    def _normalize_texts(self) -> None:
        for utt_id, text in self.data.text.items():
            if text is None:
                continue
            text = whisper_normalize_text(text)
            text = normalize_text(text, lowercase=self.lowercase)
            text = text.replace(" ", WORD_DELIMITER)
            self.data.text[utt_id] = text

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, idx: int):
        utt_id = self.utt_ids[idx]
        wav_spec = self.data.wav_scp[utt_id]
        text = self.data.text[utt_id]

        waveform = load_audio(wav_spec, self.target_sampling_rate)
        labels = self.processor.tokenizer(text, add_special_tokens=False).input_ids

        return {
            "utt_id": utt_id,
            "input_values": waveform,
            "labels": labels,
        }


class KaldiSpeechDataset(Dataset):
    def __init__(self, data_dir: str, target_sampling_rate: int) -> None:
        self.data: KaldiData = load_kaldi_dir(data_dir, require_text=False)
        self.utt_ids: List[str] = self.data.utt_ids()
        self.target_sampling_rate = target_sampling_rate

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, idx: int):
        utt_id = self.utt_ids[idx]
        wav_spec = self.data.wav_scp[utt_id]
        waveform = load_audio(wav_spec, self.target_sampling_rate)
        return {
            "utt_id": utt_id,
            "input_values": waveform,
        }
