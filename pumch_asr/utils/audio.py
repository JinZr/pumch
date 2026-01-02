"""Audio loading and resampling helpers."""

from __future__ import annotations

import io
import subprocess
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _read_waveform_from_pipe(command: str) -> Tuple[np.ndarray, int]:
    data = subprocess.check_output(command, shell=True)
    waveform, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    return waveform, sample_rate


def load_audio(wav_spec: str, target_sr: int) -> np.ndarray:
    wav_spec = wav_spec.strip()
    if wav_spec.endswith("|"):
        command = wav_spec[:-1].strip()
        waveform, sample_rate = _read_waveform_from_pipe(command)
    else:
        waveform, sample_rate = sf.read(wav_spec, dtype="float32")

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(torch.from_numpy(waveform)).numpy()

    return waveform.astype("float32")
