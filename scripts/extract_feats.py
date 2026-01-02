#!/usr/bin/env python3
"""Extract encoder features and export to Kaldi feats.ark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCTC, AutoProcessor

from pumch_asr.data.collator import DataCollatorSpeechWithPadding
from pumch_asr.data.dataset import KaldiSpeechDataset
from pumch_asr.models.loader import get_base_encoder
from pumch_asr.utils.kaldi_io import write_kaldi_feats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract encoder features to Kaldi ark")
    parser.add_argument("--data-dir", required=True, help="Kaldi data dir with wav.scp")
    parser.add_argument("--model-dir", required=True, help="Fine-tuned model directory")
    parser.add_argument("--ark", required=True, help="Output feats.ark")
    parser.add_argument("--scp", required=True, help="Output feats.scp")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden layer index (default: last)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _iter_feats(
    model,
    loader: DataLoader,
    device: str,
    layer: int,
    use_attention_mask: bool,
) -> Iterable[Tuple[str, np.ndarray]]:
    base_encoder = get_base_encoder(model)

    for batch in tqdm(loader, desc="Extracting"):
        utt_ids = batch.pop("utt_id")
        for key in list(batch.keys()):
            batch[key] = batch[key].to(device)
        attention_mask = batch.get("attention_mask")
        if not use_attention_mask and "attention_mask" in batch:
            batch.pop("attention_mask")

        with torch.no_grad():
            outputs = base_encoder(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if layer < 0:
            features = hidden_states[layer]
        else:
            features = hidden_states[layer]

        if attention_mask is None:
            lengths = [feat.shape[0] for feat in features]
        else:
            lengths = attention_mask.sum(dim=-1)
            if hasattr(model, "_get_feat_extract_output_lengths"):
                lengths = model._get_feat_extract_output_lengths(lengths)
            lengths = lengths.to("cpu").tolist()

        features = features.cpu().numpy()
        for utt_id, feat, length in zip(utt_ids, features, lengths):
            yield utt_id, feat[: int(length)]


def main() -> None:
    args = parse_args()

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForCTC.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()
    use_attention_mask = getattr(model.config, "feat_extract_norm", None) != "group"

    sampling_rate = processor.feature_extractor.sampling_rate
    dataset = KaldiSpeechDataset(args.data_dir, target_sampling_rate=sampling_rate)
    collator = DataCollatorSpeechWithPadding(processor=processor, padding=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    ark_path = Path(args.ark)
    scp_path = Path(args.scp)
    ark_path.parent.mkdir(parents=True, exist_ok=True)
    scp_path.parent.mkdir(parents=True, exist_ok=True)

    feats_iter = _iter_feats(
        model=model,
        loader=loader,
        device=args.device,
        layer=args.layer,
        use_attention_mask=use_attention_mask,
    )
    write_kaldi_feats(feats_iter, str(ark_path), str(scp_path))


if __name__ == "__main__":
    main()
