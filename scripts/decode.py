#!/usr/bin/env python3
"""Decode audio with a fine-tuned CTC model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCTC, AutoProcessor

from pumch_asr.data.collator import DataCollatorSpeechWithPadding
from pumch_asr.data.dataset import KaldiSpeechDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTC decoding for Kaldi data")
    parser.add_argument("--data-dir", required=True, help="Kaldi data dir with wav.scp")
    parser.add_argument("--model-dir", required=True, help="Fine-tuned model directory")
    parser.add_argument("--output", required=True, help="Output hypothesis text file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for batch in tqdm(loader, desc="Decoding"):
            utt_ids = batch.pop("utt_id")
            for key in list(batch.keys()):
                batch[key] = batch[key].to(args.device)
            if not use_attention_mask and "attention_mask" in batch:
                batch.pop("attention_mask")

            with torch.no_grad():
                logits = model(**batch).logits
                pred_ids = torch.argmax(logits, dim=-1)

            texts = processor.batch_decode(pred_ids)
            for utt_id, text in zip(utt_ids, texts):
                out_f.write(f"{utt_id} {text}\n")


if __name__ == "__main__":
    main()
