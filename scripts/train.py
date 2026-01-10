#!/usr/bin/env python3
"""Fine-tune HuBERT / Wav2Vec2 / WavLM with CTC on Kaldi data."""

from __future__ import annotations

import argparse

import torch
from transformers import Trainer, TrainingArguments

from pumch_asr.data.collator import DataCollatorCTCWithPadding
from pumch_asr.data.dataset import KaldiASRDataset
from pumch_asr.models.loader import (
    load_model_for_ctc,
    load_processor,
    maybe_freeze_feature_encoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CTC fine-tuning for SSL speech models"
    )
    parser.add_argument(
        "--train-dir", required=True, help="Kaldi data dir with wav.scp/text"
    )
    parser.add_argument("--valid-dir", help="Optional validation Kaldi data dir")
    parser.add_argument(
        "--model-name", required=True, help="Base HF model name or path"
    )
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--lowercase", action="store_true", help="Lowercase transcripts"
    )
    parser.add_argument("--freeze-feature-encoder", action="store_true")
    parser.add_argument("--ctc-zero-infinity", action="store_true")
    parser.add_argument(
        "--ctc-loss-reduction",
        default="mean",
        choices=["mean", "sum"],
        help="CTC loss reduction for AutoModelForCTC",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-train-epochs", type=float, default=20)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processor = load_processor(
        model_name_or_path=args.model_name,
        vocab_path=args.vocab,
        do_lower_case=args.lowercase,
    )
    sampling_rate = processor.feature_extractor.sampling_rate

    model = load_model_for_ctc(
        args.model_name,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        ctc_loss_reduction=args.ctc_loss_reduction,
    )
    if args.ctc_zero_infinity:
        model.config.ctc_zero_infinity = True
    if args.freeze_feature_encoder:
        maybe_freeze_feature_encoder(model)

    train_dataset = KaldiASRDataset(
        data_dir=args.train_dir,
        processor=processor,
        target_sampling_rate=sampling_rate,
        lowercase=args.lowercase,
    )

    eval_dataset = None
    if args.valid_dir:
        eval_dataset = KaldiASRDataset(
            data_dir=args.valid_dir,
            processor=processor,
            target_sampling_rate=sampling_rate,
            lowercase=args.lowercase,
        )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        logging_strategy="steps",
        report_to="none",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    main()
