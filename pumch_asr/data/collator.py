"""Collators for speech batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: bool | str = True

    def __call__(self, features: List[dict]) -> dict:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        if "utt_id" in features[0]:
            batch["utt_id"] = [f["utt_id"] for f in features]
        return batch


@dataclass
class DataCollatorSpeechWithPadding:
    processor: Any
    padding: bool | str = True
    return_attention_mask: bool = True

    def __call__(self, features: List[dict]) -> dict:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=self.return_attention_mask,
        )
        if "utt_id" in features[0]:
            batch["utt_id"] = [f["utt_id"] for f in features]
        return batch
