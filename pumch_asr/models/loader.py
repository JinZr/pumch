"""Model and processor utilities."""

from __future__ import annotations

from typing import Optional

from transformers import (
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
)

from pumch_asr.config import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER


def create_processor_from_vocab(
    base_model: str,
    vocab_path: str,
    do_lower_case: bool = False,
) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        word_delimiter_token=WORD_DELIMITER,
        do_lower_case=do_lower_case,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor


def load_processor(
    model_name_or_path: str,
    vocab_path: Optional[str] = None,
    do_lower_case: bool = False,
):
    if vocab_path:
        return create_processor_from_vocab(
            base_model=model_name_or_path,
            vocab_path=vocab_path,
            do_lower_case=do_lower_case,
        )
    return AutoProcessor.from_pretrained(model_name_or_path)


def load_model_for_ctc(
    model_name_or_path: str,
    vocab_size: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    ctc_loss_reduction: Optional[str] = None,
):
    kwargs = {}
    if vocab_size is not None:
        kwargs.update(
            {
                "vocab_size": vocab_size,
                "pad_token_id": pad_token_id,
                "bos_token_id": bos_token_id,
                "eos_token_id": eos_token_id,
            }
        )
    if ctc_loss_reduction is not None:
        kwargs["ctc_loss_reduction"] = ctc_loss_reduction
    return AutoModelForCTC.from_pretrained(model_name_or_path, **kwargs)


def maybe_freeze_feature_encoder(model) -> None:
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    elif hasattr(model, "freeze_feature_extractor"):
        model.freeze_feature_extractor()


def get_base_encoder(model):
    for attr in ("wav2vec2", "hubert", "wavlm"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "base_model"):
        return model.base_model
    return model
