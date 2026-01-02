# PUMCH ASR (Kaldi-format SSL fine-tuning)

This project fine-tunes Hugging Face HuBERT, wav2vec 2.0, or WavLM models for CTC char-based ASR on Kaldi-format data directories and supports decoding and encoder feature export to Kaldi `feats.ark`.

## Requirements

- Python 3.9+
- Kaldi-style data directories containing at least `wav.scp` (and `text` for training)

Install dependencies:

```bash
pip install -r requirements.txt
```

## 1) Build a char vocabulary

```bash
python scripts/build_vocab.py \
  --text /path/to/train/text \
  --output /path/to/vocab.json \
  --lowercase
```

## 2) Fine-tune (CTC)

```bash
python scripts/train.py \
  --train-dir /path/to/train \
  --valid-dir /path/to/dev \
  --model-name facebook/wav2vec2-base \
  --vocab /path/to/vocab.json \
  --output-dir /path/to/exp/wav2vec2_ctc \
  --lowercase \
  --freeze-feature-encoder
```

Swap `--model-name` for HuBERT or WavLM checkpoints, e.g.:

- `facebook/hubert-base-ls960`
- `microsoft/wavlm-base`

The fine-tuned model and processor are saved to `--output-dir`.

## 3) Decode with the fine-tuned model

```bash
python scripts/decode.py \
  --data-dir /path/to/test \
  --model-dir /path/to/exp/wav2vec2_ctc \
  --output /path/to/exp/wav2vec2_ctc/hyp.txt
```

## 4) Extract encoder features to Kaldi ark

```bash
python scripts/extract_feats.py \
  --data-dir /path/to/test \
  --model-dir /path/to/exp/wav2vec2_ctc \
  --ark /path/to/exp/wav2vec2_ctc/feats.ark \
  --scp /path/to/exp/wav2vec2_ctc/feats.scp \
  --layer -1
```

## 5) Segment videos with diarization JSON

```bash
python scripts/segment_videos.py \
  --video-dir /Volumes/Elements/pumch/第四波 \
  --diar-dir /Volumes/Elements/pumch/第四波_diar \
  --output-dir /path/to/kaldi/segments \
  --segments-dir /path/to/kaldi/segments/wav
```

The script expects JSON diarization files containing an `utterances` array with
`speaker_id`, `start_time`, `end_time`, and optional `transcript`. It writes
`wav.scp`, `text`, `utt2spk`, and `spk2utt` into `--output-dir`.

## 6) Segment m4a audio with diarization JSON

```bash
python scripts/segment_m4a.py \
  --audio-dir /Volumes/Elements/pumch/对照组36 \
  --diar-dir /Volumes/Elements/pumch/对照组36_diar \
  --output-dir /path/to/kaldi/segments_m4a \
  --segments-dir /path/to/kaldi/segments_m4a/wav
```

Behavior matches `scripts/segment_videos.py` but operates on `.m4a` inputs.

## Notes

- `wav.scp` supports direct wave paths and Kaldi-style pipes ending in `|` (requires the pipe command to output WAV bytes).
- Audio is resampled to the model's expected sampling rate (typically 16 kHz).
- For multi-GPU or advanced settings, pass extra `TrainingArguments` through `scripts/train.py` as needed.
- `scripts/segment_videos.py` relies on `ffmpeg` being available on your PATH.
- `scripts/segment_m4a.py` relies on `ffmpeg` being available on your PATH.
