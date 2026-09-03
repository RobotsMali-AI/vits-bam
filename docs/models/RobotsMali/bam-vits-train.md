---
language: [bm]
library_name: transformers
license: cc-by-4.0
datasets: [RobotsMali/afvoices-notag]
base_model: ylacombe/vits-vctk-with-discriminator
tags: [vits, text-to-speech, bambara, training-checkpoint, discriminator, low-resource]
---

# Bambara VITS — training checkpoint

`RobotsMali/bam-vits-train` is the continued-training counterpart of [`RobotsMali/bam-vits`](https://huggingface.co/RobotsMali/bam-vits). Its `VitsModelForPreTraining` weights retain both the generator and discriminator (about 323 MB rather than the inference export's 159 MB). Use this repository as `model_name_or_path` when fine-tuning; use `bam-vits` for ordinary inference.

> **Substantially undertrained.** The checkpoint received 200 epochs on a small experimental corpus and remains far below typical VITS budgets of hundreds of thousands of optimizer steps (approximately 200,000 is our reference). It is a starting point, not a converged base model.

## Continue training

Clone the RobotsMali training code, install its pinned dependencies, build monotonic alignment, then point a copied YAML config at this checkpoint and your dataset:

```bash
git clone https://github.com/RobotsMali-AI/vits-bam.git
cd vits-bam
python -m pip install -r requirements.txt
cd monotonic_align && python setup.py build_ext --inplace && cd ..
cp config/bam-vits.yaml config/my-experiment.yaml
```

At minimum, edit:

```yaml
model:
  model_name_or_path: RobotsMali/bam-vits-train
  tokenizer_name: RobotsMali/bam-vits-train
  pseudo_ipa: false
  override_speaker_embeddings: true
  override_vocabulary_embeddings: false

data:
  dataset_name: OWNER/DATASET
  dataset_config_name: null
  audio_column_name: audio
  text_column_name: text
  speaker_id_column_name: speaker_id

logging:
  output_dir: ./tmp/my-experiment
  hub_model_id: OWNER/MODEL
```

Then run:

```bash
accelerate launch run_vits_finetuning.py config/my-experiment.yaml
```

Review speaker and vocabulary settings for your data. This repository's custom training class and compiled monotonic-alignment extension are required. After training, `push_vits_model.py --inference_only` publishes a generator-only `VitsModel`; omitting the flag preserves the discriminator.

## Provenance

The model was initialized from the English VCTK [`ylacombe/vits-vctk-with-discriminator`](https://huggingface.co/ylacombe/vits-vctk-with-discriminator) and trained on `top20-speakers` from [`RobotsMali/afvoices-notag`](https://huggingface.co/datasets/RobotsMali/afvoices-notag): 21,253 training and 1,128 test examples from the 20 participants with the most utterances, excluding tagged transcripts. The corpus is spontaneous, noisy ASR data rather than studio TTS data.

The source config, [`config/bam-vits.yaml`](https://github.com/RobotsMali-AI/vits-bam/blob/main/config/bam-vits.yaml), specifies 200 epochs, per-device batch 80, learning rate 0.0005, FP16, 0.2–20-second audio, maximum 450 tokens, and seed 789. Text was lowercased; audio was resampled to 22.05 kHz.

## Limitations and responsible use

No MOS, intelligibility, pronunciation, speaker-similarity, or safety evaluation is reported. Any downstream model inherits risks from undertraining, noisy data, limited speaker coverage, minimal text normalization, and the English initialization. The source participant-to-model-index mapping was not saved. Evaluate convergence, pronunciation, memorization, bias, licensing, speaker consent, and voice similarity for every derivative.

This checkpoint is for research and fine-tuning. Do not use it to impersonate people, create deceptive audio, or support safety-critical speech without substantial additional training and evaluation. Clearly document downstream data and disclose synthetic audio.

## References

- [RobotsMali training repository](https://github.com/RobotsMali-AI/vits-bam), adapted from [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits)
- [VITS paper](https://proceedings.mlr.press/v139/kim21f.html)
