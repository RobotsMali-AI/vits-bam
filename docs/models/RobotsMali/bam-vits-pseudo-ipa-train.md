---
language: [bm]
library_name: transformers
license: cc-by-4.0
datasets: [RobotsMali/afvoices-notag]
base_model: ylacombe/vits-vctk-with-discriminator
tags: [vits, text-to-speech, bambara, pseudo-ipa, training-checkpoint, discriminator, low-resource]
---

# Bambara VITS — pseudo-IPA training checkpoint

`RobotsMali/bam-vits-pseudo-ipa-train` retains the generator and discriminator in `VitsModelForPreTraining` form for continued training. It corresponds to [`RobotsMali/bam-vits-pseudo-ipa`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa), whose discriminator was removed for inference.

> **Substantially undertrained.** It received 200 epochs on a small, noisy corpus, far below VITS training budgets of hundreds of thousands of steps (approximately 200,000 is our reference). Continue training and evaluate it; do not treat it as converged.

## Usage: required text preprocessing

The pseudo-IPA transform is the only intended experimental difference from the plain-input line. The tokenizer does not apply it automatically. Use the same cleaner for training and any diagnostic inference:

```python
import re


def clean_bambara_pseudo_ipa(text):
    """Translate Bambara text into the pseudo-IPA form used for training."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    nasal_map = {
        "aan": "ãã", "ɛɛn": "ɛ̃ɛ̃", "een": "ẽẽ", "iin": "ĩĩ",
        "ɔɔn": "ɔ̃ɔ̃", "oon": "õõ", "uun": "ũũ", "an": "ã",
        "ɛn": "ɛ̃", "en": "ẽ", "in": "ĩ", "ɔn": "ɔ̃", "on": "õ",
        "un": "ũ",
    }
    for ortho, ipa in nasal_map.items():
        text = re.sub(rf"{ortho}(?![aeɛioɔu])", ipa, text)
    for b_char, ipa_char in {"j": "dʒ", "c": "tʃ", "ɲ": "ɲ", "ŋ": "ŋ"}.items():
        text = text.replace(b_char, ipa_char)
    text = re.sub(r"[^a-zɛɔɲŋãẽĩõṹ̀̂̌̄\s.,!?ʃʒ]", "", text)
    return re.sub(r"\s+", " ", text).strip()
```

## Continue training

Use the [RobotsMali repository](https://github.com/RobotsMali-AI/vits-bam), which imports this cleaner and the custom training model:

```bash
git clone https://github.com/RobotsMali-AI/vits-bam.git
cd vits-bam
python -m pip install -r requirements.txt
cd monotonic_align && python setup.py build_ext --inplace && cd ..
cp config/bam-vits-pseudo-ipa.yaml config/my-pseudo-ipa-experiment.yaml
```

Set `model.model_name_or_path` and `model.tokenizer_name` to `RobotsMali/bam-vits-pseudo-ipa-train`, keep `model.pseudo_ipa: true`, configure your dataset columns/output path, then run:

```bash
accelerate launch run_vits_finetuning.py config/my-pseudo-ipa-experiment.yaml
```

The checkpoint is about 323 MB because it includes the discriminator. Use `push_vits_model.py --inference_only` after training to export a normal `transformers.VitsModel`.

## Provenance and experiment

Training followed English VCTK [`ylacombe/vits-vctk-with-discriminator`](https://huggingface.co/ylacombe/vits-vctk-with-discriminator) → `top20-speakers` from [`RobotsMali/afvoices-notag`](https://huggingface.co/datasets/RobotsMali/afvoices-notag). The split has 21,253 train and 1,128 test examples of spontaneous, variably noisy ASR speech. Tagged transcripts were excluded. The config uses 200 epochs, batch 80, learning rate 0.0005, FP16, and 22.05 kHz audio.

The hypothesis was that IPA-like input might improve transfer from an English checkpoint trained with English phonetics. We observed no remarkable improvement in quality or convergence, and plain Bambara often sounded slightly more natural. Bambara's largely phonetic orthography and the large acoustic mismatch may explain this. The finding is informal and not statistically validated.

## Limitations and responsible use

The cleaner is heuristic pseudo-IPA, not a complete linguistic transcription. No MOS, intelligibility, speaker similarity, safety, bias, privacy, or memorization evaluation is reported. Undertraining and noisy, limited data can propagate poor prosody, pronunciation errors, speaker leakage, and demographic imbalance. The participant-to-speaker-index mapping was not exported.

For every derivative, document data and preprocessing, evaluate quality and consent, and disclose synthetic audio. Do not use this checkpoint for impersonation, deception, or safety-critical speech.

## References

- [Training repository](https://github.com/RobotsMali-AI/vits-bam), adapted from [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits)
- [VITS paper](https://proceedings.mlr.press/v139/kim21f.html)
