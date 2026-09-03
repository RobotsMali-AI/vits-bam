---
language: [bm]
library_name: transformers
pipeline_tag: text-to-audio
license: cc-by-4.0
datasets: [RobotsMali/afvoices-notag, RobotsMali/finBamSpeech]
base_model: RobotsMali/bam-vits-pseudo-ipa
tags: [vits, text-to-speech, bambara, pseudo-ipa, fintech, low-resource]
---

# Bambara VITS — pseudo-IPA FinTech adaptation

`RobotsMali/bam-vits-pseudo-ipa-fintech` continues training [`RobotsMali/bam-vits-pseudo-ipa`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa) on the higher-quality, single-speaker [`RobotsMali/finBamSpeech`](https://huggingface.co/datasets/RobotsMali/finBamSpeech): 800 read Bambara sentences about finance, banking, and FinTech.

> **Research checkpoint — substantially undertrained.** The base was undertrained, and this adaptation added only 150 epochs over 750 training examples with batch size 80. It remains far below the hundreds of thousands of optimizer steps typical for VITS (approximately 200,000 is our reference budget). No formal evaluation is available.

## Usage

The tokenizer does not convert ordinary Bambara. Apply the exact training cleaner first:

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

```python
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

repo_id = "RobotsMali/bam-vits-pseudo-ipa-fintech"
text = clean_bambara_pseudo_ipa("Juru sarali waati sera.")
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = VitsModel.from_pretrained(repo_id)
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    waveform = model(**inputs, speaker_id=0).waveform[0]
sf.write("fintech-bambara.wav", waveform.cpu().numpy(), model.config.sampling_rate)
```

The config retains 20 speaker embeddings from the AfVoices base, although the adaptation corpus has one speaker. These are not 20 separately adapted FinTech voices; the original speaker mapping was not exported.

## Training and experiment

Training followed: English [`ylacombe/vits-vctk-with-discriminator`](https://huggingface.co/ylacombe/vits-vctk-with-discriminator) → pseudo-IPA base on `afvoices-notag` (200 epochs, batch 80) → FinBamSpeech (150 more epochs, batch 80). Audio is 22.05 kHz. The base configuration is [`config/bam-vits-pseudo-ipa.yaml`](https://github.com/RobotsMali-AI/vits-bam/blob/main/config/bam-vits-pseudo-ipa.yaml). The published 158.7 MB checkpoint is inference-only; its discriminator was removed.

The pseudo-IPA experiment tested whether IPA-like text would help transfer from an English checkpoint trained on English phonetic inputs. We observed no remarkable quality or convergence improvement; plain Bambara often sounded slightly more natural. Bambara orthography is already largely phonetic, while the English–Bambara acoustic mismatch likely limits useful transfer to the waveform generator. This informal result is not statistically validated.

## Intended use and limitations

This is a research artifact for low-resource TTS, text-representation comparisons, and domain adaptation—not a production or financial-information system. No MOS, intelligibility, financial-term accuracy, speaker similarity, safety, bias, privacy, or memorization evaluation was performed. The cleaner is heuristic pseudo-IPA, not complete linguistic transcription. The tiny, narrow corpus and short training can cause noise, mispronounced numbers/terms, memorization, and failure on code-switching or out-of-domain text.

Do not use the model for safety-critical speech, financial advice, impersonation, or deception. Disclose synthetic output and verify it by listening.

## Related resources

- [`bam-vits-pseudo-ipa`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa): base checkpoint
- [`bam-vits-fintech`](https://huggingface.co/RobotsMali/bam-vits-fintech): plain-orthography counterpart
- [Training repository](https://github.com/RobotsMali-AI/vits-bam), adapted from [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits)
- [VITS paper](https://proceedings.mlr.press/v139/kim21f.html)
