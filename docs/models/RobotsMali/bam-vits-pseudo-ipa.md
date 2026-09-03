---
language: [bm]
library_name: transformers
pipeline_tag: text-to-audio
license: cc-by-4.0
datasets: [RobotsMali/afvoices-notag]
base_model: ylacombe/vits-vctk-with-discriminator
tags: [vits, text-to-speech, bambara, pseudo-ipa, multispeaker, low-resource]
---

# Bambara VITS — pseudo-IPA experiment

`RobotsMali/bam-vits-pseudo-ipa` is an experimental 20-speaker Bambara VITS checkpoint from [RobotsMali AI4D Lab](https://robotsmali.org/). It differs from [`RobotsMali/bam-vits`](https://huggingface.co/RobotsMali/bam-vits) in its **text input representation**, not its VITS architecture: Bambara spelling is converted by a deterministic pseudo-IPA cleaner before tokenization.

> **Research checkpoint — substantially undertrained.** The model is substantially undertrained. It received 200 epochs on a small, noisy corpus, far below the hundreds of thousands of optimizer steps normally used for VITS (approximately 200,000 is our reference budget). No formal evaluation is available; expect unstable, noisy, or unintelligible output.

## Research question and finding

The experiment tested whether IPA-like inputs would improve quality or convergence when adapting an English checkpoint trained with English phonetic inputs. We observed no remarkable improvement. Plain-orthography checkpoints often sounded slightly more natural. Bambara orthography is already largely phonetic, and the large Bambara–English acoustic difference likely gave the transferred HiFi-GAN-style generator little useful guidance from either spelling scheme. This is an informal result from undertrained models, not a controlled conclusion.

## Usage

The tokenizer does **not** perform pseudo-IPA conversion itself. Apply the same function used in training before every inference request:

```python
import re


def clean_bambara_pseudo_ipa(text):
    """
    Translates pure Bambara text into pseudo-IPA form.
    Normalizes nasalizations and specific consonants (c -> tʃ, j -> dʒ).
    """
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

    sound_alignments = {"j": "dʒ", "c": "tʃ", "ɲ": "ɲ", "ŋ": "ŋ"}
    for b_char, ipa_char in sound_alignments.items():
        text = text.replace(b_char, ipa_char)

    text = re.sub(r"[^a-zɛɔɲŋãẽĩõṹ̀̂̌̄\s.,!?ʃʒ]", "", text)
    return re.sub(r"\s+", " ", text).strip()
```

```python
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

repo_id = "RobotsMali/bam-vits-pseudo-ipa"
text = clean_bambara_pseudo_ipa("An ka taa sugu la.")
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = VitsModel.from_pretrained(repo_id)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    waveform = model(**inputs, speaker_id=0).waveform[0]

sf.write("bambara.wav", waveform.cpu().numpy(), model.config.sampling_rate)
```

Use speaker IDs `0–19`. Their mapping to original AfVoices participant IDs was not exported and should not be treated as identity metadata.

## Training

The model was initialized from [`ylacombe/vits-vctk-with-discriminator`](https://huggingface.co/ylacombe/vits-vctk-with-discriminator) and trained on the 21,253-train/1,128-test `top20-speakers` split of [`RobotsMali/afvoices-notag`](https://huggingface.co/datasets/RobotsMali/afvoices-notag). This is spontaneous, variably noisy ASR speech, not studio TTS data. Transcripts with semantic/acoustic tags were excluded.

The configuration is [`config/bam-vits-pseudo-ipa.yaml`](https://github.com/RobotsMali-AI/vits-bam/blob/main/config/bam-vits-pseudo-ipa.yaml): 200 epochs, per-device batch size 80, learning rate 0.0005, FP16, 0.2–20-second audio, maximum 450 tokens, and seed 789. Audio was resampled to 22.05 kHz. The published checkpoint is inference-only `VitsModel`; its discriminator was removed.

## Intended use and limitations

This is a baseline for research into Bambara text representations, low-resource transfer, and continued fine-tuning—not a production voice. No MOS, intelligibility, pronunciation, speaker-similarity, safety, bias, memorization, or voice-similarity evaluation was performed. The dataset and short training can yield noise, poor prosody, pronunciation errors, speaker leakage, and demographic imbalance. The cleaner is only a heuristic: it is not a linguistic IPA transcription or a complete grapheme-to-phoneme system. Numbers, abbreviations, code-switching, foreign words, unusual punctuation, and long/non-Bambara text may fail.

Do not use the model for safety-critical speech, impersonation, or deceptive audio. Disclose synthetic use and verify all output by listening.

## Related resources

- [`bam-vits-pseudo-ipa-train`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa-train): discriminator retained
- [`bam-vits-pseudo-ipa-fintech`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa-fintech): additional FinBamSpeech adaptation
- [Training code](https://github.com/RobotsMali-AI/vits-bam), adapted from [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits)
- [VITS paper](https://proceedings.mlr.press/v139/kim21f.html)

Questions are welcome in the [project repository](https://github.com/RobotsMali-AI/vits-bam/issues) or this model's Community tab.
