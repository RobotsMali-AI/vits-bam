---
language: [bm]
library_name: transformers
pipeline_tag: text-to-audio
license: cc-by-4.0
datasets: [RobotsMali/afvoices-notag, RobotsMali/finBamSpeech]
base_model: RobotsMali/bam-vits
tags: [vits, text-to-speech, bambara, fintech, single-speaker, low-resource]
---

# Bambara VITS — FinTech adaptation

`RobotsMali/bam-vits-fintech` continues training [`RobotsMali/bam-vits`](https://huggingface.co/RobotsMali/bam-vits) on [`RobotsMali/finBamSpeech`](https://huggingface.co/datasets/RobotsMali/finBamSpeech), a higher-quality single-speaker corpus of 800 read Bambara sentences about finance, banking, and financial technology.

> **Research checkpoint — substantially undertrained.** The already undertrained base received only 150 additional epochs (750 training examples, batch size 80). This is still far below the hundreds of thousands of optimizer steps commonly used for VITS; approximately 200,000 steps is our reference budget. No objective or human evaluation is available.

## Model details

- **Architecture:** VITS with a HiFi-GAN-style waveform decoder
- **Checkpoint:** inference-only `transformers.VitsModel`; discriminator removed
- **Input:** lowercased plain Bambara orthography
- **Audio:** 22,050 Hz
- **Training sequence:** English VCTK checkpoint → `afvoices-notag` base → `finBamSpeech` domain adaptation
- **Code:** [RobotsMali-AI/vits-bam](https://github.com/RobotsMali-AI/vits-bam), adapted from [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits)

## Usage

```python
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

repo_id = "RobotsMali/bam-vits-fintech"
text = "Juru sarali waati sera."
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = VitsModel.from_pretrained(repo_id)
inputs = tokenizer(text.lower(), return_tensors="pt")

with torch.no_grad():
    waveform = model(**inputs, speaker_id=0).waveform[0]

sf.write("fintech-bambara.wav", waveform.cpu().numpy(), model.config.sampling_rate)
```

The model config retains 20 speaker embeddings from the AfVoices base, but FinBamSpeech contains only one speaker. The 20 IDs do **not** represent 20 separately FinTech-adapted voices, and the original participant-to-index mapping was not exported. Treat `speaker_id` as an experimental control and audition the result.

## Training data and procedure

The base checkpoint used 21,253 train and 1,128 test examples from the top 20 speakers in [`afvoices-notag`](https://huggingface.co/datasets/RobotsMali/afvoices-notag), a noisy spontaneous-speech selection intended only for experimentation. This variant was then trained for 150 more epochs with per-device batch size 80 on FinBamSpeech's 750-train/50-test split. Its read, domain-specific recordings are higher quality than the base data, but 800 samples remain very small for VITS.

Text is lowercased plain Bambara; audio is processed at 22.05 kHz. The base hyperparameters and training code are documented in [`config/bam-vits.yaml`](https://github.com/RobotsMali-AI/vits-bam/blob/main/config/bam-vits.yaml).

## Evaluation and comparison

No MOS, intelligibility, domain-term accuracy, speaker-similarity, or automated TTS metrics are reported. Domain adaptation should not be interpreted as proven improvement. In informal comparison, plain-orthography checkpoints often sounded slightly more natural than pseudo-IPA variants, with no remarkable pseudo-IPA improvement in convergence or quality.

## Intended use, risks, and limitations

This model is for research on low-resource and domain-adapted Bambara TTS. It may be useful for qualitative experiments with finance-related text, but it is not validated for financial advice, customer communication, accessibility, or production banking systems. Never treat synthesized speech as accurate financial information.

Short training and narrow data can cause noise, poor prosody, incorrect numbers or financial terms, speaker leakage, memorization, and failures on conversational, code-switched, long, or out-of-domain text. No systematic safety, bias, privacy, or memorization testing was performed. Do not impersonate speakers or create deceptive audio; disclose that output is synthetic.

## Related resources

- [`bam-vits`](https://huggingface.co/RobotsMali/bam-vits): plain-orthography base
- [`bam-vits-pseudo-ipa-fintech`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa-fintech): pseudo-IPA counterpart
- [VITS paper](https://proceedings.mlr.press/v139/kim21f.html)

Questions are welcome in the [project repository](https://github.com/RobotsMali-AI/vits-bam/issues) or this model's Community tab.
