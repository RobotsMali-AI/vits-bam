---
language: [bm]
library_name: transformers
pipeline_tag: text-to-audio
license: cc-by-4.0
datasets: [RobotsMali/afvoices-notag]
base_model: ylacombe/vits-vctk-with-discriminator
tags: [vits, text-to-speech, bambara, bamanankan, multispeaker, low-resource]
---

# Bambara VITS

`RobotsMali/bam-vits` is an experimental multi-speaker Bambara (Bamanankan, `bm`) text-to-speech checkpoint from [RobotsMali AI4D Lab](https://robotsmali.org/). It adapts VITS from [`ylacombe/vits-vctk-with-discriminator`](https://huggingface.co/ylacombe/vits-vctk-with-discriminator), an English VCTK checkpoint, to Bambara.

> **Research checkpoint — substantially undertrained.** This was produced as RobotsMali's first TTS experiment. It was trained for 200 epochs on a small, noisy subset. VITS systems are normally trained for hundreds of thousands of optimizer steps; RobotsMali uses roughly 200,000 steps as a useful reference budget. This run is far below that scale. Expect unstable pronunciation, noise, unnatural prosody, speaker leakage, and occasional unintelligible output. No objective or human evaluation is available.

## Model details

- **Architecture:** VITS with stochastic duration prediction and a HiFi-GAN-style waveform decoder
- **Checkpoint:** inference-only `transformers.VitsModel`; the discriminator was removed
- **Input:** lowercased Bambara in Latin orthography; no phonemizer or uroman step
- **Audio:** 22,050 Hz
- **Speakers:** 20 learned embeddings
- **Code:** [RobotsMali-AI/vits-bam](https://github.com/RobotsMali-AI/vits-bam), adapted from [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits)
- **Paper:** [Kim, Kong, and Son (2021)](https://proceedings.mlr.press/v139/kim21f.html)

The project upgrades the upstream dependencies while keeping its core training approach largely unchanged.

## Usage

```bash
pip install "transformers>=5.9" torch soundfile
```

```python
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

repo_id = "RobotsMali/bam-vits"
text = "An ka taa sugu la."
speaker_id = 0  # valid range: 0 through 19

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = VitsModel.from_pretrained(repo_id)
inputs = tokenizer(text.lower(), return_tensors="pt")

with torch.no_grad():
    waveform = model(**inputs, speaker_id=speaker_id).waveform[0]

sf.write("bambara.wav", waveform.cpu().numpy(), model.config.sampling_rate)
```

The training script remapped participant identifiers to contiguous indices without exporting that mapping. Model IDs `0–19` therefore cannot be reliably mapped back to named AfVoices participants. Audition them for research; do not present an ID as a verified identity.

## Training

The model used `top20-speakers` from [`RobotsMali/afvoices-notag`](https://huggingface.co/datasets/RobotsMali/afvoices-notag): 21,253 train and 1,128 test examples from the 20 AfVoices participants with the most utterances, excluding transcripts with semantic/acoustic tags. AfVoices is spontaneous, variably noisy speech collected for ASR rather than studio TTS.

The reproducible configuration is [`config/bam-vits.yaml`](https://github.com/RobotsMali-AI/vits-bam/blob/main/config/bam-vits.yaml).

| Setting | Value |
|---|---:|
| Epochs | 200 |
| Per-device train batch size | 80 |
| Learning rate | 0.0005 |
| Precision | FP16 |
| Audio duration filter | 0.2–20 s |
| Maximum token length | 450 |
| Seed | 789 |

Vocabulary and speaker embeddings were resized. Text was lowercased and audio resampled to 22.05 kHz.

## Evaluation and experimental finding

No MOS, intelligibility, pronunciation, speaker-similarity, or automated TTS results are reported. The held-out split was used during training, but losses are not perceptual evaluation.

Compared informally with [`bam-vits-pseudo-ipa`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa), pseudo-IPA produced no remarkable improvement in quality or convergence; plain Bambara often sounded slightly more natural. RobotsMali's working explanation is that Bambara orthography is already largely phonetic and that the large English–Bambara acoustic mismatch gave the transferred waveform generator little useful English-phonetic guidance. This is an observation from an undertrained experiment, not a controlled conclusion.

## Intended use, risks, and limitations

This release is for low-resource TTS research, listening experiments, baselines, and further fine-tuning. It is not production quality. The limited, non-studio corpus can cause noise, disfluencies, pronunciation errors, demographic imbalance, and speaker leakage. Numbers, abbreviations, foreign words, code-switching, unusual punctuation, long text, and non-Bambara input may fail. Safety, bias, memorization, and voice similarity have not been systematically evaluated.

Do not use it where intelligibility or identity is safety-critical, to impersonate a person, or to create deceptive audio. Use short Bambara sentences, listen critically, and disclose that output is synthetic and experimental.

## Related checkpoints

- [`bam-vits-train`](https://huggingface.co/RobotsMali/bam-vits-train): discriminator retained for continued training
- [`bam-vits-fintech`](https://huggingface.co/RobotsMali/bam-vits-fintech): adapted on single-speaker FinBamSpeech
- [`bam-vits-pseudo-ipa`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa): pseudo-IPA experiment

## Citation

Please cite VITS and identify this checkpoint by repository ID.

```bibtex
@inproceedings{kim2021vits,
  title={Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  pages={5530--5540},
  year={2021}
}
```

Questions are welcome in the [project repository](https://github.com/RobotsMali-AI/vits-bam/issues) or this model's Community tab.
