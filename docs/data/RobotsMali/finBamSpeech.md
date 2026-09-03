---
license: cc-by-4.0
language: [bm]
task_categories: [text-to-speech]
pretty_name: FinBamSpeech
tags: [bambara, speech, audio, tts, finance, banking, fintech, robotsmali]
dataset_info:
  features:
  - name: text
    dtype: string
  - name: audio
    dtype: audio
  - name: speakerID
    dtype: int64
  - name: split
    dtype: string
  splits:
  - name: train
    num_bytes: 303420224
    num_examples: 750
  - name: test
    num_bytes: 20638279
    num_examples: 50
  download_size: 324050847
  dataset_size: 324058503
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

# FinBamSpeech

FinBamSpeech is a small, higher-quality, single-speaker corpus of **800 read Bambara sentences** about finance, banking, and financial technology. RobotsMali created it to test domain adaptation of its first Bambara VITS checkpoints.

The dataset's narrow domain and single voice make it useful for controlled exploratory fine-tuning, but 800 utterances are not enough to claim broad language, speaker, or topic coverage.

## Quick facts

| Item | Value |
|---|---:|
| Train examples | 750 |
| Test examples | 50 |
| Speakers | 1 |
| Speech style | Read sentences |
| Domain | Finance, banking, and FinTech |
| Audio | Embedded WAV audio |
| Approximate packaged size | 324 MB |

## Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset("RobotsMali/finBamSpeech")
sample = dataset["train"][0]
print(sample["text"], sample["speakerID"])
decoded = sample["audio"].get_all_samples()
print(decoded.sample_rate, decoded.data.shape)
```

## Fields

- `text` (`string`): the Bambara sentence read by the speaker
- `audio` (`audio`): decoded audio and its sampling rate/path information
- `speakerID` (`int64`): constant identifier for the single speaker; it is not a public identity
- `split` (`string`): source split label, duplicating the Dataset split organization

## Intended uses

- Experimental single-speaker Bambara TTS fine-tuning
- Research on adaptation to finance-related vocabulary and read speech
- Reproduction and comparison of RobotsMali's `bam-vits-fintech` checkpoints
- Qualitative study of plain Bambara versus pseudo-IPA input

This dataset does not certify a model for financial advice, banking transactions, customer support, accessibility, or other high-stakes use.

## Limitations and responsible use

- Only one speaker is represented; voice, accent, age, and demographic diversity cannot be inferred.
- The domain is deliberately narrow and may not generalize to conversational or general-purpose Bambara.
- Read sentences do not capture spontaneous speech, dialogue, code-switching, or natural interaction.
- The train/test split is very small, and repeated sentence patterns may make results unstable.
- Higher quality than `afvoices-notag` is a relative description, not a published studio-quality measurement.
- No published audit covers pronunciation, transcription accuracy, acoustic conditions, speaker privacy, or representativeness.

Do not attempt to identify or impersonate the speaker. Evaluate memorization and voice similarity in any derived model, disclose synthetic speech, and obtain appropriate review before deployment.

## Models trained on this dataset

- [`RobotsMali/bam-vits-fintech`](https://huggingface.co/RobotsMali/bam-vits-fintech)
- [`RobotsMali/bam-vits-pseudo-ipa-fintech`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa-fintech)

Each model received only 150 additional epochs with batch size 80 on top of an already undertrained base. They remain research checkpoints and have no formal perceptual evaluation.

## Attribution

When using the dataset, cite it as `RobotsMali/finBamSpeech`, link to this repository, state the version/commit used, and describe any filtering or normalization. Questions are welcome in the dataset's Community tab.
