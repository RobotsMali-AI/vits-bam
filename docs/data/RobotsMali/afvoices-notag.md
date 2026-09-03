---
license: cc-by-4.0
language: [bm]
task_categories: [text-to-speech, automatic-speech-recognition]
pretty_name: AfVoices Top-20 Speakers without Tags
tags: [bambara, speech, audio, tts, african-next-voices, robotsmali]
dataset_info:
  config_name: top20-speakers
  features:
  - name: text
    dtype: string
  - name: audio
    dtype: string
  - name: participantId
    dtype: string
  - name: split
    dtype: string
  splits:
  - name: train
    num_bytes: 4177768.520575488
    num_examples: 21253
  - name: test
    num_bytes: 221734.47942451187
    num_examples: 1128
  download_size: 871651
  dataset_size: 4399503.0
configs:
- config_name: top20-speakers
  data_files:
  - split: train
    path: top20-speakers/train-*
  - split: test
    path: top20-speakers/test-*
---

# AfVoices Top-20 Speakers without Tags

`RobotsMali/afvoices-notag` is a small experimental TTS-oriented selection derived from [`RobotsMali/afvoices`](https://huggingface.co/datasets/RobotsMali/afvoices), the African Next Voices Bambara speech corpus. It contains the 20 participants with the highest utterance counts and excludes transcripts containing semantic/acoustic annotation tags.

This is the dataset used for RobotsMali's first Bambara VITS experiments. It is **not a high-quality studio TTS corpus**: the source is spontaneous speech collected and annotated primarily for ASR. The subset was judged sufficient to test training and text-representation hypotheses, not to build a production voice.

## Quick facts

| Item | Value |
|---|---:|
| Configuration | `top20-speakers` |
| Train examples | 21,253 |
| Test examples | 1,128 |
| Selected speakers | 20 |
| Language | Bambara (`bm`) |
| Speech style | Spontaneous |
| Audio storage | Public URLs in the `audio` string column |

The small Parquet files contain metadata and audio URLs, not embedded waveforms. Network access is therefore required to retrieve audio, and long-term use depends on those URLs remaining available.

## Load the dataset

```python
from datasets import Audio, load_dataset

dataset = load_dataset(
    "RobotsMali/afvoices-notag",
    "top20-speakers",
)

# Decode and resample URL-backed audio when it is accessed.
dataset = dataset.cast_column("audio", Audio(sampling_rate=22_050))
sample = dataset["train"][0]
print(sample["text"], sample["participantId"])
```

## Fields

- `text` (`string`): Bambara transcript after removal of examples containing semantic/acoustic tags
- `audio` (`string`): URL of the audio segment; cast it to `datasets.Audio` for decoding
- `participantId` (`string`): source participant identifier; treat it as a pseudonymous identifier, not a name
- `split` (`string`): source split label, duplicating the Dataset split organization

## Creation and relationship to AfVoices

The parent AfVoices release contains spontaneous speech collected in southern Mali through the African Next Voices project. Its transcription workflow combined automatic pre-labeling and human correction. This derivative selects the top 20 speakers by utterance count and removes examples whose transcripts include the semantic/acoustic tags used in AfVoices, such as markers for noise, code-switching, or inaudible speech.

Removing tagged transcripts does **not** guarantee clean audio: untagged background noise, recording variation, disfluencies, and transcription errors can remain. Refer to the [full AfVoices card](https://huggingface.co/datasets/RobotsMali/afvoices) and [processing repository](https://github.com/RobotsMali-AI/afvoices) for collection context.

## Intended use and limitations

The dataset is intended for exploratory TTS/ASR work, data-pipeline testing, and reproduction of the `bam-vits` experiments. It should not be described as balanced, representative, studio quality, or sufficient for production TTS.

- Selecting by utterance count overrepresents frequent contributors and excludes the broader AfVoices speaker distribution.
- The train/test split is not documented as a speaker-disjoint evaluation.
- Spontaneous ASR recordings have inconsistent prosody, environment, microphones, and transcript fidelity for TTS.
- Excluding tagged transcripts introduces selection bias and does not prove absence of acoustic events.
- The dataset may contain regional, demographic, topic, and recording-condition imbalance inherited from AfVoices.
- The public `participantId` field and recognizable voices require privacy-aware use. Do not attempt to identify speakers or build impersonation systems.

Users should listen to samples, audit text/audio alignment, check consent and licensing for their use case, and document any additional filtering.

## Models trained on this dataset

This subset trained the base [`bam-vits`](https://huggingface.co/RobotsMali/bam-vits) and [`bam-vits-pseudo-ipa`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa) experimental lines. Both are substantially undertrained and have no formal perceptual evaluation.

## Citation

Please cite the parent African Next Voices work and identify this derivative by repository ID.

```bibtex
@misc{diarra2025dealinghardfactslowresource,
  title={Dealing with the Hard Facts of Low-Resource African NLP},
  author={Yacouba Diarra and Nouhoum Souleymane Coulibaly and Panga Azazia Kamaté and Madani Amadou Tall and Emmanuel Élisé Koné and Aymane Dembélé and Michael Leventhal},
  year={2025},
  eprint={2511.18557},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2511.18557}
}
```

Questions are welcome on the dataset Community tab or in the [AfVoices repository](https://github.com/RobotsMali-AI/afvoices/issues).
