# vits-bam: experimental Bambara VITS

This repository contains RobotsMali's first text-to-speech experiments for Bambara (Bamanankan). It modernizes the dependencies and compatibility code from [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits) while leaving the core VITS adversarial training recipe largely unchanged.

The released checkpoints are research artifacts, not production TTS systems. All are substantially undertrained relative to the hundreds of thousands of optimizer steps normally used for VITS; considering 200,000 steps as a reference budget. No formal MOS, intelligibility, speaker-similarity, or safety evaluation has been completed.

## Released resources

| Resource | Purpose |
|---|---|
| [`RobotsMali/bam-vits`](https://huggingface.co/RobotsMali/bam-vits) | Plain Bambara, inference-only base |
| [`RobotsMali/bam-vits-train`](https://huggingface.co/RobotsMali/bam-vits-train) | Plain Bambara base with discriminator for continued training |
| [`RobotsMali/bam-vits-pseudo-ipa`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa) | Pseudo-IPA, inference-only base |
| [`RobotsMali/bam-vits-pseudo-ipa-train`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa-train) | Pseudo-IPA base with discriminator |
| [`RobotsMali/bam-vits-fintech`](https://huggingface.co/RobotsMali/bam-vits-fintech) | Plain Bambara, further adapted on FinBamSpeech |
| [`RobotsMali/bam-vits-pseudo-ipa-fintech`](https://huggingface.co/RobotsMali/bam-vits-pseudo-ipa-fintech) | Pseudo-IPA FinTech adaptation |
| [`RobotsMali/afvoices-notag`](https://huggingface.co/datasets/RobotsMali/afvoices-notag) | Top-20-speaker AfVoices subset without tagged transcripts |
| [`RobotsMali/finBamSpeech`](https://huggingface.co/datasets/RobotsMali/finBamSpeech) | 800 single-speaker read sentences about finance and banking |

Base models were trained for 200 epochs with per-device batch size 80 on `afvoices-notag`. The FinTech variants received 150 additional epochs at the same batch size on FinBamSpeech. See the Hub cards under [`docs/`](docs/) for full provenance, usage, limitations, and responsible-use guidance.

## Plain text versus pseudo-IPA

The two base configurations differ in the `model.pseudo_ipa` flag and resulting text inputs. The pseudo-IPA cleaner converts nasal vowel sequences and maps `c → tʃ` and `j → dʒ`; it is a deterministic heuristic, not a complete linguistic IPA transcription.

The hypothesis was that IPA-like input might help when transferring from an English checkpoint trained with English phonetics. We observed no remarkable quality or convergence improvement; plain Bambara often sounded slightly more natural. Bambara orthography is already largely phonetic, and the large English–Bambara acoustic mismatch likely leaves the transferred HiFi-GAN-style generator little useful phonetic guidance. This is an informal observation from undertrained models, not a statistically validated result.

For pseudo-IPA checkpoints, callers must run `clean_bambara_pseudo_ipa` from [`create_tokenizers.py`](create_tokenizers.py) before tokenization. The tokenizer does not apply it automatically.

## Setup

Use Python 3.12 or newer in an isolated environment:

```bash
git clone https://github.com/RobotsMali-AI/vits-bam.git
cd vits-bam
python -m pip install -r requirements.txt
```

Compile the Cython Monotonic Alignment Search extension before training:

```bash
cd monotonic_align
python setup.py build_ext --inplace
cd ..
```

Authenticate only for operations that need it:

```bash
hf auth login      # private/gated Hub access or publishing
wandb login        # when report_to is wandb
```

Never put access tokens in YAML, source files, or commits.

## Training

The two reproducible base experiment configs are:

- [`config/bam-vits.yaml`](config/bam-vits.yaml): plain Bambara orthography
- [`config/bam-vits-pseudo-ipa.yaml`](config/bam-vits-pseudo-ipa.yaml): pseudo-IPA preprocessing

Launch one process or a configured distributed run with Accelerate:

```bash
accelerate launch run_vits_finetuning.py config/bam-vits.yaml
```

Copy a config before making a new experiment. Give every run a unique `logging.output_dir` and review these fields carefully:

- `model.model_name_or_path`, `model.tokenizer_name`, and `model.pseudo_ipa`
- dataset name/config and audio, text, and speaker columns
- speaker/vocabulary embedding override flags
- sample limits, duration limits, batch size, learning rate, and output Hub ID

The training script maps source speaker identifiers to contiguous indices. That mapping is currently not exported, so published speaker IDs cannot be reliably traced to source `participantId` values. Preserve your own mapping when speaker identity matters.

### Training losses

VITS jointly optimizes a generator and discriminator. Important logs include:

- `train_loss_real_disc`, `train_loss_fake_disc`, and `train_loss_disc`: discriminator terms
- `train_loss_mel`: mel-spectrogram reconstruction term (weight 35 in the Bambara configs)
- `train_loss_kl`: latent-distribution KL term
- `train_loss_duration`: duration-prediction term
- `train_loss_fmaps`: discriminator feature-matching term
- `train_loss_gen`: generator adversarial term

Losses are diagnostics, not substitutes for listening tests or structured perceptual evaluation.

## Export and publish checkpoints

The training model includes a discriminator and is roughly twice the size needed for inference. Publish a continued-training checkpoint by omitting `--inference_only`:

```bash
python push_vits_model.py \
  --save_dir tmp/my-run \
  --repo_id OWNER/MODEL-train
```

Publish a generator-only `transformers.VitsModel` for inference with:

```bash
python push_vits_model.py \
  --save_dir tmp/my-run \
  --repo_id OWNER/MODEL \
  --inference_only
```

## Tokenizer generation

`create_tokenizers.py` builds the character vocabulary and rewrites `vits_bam_tokenizer/`. Its current entry-point dataset values are development defaults; review and update them before running:

```bash
python create_tokenizers.py
```

Inspect every tokenizer diff before committing, because a vocabulary change must match model embedding configuration and preprocessing.

## Hugging Face cards

Hub cards are source-controlled in `docs/models/` and `docs/data/`; mappings live in [`docs/huggingface.yaml`](docs/huggingface.yaml). The sync utility changes only `README.md` in the selected Hub repository.

```bash
# Pull one remote card into its manifest path
python utils/hf_cards.py pull model RobotsMali/bam-vits

# Push one reviewed local card
python utils/hf_cards.py push model RobotsMali/bam-vits
python utils/hf_cards.py push dataset RobotsMali/finBamSpeech
```

Pulling overwrites the corresponding local card. Review `git diff` before and after synchronization.

## Verification

There is no committed automated test suite. Before submitting changes:

```bash
python -m compileall -q .
cd monotonic_align && python setup.py build_ext --inplace && cd ..
```

For training changes, also run the smallest practical sample-limited experiment with a temporary config. Do not commit datasets, checkpoints, generated audio, credentials, or W&B/Hugging Face tokens.

## Attribution and license

The training implementation is derived from [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits), and the architecture comes from [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://proceedings.mlr.press/v139/kim21f.html). Repository code is available under the [MIT License](LICENSE). Dataset and model repositories have their own cards and terms; review them before use.
