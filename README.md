# vits-bam: Fine-Tuning VITS for Bambara Speech Synthesis

This repository contains scripts and configurations for fine-tuning the **VITS (Variational Inference with adversarial learning for Text-to-Speech)** architecture on low-resource African languages, with a primary focus on Bambara. 

It adapts the training framework created by `@ylacombe` to natively support modern `transformers` (v5.x+) and `accelerate` workflows, providing automated text normalization, custom token alignment, and adversarial GAN finetuning.

---

## 🚀 Quick Start Guide

Follow these steps to set up your environment, compile dependency modules, and launch your text-to-speech finetuning experiments.

### 1. Clone the Repository & Install Dependencies
First, pull down the workspace and install the underlying Python package ecosystem. Ensure you are using a Python 3.12+ environment.

```bash
git clone [https://github.com/diarray-hub/vits-bam.git](https://github.com/diarray-hub/vits-bam.git)
cd vits-bam
pip install -r requirements.txt

```

### 2. Authenticate with Hugging Face Hub

Authentication is required to download pretrained base model weight shards and interact with gated or private voice datasets (e.g., `afvoices` configurations).

```bash
git config --global credential.helper store
hf auth login

```

*When prompted, paste your Hugging Face Access Token with `write` privileges.*

### 3. Compile the Monotonic Alignment Search Module

VITS relies on a Monotonic Alignment Search (MAS) algorithm to learn structural alignment mappings between input character text embeddings and output audio mel-spectrogram target frames. To prevent severe CPU-bound training bottlenecks, you must compile the optimized Cython version of this operator:

```bash
# Navigate to the alignment directory
cd monotonic_align

# Compile the Cython extension module in-place
python setup.py build_ext --inplace

# Return to the repository root directory
cd ..

```

### 4. Authenticate Weights & Biases (W&B) Tracking

Adversarial GAN optimization requires tight monitoring across competing generator and discriminator metrics. Log into your Weights & Biases account to track live audio synthesis steps, alignment map plots, and loss parameters:

```bash
wandb login YOUR_WANDB_API_KEY

```

### 5. Launch Distributed / Single-GPU Training

Finetuning execution is controlled via standard flattened configuration schemas. Run the training orchestrator script via PyTorch's `accelerate` launcher.

```bash
accelerate launch run_vits_finetuning.py ./config/bam_exp1.yaml

```

---

## 🛠️ Experiment Configuration (`.yaml`) Overview

Finetuning runs are customized using unified config files split into four structural parameter block maps:

```yaml
model:
  model_name_or_path: "ylacombe/vits-ljs-with-discriminator"
  override_speaker_embeddings: true   # Forces embedding resizing for custom speaker spaces
  override_vocabulary_embeddings: true # Handles target vocabulary tokenizer mutations

data:
  project_name: "bam_vits_finetuning"
  dataset_name: "Panga-Azazia/all-in-one"
  dataset_config_name: "afvoices"
  audio_column_name: "audio"
  text_column_name: "text"
  max_tokens_length: 450              # Set as strict integer to avoid array slice index type errors
  do_normalize: false                 # Toggles programmatic text normalizers

training:
  do_train: true
  do_eval: true
  num_train_epochs: 200
  per_device_train_batch_size: 4
  learning_rate: 0.00002              # Written as raw decimal float for explicit PyTorch parser safety
  fp16: true

logging:
  output_dir: "./tmp/bam_vits_finetuning"
  report_to: "wandb"
  save_steps: 500

```

---

## 📈 Understanding the Adversarial Loss Trackers

During optimization, you will observe several distinct loss components inside your logging terminals and W&B dashboards. Because VITS uses a generative adversarial training lifecycle, monitor the balance between these keys:

### Discriminator Metrics

* **`train_loss_real_disc`**: Measures how effectively the multi-period and multi-scale discriminator networks identify true human speech signals from the training data.
* **`train_loss_fake_disc`**: Measures how effectively the discriminator catches synthetic audio clips produced by the generator.
* **`train_loss_disc`**: The total combined structural loss value optimized by the discriminator.

### Generator Metrics

* **`train_loss_mel`**: Evaluates the $L_1$ reconstruction penalty between the synthesized audio's mel-spectrogram and the source target properties. This carries the heaviest structural optimization weight (`weight_mel: 35.0`).
* **`train_loss_kl`**: The Kullback-Leibler divergence tracking how close the projected text encoder states match the true posterior target distributions.
* **`train_loss_duration`**: Optimizes the model's structural duration predictor to match the length constraints output by the Monotonic Alignment Search block.
* **`train_loss_fmaps`**: Feature-matching penalty calculating feature distance variations between intermediate layers of the discriminator to force the generator to mimic high-frequency acoustic details.
* **`train_loss_gen`**: The pure adversarial loss driving the generator to construct audio capable of fooling the active discriminator networks.

```
