# Repository Guidelines

## Project Structure & Module Organization

The main training entry point is `run_vits_finetuning.py`. Reusable model, feature-extraction, plotting, configuration, and romanization code lives in `utils/`. Experiment settings are YAML files under `config/`; keep reusable upstream examples in `original_training_config_examples/`. `monotonic_align/` contains the Cython alignment extension required for efficient training. Tokenizer artifacts are versioned in `vits_bam_tokenizer/`, while Hugging Face model and dataset cards live under `docs/models/` and `docs/data/`. Root-level utilities create tokenizers, convert discriminator checkpoints, and publish trained models.

## Setup, Build, and Development Commands

Use Python 3.12 or newer in an isolated environment.

```bash
python -m pip install -r requirements.txt
cd monotonic_align && python setup.py build_ext --inplace && cd ..
accelerate launch run_vits_finetuning.py config/bam-vits.yaml
python create_tokenizers.py
python push_vits_model.py --save_dir tmp/run --repo_id OWNER/MODEL --inference_only
```

The first command installs pinned dependencies. Build the Cython module before training. Training reads one flattened YAML configuration. Tokenizer generation downloads its configured Hugging Face dataset and rewrites `vits_bam_tokenizer/`; review changes before committing. Publishing requires prior Hugging Face authentication.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and `UPPER_CASE` constants. Keep imports grouped by standard library, third-party packages, then local modules. Add concise docstrings to public helpers and type hints when changing interfaces. Name experiment configs descriptively, following `bam-vits[-variant].yaml`. No formatter or linter is currently configured; keep edits PEP 8-compatible and avoid unrelated reformatting.

## Testing Guidelines

There is no committed automated test suite or coverage threshold. Before opening a PR, run a syntax check with `python -m compileall -q .`, rebuild `monotonic_align`, and exercise the smallest practical training run by limiting samples in a temporary config. For normalization or tokenizer changes, add focused tests under a new `tests/` directory using names such as `test_romanize.py`; prefer deterministic text fixtures over network-dependent datasets.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Fix bugs and update config`. Keep each commit focused and describe the affected component explicitly. PRs should explain the motivation, list commands run, identify changed configs or datasets, and link relevant issues. Include sample metrics, W&B run links, or generated-audio comparisons for training behavior changes. Never commit Hugging Face or W&B tokens, downloaded datasets, checkpoints, or large generated artifacts.
