# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Respiratory-sound classification on the SPRSound / BioCAS dataset (2022–2025). WAV + JSON annotations → class-balanced log-mel spectrograms → CNN training (PyTorch) with Optuna search and WandB logging. The project docs (README.md) are in Spanish and contain a long technical reference with known-bug notes marked ⚠; some are stale (e.g. it says `cnns.py` is empty, but it now holds `custom_cnn`). No test suite or linter is configured.

## Commands

Scripts in `workflow/` import siblings as top-level modules (`from pipeline import ...`, `from models.base import ...`), so **run them from inside `workflow/`** (or have it on `PYTHONPATH`). `train.py`'s default data paths (`data/processed/...`) are relative to the cwd.

```bash
pip install -r requirements.txt

# 1. Get data (bash; clones SPRSound, reorganizes into data/raw_data/{train,test}/{wav,json},
#    then runs data_config.py which rewrites paths/class info in config/pipeline.yaml)
bash get_data.sh

# 2. Preprocess (Hydra; any yaml key can be overridden on the CLI)
cd workflow
python run.py
python run.py data.task=1-1 stages.augmentation.enabled=false runner.chunk_size=128

# 3. Train: Optuna search, or a single run
python train.py --n-trials 20 --n-epochs 30 --train-dir data/processed/train --test-dir data/processed/test --wandb-project biocas
python train.py --no-optuna --n-epochs 100 --model efficientnet_b0
```

`workflow/config/pipeline.yaml` ships with absolute Windows paths from another developer's machine; regenerate them with `data_config.py` (or override `data.*` on the CLI) before running.

## Architecture

**Data flow:** `get_data.sh` → `data_config.py` → `PreprocessingPipeline` (`pipeline.py`) → per-sample `.npy` files + `index.csv` → `BioCASDataset` (`dataset.py`) → `train.py`.

- **Two-phase pipeline** (`pipeline.py`) — designed to avoid RAM exhaustion. Phase 1 builds lightweight metadata only (`LoadingStage` with `load_audio=False`, `EventExtractionStage` storing `start_ms`/`end_ms`, `BalancedSamplingStage`). Phase 2 processes `runner.chunk_size` events at a time: load waveform → slice → resample → concatenate → pad → augment → log-mel → SpecAugment → normalize, then drops the waveform. Heavy stages run on GPU when available. `augment=False` (test set) skips waveform augmentation and SpecAugment.
- **Stages** (`workflow/stages/`): all operate on `Batch` = list of `AudioSample` dataclasses (`stages/base.py`: waveform, `feature`, `label_int`, `label_str`, `meta`). Stages are toggled via `stages.<name>.enabled` in the yaml. Per-stage hyperparameters live in the matching yaml section.
- **Config:** Hydra/OmegaConf (`config/pipeline.yaml`). `data_config.py` edits the yaml in place with ruamel.yaml (preserving comments). `data.task` is `"1-1"` (binary) or `"1-2"` (multi-class); `allowed_classes` filters labels.
- **`pipeline_api.py`:** notebook-friendly wrapper (`load_config`, `patch_config`, `run_pipeline`, `save_dataset`, `load_dataset`, `load_or_run`). It persists as `index.csv` + `features/*.npy` — a **different format** from `run.py`'s `save_processed_dataset` (`features.npy`/`labels.npy`/`metadata.json`). `train.py` uses `load_dataset`, so it needs the `pipeline_api` format; `BioCASDataset.from_index` reads one `.npy` per `__getitem__` (safe for Windows `num_workers > 0`).
- **Model registry** (`workflow/models/`): `@register_model("name")` on a `BaseAudioClassifier` subclass populates `MODEL_REGISTRY`. Each class implements `build(trial, num_classes, in_channels)` and pulls its hyperparameters from the Optuna trial (`_suggest_or_default` falls back to defaults when `trial is None`). Registration happens at import time, so `train.py` imports `models.networks`; **a new model file must be imported there (or in `train.py`) to be registered**. Current models: `efficientnet_b0`, `resnet18` (`networks.py`), `custom_cnn` (`cnns.py`, configurable conv blocks/SE/norm). `losses.py` has a multiclass focal loss.
- **`train.py`:** each Optuna trial picks a model from the registry, suggests lr/weight-decay/batch-size, opens a WandB run, and logs per-epoch metrics plus a sample validation image. Maximizes `val_acc`.
- Notebooks: `workflow/research.ipynb`, `workflow/eda.ipynb`, `eda/eda.ipynb`.

## Repo conventions (from README)

Update `requirements.txt` when adding libraries; use descriptive file/variable names; commit before changes that may affect pipeline logic; commit messages in Spanish or English.

`.skills/pytorch-patterns/` holds an untracked local skill with PyTorch guidance.
