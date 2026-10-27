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

## Experiments (`experiments/`)

Standalone analysis scripts, separate from `workflow/`, organised by the two gates in `plan.md`: `gate1_normal_vs_abnormal/` and `gate2_event_type/` (gate 2 is evaluated in isolation, on W/FC events). Each gate holds ordered experiments `eNN_*/` with their code, `results/` (`report.md` = conclusions, `report_auto.md` = generated tables) and a git-ignored `cache/`. Read `experiments/README.md` first: it has the run order and the old→new name map (old reports and figure prefixes `g2_`/`g3_`/`g4_` refer to what are now gate2 `e01`/`e02`/`e03`).

- Shared code and tables live in `experiments/common/`; `paths.py` is the single source of paths (scripts can be run from any directory). Input data comes from `get_data.sh` (`data/raw_data/{train,test}`); run `common/extract_spectral_features.py` first.
- Run with `.venv/Scripts/python` (CPU-only torch). `librosa`/`numba` are blocked on this machine by Windows App Control, so the experiment code uses scipy/numpy for STFT/mel/MFCC.
- Validation is leave-one-year-out (2022–2025): the held-out year is never augmented nor used for early stopping or thresholds.
- Synthetic augmentation (`common/synthesis/`, experiment `gate2_event_type/e04_synthetic_augmentation/`): parametric crackles/wheezes are inserted into real SPRSound normal recordings and written in SPRSound's format (8 kHz mono PCM-16, event-level JSON) to `data/synthetic/` (git-ignored). Keep that method when extending it: never replace the real host audio, apply the dataset's response to the synthetic event only, and draw duration/level from real events of the same class and year. Synthetic data are train-only: drop the held-out year (and `host_split == "test"` for the official split) with `synthesis.sprsound_io.load_manifest`. The MEDZCOOL clips in `Auscultaion_medzcool_audio/` are morphology references only (44.1 kHz MP3 loops of 2–3 unique cycles), never audio to paste.
- `e05_cnn_synthetic/`: with the e04 synthetic data the CNN only benefits when it is used to **pretrain** and then fine-tune on real data (mixing it into real batches hurts; reuse `train_cnn_synth.py --mode pretrain`). `torch` can intermittently fail to import with a Windows App Control error (`torch_python.dll`); it worked on retry, so retry rather than work around it.

## Repo conventions (from README)

Update `requirements.txt` when adding libraries; use descriptive file/variable names; commit before changes that may affect pipeline logic; commit messages in Spanish or English.

`.skills/pytorch-patterns/` holds an untracked local skill with PyTorch guidance.
