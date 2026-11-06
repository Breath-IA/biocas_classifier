# gate 2 · e07 — progressive parameter reduction of the 3 best CNNs (Colab GPU + Weights & Biases)

`gate2_param_reduction_colab.ipynb` takes the three models with the best macro-F1 **on the 2025 fold** of e06's `cv_per_fold.csv` (`input/cv_per_fold.csv`, condition `real`, `RANK_FOLD = "2025"`):
**`efficientnet_b0` 0.746, `crnn` 0.737, `cnn_deep` 0.735** (`resnet_small` is 4th with 0.730). Set `RANK_FOLD = None` to rank by the mean over the three folds instead
(that gives `cnn_deep` 0.839, `crnn` 0.834, `resnet_small` 0.829; the 0.93 / 0.91 seen in the CSV are single-fold values of the 2024 year). For this test the models are also retrained **only on fold 2025**
(preset `fold2025`: train 2022–2024, test 2025).

Each model is scaled to **≤ 1 M trainable parameters** (the FPGA cap of `full_model`) and shrunk step by step: **1.0 → 0.7 → 0.5 → 0.35 → 0.25 → 0.15 → 0.08 M** (`TARGETS_M`). The sizes come from a width multiplier
on every channel count (the GRU size of `crnn`; the width multiplier of torchvision's EfficientNet-B0 layout, 4.0 M at 1.0). For each target the notebook picks the largest width that does not exceed it and prints the table.

> Why "from 1 M down": in e06 the three models are already below 1 M (`cnn_deep` 0.60 M, `crnn` 0.39 M, `resnet_small` 0.32 M), so the study starts each at the 1 M budget (EfficientNet-B0 is 4.0 M in e06 and has to be reduced to get there) and reduces from there.
> Change `TARGETS_M` for another range (e.g. `[0.6, 0.4, 0.25, 0.15, 0.08]`).

Same data, log-mel recipe, GPU augmentation and training recipe as e06 (real events only; leave-one-year-out folds 2023/2024/2025 with 2022 always in training; early stopping on a validation split of
real training patients; the held-out year never takes part in training or selection).

## Run (VS Code + Colab extension)

1. Open the notebook → *Select Kernel → Colab* → GPU runtime (T4). The notebook **stops with an error if CUDA is not available** (`REQUIRE_GPU`) and prints the GPU name and memory.
2. Weights & Biases (online, your account; `WANDB_MODE = "online"` is the default): get your key at https://wandb.ai/authorize and add it as a Colab secret named `WANDB_API_KEY`
   (enable notebook access), or leave it out and paste it into the hidden prompt of the setup cell. The login is verified (a wrong key stops the notebook instead of silently logging offline) and prints
   the user name. Set `WANDB_ENTITY` to a team name to log there; empty = your own account. Runs appear in project `gate2-param-reduction`.
3. Optional: upload `input/cv_per_fold.csv` to `/content/cv_per_fold.csv` to recompute the ranking (the values are also embedded in the options cell). `TOP_K` selects how many models.
4. Run with `PRESET = "smoke"` (1 seed, 2 sizes, 3 epochs: minutes), then `"fold2025"` (**this test**: fold 2025 only, 2 seeds × 7 sizes × 3 models = 42 trainings, ≤ 30 epochs); `"long"` / `"full"` use the three folds.
   `TIME_BUDGET_MIN` ends the loop cleanly before the session limit; a checkpoint after each (model, size, fold) lets the training cell resume after a disconnection.

## Weights & Biases

- One run per training: `{model}-{size}k-fold{year}-s{seed}`, grouped by model. Logged per epoch: `train_loss`, `val_auc`, `val_macro_f1`, `best_val_auc`; summary: all held-out metrics, parameters, epochs, seconds;
  and the **confusion matrix of that test** (image + interactive W&B confusion matrix).
- A final run `summary-<preset>` with the tables (`report_final`, `report_per_run`, confusion-matrix counts), the parameters-vs-performance chart, one confusion-matrix figure per model and the CSV files.

## Outputs (`/content/outputs`, zipped in `/content/gate2_param_reduction_results.zip`)

| file | content |
|---|---|
| `report_per_run.csv` | every training (model, size, fold, seed): validation AUC, held-out AUC / precision / recall / F1 / macro-F1, confusion-matrix counts, epochs, seconds |
| `confusion_matrices.csv` | confusion-matrix counts (W→W, W→FC, FC→W, FC→FC) of every test, per seed and for the seed ensemble |
| `confusion_matrices/cm_<model>.png` | confusion matrices, rows = size, columns = held-out year (seed ensemble) |
| `report_per_fold.csv` | seed ensemble per (model, size, fold) |
| `report_summary.csv` | mean ± std over the folds per (model, size) |
| `report_final.csv` | **last step:** the summary plus `recommended` (smallest size within `TOLERANCE` = 0.01 macro-F1 of the model's best), `rank_macro_f1`, `under_1M` |
| `params_vs_performance.png` | macro-F1 and AUC against parameters (log axis) for each model |

Status: the notebook logic was run end to end on a CPU with synthetic data and W&B offline (training, checkpoint, reports, CSVs, charts); it has not been run on a Colab GPU or with real data.
