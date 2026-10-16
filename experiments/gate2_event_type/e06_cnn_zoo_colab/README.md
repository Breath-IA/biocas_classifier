# gate 2 · e06 — CNN zoo on a Colab GPU (VS Code Colab extension)

`gate2_cnn_zoo_colab.ipynb` trains eight CNNs on the SPRSound W / FC events, compares them with leave-one-year-out cross-validation
(held-out years 2023, 2024, 2025; 2022 holds most of the data and is always in the training set), ranks them by mean validation AUC and exports the
best one (3-seed ensemble, official fold 2025) to ONNX. Models: `cnn_baseline`, `cnn_se`, `cnn_wide`, `crnn`, `resnet_small`, `cnn_deep`,
`efficientnet_b0`, `resnet18`. Each is trained with real data only (`real`) and, if the synthetic set is available, with synthetic pretraining (`pretrain`).

## Run it from VS Code

1. Install the **Google Colab** extension in VS Code (and the Jupyter extension) and open `gate2_cnn_zoo_colab.ipynb`.
2. *Select Kernel → Colab* and choose a GPU runtime (T4 is enough). Everything runs on the Colab machine (`/content`); the notebook downloads SPRSound itself
   and installs what it needs. It does not need this repository.
3. Edit the options cell: `PRESET`, `TIME_BUDGET_MIN`, and (for the `pretrain` condition) `SYNTH_URL` / `SYNTH_NPZ`.
4. Run all cells. The last cell zips the results (`best_model.onnx`, `best_metrics.json`, confusion matrices, `cv_summary.csv`, `cv_per_fold.csv`, `cv_comparison.png`).
   Download the zip from the Colab machine before closing the session: `/content` is erased when the runtime is recycled.
   If the extension does not offer a file download, set `SAVE_TO_DRIVE = True` (works only if the extension supports the Drive mount).

Synthetic set (optional, makes `pretrain` possible): build `synth_v2.npz` locally with the e05 instructions, upload it somewhere downloadable (a Google Drive file shared by link) and
put the link in `SYNTH_URL`. Without it only the `real` condition runs.

## Suggested test sequence

| step | `PRESET` | what runs | purpose |
|---|---|---|---|
| 1 | `smoke` | fold 2025, 1 seed, 3 models, 5 epochs | a few minutes: checks data download, GPU, outputs; read the time per model printed in the log |
| 2 | `long` (**the long starting test**) | folds 2023–2025, 2 seeds, all 8 models, up to 30 epochs, both conditions | first serious comparison; set `TIME_BUDGET_MIN` from the time measured in step 1 |
| 3 | `full` | same with 3 seeds and up to 40 epochs | final numbers for the best candidates only (reduce `MODELS` to the top 3 of step 2) |

`long` is 8 models × 3 folds × 2 seeds × 2 conditions = 96 trainings (48 without the synthetic set). I could only time it on a CPU, not on a GPU, so I can't give a reliable GPU duration:
the notebook prints the elapsed time per model and the minutes left, so measure it with `smoke` first. ResNet-18 and EfficientNet-B0 are by far the most expensive models.

## Keeping the GPU session alive until the run ends

Colab session limits and idle disconnections cannot be switched off, and keep-alive tricks are against Colab's rules, so the notebook is built to tolerate them:

- **Time budget:** `TIME_BUDGET_MIN` stops the training loop cleanly before the limit (it also avoids starting a run that would not fit), and the comparison / report / ONNX cells then work with the models that finished.
  Choose a value below your session limit.
- **Order of work:** models run from cheap to expensive, and each finishes all its folds (official fold first) before the next one starts, so an interruption always leaves complete models and an ONNX can be exported.
- **Checkpoint and resume:** `runs_checkpoint.pt` is written after every model. After a disconnection reconnect, run the options, setup and data cells, then the training cell: it resumes
  (same settings only). `SAVE_TO_DRIVE = True` also keeps the checkpoint on Drive.
- Keep VS Code open and the computer awake while it trains, and run the `long` preset in one sitting if you are on the free plan, or split it with several resumes.
