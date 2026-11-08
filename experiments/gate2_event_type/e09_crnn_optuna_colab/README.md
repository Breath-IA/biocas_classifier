# gate 2 · e09 — CRNN tuned with Optuna for wheeze recall (Colab GPU, ONNX export)

`gate2_crnn_optuna_colab.ipynb` tunes the CRNN (the best compromise in the e06–e08 confusion matrices, ~0.3–0.5 M parameters) with Optuna and **prioritises wheeze (W) recall without neglecting fine-crackle (FC) performance**.
Train 2022–2024, test 2025 (official split). No Weights & Biases: every graph is drawn in the notebook and saved to `/content/outputs`.

## Run (VS Code + Colab extension)
*Select Kernel → Colab → GPU (T4)*. The notebook stops with an error without CUDA. Run with `PRESET = "smoke"` first (minutes), then `"standard"` (40 trials, 3 seeds in the final models) or `"long"` (100 trials, 5 seeds).
The Optuna study is stored in SQLite (`/content/outputs/optuna_crnn.db`): re-running the search cell resumes it.

## What it does
1. **Search** (TPE + median pruner): channels (`c1`, ratio), GRU hidden size and layers, pooling over time (mean / max), dropout, head units, learning rate, weight decay, batch size and an **FC loss weight** (<1 makes a missed wheeze more expensive).
   Only networks of 250 k–700 k parameters are trained (others are rejected before training). The first trial is the e06 CRNN.
2. **Objective (validation patients only):** `0.65·recall_W + 0.35·recall_FC − 2·max(0, 0.80 − recall_FC)` at the best threshold, so wheezes count almost twice as much, and an FC recall under 0.80 is penalised.
   The 2025 test year is never used to choose architecture, loss weight or threshold.
3. **Final training** of the e06 CRNN (reference) and the tuned CRNN with several seeds; training loss / validation AUC / validation score curves are displayed per seed.
4. **Test report 2025:** AUC, recall / precision / F1 of W and FC, balanced accuracy, confusion matrices (threshold 0.5 and tuned), ROC and the W-recall vs FC-recall trade-off, 95 % patient-clustered bootstrap intervals.
5. **State-of-the-art table and chart** (`SPRSound_SOTA_summary.md`): challenge-style SE/SP/Score computed with SE = recall W and SP = recall FC, plus Mamba-KAN's per-class precision/recall/F1. The literature tasks differ (normal vs abnormal, 7 classes, 4 classes), so this is an orientation, not a like-for-like ranking.
6. **ONNX export** of the best seed (input `log_mel` `(batch,1,64,95)`, output `p_fine_crackle`), verified against PyTorch with onnxruntime; `threshold.json` holds the decision rule and the preprocessing.
7. **Model report:** per-layer table (shapes, parameters, MACs), latency, `report_model.md`, and an architecture drawing (`model_diagram.png`). Open the `.onnx` in https://netron.app for an interactive graph.

## Outputs (`/content/outputs`, zipped in `/content/gate2_crnn_optuna_results.zip`)
`crnn_w_vs_fc.onnx`, `threshold.json`, `report_model.md`, `model_diagram.png`, `model_layers.csv`, `results_test2025.csv`, `comparison_sota.csv/.png`, `optuna_trials.csv`, `best_params.json`,
`training_histories.csv`, `training_curves.png`, `confusion_matrices.png`, `roc_tradeoff.png`.

Status: the notebook ran end to end on a CPU with synthetic data (search, final training, metrics, bootstrap, ONNX check, report, drawing, zip). It has not been run on a Colab GPU with the real data, so the metrics of that test are meaningless.
