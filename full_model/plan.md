# Full model (gate 1 + gate 2): long GPU search under an FPGA parameter budget

Plan to turn the isolated gate experiments into **one complete cascade** (N / W / FC) and to search many small CNN variants
unattended for ~12 h on a CUDA machine. Companion files: [`../Makefile`](../Makefile) (whole process) and
[`README.md`](README.md) (commands and flags). The repo-root [`plan.md`](../plan.md) (physics-informed MoE protocol) is unchanged;
this plan implements its gate 1 → gate 2 hierarchy, steps 1 and 3 of its protocol, under new constraints.

**Status:** phases 0–3 and 5 are implemented and were run end to end with tiny settings (`make selftest`); the 12 h GPU run (phase 4) and the MCP server (phase 6, `mcp_server.py`, needs `pip install mcp`) are written but not yet exercised at scale. Section 10 lists what is built in each phase.

## 1. Requirements and decisions

| # | Requirement / decision | Consequence |
|---|---|---|
| R1 | **Every deployable model < 1 000 000 trainable parameters** (future FPGA test) | Hard limit, enforced in code (§4), not a soft preference |
| R2 | Train many CNN variants for ~12 h on a local CUDA machine | Resumable Optuna search, GPU-resident data, time budget per gate (§5, §6) |
| R3 | Complete model = gate 1 (N vs abnormal) → gate 2 (W vs FC) | Two searches + a cascade-composition stage (§7) |
| R4 | **Test set balanced: same number of N, W and FC** | Fixed, patient-disjoint, frozen test set (§3) |
| R5 | A Makefile for the whole process on another PC; flags documented in a README | `../Makefile`, `README.md` |
| R6 | Model Context Protocol integration "at some point" | Last phase, read-mostly server over the search results (§9) |

Assumptions I made (change them here if wrong):

- **CUDA means an NVIDIA GPU.** CUDA does not run on Intel/AMD integrated GPUs. If the other PC only has an iGPU, everything still runs with
  `DEVICE=cpu` (much slower: expect the 12 h to buy far fewer trials). `make check-gpu` reports what torch sees.
- **The 1 M limit applies to the deployed system:** `params(gate 1 net) + params(gate 2 net) < 1 M`. Each net is searched under a per-net cap
  (`MAX_PARAMS_NET`, default 500 000) so that any pair fits. Seed ensembles (as in e05) are an evaluation device only; a deployed ensemble would count all its members.
- "Trainable" = `requires_grad` parameters, BatchNorm scale/shift included. The count with BatchNorm folded into the convolutions (what the FPGA sees) is also reported.
- Gate 2 stays binary W vs FC (as e05). Events labelled WC, CC, R, S ("other abnormal") are abnormal for gate 1 training, and are **not** in the 3-class test set or in gate 2 training.
- Input is the existing log-mel recipe (4 kHz, 3 s window, n_fft 512, hop 128, per-event normalisation); band (50–2000 Hz / 64 mels or 60–900 Hz / 40 mels) is a searched option.
- Recordings flagged `poor_quality` are left out of the test pool (label noise), and stay in training.

## 2. What is reused and what changes

| Existing | Use |
|---|---|
| `experiments/common/features/segments_spectral.csv` | Event table (stem, year, patient, class, start/end) for the splits |
| `e05_cnn_synthetic/cnn_data.py`, `train_best_cnn.py::to_mel` | Mel recipe; the new cache must reproduce it exactly (checked by a unit comparison, §10 phase 1) |
| `e05_cnn_synthetic/cache/synth_v2.npz` | Synthetic W/FC events, used **only** for optional gate 2 pretraining, never in val/test |
| `e04`/`e05` synthesis method | Unchanged. Not used for gate 1 (root plan: don't inflate "abnormal" with synthetic data) |
| `e06_cnn_zoo_colab` (`efficientnet_b0` 4 M, `resnet18` 11 M) | **Excluded**: over budget. Its small models (`cnn_baseline`, `cnn_se`, `cnn_wide`, `cnn_deep`, `resnet_small`) become seeds of the search space |
| `workflow/` (Hydra pipeline, `train.py`) | Not touched; it belongs to the older whole-recording pipeline |
| e03/e05 per-sample Python `augment` loop | Replaced by batched GPU augmentation (the CPU loop would starve a GPU) |

## 3. Data and splits (R4)

Event counts available (all years, all splits): **N 18 772 · FC 3 530 · W 1 505** · other abnormal 771 (WC 303, R 217, CC 177, S 74).
Wheeze is the scarce class (141 patients), so it fixes the size of the balanced test set.

`make_splits.py` builds, once, with a fixed seed, and writes small CSVs that are committed so every machine uses the same test set:

1. **Balanced test set** (`splits/balanced_test.csv`): exactly `K` events of each of N, W and FC (default `K = 250`, so 750 events).
   - Whole **patients** are assigned to test (greedy, until each class has ≥ K candidates), then events are subsampled to exactly K per class
     with at most `max_per_patient` (default 8) events per patient per class, so a few patients cannot dominate. Leftover events of test patients are dropped, not used for training.
   - Each class must come from ≥ 3 of the 4 years; the per-year composition is reported (2025 has a known domain shift, gate 1 AUC 0.63 in e01).
   - Test patients never appear in train/val (the root-plan rule "splits by patient" is checked by an assertion).
2. **Training pool** (`splits/train_pool.csv`): all remaining patients, all classes (N, W, FC, other abnormal).
3. **Validation**: a fixed patient-disjoint split of the pool (`val_fraction` 0.15) used for early stopping, pruning, thresholds and model selection during the search;
   plus `cv_folds` (default 5) patient-grouped folds used only to refit the finalists (§7).
   Validation metrics are computed on a **class-balanced resample** of the validation events, so the search optimises the same quantity that is reported on test.
4. **Natural-distribution check:** the official 2025 test split (unbalanced) is also reported, as a secondary number. It overlaps the training pool, so it is only used for the leave-one-year-out robustness table (§7), never to choose models.

Training-time balancing (the pool is ~15:1 N:W): class-balanced sampler (gate 1: 50 % N, 50 % abnormal split evenly over W / FC / other; gate 2: 50 / 50), with a fixed number of samples per "epoch"
so trial duration is predictable. Per-patient caps on N events avoid a few long recordings dominating.

**Test discipline:** the test set is opened exactly once, by `final_test.py`, which writes a lock file and refuses to run again without `--allow-retest`.
Nothing in the search, the cascade stage or the refit reads it.

## 4. Models and the parameter budget (R1, FPGA)

One builder, `models_small.py`, produces every variant from a config dict; `search_space.py` samples configs. All variants are conv-net based and FPGA-friendly:

| Dimension | Options |
|---|---|
| Depth / widths | 3–6 stages; base width {8, 12, 16, 24, 32}, growth {×1.5, ×2}; widths capped by the budget |
| Block type | plain Conv-BN-ReLU · depthwise-separable · residual (identity or 1×1 shortcut) · optional squeeze-excite (flagged "non-FPGA-friendly", off by default in the first half of the search) |
| Kernel | 3×3 · 5×5 (first stage only) |
| Activation | ReLU · ReLU6 · LeakyReLU (no Swish/GELU: costly in hardware) |
| Pooling | max · average between stages; global average pooling before the head |
| Head | linear · one hidden layer {16, 32, 64}; dropout {0 … 0.5} |
| Input | band {2000/64 mel, 900/40 mel}; 3 s window fixed |
| Gate-2 extras | synthetic pretraining on/off (`synth_v2`), ratio {0.5, 1.0} |
| Training | AdamW/SGD, lr (log), weight decay (log), batch {64, 128, 256}, schedule {cosine, one-cycle}, loss {BCE, focal(γ), label smoothing} (no pos_weight: the class-balanced sampler already equalises the classes), SpecAugment strengths, spectrogram mixup |

**Budget enforcement (three layers):**

1. `count_params(model)` (trainable, and BN-folded) is evaluated at *build time*, before any GPU work. A trial over `MAX_PARAMS_NET` is rejected with `optuna.TrialPruned`
   and recorded with its count, so the sampler learns the frontier at no training cost.
2. `make check-budget` samples thousands of configs and asserts the builder never exceeds the cap without being rejected, and that rejected configs are a minority
   (otherwise the search wastes trials: shrink the ranges).
3. The cascade stage asserts `params(g1) + params(g2) < MAX_PARAMS` and `final_test.py` and `fpga_report.py` re-assert it on the exported ONNX file by counting initializer elements
   (same method as the report on `best_model.onnx`: 316 419 for the 3-seed ensemble, 105 473 per member).

Reference points: the e05 best model is 105 k per member. Width (32, 64, 128, 256) with 3×3 convs is ~390 k conv weights, which is why 500 k per net is a sensible cap.

## 5. Making the training fast on the GPU (R2)

The e03/e05 trainers run on CPU with a Python loop per sample. For a 12 h run the loop is the bottleneck, so:

- **Data on device:** the whole event cache (≈ 24 k events × 64 × 95, float16 ≈ 0.3 GB) is loaded to GPU once per process. If GPU memory is below ~2 GB it stays in pinned CPU memory.
- **Batched augmentation on GPU:** circular time shift, frequency/time masking, mixup, vectorised (no per-sample loop).
- **AMP** (`torch.autocast`, bf16/fp16) and `channels_last`; export and the final evaluation stay FP32 so ONNX results match PyTorch.
- **No DataLoader workers** (data already on device), which also sidesteps the Windows `num_workers` problems noted in CLAUDE.md.
- **Parallel trials:** `--n-jobs 2` runs two worker processes sharing one GPU (the nets are tiny and do not fill it). Optuna storage is SQLite, so workers coordinate through it.
- **Seeds:** fixed `numpy`/`torch`/`random` per trial; cudnn deterministic off during the search (speed), on for the final refit.
- Windows/torch note from CLAUDE.md: `import torch` may fail intermittently with a Windows App Control error; the search runner retries the import before giving up.

## 6. The 12 h run (R2)

Default schedule (`make full-run`, all knobs are Makefile variables):

| Slot | Default | What | Stops on |
|---|---|---|---|
| Search gate 1 | 5 h | Optuna TPE + Hyperband pruner, objective = balanced accuracy of N vs abnormal on the validation set (AUC as tie-break) | `HOURS_G1`, `STOP` file, or `MAX_TRIALS` |
| Search gate 2 | 5 h | same, objective = balanced accuracy of W vs FC; half of the trials with synthetic pretraining on/off | `HOURS_G2` |
| Cascade + refit | ~1.5 h | top-k per gate refit with 5 folds × 3 seeds; best valid pairs composed | finishes |
| Final test + FPGA report | ~0.5 h | one-time test evaluation, ONNX export, quantisation check | finishes |

Behaviour required of the runner (`search.py`):

- **Resumable:** study in `sqlite:///full_model/results/<study>.db`, `load_if_exists`. Re-running the same command continues; every finished trial also appends a row to `results/<study>_trials.csv` (params, MACs, val metrics, seconds).
- **Bounded trials:** per-trial wall-clock cap (`--trial-max-min`, default 15), early stopping by patience on validation, Hyperband pruning on intermediate validation scores.
- **Graceful stop:** `make stop` creates `STOP`; the runner finishes the current trial and exits. The overall `--hours` budget also refuses to start a trial that cannot finish in the remaining time (same rule as e06).
- **Logging:** a plain log under `logs/` and a heartbeat line per trial (best so far, trials done, time left). W&B is optional (`WANDB=1`), off by default for an unattended machine.
- **Machine:** disable sleep/hibernation on the other PC (`powercfg /change standby-timeout-ac 0`) and keep it on AC power; documented in the README.
- Expected throughput: with GPU-resident data these nets should take on the order of a minute per trial, i.e. hundreds of trials per gate in 5 h. This is an estimate; `make smoke` prints the real seconds per trial and the README says how to set `HOURS_*` from it.

## 7. Composing and evaluating the complete model (R3)

1. **Finalists:** top `TOP_K` (default 5) distinct configs per gate from the search are **refit** with patient-grouped 5-fold CV × 3 seeds on the training pool (the validation split is merged in);
   mean ± std of the fold metrics decides the ranking, not the single search validation score (which is optimistic after hundreds of trials).
2. **Cascade rule:** `p1 = gate1(x)`, `p2 = gate2(x)`; predict **N** if `p1 < t1`, else **FC** if `p2 ≥ t2`, else **W**. Thresholds `t1, t2` are chosen on pooled out-of-fold predictions
   (maximising macro-F1 on the balanced resample), never on test.
3. **Pair selection:** all valid (g1, g2) pairs among finalists with `params(g1)+params(g2) < MAX_PARAMS` are scored end-to-end on the out-of-fold predictions — error compounding means the best
   gate-1 and gate-2 nets individually are not necessarily the best pair. The result is the Pareto set (macro-F1 vs total parameters), and the smallest pair within one standard error of the best is the default choice.
4. **Final test (once):** the chosen pair is retrained on the full training pool (3 seeds, only for variance reporting; the deployable artefact is a single net per gate) and evaluated on the balanced test set. Report:
   - 3×3 confusion matrix (rows true, columns predicted), per-class sensitivity and specificity/precision, macro-F1, balanced accuracy;
   - gate-level AUC and balanced accuracy (gate 1 N vs abnormal, gate 2 W vs FC given a correct gate 1);
   - patient-clustered bootstrap 95 % CIs (resample test patients, not events);
   - per-year breakdown and a leave-one-year-out table for the chosen configs (project convention; the held-out year is never augmented and not used for early stopping).
5. **Ablations required by the root plan protocol:** chosen config with/without synthetic pretraining (gate 2) — one line each, same seeds — so the claim "synthetic only helps as pretraining" is rechecked on the new split.

## 8. FPGA readiness (R1)

`fpga_report.py` takes the exported ONNX files and writes `results/fpga_report.csv/.md`:

- trainable parameters (raw and BN-folded), MACs per inference, weight memory at int8 (KB), peak activation memory (KB) for the 64×95 (or 40×95) input;
- operator set check: only `Conv`, `Relu/Clip`, `MaxPool/AveragePool/GlobalAveragePool`, `Gemm`, `Add`, `Flatten` (+ `Sigmoid` at the output) allowed; anything else is listed as a blocker;
- **int8 post-training quantisation** with `onnxruntime.quantization` (static, calibrated on training events), and the macro-F1 / AUC change on validation. A drop above a set tolerance (default 2 points) marks the model "needs QAT";
- the front-end (STFT + mel + log + normalisation) is **not** in the ONNX file. It is documented in `best_model.json`-style metadata so the hardware side can implement it; its cost is out of scope for the parameter budget.

## 9. Model Context Protocol (R6, last phase)

Goal: let a Claude session (Claude Code or Desktop) watch and steer the long run, and query the final model, without shell access to the machine.
`mcp_server.py` uses the official `mcp` Python SDK (stdio transport), registered in a project `.mcp.json`:

| Kind | Name | Behaviour |
|---|---|---|
| Tool (read) | `run_status` | Study names, trials done / running / pruned, best value, time left, last log lines |
| Tool (read) | `leaderboard(gate, top_k)` | Best trials with params, MACs, val metrics, config |
| Tool (read) | `trial_detail(study, number)` | Full config + learning curve |
| Tool (read) | `check_budget(config)` | Parameter count of a config against the caps, without training |
| Tool (read) | `classify_event(wav, start_ms, end_ms)` | Runs the exported cascade ONNX on one event; returns p(N/W/FC) |
| Tool (write, off by default) | `enqueue_trial(config)` | Adds a config to the study queue; only with `--allow-enqueue`, validated by `check_budget` first |
| Tool (write, off by default) | `request_stop()` | Creates `STOP` |
| Resources | `results/*.md`, `fpga_report.md`, `confusion_matrix.md` | Read-only reports |

Safeguards: no arbitrary shell or file access (paths resolved inside `full_model/` and `data/`), write tools disabled unless explicitly flagged, the test set is not exposed by any tool
(`classify_event` is for single user-supplied events, and refuses events that belong to the frozen test CSV so it cannot be used to peek). Data returned by tools is treated as data, not instructions.

## 10. Phases, deliverables, acceptance

| Phase | Build | Done when |
|---|---|---|
| 0. Housekeeping | `requirements.txt` completed (scikit-learn, scipy, soundfile, joblib were missing; `mcp` added as optional); this plan, Makefile, README | `make help` runs on the other PC |
| 1. Data | `make_splits.py`, `build_event_cache.py` | Test CSV has exactly K per class; zero patient overlap (assert); cache mels equal `cnn_data.to_mel` output (max abs diff < 1e-2 on 200 random events: the cache is float16, rounding error ≈ 1e-3) |
| 2. Models + budget | `search_space.py`, `models_small.py`, `check_budget.py` | `make check-budget` passes; 0 builds above the cap reach training |
| 3. GPU trainer + search | `train_lib.py`, `search.py` | `make smoke` finishes in minutes on GPU, resumes after Ctrl+C, honours `STOP` and `--hours`; seconds/trial printed |
| 4. 12 h run | run `make full-run`, then `refit_top.py`, `cascade_eval.py` | Finalists ranked by CV; pair Pareto set written |
| 5. Final + FPGA | `final_test.py`, `fpga_report.py` | One balanced-test report with CIs; ONNX under the cap; quantisation delta reported |
| 6. MCP | `mcp_server.py`, `.mcp.json` | A Claude session can read `run_status` during a run |

Phases 1–3 should be validated on this machine (CPU, tiny `--hours`) before the 12 h run is launched on the other PC.

## 11. Risks and what to watch

- **Few wheezes:** K = 250 W events from ~28 patients gives wide intervals; that is why the bootstrap is clustered by patient. Raising K to 300 uses ~20 % of all wheeze events and thins the training set; check `make_splits.py`'s report before changing it.
- **Search overfits validation:** hundreds of trials on one validation split inflate the best score. Mitigation: CV refit of finalists (§7) and reporting only the refit/test numbers.
- **Domain shift by year/device:** e01 showed gate 1 collapsing on 2025 (AUC 0.63 with hand features). The per-year and leave-one-year-out tables exist to expose this; a good pooled score with a bad 2025 row is a finding, not noise.
- **Synthetic data leakage:** `synth_v2.npz` rows carry host year/patient; the pretraining sampler must drop test and validation patients (the e05 guard, reimplemented and unit-tested).
- **Param-count games:** a model under 1 M that needs a huge front-end or activation memory is not FPGA-friendly; that is why MACs and activation memory are reported next to parameters.
- **Unattended machine:** disk fill from checkpoints (keep only best weights per finished trial, prune the rest), Windows update reboots (resume works), GPU thermal throttling (the heartbeat shows seconds/trial drifting up).

## 12. Layout

```
full_model/
├── plan.md  README.md
├── fm_common.py  make_splits.py  build_event_cache.py  selftest.py
├── search_space.py  models_small.py  train_lib.py  search.py  check_budget.py
├── refit_top.py  cascade_eval.py  final_test.py  fpga_report.py  mcp_server.py
├── splits/      balanced_test.csv, train_pool.csv, val_split.csv, cv_folds.csv   (committed)
├── results/     <study>.db (git-ignored), <study>_trials.csv, reports, ONNX, confusion matrices
├── cache/       events_mel_<band>.npz   (git-ignored)
└── logs/        (git-ignored)
../Makefile   ../.mcp.json (phase 6)
```
