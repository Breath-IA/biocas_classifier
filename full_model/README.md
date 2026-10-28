# full_model — complete gate 1 + gate 2 cascade, long GPU search, < 1 M parameters

Trains many small CNN variants for hours on a CUDA machine, composes the best gate 1 (N vs abnormal) and gate 2 (W vs FC) networks into one
cascade, and evaluates it once on a **class-balanced test set** (same number of N, W and FC events). Every deployable model stays under
1 000 000 trainable parameters, for later FPGA tests. Design and reasoning: [plan.md](plan.md). The process is driven by the [Makefile](../Makefile) at the repo root.

> **Status:** every script of the Makefile exists and the whole process was run end to end (`run_full_model_test.bat` / `make selftest`, tiny settings, CPU).
> The 12-hour GPU run itself has not been done yet. First thing to do on a new machine: `make selftest` (a few minutes) — if it prints
> `ALL STAGES PASSED`, `make smoke` and `make full-run` will work too.

## 1. Set up the other PC

Requirements: Python 3.10+, git, **GNU make and bash**, and for GPU training an **NVIDIA GPU with a recent driver**. CUDA does not run on Intel/AMD integrated GPUs;
there, use `DEVICE=cpu` (works, much slower).

```bash
git clone <repo> && cd biocas_classifier
# Windows: use Git Bash. Install make once:  choco install make   (or: winget install GnuWin32.Make)
make install            # .venv + requirements.txt (CPU torch)
make install-cuda       # swaps torch for the CUDA build; CUDA_TAG=cu121 by default (cu118, cu124 ... see pytorch.org)
make check-gpu          # must print "cuda available: True"
make data features synth   # SPRSound download (~GB), per-event table, synthetic set for gate 2 pretraining
make selftest              # whole process with tiny settings on CPU (~10 min): proves the install works. Windows without make: run_full_model_test.bat
```

Linux notes: use the repo as cloned (`.gitattributes` forces LF in `*.sh` and the Makefile; a CRLF `get_data.sh` fails with `$'': command not found`).
If `make` stops with "No such file" for a script, `git pull` — all scripts live in `full_model/`. `make data features synth splits cache` are needed once;
`data/`, `cache/`, the Optuna `.db` files and `experiments/**/cache` are git-ignored and rebuilt per machine, `full_model/splits/` is committed.

Keep the machine awake for the whole run: on Windows `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0`; keep it on AC power.

### GPU use

- Data lives on the GPU (mel caches ≈ 0.5 GB for both bands), augmentation is batched on the GPU, training uses mixed precision (`--amp`, default) and `channels_last`; there are no DataLoader workers.
- These nets are tiny, so one process leaves the GPU mostly idle (Python/kernel-launch bound). Use **`N_JOBS=2`** (or 3–4 on a large GPU): the workers share the Optuna SQLite study. Check with `nvidia-smi -l 2` — if utilisation is under ~70 %, add workers; if memory runs out, lower it or use `ARGS="--cache-on-cpu"`.
- Test the GPU path first: `make install-cuda && make check-gpu && make selftest DEVICE=cuda` (the selftest then runs with mixed precision). If the mixed-precision path ever misbehaves, `ARGS="--no-amp"` turns it off.
- Choose several GPUs by running one search per card: `make search-g1 DEVICE=cuda:0 STUDY=a` and `make search-g2 DEVICE=cuda:1 STUDY=a` in two terminals.

## 2. Run order

```bash
make splits cache          # frozen balanced test set + train/val/CV splits; log-mel cache of every event
make check-budget          # asserts no sampled config above the parameter cap reaches training
make smoke                 # ~5 min; prints seconds per trial. Use it to choose HOURS_G1 / HOURS_G2

make search-bg             # both searches, detached, log in full_model/logs/ (default 5 h + 5 h)
make status                # progress at any time
make stop                  # graceful stop after the current trial (then: rm full_model/STOP)

make refit cascade         # finalists with CV, then best (gate 1, gate 2) pair under the 1 M budget
make final CONFIRM_FINAL=1 # the ONE evaluation on the balanced test set
make fpga report           # parameters/MACs/int8 size/operator check and the key reports
```

Everything unattended (~12 h with the defaults): `make full-run` (prepare, search both gates, refit, cascade, final, FPGA report). To run it detached:
`nohup make full-run > full_run.log 2>&1 &`.

Interrupted? Run the same command again: searches resume from the SQLite study, finished steps with output files are skipped.

## 3. Makefile targets

| Target | What it does |
|---|---|
| `help` | List targets and the current values of the main variables |
| `install` / `install-cuda` / `check-gpu` | Environment (see §1) |
| `data` | `get_data.sh`: clone SPRSound, organise `data/raw_data/{train,test}/{wav,json}` |
| `features` | `extract_spectral_features.py` + `physics_features.py` in `experiments/common/` (event tables; the physics table is read by the synthetic builder) |
| `synth` | Build `e05_cnn_synthetic/cache/synth_v2.npz` (gate 2 pretraining mels); needs the committed `calibration_map_v2.json` |
| `splits` | `make_splits.py`: balanced test set, training pool, validation split, CV folds (written to `splits/`) |
| `cache` | `build_event_cache.py`: log-mel of every event for both bands |
| `check-budget` | `check_budget.py`: parameter-cap sanity check over thousands of sampled configs |
| `smoke` | 3 trials per gate, minutes; validates GPU, data, resume |
| `search-g1` / `search-g2` / `search` | Optuna searches (hours set by `HOURS_G1`, `HOURS_G2`) |
| `search-bg` | `search` detached with `nohup`; PID in `logs/search.pid` |
| `stop` / `status` | Graceful stop / progress report |
| `refit` | Top `TOP_K` trials per gate, patient-grouped `CV_FOLDS`-fold CV × `REFIT_SEEDS` seeds |
| `cascade` | Score all valid pairs; `params(g1)+params(g2) < MAX_PARAMS`; writes the Pareto table |
| `final` | One-time balanced-test evaluation; refuses to run without `CONFIRM_FINAL=1` |
| `fpga` | Parameter / MAC / memory / operator / int8 report on the exported ONNX |
| `report` | Print cascade, final and FPGA reports |
| `mcp` | Start the MCP server (needs `pip install mcp`) |
| `selftest` | Whole process with tiny settings on CPU, writing only to `results_test/` and `splits_test/`; `DEVICE=cuda` to include the GPU |
| `all` | Prepare everything, no training |
| `full-run` | `all` + search + refit + cascade + final + fpga + report |
| `clean` | Delete `cache/`, `logs/`, `STOP` (keeps `splits/` and `results/`) |

## 4. Makefile variables (`make <target> VAR=value`)

| Variable | Default | Meaning |
|---|---|---|
| `DEVICE` | `cuda` | `cuda`, `cuda:1` or `cpu` |
| `SEED` | `42` | Seed for splits, search, refit and final test. `42` uses the committed `splits/` and `results/`; any other value (e.g. `SEED=7`) uses its own `splits_s7/` and `results_s7/`, so the first seed is never overwritten. Run every target with the same `SEED` |
| `K_PER_CLASS` | `250` | Events per class (N, W, FC) in the balanced test set. Wheeze is the limiting class (1 505 events in total) |
| `MAX_PER_PATIENT` | `8` | Max test events per patient and class |
| `MAX_PARAMS` | `1000000` | Limit for the deployed cascade: gate 1 net + gate 2 net |
| `MAX_PARAMS_NET` | `500000` | Limit for each searched net (so any pair fits) |
| `HOURS_G1`, `HOURS_G2` | `5`, `5` | Wall-clock budget of each search |
| `MAX_TRIALS` | `0` | Stop after this many trials per gate (`0` = only the time budget) |
| `TRIAL_MAX_MIN` | `15` | Wall-clock cap of one trial |
| `N_JOBS` | `1` | Parallel worker processes sharing the GPU (2 is usually fine for these small nets) |
| `STUDY` | `full` | Study name prefix (`full_g1`, `full_g2`); use a new name for a fresh search |
| `TOP_K` | `5` | Finalists per gate |
| `CV_FOLDS`, `REFIT_SEEDS` | `5`, `3` | Refit protocol |
| `WANDB` | `0` | `1` logs trials to Weights & Biases (needs `wandb login`) |
| `CONFIRM_FINAL` | `0` | Must be `1` for `make final` |
| `CUDA_TAG` | `cu121` | PyTorch CUDA wheel index for `install-cuda` |
| `PY` | `.venv/...python` | Python interpreter to use |
| `SMOKE_ARGS` | `--epochs-max 8 --n-per-epoch 3000` | Epoch limits used by `make smoke` |
| `ARGS` | empty | Extra flags appended to the script of the target (e.g. `ARGS="--pruner none"`) |

Choosing the hours: run `make smoke`, read the seconds per trial `s`; trials ≈ `3600 × hours / s` (times `N_JOBS` if the GPU is not saturated).

## 5. Script flags

Run from anywhere (paths come from the repo root). Scripts are driven by the Makefile; call them directly only for debugging.

**`make_splits.py`**: `--k-per-class 250` · `--max-per-patient 8` · `--seed 42` · `--val-fraction 0.15` · `--cv-folds 5` · `--keep-poor-quality` (default: poor-quality recordings are excluded from the test pool only) · `--out splits/`

**`build_event_cache.py`**: `--band {2000,900}` (50–2000 Hz/64 mels or 60–900 Hz/40 mels) · `--jobs 4` · `--check 200` (compare with `cnn_data.to_mel` on N random events)

**`check_budget.py`**: `--n-samples 2000` · `--max-params-net 500000` · `--max-params 1000000` · `--seed 0`

**`search.py`**
| Flag | Default | Meaning |
|---|---|---|
| `--gate {1,2}` | required | 1 = N vs abnormal; 2 = W vs FC |
| `--study NAME` | required | Study name (SQLite, resumable) |
| `--storage URL` | `sqlite:///results/<study>.db` | Optuna storage |
| `--hours H` | `5` | Total time budget; no trial is started that cannot finish in what is left |
| `--max-trials N` | `0` | `0` = unlimited |
| `--trial-max-min M` | `15` | Per-trial cap |
| `--max-params-net P` | `500000` | Configs above this are rejected before training and recorded as pruned |
| `--device` | `cuda` | `cuda`, `cuda:N`, `cpu` |
| `--n-jobs N` | `1` | Worker processes |
| `--pruner {hyperband,median,none}` | `hyperband` | Early pruning on validation scores |
| `--sampler {tpe,random}` | `tpe` | |
| `--synth-pretrain {auto,0,1}` | `auto` | Gate 2 only: searched / forced off / forced on |
| `--amp / --no-amp` | `--amp` | Mixed precision during the search |
| `--no-se` | off | Exclude squeeze-excite blocks (more FPGA-friendly space) |
| `--cache-on-cpu` | off | Keep the mel cache in pinned CPU memory instead of on the GPU (small GPUs) |
| `--wandb {0,1}` | `0` | |
| `--status` | | Print progress of the study and exit |
| `--seed` | `42` | A resumed study continues with a different random sequence (seed + trials done) |
| `--epochs-max N` | `40` | Maximum epochs of one trial (early stopping usually ends it sooner) |
| `--n-per-epoch N` | `6000` | Samples drawn per "epoch" (class-balanced, with replacement): fixes the time of a trial |

Creating the file `full_model/STOP` stops a running search after its current trial.

**`refit_top.py`**: `--gate` · `--study` · `--storage` · `--top-k 5` · `--cv-folds 5` · `--seeds 3` · `--device` · `--epochs-max 60` · `--n-per-epoch 6000` · `--trial-max-min 15`
**`cascade_eval.py`**: `--study` · `--max-params 1000000` · `--top-k 5` · `--size-tolerance-se 1.0` (choose the smallest pair within this many standard errors of the best)
**`final_test.py`**: `--study` · `--max-params` · `--device` · `--confirm` (required) · `--allow-retest` (the test set is normally opened once; a lock file records it) · `--bootstrap 2000` (patient-clustered) · `--seeds 3` · `--no-loyo` · `--no-ablation`
**`fpga_report.py`**: `--study` · `--max-params` · `--int8` · `--qat-tolerance 0.02`
**`selftest.py`**: `--seed N` (own `results_test_s<N>/` folders; `run_full_model_test.bat 7` does the same) · `--device cpu|cuda` · `--keep` (reuse `results_test/`) · `--skip-cache` · `--k-per-class 40`. Environment variables `FM_SPLITS_DIR` / `FM_RESULTS_DIR` redirect `splits/` and `results/` for any script.
**`mcp_server.py`**: `--study` · `--storage` · `--allow-enqueue` · `--allow-stop` (both off by default; read-only otherwise)

## 6. Outputs

```
full_model/
├── splits/    balanced_test.csv, train_pool.csv, val_split.csv, cv_folds.csv   (commit these: same test set on every machine)
├── cache/     events_mel_2000.npz, events_mel_900.npz                          (git-ignored)
├── logs/      search_<timestamp>.log, search.pid                                (git-ignored)
└── results/
    ├── <study>.db                     Optuna studies                            (git-ignored)
    ├── <study>_trials.csv             one row per trial: config, params, MACs, validation metrics, seconds
    ├── refit_g1.csv, refit_g2.csv     finalists with CV mean ± std
    ├── cascade_pairs.md               all valid pairs, Pareto set (macro-F1 vs parameters)
    ├── final_report.md, final_metrics.json, confusion_matrix.{csv,png}   balanced-test results with patient-clustered 95 % CIs, per-year table
    ├── gate1.onnx, gate2.onnx, cascade.json                               deployable models and their preprocessing/thresholds
    └── fpga_report.md/.csv            parameters, MACs, int8 memory, operator check, quantisation delta
```

## 7. Rules the pipeline enforces

- Splits are by **patient**; test patients never appear in train or validation (asserted).
- The balanced test set is created once and committed; `final` runs once (lock file) and nothing before it reads the test events.
- Synthetic events are used only to pretrain gate 2, never for validation or test, and never from validation/test patients.
- Any net above `MAX_PARAMS_NET`, and any pair above `MAX_PARAMS`, is rejected; the ONNX files are re-counted at the end.
- Model selection uses cross-validated refits of finalists, not the single search-validation score.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `cuda available: False` after `install-cuda` | Update the NVIDIA driver; check `nvidia-smi`; try another `CUDA_TAG`; with only an integrated GPU use `DEVICE=cpu` |
| `import torch` fails with a `torch_python.dll` / App Control error (Windows) | Retry; it is intermittent (the search runner also retries) |
| Out of GPU memory | Lower `N_JOBS` to 1; the runner falls back to keeping the cache in pinned CPU memory with `ARGS="--cache-on-cpu"` |
| `make: command not found` on Windows | Install make and run from Git Bash (§1) |
| Search does not start after `make stop` | Delete `full_model/STOP` |
| `bash: get_data.sh: $'': command not found` | CRLF line endings: `sed -i 's/$//' get_data.sh Makefile` (the repo's `.gitattributes` prevents it on a fresh clone) |
| `segments_spectral.csv not found` / `events_mel_*.npz not found` | Run `make features` / `make cache` (they are git-ignored or untracked and built per machine) |
| `selftest` stops at a stage | The log is `full_model/logs/selftest_<time>.log`; the stage name tells which script failed |
| Most trials end "pruned: over parameter cap" | `make check-budget` shows the rejected fraction; narrow the width/depth ranges in `search_space.py` |
