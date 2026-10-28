# Makefile — whole process: data → features → synthetic set → balanced splits → GPU search (gate 1, gate 2)
#              → cascade → final test → FPGA report.   Plan: full_model/plan.md · Flags: full_model/README.md
#
# Needs GNU make and bash (Linux/macOS: built in; Windows: Git Bash + `choco install make` or `winget install GnuWin32.Make`,
# and run make from Git Bash). Every variable below can be overridden on the command line:  make search HOURS_G1=3 DEVICE=cpu

SHELL := bash
.DEFAULT_GOAL := help
.NOTPARALLEL:

# ── Python / virtualenv ──────────────────────────────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
  BOOT_PY ?= python
  VENV_PY := .venv/Scripts/python
else
  BOOT_PY ?= python3
  VENV_PY := .venv/bin/python
endif
PY ?= $(VENV_PY)
CUDA_TAG ?= cu121

# ── Run configuration (documented in full_model/README.md) ────────────────────────
DEVICE         ?= cuda
SEED           ?= 42
K_PER_CLASS    ?= 250
MAX_PER_PATIENT?= 8
MAX_PARAMS     ?= 1000000
MAX_PARAMS_NET ?= 500000
HOURS_G1       ?= 5
HOURS_G2       ?= 5
MAX_TRIALS     ?= 0
TRIAL_MAX_MIN  ?= 15
N_JOBS         ?= 1
TOP_K          ?= 5
CV_FOLDS       ?= 5
REFIT_SEEDS    ?= 3
STUDY          ?= full
WANDB          ?= 0
CONFIRM_FINAL  ?= 0
ARGS           ?=
SMOKE_ARGS     ?= --epochs-max 8 --n-per-epoch 3000

FM        := full_model
# SEED=42 (default) uses the committed full_model/splits and full_model/results. Any other SEED gets its own folders, so a second
# seed never overwrites the first:  make splits cache search ... SEED=7   ->  full_model/splits_s7, full_model/results_s7
ifeq ($(SEED),42)
  SPLITS_DIR  := $(FM)/splits
  RESULTS_DIR := $(FM)/results
else
  SPLITS_DIR  := $(FM)/splits_s$(SEED)
  RESULTS_DIR := $(FM)/results_s$(SEED)
endif
export FM_SPLITS_DIR  := $(SPLITS_DIR)
export FM_RESULTS_DIR := $(RESULTS_DIR)
E4        := experiments/gate2_event_type/e04_synthetic_augmentation
E5        := experiments/gate2_event_type/e05_cnn_synthetic
STORAGE   := sqlite:///$(RESULTS_DIR)/$(STUDY).db
STAMP     := $(shell date +%Y%m%d_%H%M%S)
COMMON_SEARCH = --device $(DEVICE) --seed $(SEED) --max-params-net $(MAX_PARAMS_NET) --n-jobs $(N_JOBS) \
                --max-trials $(MAX_TRIALS) --trial-max-min $(TRIAL_MAX_MIN) --storage $(STORAGE) --wandb $(WANDB)

RAW_STAMP := data/raw_data/train/wav
FEATURES  := experiments/common/features/segments_spectral.csv
PHYSICS   := experiments/common/features/physics_features.csv
SYNTH_NPZ := $(E5)/cache/synth_v2.npz
SPLIT_CSV := $(SPLITS_DIR)/balanced_test.csv
CACHE_NPZ := $(FM)/cache/events_mel_2000.npz

.PHONY: venv help install install-cuda check-gpu data features synth splits cache check-budget smoke \
        search-g1 search-g2 search search-bg stop status refit cascade final fpga report mcp \
        all full-run clean selftest

help: ## list the targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | awk -F'\t' '{printf "  %-14s %s\n", $$1, $$2}'
	@echo ""
	@echo "Common variables (current value): DEVICE=$(DEVICE) HOURS_G1=$(HOURS_G1) HOURS_G2=$(HOURS_G2) MAX_PARAMS=$(MAX_PARAMS)"
	@echo "  MAX_PARAMS_NET=$(MAX_PARAMS_NET) K_PER_CLASS=$(K_PER_CLASS) N_JOBS=$(N_JOBS) STUDY=$(STUDY) SEED=$(SEED)"

# ── Setup ─────────────────────────────────────────────────────────────────────────────────────
# Every target that runs Python depends (order-only) on `venv`: if there is no virtualenv yet it is created with `make install`,
# so `make data features synth` works on a fresh clone without running `make install` first.
venv:
	@test -x $(PY) || test -x .venv/Scripts/python.exe || $(MAKE) install

install: ## create .venv and install requirements.txt (CPU torch by default)
	test -x $(VENV_PY) || test -x .venv/Scripts/python.exe || $(BOOT_PY) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

install-cuda: install ## replace torch by the CUDA build (CUDA_TAG=cu121; see https://pytorch.org/get-started/locally)
	$(PY) -m pip install --force-reinstall --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$(CUDA_TAG)
	$(MAKE) check-gpu

check-gpu: ## print torch version and whether CUDA is available
	$(PY) -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); \
	print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (use DEVICE=cpu)'); \
	print('memory GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1) if torch.cuda.is_available() else '-')"

# ── Data and features (existing scripts) ──────────────────────────────────────────────────────
$(RAW_STAMP): | venv
	PYTHON=$(abspath $(PY)) bash get_data.sh

data: $(RAW_STAMP) ## download and reorganise SPRSound into data/raw_data

$(FEATURES): | $(RAW_STAMP) venv
	$(PY) experiments/common/extract_spectral_features.py

features: $(FEATURES) $(PHYSICS) ## per-event spectral and physics tables (~4 min)

$(PHYSICS): $(FEATURES) | venv
	$(PY) experiments/common/physics_features.py

# Synthetic mels for the optional gate 2 pretraining (e05 recipe, v2 calibration). Needs the committed e05
# results/calibration_map_v2.json and the physics table (builder.py reads it). No synthetic WAVs are needed.
$(SYNTH_NPZ): $(PHYSICS) | $(RAW_STAMP) venv
	test -f $(E5)/results/calibration_map_v2.json
	$(PY) $(E5)/build_synth_mels.py --name v2 --calibration $(E5)/results/calibration_map_v2.json --n-per-class 2500 --seed 22 --match-hosts

synth: $(SYNTH_NPZ) ## synthetic W/FC mels (gate 2 pretraining only)

# ── New pipeline (full_model) ─────────────────────────────────────────────────────
$(SPLIT_CSV): $(FEATURES) $(FM)/make_splits.py | venv
	$(PY) $(FM)/make_splits.py --k-per-class $(K_PER_CLASS) --max-per-patient $(MAX_PER_PATIENT) --seed $(SEED) $(ARGS)

splits: $(SPLIT_CSV) ## frozen balanced test set (K per class N/W/FC) + patient-disjoint train/val/CV folds

$(CACHE_NPZ): $(FEATURES) $(FM)/build_event_cache.py | venv
	$(PY) $(FM)/build_event_cache.py --band 2000
	$(PY) $(FM)/build_event_cache.py --band 900

cache: $(CACHE_NPZ) ## log-mel cache of every event (both bands)

check-budget: | venv ## sample many configs, assert none above the parameter cap reaches training
	$(PY) $(FM)/check_budget.py --max-params-net $(MAX_PARAMS_NET) --max-params $(MAX_PARAMS) --n-samples 2000

smoke: splits cache check-budget synth ## 5-minute end-to-end check on a few trials (prints seconds per trial)
	$(PY) $(FM)/search.py --gate 1 --hours 0.05 --study smoke_g1 $(COMMON_SEARCH) --max-trials 3 $(SMOKE_ARGS) $(ARGS)
	$(PY) $(FM)/search.py --gate 2 --hours 0.05 --study smoke_g2 $(COMMON_SEARCH) --max-trials 3 $(SMOKE_ARGS) $(ARGS)

# ── Long search ───────────────────────────────────────────────────────────────────────────────
search-g1: splits cache ## Optuna search, gate 1 (N vs abnormal) for HOURS_G1 hours; resumable
	$(PY) $(FM)/search.py --gate 1 --hours $(HOURS_G1) --study $(STUDY)_g1 $(COMMON_SEARCH) $(ARGS)

search-g2: splits cache synth ## Optuna search, gate 2 (W vs FC) for HOURS_G2 hours; resumable
	$(PY) $(FM)/search.py --gate 2 --hours $(HOURS_G2) --study $(STUDY)_g2 $(COMMON_SEARCH) $(ARGS)

search: search-g1 search-g2 ## both searches, one after the other

search-bg: ## run `make search` detached (nohup) with a log in full_model/logs/
	mkdir -p $(FM)/logs
	nohup $(MAKE) search > $(FM)/logs/search_$(STAMP).log 2>&1 & echo $$! > $(FM)/logs/search.pid
	@echo "started, pid $$(cat $(FM)/logs/search.pid); follow with: tail -f $(FM)/logs/search_$(STAMP).log"

stop: ## ask the running search to finish its current trial and exit
	touch $(FM)/STOP
	@echo "STOP file created; remove it ($(FM)/STOP) before launching another search"

status: ## trials done, best values, time left, last log lines
	$(PY) $(FM)/search.py --status --study $(STUDY)_g1 --storage $(STORAGE) || true
	$(PY) $(FM)/search.py --status --study $(STUDY)_g2 --storage $(STORAGE) || true
	@ls -t $(FM)/logs/*.log 2>/dev/null | head -1 | xargs -r tail -n 5

# ── Finalists, cascade, final test ────────────────────────────────────────────────────────────
refit: ## refit the TOP_K trials of each gate with CV_FOLDS folds x REFIT_SEEDS seeds
	$(PY) $(FM)/refit_top.py --gate 1 --study $(STUDY)_g1 --storage $(STORAGE) --top-k $(TOP_K) --cv-folds $(CV_FOLDS) --seeds $(REFIT_SEEDS) --device $(DEVICE) --seed $(SEED) $(ARGS)
	$(PY) $(FM)/refit_top.py --gate 2 --study $(STUDY)_g2 --storage $(STORAGE) --top-k $(TOP_K) --cv-folds $(CV_FOLDS) --seeds $(REFIT_SEEDS) --device $(DEVICE) --seed $(SEED) $(ARGS)

cascade: ## score every valid (gate1, gate2) pair on out-of-fold predictions; params(g1)+params(g2) < MAX_PARAMS
	$(PY) $(FM)/cascade_eval.py --study $(STUDY) --max-params $(MAX_PARAMS) --top-k $(TOP_K) $(ARGS)

final: ## evaluate the chosen pair ONCE on the balanced test set (needs CONFIRM_FINAL=1)
	@test "$(CONFIRM_FINAL)" = "1" || { echo "The test set is opened once. Re-run with CONFIRM_FINAL=1"; exit 1; }
	$(PY) $(FM)/final_test.py --study $(STUDY) --max-params $(MAX_PARAMS) --device $(DEVICE) --seed $(SEED) --confirm $(ARGS)

fpga: ## parameters, MACs, int8 size, operator check and quantisation delta of the exported ONNX
	$(PY) $(FM)/fpga_report.py --study $(STUDY) --max-params $(MAX_PARAMS) --int8 $(ARGS)

report: ## print the key result files
	@for f in $(RESULTS_DIR)/cascade_pairs.md $(RESULTS_DIR)/final_report.md $(RESULTS_DIR)/fpga_report.md; do \
	  if [ -f $$f ]; then echo "=== $$f"; cat $$f; else echo "(missing $$f)"; fi; done

mcp: ## phase 6: start the MCP server over stdio (read-only; add ARGS=--allow-enqueue to enable writes)
	test -f $(FM)/mcp_server.py || { echo "mcp_server.py is built in phase 6 of plan.md"; exit 1; }
	$(PY) $(FM)/mcp_server.py --study $(STUDY) --storage $(STORAGE) $(ARGS)

# ── Everything ────────────────────────────────────────────────────────────────────────────────
all: data features synth splits cache check-budget ## prepare everything, no training

full-run: all search refit cascade ## unattended ~12 h run: prepare, search both gates, refit, cascade
	$(MAKE) final CONFIRM_FINAL=1
	$(MAKE) fpga report

selftest: | venv ## whole process with tiny settings on CPU (a few minutes); writes only to *_test folders (DEVICE=cuda to use the GPU)
	$(PY) $(FM)/selftest.py --device $(if $(filter cuda%,$(DEVICE)),$(DEVICE),cpu) --seed $(SEED)

clean: ## remove caches and logs of the full_model pipeline (keeps splits/ and results/)
	rm -rf $(FM)/cache $(FM)/logs $(FM)/STOP $(FM)/results_test* $(FM)/splits_test*
