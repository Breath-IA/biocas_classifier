# gate 2 · e04 — Synthetic augmentation (N, FC, W)

First version of plan.md's synthetic augmentation. Parametric crackles and wheezes are inserted into **real SPRSound normal
recordings** and written in the dataset's own format (8 kHz, mono, PCM-16, event-level JSON), keeping the dataset's recording method.
**No model is trained in this experiment.** The MEDZCOOL reference clips are used only to estimate source morphology.

- **Library:** `experiments/common/synthesis/` (`channel`, `generators`, `reference`, `insertion`, `builder`, `sprsound_io`)
- **Scripts (run in this order):**
  1. `analyze_references.py` — first analysis of each reference label (FC, CC for contrast, W) on unique cycles
  2. `analyze_sprsound_events.py` — the same statistics on real SPRSound events, first channel estimate, per-year PSD
  3. `calibrate_synthesis.py` — chain, rate, level and wheeze parameters from filter-independent features (`--resume` skips the chain search)
  4. `build_synthetic.py` — writes `data/synthetic/{wav,json}` and `manifest.csv` (`--n-per-class`, `--years`, `--seed`, `--overwrite`)
  5. `qc_synthetic.py` — integrity and realism checks, figures
- **Inputs:** `Auscultaion_medzcool_audio/*.mp3`, `data/raw_data` (`get_data.sh`), `common/features/{segments_spectral,physics_features}.csv`
- **Outputs:** `results/report.md` (steps, results and suggestions), `report_auto.md`, `calibration.md`, `reference_analysis.md`, `sprsound_event_stats.md`,
  `calibration.json` (parameters used by the builder), `manifest.csv`, figures; `cache/` holds earlier calibrations and logs (git-ignored);
  the synthetic recordings are in `data/synthetic/` (git-ignored, 167 MB for 900 recordings)
- **Use in a leave-one-year-out experiment** (synthetic data are train-only):
  ```python
  from synthesis.sprsound_io import load_manifest
  m = load_manifest(paths.SYNTH_DIR / "manifest.csv", exclude_years=["2025"])   # drops synthetic data derived from the held-out year
  m = m[m["host_split"] == "train"]                                              # official split: also drop hosts from the 2025 test set
  ```

## Key results

- Effective SPRSound response calibrated on features: **~250 Hz (6th-order) low-pass for crackles, ~500 Hz for wheezes**; independent check: synthetic fine-crackle
  two-cycle duration 7.9 ms vs real 8.1 ms (the reference clip's 3 ms, stretched by the chain).
- 900 recordings (300 per class). Integrity: format, schema and clipping errors 0; the real host audio is bit-identical outside the inserted event (100 %).
- Realism, median |2·AUC−1| between real and synthetic events: **N 0.04, FC 0.15, W 0.21**. Weakest points: wheeze tonal persistence and envelope irregularity,
  crackle decay time, high-frequency excess.
- Reference clips are loops of 2–3 unique cycles; fine and coarse crackles become nearly indistinguishable after SPRSound's chain; 2022 is a different device state.
