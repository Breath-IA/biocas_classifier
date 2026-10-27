# Gate 2 · e05 — CNN with the synthetic data (W vs FC), improvements and re-test

**Question.** Does the synthetic augmentation of e04 help the e03 CNN (W vs FC, leave-one-year-out)? What happens after applying the
improvements suggested in the e04 report? Everything below was trained and evaluated in this experiment; tables:
[report_auto.md](report_auto.md), figures `fig_auc_by_fold.png`, `fig_delta_auc.png`, calibration notes `calibration_v2_*.md`, realism probe `probe.md`.

## 1. Summary

1. **Mixing synthetic events into the training batches does not help; it hurts.** At a 1:1 share with the real data the mean AUC drops from
   0.923 to 0.905 with the e04 generator (v1) and to 0.911 with the improved one (v2). The loss is largest in 2025, the year most
   shifted from the training years (−0.029 / −0.019).
2. **Using the synthetic data to pretrain, then fine-tuning on real data only, helps.** With v2: mean AUC **0.931 vs 0.923** (+0.008, positive in
   11 of 12 year×seed pairs), **2025: 0.878 vs 0.863**; seed-ensemble AUC 0.937 vs 0.929 (2025: 0.889 vs 0.871). A real-only control with the same
   fine-tuning learning rate gives 0.923, so the gain is not an effect of the learning rate.
3. **Pretraining also stabilises the decision threshold.** Balanced accuracy at the fixed 0.5 threshold rises from 0.795 to 0.854 and its standard
   deviation across seeds falls from 0.078 to 0.011. About 0.02 of that rise comes from the lower learning rate alone (control: 0.816), the rest from the pretraining.
4. **The amount of synthetic data matters when it is mixed in.** AUC with v2 mixed in at 100 / 50 / 25 / 10 % of the larger real class: 0.911 / 0.921 / 0.920 / 0.928.
   Only the 10 % share is above the baseline (2025: 0.877).
5. **Improvements that worked:** per-device-state calibration (2022 vs 2023–25), irregular wheezes, host matching: v2 is more realistic than v1
   (real-vs-synthetic probe AUC for wheezes 0.90 → 0.84) and performs slightly better in two of the three equal-settings comparisons
   (+0.003 with pretraining, +0.007 mixed 1:1; −0.001 mixed at 25 %); these differences are small and not conclusive on their own.
   **Improvements that did not:** a longer crackle decay (the search kept 4.1 ms) and the narrower 60–900 Hz mel band (−0.005 AUC, and worse with synthetic data).
6. **Size of the effects.** The gains are modest (≈ +0.01 AUC) and the seed noise is of the same order (std 0.002–0.03 per year), so the conclusions rest on
   consistency across years and seeds, not on the size of one number. All conditions were compared on the same four folds that were used to choose among them;
   there is no untouched test set.

## 2. What was done

### 2.1 Setup shared by every condition
- **Task and validation:** W vs FC on the 5 035 real events, leave-one-year-out (2022–2025), 3 seeds (0, 1, 2), the e03 CNN (4 conv blocks on log-mel, SpecAugment,
  AdamW, balanced BCE, ≤ 20 epochs, patience 4). The real-only run (`real_b2000`) reproduces e03's `cnn` run for run (e.g. 2022 seed 0: AUC 0.945, 19 epochs).
- **Leakage guards** (`train_cnn_synth.py`): synthetic events whose host recording is from the held-out year are never used; early stopping uses **real** validation events only
  (15 % of the real training patients, as in e03); synthetic events whose host patient is in that validation set are dropped; the test year is always real.
  The same number of synthetic events per class is added for W and FC, so "synthetic" carries no label information.
- **Synthetic sets** (`build_synth_mels.py`): 2 500 FC + 2 500 W events per generator version, stored as log-mel inputs for both bands (the same code as e04's builder
  produces the recordings). Events per training fold at 1:1: ≈ 3 100.

### 2.2 Stage 1 — CNN with the e04 synthetic data (v1)
Real-only baseline and real + v1 at 1:1: **v1 lowers AUC in all four years** (−0.017, −0.009, −0.016, −0.029).

### 2.3 Stage 2 — improvements applied (v2)
Chosen from the e04 report because they could be implemented and tested without new data:

| Suggestion (e04) | What was done | Outcome |
|---|---|---|
| Calibrate per device state (2022 differs: PSD peak −9 dB, no high-frequency noise) | `calibrate_v2.py` run separately for 2022 and for 2023–25 | 2022: crackle chain low-pass 250 → **300 Hz**, high-pass 70 → 50 Hz, rate scale 0.78; distances D_FC 0.28 → **0.21**, D_W 0.39 → **0.24**. 2023–25: crackle chain unchanged (D_FC 0.27 → 0.27), D_W 0.30 → **0.22** |
| Raise the crackle source decay τ (physics τ too short) | τ searched in {4.1, 5.0, 5.7, 7.0} ms | **not supported**: both groups kept 4.1 ms; not applied |
| Irregular wheezes (more transients, flatter envelope in the synthetic ones) | burst gap, transient clicks and slow envelope searched | 2023–25: gap 0.3 s, slow envelope depth 0.6, no clicks, wheeze chain 400 Hz; 2022: gap 0.3 s, 3 clicks/s, depth 0.6, chain 500 Hz |
| Match hosts to the real events | host chosen with the profile event's chest location and age ±3 years (same year) | applied in v2 |
| Band-limit the ambient noise of the N variant | same device response applied to the noise | applied (N is not used in the W-vs-FC CNN; kept in the library) |
| Narrow the analysis band to the data (≈ 60–900 Hz) | CNN input with 40 mels over 60–900 Hz | tested as a separate factor (section 3.4) |
| Real-vs-synthetic probe | `qc_probe.py`: RF and logistic regression, 5 folds grouped by patient | v2 more realistic than v1 (below) |

Independent check kept from e04 (fitted crackle τ / 2CD, not a calibration target): v2 2023–25 gives τ 5.9 ms / 2CD 8.2 ms and 2022 gives 4.6 / 6.8 ms against the real SPRSound
values 6.8 / 8.1 ms (pooled over years; the 2022 real fine crackles have τ ≈ 6.2 ms), so the 2022 crackles are still somewhat too short.

**Real-vs-synthetic probe** (AUC of a classifier trained to tell real from synthetic; 0.5 = indistinguishable; 400 synthetic events per class):

| set | FC (RF / LR) | W (RF / LR) | median univariate gap FC / W |
|---|---|---|---|
| v1 (e04 generator) | 0.935 / 0.929 | 0.904 / 0.936 | 0.146 / 0.204 |
| v2 (improved) | 0.916 / 0.921 | **0.835 / 0.858** | 0.131 / 0.181 |

Synthetic events are still clearly detectable (AUC ≥ 0.84), especially fine crackles. The features that betray them: `mfcc_10_mean`, `ph_persist_cos_mean`, the physics τ
(`ph_tau_med/p25`) and `band_600_800` for crackles; `ph_persist_cos_mean`, `tonal_run_s`, `ph_wh_peak_db` and `ph_if_*` for wheezes (see `probe_top_features.csv`).

### 2.4 Stage 3 — how the synthetic data are used (follow-up forced by the stage-2 results)
Because 1:1 mixing hurt even with v2, three things were tested: the amount of synthetic data (mixed at 50, 25 and 10 %), v1 vs v2 at equal settings, and a different way of using
the data — **pretraining** on the synthetic events (5 epochs, balanced) followed by the usual training on real data only, with a lower fine-tuning learning rate (3e-4). The
real-only run at the same learning rate is the control for the latter.

## 3. Results

### 3.1 ROC AUC per held-out year (mean of 3 seeds)

| condition | 2022 | 2023 | 2024 | 2025 | mean | seed-ensemble mean |
|---|---:|---:|---:|---:|---:|---:|
| real only (e03 recipe) | 0.932 | 0.933 | 0.961 | 0.863 | 0.923 | 0.929 |
| real only, lr 3e-4 (control) | 0.935 | 0.931 | 0.960 | 0.865 | 0.923 | 0.928 |
| mixed 1:1, v1 | 0.915 | 0.925 | 0.945 | 0.835 | 0.905 | 0.917 |
| mixed 1:1, v2 | 0.915 | 0.930 | 0.957 | 0.844 | 0.911 | 0.921 |
| mixed 50 %, v2 | 0.927 | 0.946 | 0.962 | 0.851 | 0.921 | 0.928 |
| mixed 25 %, v1 | 0.925 | 0.942 | 0.950 | 0.867 | 0.921 | 0.928 |
| mixed 25 %, v2 | 0.938 | 0.938 | 0.956 | 0.846 | 0.920 | 0.928 |
| mixed 10 %, v2 | 0.932 | 0.939 | 0.964 | 0.877 | 0.928 | **0.938** |
| **pretrain on v1, fine-tune on real** | 0.933 | 0.942 | 0.960 | 0.876 | 0.928 | 0.934 |
| **pretrain on v2, fine-tune on real** | **0.940** | **0.942** | **0.963** | **0.878** | **0.931** | 0.937 |

### 3.2 Paired differences (same year and seed; `fig_delta_auc.png`)

| comparison | mean ΔAUC | pairs > 0 | Wilcoxon p* | 2022 / 2023 / 2024 / 2025 |
|---|---:|---:|---:|---|
| pretrain v2 − real-only (e03 recipe) | **+0.008** | 11 / 12 | 0.021 | +0.008 / +0.009 / +0.002 / +0.015 |
| pretrain v2 − real-only, same learning rate | **+0.008** | 10 / 12 | 0.002 | +0.005 / +0.011 / +0.003 / +0.014 |
| pretrain v1 − real-only, same learning rate | +0.005 | 9 / 12 | 0.077 | −0.002 / +0.011 / 0.000 / +0.011 |
| pretrain v2 − pretrain v1 | +0.003 | 10 / 12 | 0.016 | +0.007 / 0.000 / +0.003 / +0.002 |
| mixed 1:1 v1 − real-only | −0.018 | 2 / 12 | – | −0.017 / −0.009 / −0.016 / −0.029 |
| mixed 1:1 v2 − real-only | −0.011 | 3 / 12 | – | −0.017 / −0.003 / −0.004 / −0.019 |
| mixed 1:1 v2 − mixed 1:1 v1 | +0.007 | 8 / 12 | 0.42 | 0.000 / +0.005 / +0.012 / +0.009 |
| mixed 10 % v2 − real-only | +0.006 | 7 / 12 | – | 0.000 / +0.006 / +0.003 / +0.014 |
| learning rate 3e-4 − 1e-3, real-only | 0.000 | 5 / 12 | 0.79 | +0.003 / −0.002 / −0.001 / +0.002 |

\*Paired signed-rank test over the 12 year×seed pairs. The 3 seeds within a year share the same data, so this is a rough guide, not a confirmation.

### 3.3 Balanced accuracy at the fixed 0.5 threshold
Real-only 0.795 (std across seeds 0.078; recall of FC from 0.28 to 0.75 in 2025 across seeds); real-only at lr 3e-4 0.816 (0.037); pretrain v1 0.824 (0.038);
**pretrain v2 0.854 (0.011)**; mixed variants 0.75–0.80 (std 0.05–0.13). Per year for pretrain v2: 0.858 / 0.865 / 0.905 / 0.789 against 0.742 / 0.849 / 0.892 / 0.699 for real-only.

### 3.4 Narrow mel band (60–900 Hz, 40 mels)
Real-only: AUC 0.918 vs 0.923 (−0.005; p = 0.91, 2025: 0.859 vs 0.863). With v2 mixed 1:1 it is worse in 2025 (0.817, −0.042 vs the same-band baseline). No benefit was found;
the 50–2000 Hz input stays the default.

## 4. Interpretation (hypotheses, not tested)
- **Why mixing fails.** The probe shows that a classifier separates real from synthetic events easily. In a mixed batch the CNN can use those artefacts to separate synthetic W from
  synthetic FC, which does not transfer to real events; the more synthetic data, the larger the effect. This matches the loss growing with the synthetic share and being largest in 2025.
- **Why pretraining helps.** The synthetic data teach generic crackle-versus-tone filters; fine-tuning then corrects the artefacts on real data, and the early-stopping set is real, so artefacts
  cannot be selected. The much smaller seed variance of the threshold-dependent metrics suggests the pretrained start is more reliable.
- **v2 vs v1.** The improvements reduce the artefacts (probe) and give consistent but small gains; the three v2 changes (per-state calibration, irregular wheezes, matched hosts) were
  applied together, so their individual contributions cannot be separated.

## 5. Limitations
- 3 seeds per condition; effects of ≈ 0.01 AUC are of the same size as the seed noise per pair, and the Wilcoxon p-values treat dependent pairs as independent.
- The conditions were compared on the folds used to choose among them (ratio, mode); the best recipe (pretrain v2) is one of 12 compared settings. There is no untouched final test.
- Pretraining hyper-parameters (5 epochs, fine-tuning lr 3e-4, ratio 1.0) were set once and not tuned.
- The v2 generator bundles several changes; the 2022 crackles remain shorter than real (τ 4.6 vs ≈ 6.2 ms) and synthetic events remain easy to detect (probe AUC 0.84–0.92).
- Only W vs FC was tested. Synthetic N, CC, WC and the flat N/W/FC problem were not trained; hosts are normal recordings that may contain unannotated sounds.
- Synthetic recordings from 2025 hosts exist in the sets but are excluded whenever 2025 is the held-out year; for deployment-like use, the same exclusion rule applies to whichever year is unseen.

## 6. Suggestions
1. **Adopt synthetic pretraining (v2) → real fine-tuning as the default CNN recipe for gate 2**, with seed ensembling; do not mix synthetic events into real batches except at ≈ 10 %.
2. **Tune the pretraining recipe on the existing folds, then confirm on data not used for selection**: pretraining epochs (2 / 5 / 10), synthetic set size, fine-tuning learning rate, and freezing the first blocks.
   A new year or another dataset (e.g. ICBHI) would give the untouched test the present comparison lacks.
3. **Reduce the artefacts the probe still finds** and re-run the probe as the realism metric: crackle decay time (physics τ), spectral persistence, MFCC bins, wheeze tonal persistence. A discriminator-guided
   adjustment of the generator parameters (the probe's own importances) is a direct next step.
4. **Apply the same recipe to the remaining problems**: the flat N/W/FC problem of e01 (needs synthetic N, which the library already generates), gate 1 (normal vs abnormal) with hard normals, and the physics-fusion variant of e03 on top of the pretrained network.
5. **Keep the e03 input band (50–2000 Hz)**: the narrower band gave no benefit. Revisit it only with the pretrained network.
6. **If a deployment-like year is available without labels, use its host recordings** to build synthetic events from that year's own device state (the per-state calibration helped); this is a transductive choice that should be decided explicitly.
7. **Report AUC together with balanced accuracy and its seed spread**: the threshold stabilisation (std 0.078 → 0.011) is one of the clearest benefits and is invisible in AUC alone.

## 7. Reproduction

```bash
E5=experiments/gate2_event_type/e05_cnn_synthetic
.venv/Scripts/python $E5/build_synth_mels.py --name v1 --calibration $E5/results/calibration_v1_e04.json --n-per-class 2500 --seed 21
.venv/Scripts/python $E5/calibrate_v2.py --tag 2023_25 --years 2023 2024 2025 ; .venv/Scripts/python $E5/calibrate_v2.py --tag 2022 --years 2022
.venv/Scripts/python $E5/build_synth_mels.py --name v2 --calibration $E5/results/calibration_map_v2.json --n-per-class 2500 --seed 22 --match-hosts
.venv/Scripts/python $E5/qc_probe.py --n 400
.venv/Scripts/python $E5/train_cnn_synth.py --name real_b2000                                        # baseline (= e03 cnn)
.venv/Scripts/python $E5/train_cnn_synth.py --name v2_r1_b2000 --synth v2 --ratio 1.0                # mixed (ratio 0.1 / 0.25 / 0.5 / 1.0)
.venv/Scripts/python $E5/train_cnn_synth.py --name v2_pre_b2000 --synth v2 --ratio 1.0 --mode pretrain --pre-epochs 5 --ft-lr 3e-4
.venv/Scripts/python $E5/train_cnn_synth.py --name real_lr3e4_b2000 --lr 3e-4                        # learning-rate control
.venv/Scripts/python $E5/train_cnn_synth.py --name real_b900 --band 900                              # narrow band
.venv/Scripts/python $E5/report.py
```
Each training condition takes 15–45 min on the CPU (12 runs: 4 years × 3 seeds).
