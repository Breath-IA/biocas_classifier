# Gate 2 · e04 — Synthetic augmentation for N, FC and W: steps, results and suggestions

**Scope.** First version of the synthetic augmentation of plan.md (component 2), for the classes N, FC (fine crackle) and
W (wheeze) only. **No model was trained.** Realism is measured with rank statistics on the same features used for real events.
Code: `experiments/common/synthesis/` (library) and the scripts in `experiments/gate2_event_type/e04_synthetic_augmentation/`.
Automatic tables: [report_auto.md](report_auto.md), [calibration.md](calibration.md),
[reference_analysis.md](reference_analysis.md), [sprsound_event_stats.md](sprsound_event_stats.md).

## 1. Summary

1. **What was built.** A generator that puts parametric crackles and wheezes into *real* SPRSound normal recordings, plus a natural
   variation of the normal recordings themselves, in exactly the dataset's format (8 kHz, mono, PCM-16, event-level JSON).
   900 recordings (300 per class) were generated in 42 s; the library scales to any number.
2. **How the SPRSound auscultation method is kept.** Real host recordings (never simulated backgrounds), the dataset's own
   frequency response (causal filter applied to the synthetic event only), event durations / levels / crackle rates drawn from real
   events of the same class and year, events placed inside breath-sound bursts, no loudness normalization, no resampling of
   the host, and the dataset's annotation conventions. Verified: **the host audio is bit-identical outside the inserted event
   (100 % of 60 sampled recordings)**.
3. **The reference clips were used only for source morphology.** They are 44.1 kHz MP3 teaching clips from another device and
   are *loops of only 2–3 unique breath cycles*, so they are never pasted into SPRSound audio.
4. **Main finding about the data.** SPRSound crackles are not the crackles of the textbook: the recorded transient spectrum falls
   ~40 dB between 200 and 500 Hz. Calibration finds an effective response of a **~250 Hz, 6th-order low-pass for crackles** and
   **~500 Hz for wheezes**. After it, the reference crackle's two-cycle duration (3 ms) becomes 8 ms, **matching the real SPRSound
   value (8.1 ms)** although that value was not a calibration target.
5. **Realism.** Median feature gap between real and synthetic events (|2·AUC−1|; 0 = indistinguishable): **N 0.04, FC 0.15,
   W 0.21**. All integrity checks pass. Remaining weaknesses: wheeze tonal persistence and envelope irregularity, crackle decay
   time and a high-frequency excess (section 6).
6. **Not claimed.** Realism here means *marginal feature distributions*; the plan's classifier-based real-vs-synthetic probe needs a
   trained model and is left for when training is allowed. Whether the synthetic data help a classifier is untested.

## 2. Constraints respected (the "natural auscultation method")

| SPRSound property | How it is kept |
|---|---|
| 8 kHz, mono, PCM-16 | generated at 8 kHz; written as `PCM_16`; checked on all 900 files |
| Real recordings with real noise floor, patient, location, age, gender, year | every synthetic file is a real normal host recording; its year, patient, age, gender and chest location (p1–p6) are inherited |
| Electronic-stethoscope band-limited response | causal band-pass applied **to the synthetic event only** (not to the host); causal so crackle onsets stay asymmetric like in real data |
| Event-level labels (start/end in ms, type) and record tag | same JSON schema (`Fine Crackle`, `Wheeze`, `Normal`; tags `DAS`, `CAS`, `Normal`), start/end as millisecond strings |
| Annotation style of each year (event length changed from ~0.6 s to ~1.9 s between 2022 and 2025) | event duration drawn from a real event of the same class **and year** |
| Natural level of events | level (`snr_vs_floor_db`) drawn from a real event of the same class and year and solved with the extractor's own definition; achieved − target error 0.00 dB |
| Sounds occur with breathing | 94 % of the events are placed inside a detected breath burst (crackles early-to-mid burst, wheezes late) |
| No artificial loudness or pitch change | no normalization, no pitch shift, no time stretch |
| Train-only | manifest keeps host year and split; `load_manifest(exclude_years=...)` drops the held-out year; 137 recordings have a 2025 test host and must be dropped for the official split |
| No label leakage through "being synthetic" | N, FC and W recordings go through the same planner and final steps (N also gets a real-noise and gain variation) |

## 3. Steps done

### Step 0 — audit of the SPRSound acquisition format

- All sampled WAV: 8 kHz, 1 channel, PCM-16. Labels: events `Normal`, `Fine Crackle`, `Coarse Crackle`, `Wheeze`, `Wheeze+Crackle`,
  `Rhonchi`, `Stridor`; record tags `Normal`, `DAS`, `CAS`, `CAS & DAS`, `Poor Quality` (or absent).
- The median spectrum of the recordings peaks near 130 Hz and falls to the 16-bit quantization floor (−135 dB) above ~1.2 kHz.
  The pipeline's mel range of 50–2000 Hz therefore covers mostly noise floor.
- **2022 is a different device/gain state**: its median PSD peaks ~9 dB lower than 2023–2025 (−66 vs −57 dB) and its noisier
  recordings have almost no high-frequency noise (flat at the quantization floor from ~1.2 kHz), while 2023–2025 have −90 to −115 dB
  there (`sprsound_psd_and_trigger.png`). This is a likely cause of the 2022 shift seen in earlier experiments.

### Step 1 — first analysis of each reference label (all except normal)

Clips: fine crackles (FC), coarse crackles (CC, for contrast) and expiratory wheeze (W). Analysis at 8 kHz on unique cycles.

| | FC | CC | W |
|---|---|---|---|
| native format | 44.1 kHz stereo MP3 (L/R correlation 1.000) | same | same |
| 99 % of the energy below | 904 Hz | 506 Hz | 345 Hz |
| active breath segments → **unique** | 24 → **3** | 34 → **2** | 33 → **2** |
| crackle f0 (median, p10–p90) | 612 Hz (426–661) | 347 Hz (236–377) | – |
| decay τ | 4.1 ms (2.6–8.7) | 5.4 ms (3.2–8.5) | – |
| two-cycle duration (2CD) | 3.0 ms | 5.1 ms | – |
| sensitivity to detection settings (18 settings) | f0 474–625 Hz, τ 3.4–4.3 ms | f0 195–351 Hz, 2CD 4.8–10.9 ms | – |
| rate inside a breath segment | ≈ 27 /s (gamma shape of the intervals ≈ 1.8) | – | – |
| wheeze f0, tone length | – | – | 268 Hz (n = 2 cycles), 1.26 s |

- Damped-sine fits are good (R² 0.73–0.89 on the best examples, `ref_crackle_fits.png`), which supports plan.md's physical model.
  The fits give f0 ≈ 612 Hz (fine) and ≈ 347 Hz (coarse), close to the 650 / 350 Hz in plan.md.
- **The clips are loops**: many cycle pairs have cross-correlation 1.00. Statistics rest on 2–3 independent cycles.
- The wheeze clip has partials that are not integer multiples of f0 (tracks near 150, 370, 580, 770 Hz) and 50 % of its
  cycles are polyphonic; its harmonic ratio estimates are unreliable with n = 2.

### Step 2 — the same statistics on real SPRSound events

300 FC events and 220 W events with a tracked tone (`sprsound_event_stats.md`):

- **Crackle f0 is not identifiable at SPRSound's bandwidth.** The fitted f0 follows the analysis high-pass (100 / 150 / 250 Hz →
  median f0 174 / 216 / 288 Hz; 38 / 46 / 67 % of the fits lie within 60 Hz of the cutoff). The crackle energy sits in a very low band.
- Real crackles are longer than the reference: τ 6.8 ms and 2CD 8.1 ms (reference 4.1 and 3.0 ms): the chain stretches them.
- Real wheeze tones are short and low: f0 median 203 Hz (p10–p90 129–514), tracked tone 0.15 s, jitter CV 0.010, 2f0 at −24 dB and 3f0 at
  −48 dB, polyphonic in 36 %.
- First estimate of the chain from the SPRSound / MEDZCOOL crackle-spectrum ratio: low-pass ≈ 375 Hz, order 2 (fit RMSE 12.8 dB, used only as
  a starting point).

### Step 3 — design of the synthesis (`common/synthesis/`)

- **Crackle** `y(t) = A·exp(−t/τ)·sin(2π f0 t + φ)` as a train: gamma-distributed intervals with a refractory time, log-normal amplitudes,
  per-crackle (f0, τ) and **random polarity**. Source parameters come from the reference fits (f0 612 Hz, τ 4.1 ms).
- **Wheeze**: a train of **short gliding tonal bursts** (median 0.25 s, glide ±25 %), frequency-modulated harmonics or a van der Pol oscillator,
  with a parallel second tone in 40 % of the events. f0 comes from the real W events.
- **Channel** (`channel.py`): causal Butterworth band-pass, applied to the event only.
- **Insertion** (`insertion.py`): breath-burst detection → placement → gain solved so that the annotated window reaches the real event's
  `snr_vs_floor_db` → PCM-16. Up to 8 placements are tried so the host window leaves room for the target level (an event can only add energy).
- **N**: a real normal recording with ±gain (σ 1.5 dB), real ambient noise (the quietest second of another normal recording of the same
  year) at the host's own floor ±3 dB, and annotation-boundary jitter ±10 %.

### Step 4 — calibration (`calibrate_synthesis.py`, `calibration.md`)

Because f0 is not identifiable, targets are **filter-independent features measured with the same extractors as for real events**
(spectral shape, kurtosis, crest factor, transient counts, physics features, band-relative levels). Distance D = mean |median(synthetic) −
median(real)| / IQR(real), over the same hosts and seeds for every candidate. This is a search over generator hyper-parameters, not training.

| Parameter | Result | D |
|---|---|---|
| crackle chain low-pass | **250 Hz, order 6** (order 2: 0.90, order 4: 0.49, order 6: 0.26 at 250 Hz) | 0.26 |
| crackle chain high-pass | 70 Hz (flat between 20 and 70 Hz) | 0.25 |
| crackle rate scale / level offset | 0.99 / −1.7 dB | 0.20 |
| wheeze chain low-pass | **500 Hz, order 6** (crackles: 250 Hz) | – |
| wheeze bursts | parallel tone 0.4, burst median 0.25 s, glide 0.25, harmonics (1, 0.2, 0.04) | 0.30 |

The distances of the chain stages were computed with the first feature set; the final D values use a larger set that also includes the band-relative levels
(`rel_low/mid/high_db`, `level_vs_rec_db`), so they are stricter and not directly comparable with the stage-1/2 values.

**Independent check (not a target).** Fitted decay of synthetic fine crackles vs real SPRSound fine crackles: **2CD 7.9 vs 8.1 ms**, τ 5.8 vs 6.8 ms
(419 fits). The reference clip's 3 ms / 4 ms were stretched by the calibrated chain to the real values, which supports both the reference-derived
morphology and the chain.

**Why crackles and wheezes get different responses.** One band-pass could not fit both: real wheezes keep more energy at 300–600 Hz than a 250 Hz
low-pass allows (the first version was ~8 dB too low there), whereas real crackle transients are much lower in frequency. Physically, the device
response (~500–800 Hz, close to the knee of the median spectrum) applies to both, and crackles are additionally low-passed by lung tissue. The
two responses are *effective* ones (device + propagation), not a single measured transfer function.

**Corrections made on the way** (all kept in the code and cache):
the first grid stopped at the optimum's edge, so it was extended to 120–400 Hz; the level offset ran away (−6 dB with no effect) because an event
can never lower the host level, so it is now clipped to ±3 dB and the placement is chosen so the host window is quiet enough; the first
steady-tone wheeze was unnatural (long and perfectly stable, D_W 0.34), so it became the gliding-burst model (D_W 0.22 on the same features);
all crackles started with the same polarity (skewness −0.55 vs +0.07 real), so polarity is now random. Earlier calibrations are in `cache/`.

### Step 5 — dataset build (`build_synthetic.py`)

900 recordings (300 N, 300 FC, 300 W) from 812 distinct hosts (no host used more than twice), 167 MB in `data/synthetic/`
(`wav/`, `json/`, `manifest.csv`, `calibration_used.json`; git-ignored).

| year | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| N | 115 | 30 | 106 | 49 |
| FC | 126 | 30 | 97 | 47 |
| W | 134 | 28 | 97 | 41 |

The year of each recording is that of its host, so the mix follows the number of available hosts. For minority classes or balanced years raise
`--n-per-class` or restrict `--years`.

### Step 6 — quality control (`qc_synthetic.py`)

**Integrity (900 recordings):** format errors 0; schema errors 0; events outside the recording 0; clipping introduced by the synthesis 0
(60 sampled); host samples identical outside the inserted event **100 %**; 17 recordings with overlapping events, **all already overlapping in the real host
annotation** (none introduced).

**Realism**, |2·AUC−1| between real and synthetic events:

| class | features compared | median gap | share of features with gap < 0.2 | share with gap > 0.5 |
|---|---:|---:|---:|---:|
| N | 69 | **0.04** | 94 % | 0 % |
| FC | 97 | **0.15** | 61 % | 1 % |
| W | 97 | **0.21** | 50 % | 1 % |

Median gap of the key features by year (FC / W): 2022 0.20 / 0.17, 2023 0.22 / 0.34, 2024 0.06 / 0.25, 2025 0.05 / 0.18. Crackle impulsiveness is reproduced
(crest factor 9.85 vs 9.84, transient count 16 vs 17, envelope kurtosis 21 vs 25 for synthetic vs real FC). Placement: achieved level equals target (0.00 dB),
94 % inside a breath burst. Duration (synthetic vs real median): FC 1.36 vs 1.58 s, W 1.03 vs 0.78 s.

## 4. Figures

`ref_overview`, `ref_crackle_fits`, `ref_crackle_distributions`, `ref_wheeze` (references) · `sprsound_psd_and_trigger`, `sprsound_event_stats`
(real data) · `calibration` · `qc_feature_gaps`, `qc_distributions`, `qc_spectra`, `qc_gallery` (real vs synthetic spectrograms), `qc_crackle_zoom` (50 ms around the
strongest transient).

## 5. Observations that go beyond the augmentation

- **Fine and coarse crackles become almost indistinguishable after SPRSound's chain.** Passing the reference clips through the calibrated chain gives f0 221 vs
  218 Hz, τ 6.8 vs 7.3 ms (2CD 7.8 vs 10.2 ms; only 7 and 4 fits). Synthetic pairs with the reference's source parameters give f0 224 vs 250 Hz, τ 5.2 vs 5.6 ms
  and 2CD that even reverses (9.1 vs 6.4 ms). This is consistent with the absence of a stable FC-vs-CC feature in e01, and suggests the limit is the
  recording chain, not the features. It is a hypothesis supported by few fits, not a proof.
- **The pipeline's analysis band is much wider than the data.** Mel 50–2000 Hz and the crackle matched-filter bank at 200–1000 Hz look at a band where
  SPRSound has noise floor; the useful band is ≈ 70–800 Hz.
- **Hosts labelled all-Normal can contain unannotated sounds.** In the gallery, the first synthetic FC shows a sweeping tonal pattern outside its annotated
  window that was already in the host.

## 6. Known weaknesses of this first version

| Class | Gap | Detail (median real vs synthetic) |
|---|---|---|
| FC | decay time of the physics estimate | `ph_tau_med` 11.4 vs 5.8 ms; `ph_decay_frac` 0.67 vs 0.88: synthetic decays faster than real, although the fitted τ is close (5.8 vs 6.8 ms) |
| FC | high-frequency excess | median spectrum ~5 dB above real at 450–900 Hz; `max_peak_prom_db` 13.0 vs 9.1 |
| W | tonal persistence and irregularity | `tonal_run_s` 0.096 vs 0.208 s; `env_cv` 1.11 vs 0.87; `env_peak_to_median` 12.6 vs 6.6; `kurtosis` 5.6 vs 2.1; `transient_rate` 9.1 vs 3.0 /s: real wheeze windows are more irregular and contain more transients |
| W | spectrum | ~3 dB below real at 300–700 Hz |
| N | high frequencies | +5 dB at 800–2000 Hz (the added ambient noise has more high-frequency content than the host floor) |
| all | a few features are inherently noisy | `ph_persist_cos_mean`, several MFCC means: gap ≈ 0.5 in every class |

## 7. Limitations

- Realism is only marginal: it says nothing about the joint structure of the features or about whether a classifier can separate synthetic from real.
- Reference clips: 2–3 unique cycles each, adult teaching recordings; the wheeze statistics (n = 2) were not used for the wheeze f0, which comes from SPRSound.
- The channel responses are effective ones fitted on SPRSound features; they were fitted on events pooled over all years, although 2022 has a different device state.
- Hosts are "all events Normal", which does not guarantee absence of unannotated adventitious sounds; 4 487 hosts are available and only 812 were used.
- Breath-phase awareness is limited to "inside a breath burst, early or late": the recordings do not mark inspiration vs expiration and no phase detector was built.
- The chain, rates and levels were calibrated on fine crackles and wheezes only; coarse crackles, Wheeze+Crackle, rhonchi and stridor are not generated.
- Calibration used 90 recordings per candidate; the distances are medians over that sample.

## 8. Suggestions

1. **Fix the first weaknesses before using the data in training** (cheap, local): raise the crackle source τ by ≈ 40 % (reference 4.1 → ≈ 6 ms; it was left untouched to keep the independent
   check meaningful, so re-check τ/2CD afterwards); add amplitude irregularity and transient content to the wheeze (breath-synchronous envelope, a few low-level clicks); band-limit the added
   ambient noise of the N variant with the same response; and lower the wheeze harmonics.
2. **Calibrate per device state**: 2022 differs from 2023–2025 (PSD peak −9 dB, no high-frequency noise). Fit the chain separately for 2022 and for 2023–2025, or add a per-year gain, and
   compare. This may also explain part of the year effects in e01–e03.
3. **When training is allowed, run the plan's step 3 as a controlled experiment** on top of e02/e03: train with the synthetic data only in the training years, report macro-F1 / per-class
   metrics per held-out year, and compare against the same models without synthetic data. Always drop the held-out year (`load_manifest(exclude_years=[...])`) and, for the official split,
   `host_split == "test"`. Start with synthetic W and FC to balance gate 2; keep the synthetic N as the control that makes "synthetic" uninformative.
4. **Run the plan's real-vs-synthetic probe** (a simple classifier on embeddings or features) at that point. If it separates the two with high accuracy, increase variability (more crackle
   morphology, amplitude irregularity) rather than adding more samples.
5. **Match the analysis band to the data**: use ≈ 60–900 Hz for mel features (finer resolution below 400 Hz) and redesign the crackle matched-filter bank around the real response
   (carriers 100–300 Hz, τ 3–12 ms). The current bank sits where the data has noise floor.
6. **Extend the generator to the remaining classes with the same machinery**: CC, WC (a crackle train plus a wheeze in the same event), rhonchi and stridor. After the chain, CC and FC may need a
   different handle than f0 (event rate, duration, amplitude pattern), so check first whether they are separable at all.
7. **Use more of the available hosts and diversify them**: only 812 of 4 487 were used; select hosts by matching age, location and noise level to the real minority events, and filter hosts with
   unannotated transients (e.g. high `n_transients` or tonal persistence) to cleaner ones.
8. **A real template option** (not done, by design): isolated crackles from the reference clips could be used as templates after passing through the chain; it is an alternative to the parametric
   source if you prefer morphology that is not parametric. Because the clips are loops, they would provide only a handful of distinct crackles.

## 9. Reproduction

```bash
.venv/Scripts/python experiments/gate2_event_type/e04_synthetic_augmentation/analyze_references.py      # step 1 (~1.5 min)
.venv/Scripts/python experiments/gate2_event_type/e04_synthetic_augmentation/analyze_sprsound_events.py # step 2 (~4 min)
.venv/Scripts/python experiments/gate2_event_type/e04_synthetic_augmentation/calibrate_synthesis.py     # step 4 (~25 min; --resume skips the chain search)
.venv/Scripts/python experiments/gate2_event_type/e04_synthetic_augmentation/build_synthetic.py --n-per-class 300 --overwrite   # step 5 (~45 s)
.venv/Scripts/python experiments/gate2_event_type/e04_synthetic_augmentation/qc_synthetic.py            # step 6 (~1 min)
```
