# gate 2 · e05 — CNN with synthetic data (W vs FC)

Wheeze (W) vs fine-crackle (FC) classification of SPRSound events. The experiment tests whether synthetic events (parametric crackles and
wheezes inserted into real normal recordings) help a small CNN. Result: mixing them into the real batches hurts, but **pretraining on them
and then fine-tuning on real data only helps** (mean leave-one-year-out AUC 0.931 vs 0.923; balanced accuracy at threshold 0.5: 0.854 vs 0.795).
The full study is in `results/report.md`; this README explains how to reproduce the best model.

## Best model

Pretraining on synthetic events (5 epochs) → fine-tuning on real events only → ensemble of 3 seeds → one ONNX file.
Official split: train 2022–2024, test 2025, threshold 0.5. Result on the 2025 test set (rows = true, columns = predicted):

| | predicted W | predicted FC | precision | recall | F1 | support |
|---|---:|---:|---:|---:|---:|---:|
| **W** | 233 | 72 | 0.456 | 0.764 | 0.571 | 305 |
| **FC** | 278 | 1607 | 0.957 | 0.853 | 0.902 | 1885 |
| macro avg | | | 0.707 | 0.808 | 0.736 | 2190 |
| weighted avg | | | 0.887 | 0.840 | 0.856 | 2190 |

Accuracy 0.840, AUC 0.889. The test set is 86 % FC, so read the macro average and the per-class recall rather than the accuracy.

## Reproduce it (augmentation → training → ONNX)

### 0. Setup

Python 3.10+. From the repository root:

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # needs torch, numpy, scipy, pandas, scikit-learn, soundfile, joblib, matplotlib, onnx, onnxruntime
bash get_data.sh                  # clones SPRSound -> data/raw_data/{train,test}/{wav,json}
```

The scripts resolve their own paths, so they can be run from any directory. Commands below are written from the repository root.

### 1. Feature tables (inputs of the synthetic generator)

The generator draws event durations and levels from real events and needs two tables in `experiments/common/features/`:

```bash
python experiments/common/extract_spectral_features.py   # -> segments_spectral.csv (all real events)
python experiments/common/physics_features.py            # -> physics_features.csv  (W and FC events)
```

### 2. Synthetic augmentation

```bash
E5=experiments/gate2_event_type/e05_cnn_synthetic
python $E5/build_synth_mels.py --name v2 --calibration $E5/results/calibration_map_v2.json \
       --n-per-class 2500 --seed 22 --match-hosts
```

This creates 2 500 FC + 2 500 W synthetic events and stores their log-mel inputs in `$E5/cache/synth_v2.npz` (about 200 MB, git-ignored) plus
`$E5/results/synth_v2_manifest.csv` with the provenance of each event. How they are made (details in `common/synthesis/`):

- A **real normal recording** of SPRSound is used as host; it is never replaced.
- A synthetic **crackle train** (damped sinusoids) or **wheeze** (gliding tonal bursts) is generated and passed through the stethoscope / device
  response of SPRSound (calibrated per device state in `results/calibration_v2_*.json`), then added to the host at a realistic level.
- Duration and level are drawn from real events of the same class and year; the recording is quantised to 8 kHz PCM-16 like the dataset.
- The MEDZCOOL clips in `Auscultaion_medzcool_audio/` were only morphology references; they are not needed to run anything.

### 3. Train the best CNN and export the ONNX

```bash
python $E5/train_best_cnn.py --data-root data/raw_data --synth-npz $E5/cache/synth_v2.npz --out outputs/best_cnn
```

About 5–6 minutes on a CPU. It writes only:

| file | content |
|---|---|
| `best_model.onnx` | the 3-seed ensemble as one model: input `mel` (batch, 1, 64, 95) → output `p_fc` (batch,), probability of FC |
| `confusion_matrix.png` | confusion matrix on the held-out year |
| `metrics.json` | confusion matrix, precision / recall / F1 per class, macro and weighted averages, accuracy |

The same metrics are printed in the console, and the script checks the ONNX output against PyTorch (difference ≈ 1e-6).
Options: `--held-out 2024` (any year, trained on the others), `--seeds 0 1 2`, `--threshold 0.5`, `--pre-epochs 5`, `--ratio 1.0`.
With the defaults the numbers in the table above are reproduced exactly (seeds are fixed).

Training recipe: patients of the training years are split 85 / 15 (by patient) into train / validation, validation uses real events only.
Synthetic events from the held-out year or from validation patients are never used. **Stage 1:** 5 epochs on synthetic events only
(AdamW, lr 1e-3, weight decay 1e-3). **Stage 2:** up to 20 epochs on real events only (AdamW, lr 3e-4, class-weighted BCE,
early stopping with patience 4 on the validation AUC, best weights restored). Both stages use batches of 64 and the augmentation described below.

### 4. Use the ONNX model

`onnx_inference.py` needs only `numpy scipy soundfile onnxruntime` (no PyTorch, no repository code), so it can be copied into another project
next to `best_model.onnx`:

```bash
python $E5/onnx_inference.py --model outputs/best_cnn/best_model.onnx --wav rec.wav --start-ms 1200 --end-ms 2900
python $E5/onnx_inference.py --model outputs/best_cnn/best_model.onnx --events-csv events.csv   # columns wav,start_ms,end_ms[,label]
```

It prints `p(FC)` and the class (FC if p ≥ 0.5, otherwise W); with a `label` column (0 = W, 1 = FC) it also prints the confusion matrix.

## Model input (log-mel)

Each annotated event becomes a 1 × 64 × 95 tensor:

1. first channel, resampled to 4 kHz, event cut by its start / end;
2. tiled (shorter than 3 s) or centre-cropped (longer) to 3 s;
3. STFT with `n_fft` 512, hop 128, Hann window;
4. 64 mel filters between 50 and 2000 Hz, power converted to dB (`10·log10`, floor 1e-10);
5. standardised per event: `(x − mean) / (std + 1e-6)`.

## CNN architecture

A small 2-D CNN (105 713 parameters per network; the ONNX file holds three, 1.2 MB) over the mel image (frequency × time):

| block | layers | output shape (channels × mel × frames) |
|---|---|---|
| input | log-mel | 1 × 64 × 95 |
| 1 | Conv 3×3 (16, pad 1, no bias) → BatchNorm → ReLU → MaxPool 2×2 | 16 × 32 × 47 |
| 2 | Conv 3×3 (32) → BatchNorm → ReLU → MaxPool 2×2 | 32 × 16 × 23 |
| 3 | Conv 3×3 (64) → BatchNorm → ReLU → MaxPool 2×2 | 64 × 8 × 11 |
| 4 | Conv 3×3 (128) → BatchNorm → ReLU → MaxPool 2×2 | 128 × 4 × 5 |
| pooling | global average pooling | 128 |
| head | Dropout 0.3 → Linear 128→64 → ReLU → Linear 64→1 | 1 logit |

- Each **convolutional block** learns local time-frequency patterns (clicks for crackles, narrow horizontal ridges for wheezes); BatchNorm stabilises
  training and MaxPool halves both axes, so deeper blocks see longer time spans and wider frequency ranges while the channels double.
- **Global average pooling** makes the head independent of the event position inside the 3 s window and keeps the model small.
- The output is a single logit for FC; `sigmoid(logit)` = p(FC), W = 1 − p. The exported ONNX model averages the sigmoid of the three seeds.
- **Training augmentation** (SpecAugment-like, applied per sample): circular time shift, one frequency mask (up to 8 mel bins) and one time mask
  (up to 10 frames).

## Files

| file | role |
|---|---|
| `train_best_cnn.py` | self-contained trainer: real events → synthetic pretraining → fine-tuning → metrics, confusion matrix, ONNX |
| `onnx_inference.py` | standalone inference (WAV + event times → p(FC)) with the ONNX model |
| `build_synth_mels.py`, `cnn_data.py` | create the synthetic mel set (`cache/synth_v2.npz`) |
| `results/calibration_*v2*.json` | per-device-state calibration of the synthetic generator |
| `../../common/synthesis/` | the synthetic-event generator (channel model, crackle / wheeze sources, insertion) |
| `../../common/extract_spectral_features.py`, `physics_features.py`, `paths.py` | real-event tables and paths used by the generator |

Further code of the full study (not needed for the best model): `train_cnn_synth.py` (all conditions), `calibrate_v2.py`, `qc_probe.py`, `report.py`.
