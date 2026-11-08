# Tuned CRNN for wheeze vs fine crackle (SPRSound, test year 2025)

**Task.** Event-level wheeze (W) vs fine crackle (FC). Train 2022-2024, test 2025, validation = 20 % of the training patients. Priority: wheeze recall (weight 0.65) without letting FC recall fall under 0.8.

**Model.** Log-mel (1 x 64 x 95) -> 3 x [Conv3x3-BN-ReLU-MaxPool] with channels (32, 64, 128) -> bidirectional GRU (hidden 64, 1 layer(s)) -> mean pooling over time -> Dropout 0.29 -> Linear 32 -> ReLU -> Linear 1 -> sigmoid.
515,617 parameters, 66.2 M multiply-accumulates per 3-s event, ONNX file 2067 kB, CPU latency 1.3 ms (onnxruntime, batch 1).

**Training.** AdamW (lr 7.20e-04, weight decay 2.19e-04), batch 64, FC loss weight x1.14 of the balanced weight, SpecAugment-style masks and circular shift, early stopping on validation AUC. Search: Optuna TPE, 40 trials, 250,000-700,000 parameters.

**Test results 2025 (3-seed ensemble, threshold 0.726 chosen on validation)** - tuned CRNN vs the e06 CRNN:

| metric | tuned CRNN | e06 CRNN | 95 % CI tuned |
|---|---:|---:|---|
| AUC | 0.875 | 0.867 | 0.790-0.929 |
| recall W | 0.862 | 0.875 | 0.714-0.954 |
| precision W | 0.312 | 0.285 | 0.132-0.464 |
| recall FC | 0.692 | 0.645 | 0.629-0.749 |
| precision FC | 0.969 | 0.970 | 0.931-0.992 |
| priority score | 0.587 | 0.485 | 0.433-0.707 |

The exported network (seed 0, threshold 0.846): AUC 0.855, recall W 0.807, recall FC 0.779.

**Caveats.** One test year with few patients (see the intervals); the threshold, architecture and loss weight were chosen on validation data only; literature numbers are for different tasks (see the comparison table).
