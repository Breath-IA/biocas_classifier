# gate 2 · e11 — e09 CRNN cut after its last 3×3 convolution (reference for the FPGA)

`export_last_conv.py` (CPU, onnxruntime) truncates `../e09_crnn_optuna_colab/crnn_w_vs_fc.onnx` and writes `crnn_last_conv3x3.onnx`. Same graph, same weights (no retraining); the BatchNorm is already folded into the convolutions.
Graph: `Conv3×3(1→32)-ReLU-MaxPool2×2 → Conv3×3(32→64)-ReLU-MaxPool2×2 → Conv3×3(64→128)-ReLU-MaxPool(2,1)`, padding 1, stride 1. Input `log_mel` (B,1,64,95).

| output | shape | where |
|---|---|---|
| `conv3_out` (main) | (B,128,16,23) | output of the last Conv3×3 (+ folded BN, bias), **before ReLU** |
| `conv3_relu` | (B,128,16,23) | after ReLU |
| `conv3_pool` | (B,128,8,23) | after the (2,1) max-pool = what the GRU receives |

Check done: the three tensors are bit-identical (max error 0.0) to the same tensors inside the full e09 model, on the 9 samples.

Reference data for the comparison: `input_log_mel.npy` (9,1,64,95: 8 real 2025 events + 1 random tile, seed 0), `output_conv3_out.npy`, `output_conv3_relu.npy`, `output_conv3_pool.npy`, all in one `fpga_reference.npz` (also has the event indices), and for sample 0 plain text, one float per line in (channel, row, column) order: `sample0_input.txt`, `sample0_conv3_out.txt`.
Floating point float32; if the FPGA is fixed point, compare with a tolerance.
