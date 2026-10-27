# Export check

Held-out year 2025; official split (trained on the other years).

## Retrained networks vs. the saved e05 runs

- seed 0: AUC 0.8716 (saved e05 run: 0.8716, |diff| 0.0e+00); epochs 9; 2698 synthetic events used; 86 s
- seed 1: AUC 0.8944 (saved e05 run: 0.8944, |diff| 0.0e+00); epochs 15; 2818 synthetic events used; 123 s
- seed 2: AUC 0.8689 (saved e05 run: 0.8689, |diff| 0.0e+00); epochs 8; 2924 synthetic events used; 86 s

## Ensemble
- PyTorch: AUC 0.8887, balanced accuracy 0.8082, confusion matrix [[233, 72], [278, 1607]]
- ONNX (onnxruntime, CPU): AUC 0.8887, balanced accuracy 0.8082, confusion matrix [[233, 72], [278, 1607]]
- max |p_onnx − p_torch| over 2190 events: 1.04e-06; batch-of-1 output shape (1,)
- identical to the confusion matrix computed from the saved e05 predictions (`confusion_matrix.csv`): True
- ONNX file: best_model.onnx, 1247.5 KB, opset 17