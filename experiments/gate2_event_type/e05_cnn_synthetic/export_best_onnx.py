"""
export_best_onnx.py  (gate 2 · e05)
───────────────────────────────────
Retrains the best e05 recipe once (the e05 runs did not keep the weights) and exports it to ONNX:

    pretrain on the v2 synthetic events (5 epochs) → fine-tune on real events only (lr 3e-4, early stopping on real validation patients)

for the OFFICIAL split (train 2022–2024, test 2025), with the same 3 seeds as e05. The three networks are exported as ONE ONNX model that
averages their probabilities (the seed ensemble that gave the confusion matrix). The script also checks that

  * the retrained networks reproduce the saved e05 results (AUC per seed, confusion matrix of the ensemble),
  * the ONNX outputs match PyTorch (max absolute difference),
  * and writes the 2025 event list for `onnx_inference.py --events-csv`, which repeats the evaluation from raw audio through ONNX.

Usage : python export_best_onnx.py [--held-out 2025] [--seeds 0 1 2]
Output: results/best_model/{best_model.onnx, cnn_seed<k>.pt, best_model.json, events_<year>.csv, export_check.md}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_cnn_synth as t  # noqa: E402  (loads e03's CNN, the real/synthetic data and run_one)

OUT = HERE / "results" / "best_model"
paths = t.paths


class Ensemble(nn.Module):
    """Mean of the sigmoid outputs of several CNNs: (N, 1, 64, 95) log-mel → (N,) probability of FC."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, mel):
        return torch.stack([torch.sigmoid(m(mel)) for m in self.models]).mean(0)


def cm_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return cm, {"auc": float(roc_auc_score(y, p)), "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", default="2025")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--synth", default="v2")
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--pre-epochs", type=int, default=5)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    H = a.held_out

    ev, X, y = t.load_real(2000)
    synth = t.load_synth(a.synth, 2000)
    te = (ev["source"] == H).values
    saved = pd.read_csv(HERE / "results" / "v2_pre_b2000_results.csv")
    saved["fold"] = saved["fold"].astype(str)

    models, lines = [], []
    for seed in a.seeds:
        t0 = time.time()
        p, best, ep, n_syn, n_real, model = t.run_one(ev, X, y, synth, H, seed, a.ratio, 20, 4, mode="pretrain", pre_epochs=a.pre_epochs,
                                                      ft_lr=a.ft_lr, return_model=True)
        model.eval()
        torch.save(model.state_dict(), OUT / f"cnn_seed{seed}.pt")
        models.append(model)
        ref = saved[(saved.fold == H) & (saved.seed == seed)]["auc"]
        auc = roc_auc_score(y[te], p)
        lines.append(f"- seed {seed}: AUC {auc:.4f} (saved e05 run: {ref.iloc[0]:.4f}, |diff| {abs(auc - ref.iloc[0]):.1e}); epochs {ep}; "
                     f"{n_syn} synthetic events used; {time.time() - t0:.0f} s")
        print(lines[-1], flush=True)

    ens = Ensemble(models).eval()
    Xe = torch.from_numpy(X[te])[:, None]
    with torch.no_grad():
        p_torch = torch.cat([ens(Xe[i:i + 256]) for i in range(0, len(Xe), 256)]).numpy()
    cm_t, m_t = cm_metrics(y[te], p_torch)

    onnx_path = OUT / "best_model.onnx"
    torch.onnx.export(ens, torch.zeros(1, 1, 64, 95), str(onnx_path), input_names=["mel"], output_names=["p_fc"],
                      dynamic_axes={"mel": {0: "batch"}, "p_fc": {0: "batch"}}, opset_version=a.opset, dynamo=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    p_onnx = np.concatenate([sess.run(None, {"mel": X[te][i:i + 256][:, None]})[0] for i in range(0, int(te.sum()), 256)])
    diff = float(np.max(np.abs(p_onnx - p_torch)))
    cm_o, m_o = cm_metrics(y[te], p_onnx)
    one = sess.run(None, {"mel": X[te][:1][:, None]})[0]                          # batch of 1 works (dynamic batch axis)

    # confusion matrix saved by confusion_matrix.py (from the e05 predictions) for the same held-out year
    cmf = pd.read_csv(OUT / "confusion_matrix.csv", index_col=0) if (OUT / "confusion_matrix.csv").exists() else None
    same = None if cmf is None or H not in cmf.index else bool((cmf.loc[H].values.reshape(2, 2) == cm_o).all())

    ev_te = ev[te].copy()
    pd.DataFrame({"wav": [str(paths.RAW / s / "wav" / f"{st}.wav") for s, st in zip(ev_te["split"], ev_te["stem"])], "start_ms": ev_te["start_ms"].values,
                  "end_ms": ev_te["end_ms"].values, "label": y[te], "stem": ev_te["stem"].values, "event_idx": ev_te["event_idx"].values}
                 ).to_csv(OUT / f"events_{H}.csv", index=False)
    meta = {"model": "e05 best: synthetic (v2) pretraining -> real fine-tuning, seed ensemble", "held_out_year": H, "seeds": a.seeds,
            "classes": {"0": "W (wheeze)", "1": "FC (fine crackle)"}, "output": "p_fc = probability of FC, mean over the seeds", "threshold": 0.5,
            "input": {"name": "mel", "shape": ["batch", 1, 64, 95], "dtype": "float32"},
            "preprocessing": {"sample_rate_hz": 4000, "window_s": 3.0, "short_events": "tiled to 3 s", "long_events": "centre crop to 3 s", "n_fft": 512,
                              "hop": 128, "stft_window": "hann (scipy.signal.stft, boundary=even, padded)", "mel": "64 HTK-style triangular filters, 50-2000 Hz",
                              "log": "10*log10(max(power, 1e-10))", "normalisation": "per event: (x - mean) / (std + 1e-6)"},
            "training": {"pretrain_epochs": a.pre_epochs, "finetune_lr": a.ft_lr, "synthetic_ratio": a.ratio, "synthetic_set": a.synth, "optimizer": "AdamW wd 1e-3"},
            "metrics_on_held_out_year": {"auc": m_o["auc"], "balanced_accuracy": m_o["balanced_accuracy"], "confusion_matrix_rows_true_cols_pred": cm_o.tolist()},
            "onnx": {"opset": a.opset, "max_abs_diff_vs_pytorch": diff, "size_kb": round(onnx_path.stat().st_size / 1024, 1)}}
    (OUT / "best_model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report = ["# Export check\n", f"Held-out year {H}; official split (trained on the other years).\n", "## Retrained networks vs. the saved e05 runs\n", *lines,
              "\n## Ensemble", f"- PyTorch: AUC {m_t['auc']:.4f}, balanced accuracy {m_t['balanced_accuracy']:.4f}, confusion matrix {cm_t.tolist()}",
              f"- ONNX (onnxruntime, CPU): AUC {m_o['auc']:.4f}, balanced accuracy {m_o['balanced_accuracy']:.4f}, confusion matrix {cm_o.tolist()}",
              f"- max |p_onnx − p_torch| over {int(te.sum())} events: {diff:.2e}; batch-of-1 output shape {one.shape}",
              f"- identical to the confusion matrix computed from the saved e05 predictions (`confusion_matrix.csv`): {same}",
              f"- ONNX file: {onnx_path.name}, {meta['onnx']['size_kb']} KB, opset {a.opset}"]
    (OUT / "export_check.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
