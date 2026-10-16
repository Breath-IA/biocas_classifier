"""
train_best_cnn.py  (gate 2 · e05) — best CNN: wheeze (W) vs fine crackle (FC)
─────────────────────────────────────────────────────────────────────────────
Trains the best model of e05 and writes ONLY these outputs:

    best_model.onnx       the seed ensemble as one ONNX model:  mel (batch, 1, 64, 95) -> p_fc (batch,)
    confusion_matrix.png  confusion matrix of the held-out year
    metrics.json          confusion matrix + precision, recall, F1 per class and their macro / weighted averages (also printed)

Recipe: 4-block CNN on log-mel; pretraining on SYNTHETIC events (5 epochs), then fine-tuning on REAL events only (AdamW, lr 3e-4,
early stopping on a real validation split of the training patients); 3 seeds, probabilities averaged. Official split: train on the
other years, evaluate on the held-out year (default 2025), threshold 0.5. W = class 0, FC = class 1.

Inputs
  --data-root   SPRSound folder with train/{wav,json} and test/{wav,json} (the output of get_data.sh: data/raw_data)
  --synth-npz   synthetic mel set (arrays y, year, patient, mel2000) created by build_synth_mels.py (e05); it is required because
                the best recipe pretrains on it. Synthetic events from the held-out year are never used.

Usage:  python train_best_cnn.py --data-root data/raw_data --synth-npz cache/synth_v2.npz --out outputs/best_cnn
"""
from __future__ import annotations

import argparse
import json
import time
from math import gcd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy import signal
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

torch.set_num_threads(4)
SR, WIN_S, N_FFT, HOP, N_MELS, FMIN, FMAX = 4000, 3.0, 512, 128, 64, 50.0, 2000.0
CLASSES = ["W", "FC"]
EVENT_TYPES = {"wheeze": 0, "fine crackle": 1}


# ── preprocessing: event → log-mel (1 × 64 × 95), identical to onnx_inference.py ──────────────────────────
def mel_filterbank() -> np.ndarray:
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = inv(np.linspace(mel(FMIN), mel(FMAX), N_MELS + 2))
    fb = np.zeros((N_MELS, len(freqs)))
    for i in range(N_MELS):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0, np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)))
    return fb


_FB = mel_filterbank()


def to_mel(seg: np.ndarray) -> np.ndarray:
    n = int(WIN_S * SR)
    w = seg[(len(seg) - n) // 2:(len(seg) - n) // 2 + n] if len(seg) >= n else np.tile(seg, int(np.ceil(n / len(seg))))[:n]
    _, _, z = signal.stft(w, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann", boundary="even", padded=True)
    d = 10 * np.log10(np.maximum(_FB @ (np.abs(z) ** 2), 1e-10))
    return ((d - d.mean()) / (d.std() + 1e-6)).astype(np.float32)


def _recording(wav: Path, js: Path, year: str, patient: str):
    ann = json.loads(js.read_text(encoding="utf-8"))
    ev = []
    for e in ann.get("event_annotation", []):
        label = EVENT_TYPES.get(str(e.get("type", "")).strip().lower())
        try:
            s, t = int(e["start"]), int(e["end"])
        except (KeyError, ValueError):
            continue
        if label is not None and t - s >= 20:
            ev.append((label, s, t))
    if not ev:
        return []
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        g = gcd(int(sr), SR)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    out = []
    for label, s, t in ev:
        seg = x[int(s * SR / 1000):int(t * SR / 1000)]
        if len(seg) >= 64:
            out.append((to_mel(seg), label, year, patient))
    return out


def load_real_events(root: Path):
    """W and FC events of every recording (train/ then test/, sorted by file name, events in JSON order)."""
    jobs = []
    for split in ("train", "test"):
        for wav in sorted((root / split / "wav").glob("*.wav")):
            js = root / split / "json" / f"{wav.stem}.json"
            year, rest = wav.stem.split("__", 1)
            pid = rest.split("_")[0]
            if js.exists():
                # patient keys as in the synthetic set: float strings ("40638274.0"), "nan" when the id is not numeric
                jobs.append((wav, js, year, str(float(pid)) if pid.isdigit() else "nan"))
    res = Parallel(n_jobs=4, verbose=0)(delayed(_recording)(*j) for j in jobs)
    rows = [r for rs in res for r in rs]
    X = np.stack([r[0] for r in rows])
    y, year, patient = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    return X, y, year, patient


# ── model and training ────────────────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    """4 × (Conv3×3 – BatchNorm – ReLU – MaxPool) → global average pooling → MLP; output = logit of FC."""

    def __init__(self, ch=(16, 32, 64, 128), drop=0.3):
        super().__init__()
        blocks, c_in = [], 1
        for c in ch:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.conv, self.pool = nn.Sequential(*blocks), nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(ch[-1], 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.pool(self.conv(x)).flatten(1)).squeeze(1)


class Ensemble(nn.Module):
    """Mean of the sigmoid outputs of several CNNs: (N, 1, 64, 95) → (N,) probability of FC."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, mel):
        return torch.stack([torch.sigmoid(m(mel)) for m in self.models]).mean(0)


def augment(x, rng):
    """Circular time shift (the event is tiled) + one frequency mask + one time mask (SpecAugment)."""
    out = x.clone()
    _, _, F, T = out.shape
    for i in range(out.shape[0]):
        out[i] = torch.roll(out[i], int(rng.integers(0, T)), dims=-1)
        f = int(rng.integers(0, 9)); f0 = int(rng.integers(0, F - f))
        out[i, :, f0:f0 + f] = 0
        t = int(rng.integers(0, 11)); t0 = int(rng.integers(0, T - t))
        out[i, :, :, t0:t0 + t] = 0
    return out


def train_one(X, y, year, patient, synth, held_out, seed, a):
    tr = year != held_out
    Xtr, ytr, gtr = X[tr], y[tr], patient[tr]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr, ytr, gtr))   # validation = real patients
    X_real, y_real = Xtr[tr_i], ytr[tr_i]

    # synthetic events for the pretraining: never from the held-out year nor from validation patients; equal number per class
    ok = (synth["year"] != held_out) & ~np.isin(synth["patient"], np.unique(gtr[va_i]))
    n_per = int(round(a.ratio * max((y_real == 0).sum(), (y_real == 1).sum())))
    srng = np.random.default_rng(seed + 1000)
    pick = np.concatenate([srng.choice(np.flatnonzero(ok & (synth["y"] == c)), size=min(n_per, int((ok & (synth["y"] == c)).sum())), replace=False)
                           for c in (0, 1)])
    T = lambda arr: torch.from_numpy(arr)[:, None]  # noqa: E731
    Xs, ys = T(synth["X"][pick]), torch.from_numpy(synth["y"][pick].astype(np.float32))
    Xa, ya = T(X_real), torch.from_numpy(y_real.astype(np.float32))
    Xv, Xe = T(Xtr[va_i]), T(X[~tr])

    model = CNN()

    def predict(Xx):
        model.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(model(Xx[i:i + 256])).numpy() for i in range(0, len(Xx), 256)])

    # stage 1: pretraining on synthetic events only
    prng = np.random.default_rng(seed + 2000)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(a.pre_epochs):
        model.train()
        perm = prng.permutation(len(Xs))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            if len(b) >= 4:
                loss = loss_fn(model(augment(Xs[b], prng)), ys[b])
                opt.zero_grad(); loss.backward(); opt.step()

    # stage 2: fine-tuning on real events only, early stopping on the real validation patients
    opt = torch.optim.AdamW(model.parameters(), lr=a.ft_lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y_real == 0).sum() / max((y_real == 1).sum(), 1))))
    best, best_state, bad = -1.0, None, 0
    for _ in range(a.epochs):
        model.train()
        perm = rng.permutation(len(Xa))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            if len(b) >= 4:
                loss = loss_fn(model(augment(Xa[b], rng)), ya[b])
                opt.zero_grad(); loss.backward(); opt.step()
        va = roc_auc_score(ytr[va_i], predict(Xv))
        if va > best + 1e-4:
            best, bad, best_state = va, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), best, len(pick)


# ── metrics, figure, ONNX ───────────────────────────────────────────────────────────────────────────────
def write_metrics(y, p, thr, out: Path, held_out):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    r = classification_report(y, pred, labels=[0, 1], target_names=CLASSES, output_dict=True, zero_division=0)
    pick = lambda d: {"precision": d["precision"], "recall": d["recall"], "f1": d["f1-score"], "support": int(d["support"])}  # noqa: E731
    metrics = {"held_out_year": held_out, "threshold": thr, "classes": CLASSES,
               "confusion_matrix_rows_true_cols_pred": cm.tolist(), "per_class": {c: pick(r[c]) for c in CLASSES},
               "accuracy": r["accuracy"], "macro_avg": pick(r["macro avg"]), "weighted_avg": pick(r["weighted avg"])}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.1%}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion matrix · held-out {held_out}")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=120); plt.close()

    print(f"\nConfusion matrix (rows = true, columns = predicted), held-out {held_out}, threshold {thr}\n          W     FC")
    print(f"  W   {cm[0, 0]:6d} {cm[0, 1]:6d}\n  FC  {cm[1, 0]:6d} {cm[1, 1]:6d}\n")
    print(f"{'':14s}{'precision':>10s}{'recall':>9s}{'f1-score':>10s}{'support':>9s}")
    for k, v in [*metrics["per_class"].items(), ("macro avg", metrics["macro_avg"]), ("weighted avg", metrics["weighted_avg"])]:
        print(f"{k:14s}{v['precision']:10.3f}{v['recall']:9.3f}{v['f1']:10.3f}{v['support']:9d}")
    print(f"{'accuracy':14s}{'':10s}{'':9s}{metrics['accuracy']:10.3f}{int(cm.sum()):9d}")


def export_onnx(models, out: Path, Xe: np.ndarray):
    import onnx
    import onnxruntime as ort

    ens = Ensemble(models).eval()
    path = out / "best_model.onnx"
    torch.onnx.export(ens, torch.zeros(1, 1, 64, 95), str(path), input_names=["mel"], output_names=["p_fc"],
                      dynamic_axes={"mel": {0: "batch"}, "p_fc": {0: "batch"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    xs = Xe[:256][:, None]
    with torch.no_grad():
        ref = ens(torch.from_numpy(xs)).numpy()
    diff = float(np.max(np.abs(sess.run(None, {"mel": xs})[0] - ref)))
    print(f"\nONNX: {path} ({path.stat().st_size / 1024:.0f} KB); max |ONNX - PyTorch| on {len(xs)} events = {diff:.1e}")
    return ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/raw_data"))
    ap.add_argument("--synth-npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/best_cnn"))
    ap.add_argument("--held-out", default="2025")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ratio", type=float, default=1.0, help="synthetic events per class = ratio × the larger real training class")
    ap.add_argument("--pre-epochs", type=int, default=5)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.synth_npz.exists():
        raise SystemExit(f"{a.synth_npz} not found: create it with build_synth_mels.py (the best recipe pretrains on synthetic events).")

    X, y, year, patient = load_real_events(a.data_root)
    z = np.load(a.synth_npz, allow_pickle=True)
    synth = {"X": z["mel2000"], "y": z["y"].astype(int), "year": z["year"].astype(str), "patient": z["patient"].astype(str)}
    print(f"{len(y)} real events (W {int((y == 0).sum())}, FC {int((y == 1).sum())}), {len(synth['y'])} synthetic events; held-out year {a.held_out}", flush=True)

    models = []
    for seed in a.seeds:
        t0 = time.time()
        m, best, n_syn = train_one(X, y, year, patient, synth, a.held_out, seed, a)
        models.append(m)
        print(f"seed {seed}: validation AUC {best:.3f}, {n_syn} synthetic events for pretraining, {time.time() - t0:.0f} s", flush=True)

    te = year == a.held_out
    Xe = X[te]
    ens = Ensemble(models).eval()
    with torch.no_grad():
        p = torch.cat([ens(torch.from_numpy(Xe[i:i + 256])[:, None]) for i in range(0, len(Xe), 256)]).numpy()
    write_metrics(y[te], p, a.threshold, a.out, a.held_out)
    export_onnx(models, a.out, Xe)


if __name__ == "__main__":
    main()
