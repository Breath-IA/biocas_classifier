"""
onnx_inference.py  (gate 2 · e05)
─────────────────────────────────
Standalone inference for the exported e05 model (wheeze vs fine crackle). Needs only numpy, scipy, soundfile and onnxruntime:
no PyTorch and no repository code, so it can be copied into another project together with `best_model.onnx`.

Input  : an annotated event of a SPRSound-style recording (WAV + start/end in ms)
Output : probability of FC (fine crackle); W (wheeze) = 1 − p. Decision threshold 0.5.

Preprocessing (identical to the training pipeline; see best_model.json):
  first channel → 4 kHz (polyphase) → event segment → tiled (short) or centre-cropped (long) to 3 s → STFT (n_fft 512, hop 128, Hann)
  → 64 HTK-style mel filters, 50–2000 Hz → 10·log10(power) → per-event standardisation → tensor (1, 1, 64, 95)

Usage
  python onnx_inference.py --wav rec.wav --start-ms 1200 --end-ms 2900
  python onnx_inference.py --events-csv results/best_model/events_2025.csv      # columns wav,start_ms,end_ms[,label]; with label: confusion matrix
"""
from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy import signal

SR, WIN_S, N_FFT, HOP, N_MELS, FMIN, FMAX = 4000, 3.0, 512, 128, 64, 50.0, 2000.0
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "results" / "best_model" / "best_model.onnx"


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


def load_event(wav: str, start_ms: float, end_ms: float) -> np.ndarray:
    """Event segment at 4 kHz (float32)."""
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        g = gcd(int(sr), SR)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    seg = x[int(start_ms * SR / 1000):int(end_ms * SR / 1000)]
    if len(seg) < 64:
        raise ValueError(f"event too short ({len(seg)} samples at {SR} Hz)")
    return seg


def to_mel(seg: np.ndarray) -> np.ndarray:
    n = int(WIN_S * SR)
    w = seg[(len(seg) - n) // 2:(len(seg) - n) // 2 + n] if len(seg) >= n else np.tile(seg, int(np.ceil(n / len(seg))))[:n]
    _, _, z = signal.stft(w, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann", boundary="even", padded=True)
    d = 10 * np.log10(np.maximum(_FB @ (np.abs(z) ** 2), 1e-10))
    return ((d - d.mean()) / (d.std() + 1e-6)).astype(np.float32)


def predict(session: ort.InferenceSession, mels: np.ndarray, batch: int = 256) -> np.ndarray:
    """mels: (N, 64, 95) → probability of FC, shape (N,)."""
    return np.concatenate([session.run(None, {"mel": mels[i:i + batch][:, None]})[0] for i in range(0, len(mels), batch)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--wav")
    ap.add_argument("--start-ms", type=float)
    ap.add_argument("--end-ms", type=float)
    ap.add_argument("--events-csv", type=Path)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, help="CSV with the probabilities (events-csv mode)")
    a = ap.parse_args()
    sess = ort.InferenceSession(str(a.model), providers=["CPUExecutionProvider"])

    if a.wav:
        p = float(predict(sess, to_mel(load_event(a.wav, a.start_ms, a.end_ms))[None])[0])
        print(f"p(FC) = {p:.4f} → {'FC (fine crackle)' if p >= a.threshold else 'W (wheeze)'}")
        return
    import csv

    with open(a.events_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mels = np.stack([to_mel(load_event(r["wav"], float(r["start_ms"]), float(r["end_ms"]))) for r in rows])
    p = predict(sess, mels)
    print(f"{len(rows)} events scored")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wav", "start_ms", "end_ms", "p_fc", "pred"])
            for r, v in zip(rows, p):
                w.writerow([r["wav"], r["start_ms"], r["end_ms"], f"{v:.6f}", "FC" if v >= a.threshold else "W"])
    if "label" in rows[0]:
        y = np.array([int(r["label"]) for r in rows])
        pred = (p >= a.threshold).astype(int)
        cm = np.array([[np.sum((y == i) & (pred == j)) for j in (0, 1)] for i in (0, 1)])
        print("confusion matrix (rows = true, columns = predicted)\n        W    FC")
        print(f"  W   {cm[0, 0]:5d} {cm[0, 1]:5d}\n  FC  {cm[1, 0]:5d} {cm[1, 1]:5d}")
        print(f"recall W {cm[0, 0] / cm[0].sum():.3f} | recall FC {cm[1, 1] / cm[1].sum():.3f} | balanced accuracy "
              f"{0.5 * (cm[0, 0] / cm[0].sum() + cm[1, 1] / cm[1].sum()):.3f}")


if __name__ == "__main__":
    main()
