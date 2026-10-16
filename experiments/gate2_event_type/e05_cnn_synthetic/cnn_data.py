"""
cnn_data.py — log-mel inputs for real and synthetic events.

Same recipe as e03: the event is cut from the 4 kHz recording, tiled / centre-cropped to 3 s, STFT (n_fft 512, hop 128),
mel filter bank, dB, instance normalisation. Two analysis bands:

  2000 : 50–2000 Hz, 64 mels   (e03; identical to the cached e03 inputs)
  900  : 60–900 Hz, 40 mels    (suggestion of e04: SPRSound has no signal above ~900 Hz; 40 mels so that no filter is narrower
                                than one FFT bin)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import extract_spectral_features as ex  # noqa: E402

SR, DUR, N_FFT, HOP = ex.SR, 3.0, 512, 128
BANDS = {2000: dict(fmin=50, fmax=2000, n_mels=64), 900: dict(fmin=60, fmax=900, n_mels=40)}
_FB = {}


def filter_bank(band: int) -> np.ndarray:
    if band not in _FB:
        b = BANDS[band]
        fb = ex._mel_fb(n_mels=b["n_mels"], fmin=b["fmin"], fmax=b["fmax"])
        assert fb.sum(1).min() > 0, f"band {band}: empty mel filter"
        _FB[band] = fb
    return _FB[band]


def to_mel(seg: np.ndarray, band: int = 2000) -> np.ndarray:
    L = int(DUR * SR)
    w = seg[(len(seg) - L) // 2:(len(seg) - L) // 2 + L] if len(seg) >= L else np.tile(seg, int(np.ceil(L / len(seg))))[:L]
    _, _, Z = signal.stft(w, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann", boundary="even", padded=True)
    D = 10 * np.log10(np.maximum(filter_bank(band) @ (np.abs(Z) ** 2), 1e-10))
    return ((D - D.mean()) / (D.std() + 1e-6)).astype(np.float32)


def _job(stem, split, events, band):
    try:
        x = ex.load_wav(ex.RAW / split / "wav" / f"{stem}.wav")
    except Exception:
        return []
    return [(k, to_mel(x[int(s * SR / 1000):int(e * SR / 1000)], band)) for k, s, e in events
            if int(e * SR / 1000) - int(s * SR / 1000) >= 64]


def real_mels(ev, band: int) -> np.ndarray:
    """Mels of the real events in `ev` (columns stem, split, event_idx, start_ms, end_ms), in the row order of `ev`."""
    jobs = [(s, g["split"].iloc[0], list(zip(g.event_idx, g.start_ms, g.end_ms)), band) for s, g in ev.groupby("stem")]
    res = Parallel(n_jobs=4)(delayed(_job)(*j) for j in jobs)
    got = {f"{j[0]}|{k}": m for j, r in zip(jobs, res) for k, m in r}
    return np.stack([got[f"{s}|{k}"] for s, k in zip(ev["stem"], ev["event_idx"])])
