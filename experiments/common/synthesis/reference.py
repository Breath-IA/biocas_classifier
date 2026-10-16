"""
reference.py — analysis of the MEDZCOOL reference clips (fine crackles, coarse crackles, expiratory wheeze).

The clips are teaching recordings (44.1 kHz stereo MP3), i.e. a different acquisition chain from SPRSound, and
their cycles are looped. They are therefore used ONLY to estimate source morphology; they are never pasted into
SPRSound audio. Analysis is done after resampling to the dataset's native 8 kHz.

Outputs are plain dictionaries / arrays so that the generators can be parameterised from them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import optimize, signal
from scipy.ndimage import label, uniform_filter1d

from . import FS
from .channel import to_native_rate


# ── loading and cycle structure ───────────────────────────────────────────────
def load_reference(path) -> Dict:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if x.shape[1] > 1 else 1.0
    mono = x.mean(1).astype(np.float64)
    # bandwidth of the native clip: frequency below which 99 % of the energy lies
    f, P = signal.welch(mono, fs=sr, nperseg=4096)
    cum = np.cumsum(P) / P.sum()
    return {"sr": int(sr), "duration_s": len(mono) / sr, "lr_corr": corr, "peak": float(np.abs(x).max()),
            "bw99_hz": float(f[np.searchsorted(cum, 0.99)]), "y": to_native_rate(mono, sr, FS)}


def breath_segments(y: np.ndarray, fs: int = FS, band=(60, 900), min_len_s: float = 0.4,
                    rel_thr: float = 0.12) -> List[Tuple[int, int]]:
    """Active breath-sound segments from the smoothed band envelope."""
    sos = signal.butter(4, band, "bandpass", fs=fs, output="sos")
    env = uniform_filter1d(np.abs(signal.hilbert(signal.sosfiltfilt(sos, y))), int(0.15 * fs))
    act = env > rel_thr * np.percentile(env, 95)
    lab, n = label(act)
    segs = []
    for i in range(1, n + 1):
        idx = np.flatnonzero(lab == i)
        if (idx[-1] - idx[0]) / fs >= min_len_s:
            segs.append((int(idx[0]), int(idx[-1])))
    return segs


def unique_segments(y: np.ndarray, segs, fs: int = FS, sim_thr: float = 0.95, win_s: float = 0.8):
    """
    Cluster segments whose first `win_s` seconds are (almost) identical copies (max normalized cross-correlation
    ≥ sim_thr). Returns (representative indices, cluster sizes, list of clusters).
    """
    L = int(win_s * fs)
    wav = [y[a:a + L] for a, _ in segs]
    ok = [len(w) == L for w in wav]
    reps, clusters = [], []
    for i, w in enumerate(wav):
        if not ok[i]:
            continue
        placed = False
        for c, r in enumerate(reps):
            sim = np.max(np.abs(signal.fftconvolve(w, wav[r][::-1], mode="full"))) / (
                np.linalg.norm(w) * np.linalg.norm(wav[r]) + 1e-12)
            if sim >= sim_thr:
                clusters[c].append(i)
                placed = True
                break
        if not placed:
            reps.append(i)
            clusters.append([i])
    return reps, [len(c) for c in clusters], clusters


# ── crackles ──────────────────────────────────────────────────────────────────
def _damped_sine(t, A, f0, tau, phi):
    return A * np.exp(-t / tau) * np.sin(2 * np.pi * f0 * t + phi)


def detect_crackles(y: np.ndarray, seg: Tuple[int, int], fs: int = FS, k: float = 6.0, min_dist_ms: float = 8.0,
                    hp_hz: float = 150.0):
    """
    Transient detection inside one segment on the high-passed envelope. Only the segment (plus 0.25 s of padding) is
    filtered, so this is cheap on long recordings. Returns (hp signal, same length as y, zero outside the padded
    window; peak indices in y coordinates).
    """
    a, b = seg
    pad = int(0.25 * fs)
    lo, hi = max(a - pad, 0), min(b + pad, len(y))
    sos = signal.butter(4, hp_hz, "highpass", fs=fs, output="sos")
    hp = np.zeros_like(y, dtype=np.float64)
    hp[lo:hi] = signal.sosfiltfilt(sos, y[lo:hi])
    env = uniform_filter1d(np.abs(signal.hilbert(hp[a:b])), max(int(0.0005 * fs), 1))
    med = np.median(env)
    mad = 1.4826 * np.median(np.abs(env - med)) + 1e-12
    pk, _ = signal.find_peaks(env, height=med + k * mad, distance=int(min_dist_ms / 1000 * fs))
    return hp, pk + a


def fit_crackle(hp: np.ndarray, peak: int, fs: int = FS, pre_ms: float = 2.0, post_ms: float = 12.0,
                r2_min: float = 0.4) -> Optional[Dict]:
    """
    Fit A·exp(-t/τ)·sin(2π f0 t + φ) to the crackle around `peak` and measure IDW/2CD from zero crossings.
    Returns None when the fit is poor (R² < r2_min) or the window is truncated.
    """
    a, b = peak - int(pre_ms / 1000 * fs), peak + int(post_ms / 1000 * fs)
    if a < 0 or b > len(hp):
        return None
    w = hp[a:b]
    mx = np.max(np.abs(w))
    if mx <= 0:
        return None
    on = int(np.argmax(np.abs(w) > 0.15 * mx))               # onset = first sample above 15 % of the peak
    seg = w[on:] / mx
    t = np.arange(len(seg)) / fs
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1024))
    f_init = float(np.fft.rfftfreq(1024, 1 / fs)[1 + int(np.argmax(spec[1:]))])
    best = None
    for phi0 in (0.0, 1.5, 3.0, -1.5):
        try:
            popt, _ = optimize.curve_fit(_damped_sine, t, seg, p0=[1.0, np.clip(f_init, 200, 3000), 0.003, phi0],
                                         bounds=([0, 150, 3e-4, -np.pi], [5, 3500, 0.015, np.pi]), maxfev=2000)
        except Exception:
            continue
        sse = float(np.sum((seg - _damped_sine(t, *popt)) ** 2))
        if best is None or sse < best[0]:
            best = (sse, popt)
    if best is None:
        return None
    sse, (A, f0, tau, phi) = best
    r2 = 1 - sse / (np.sum((seg - seg.mean()) ** 2) + 1e-12)
    if r2 < r2_min:
        return None
    zc = np.flatnonzero(np.diff(np.signbit(seg).astype(int)))
    idw = zc[0] / fs * 1000 if len(zc) >= 1 else np.nan
    cd2 = zc[3] / fs * 1000 if len(zc) >= 4 else np.nan       # 4 half-waves = two cycles
    return {"f0_hz": float(f0), "tau_ms": float(tau * 1000), "phi": float(phi), "r2": float(r2),
            "idw_ms": float(idw), "twocd_ms": float(cd2), "amp_rel": float(mx), "peak": int(peak)}


def analyze_crackles(y: np.ndarray, segs, fs: int = FS, k: float = 4.0, hp_hz: float = 250.0, post_ms: float = 12.0,
                     r2_min: float = 0.4) -> Dict:
    """Crackle statistics over the given (unique) breath segments."""
    fits, icis, rates, pos, amps = [], [], [], [], []
    ev_spectra = []
    n_det = 0
    for a, b in segs:
        hp, pk = detect_crackles(y, (a, b), fs, k, hp_hz=hp_hz)
        n_det += len(pk)
        dur = (b - a) / fs
        rates.append(len(pk) / dur)
        pos += [float((p - a) / (b - a)) for p in pk]
        ici = np.diff(pk) / fs * 1000
        icis += [float(v) for v in ici if v < 150]
        loc_rms = [np.sqrt(np.mean(hp[max(p - int(0.05 * fs), 0):p + int(0.05 * fs)] ** 2)) + 1e-12 for p in pk]
        amps += [float(abs(hp[p]) / r) for p, r in zip(pk, loc_rms)]
        for i, p in enumerate(pk):
            iso = (i == 0 or (p - pk[i - 1]) / fs > 0.015) and (i == len(pk) - 1 or (pk[i + 1] - p) / fs > 0.015)
            if iso:
                f = fit_crackle(hp, p, fs, post_ms=post_ms, r2_min=r2_min)
                if f:
                    fits.append(f)
                w = hp[p - int(0.008 * fs):p + int(0.008 * fs)]
                if len(w) == int(0.016 * fs):
                    ev_spectra.append(np.abs(np.fft.rfft(w * np.hanning(len(w)), 256)) ** 2)
    out = {"n_segments": len(segs), "n_detected": n_det, "n_isolated_fits": len(fits),
           "rate_hz_per_segment": rates, "ici_ms": icis, "position_in_segment": pos, "amp_over_local_rms": amps,
           "fits": fits}
    if ev_spectra:
        out["trigger_spectrum"] = {"freq_hz": np.fft.rfftfreq(256, 1 / fs).tolist(),
                                   "power": np.mean(ev_spectra, 0).tolist()}
    return out


def summarize_crackles(res: Dict) -> Dict:
    """Parameters used by the generator: log-normal fits and descriptive quantiles."""
    def lognorm(v):
        v = np.array([x for x in v if np.isfinite(x) and x > 0])
        return ({"median": float(np.median(v)), "log_sd": float(np.std(np.log(v))), "p10": float(np.percentile(v, 10)),
                 "p90": float(np.percentile(v, 90)), "n": int(len(v))} if len(v) >= 3 else None)
    fits = res["fits"]
    ici = np.array(res["ici_ms"])
    s = {"f0_hz": lognorm([f["f0_hz"] for f in fits]), "tau_ms": lognorm([f["tau_ms"] for f in fits]),
         "idw_ms": lognorm([f["idw_ms"] for f in fits]), "twocd_ms": lognorm([f["twocd_ms"] for f in fits]),
         "amp_over_local_rms": lognorm(res["amp_over_local_rms"]),
         "rate_hz": lognorm(res["rate_hz_per_segment"]), "ici_ms": lognorm(ici.tolist()),
         "phase_rad": {"p10": float(np.percentile([f["phi"] for f in fits], 10)),
                       "p90": float(np.percentile([f["phi"] for f in fits], 90))} if fits else None,
         "n_detected": res["n_detected"], "n_isolated_fits": res["n_isolated_fits"]}
    if len(ici) >= 8:                       # gamma shape of the inter-crackle interval above the refractory time
        m, v = np.mean(ici), np.var(ici)
        s["ici_gamma_shape"] = float(np.clip(m * m / (v + 1e-9), 0.8, 6.0))
    pos = np.array(res["position_in_segment"])
    if len(pos):
        s["position_thirds"] = [float(np.mean(pos < 1 / 3)), float(np.mean((pos >= 1 / 3) & (pos < 2 / 3))),
                                float(np.mean(pos >= 2 / 3))]
    return s


# ── wheezes ───────────────────────────────────────────────────────────────────
def track_wheeze(y: np.ndarray, seg: Tuple[int, int], fs: int = FS, n_fft: int = 2048, hop: int = 200,
                 prom_db: float = 10.0, max_jump_hz: float = 35.0, min_dur_s: float = 0.08, fmin=80, fmax=1500):
    """
    Spectral-peak tracking inside one breath segment. Peaks must stand `prom_db` above a local (±120 Hz) median.
    Returns a list of tracks {t, f, a_db} and the STFT axes.
    """
    a, b = seg
    f, t, Z = signal.stft(y[a:b], fs=fs, nperseg=n_fft, noverlap=n_fft - hop, window="hann", boundary=None, padded=False)
    S = 10 * np.log10(np.abs(Z) ** 2 + 1e-14)
    df = f[1] - f[0]
    k = int(120 / df) | 1
    base = signal.medfilt2d(S, [k, 1]) if k < S.shape[0] else np.median(S, 0, keepdims=True)
    band = (f >= fmin) & (f <= fmax)
    tracks, active = [], []
    for j in range(S.shape[1]):
        col = S[:, j] - base[:, j]
        pk, _ = signal.find_peaks(np.where(band, col, -99), height=prom_db, distance=int(30 / df))
        used = set()
        new_active = []
        for tr in active:
            cand = [p for p in pk if p not in used and abs(f[p] - tr["f"][-1]) <= max_jump_hz]
            if cand:
                p = min(cand, key=lambda q: abs(f[q] - tr["f"][-1]))
                used.add(p)
                tr["t"].append(t[j]); tr["f"].append(f[p]); tr["a_db"].append(S[p, j])
                new_active.append(tr)
            else:
                tracks.append(tr)
        for p in pk:
            if p not in used:
                new_active.append({"t": [t[j]], "f": [f[p]], "a_db": [S[p, j]]})
        active = new_active
    tracks += active
    tracks = [tr for tr in tracks if tr["t"][-1] - tr["t"][0] >= min_dur_s]
    return tracks, (f, t, S)


def analyze_wheeze(y: np.ndarray, segs, fs: int = FS) -> Dict:
    """Wheeze statistics over unique breath segments: f0 (lowest strong track), harmonics, jitter, drift, polyphony."""
    rows = []
    for a, b in segs:
        tracks, (f, t, S) = track_wheeze(y, (a, b), fs)
        if not tracks:
            continue
        # f0 = dominant persistent tone (duration x energy) between 120 and 1200 Hz. The lowest track is not used:
        # in real recordings it is often breath / heart-sound rumble rather than a wheeze.
        cand = [tr for tr in tracks if 120 <= np.median(tr["f"]) <= 1200]
        if not cand:
            continue
        base = max(cand, key=lambda tr: (tr["t"][-1] - tr["t"][0]) * 10 ** (np.mean(tr["a_db"]) / 10))
        strong = [tr for tr in tracks if np.mean(tr["a_db"]) > np.mean(base["a_db"]) - 15]
        f0 = float(np.median(base["f"]))
        tt, ff = np.array(base["t"]), np.array(base["f"])
        slope = float(np.polyfit(tt, ff, 1)[0]) if len(tt) >= 3 else 0.0
        resid = ff - np.polyval(np.polyfit(tt, ff, 1), tt) if len(tt) >= 3 else np.zeros_like(ff)
        harm_db = []
        for h in (2, 3, 4):
            fh = h * f0
            if fh < fs / 2 - 100:
                jj = np.argmin(np.abs(f - fh))
                col = [S[max(jj - 2, 0):jj + 3, np.argmin(np.abs(t - tx))].max() for tx in tt]
                col0 = [S[np.argmin(np.abs(f - fx)), np.argmin(np.abs(t - tx))] for fx, tx in zip(ff, tt)]
                harm_db.append(float(np.mean(col) - np.mean(col0)))
            else:
                harm_db.append(np.nan)
        n_par = []                                                                 # simultaneous tracks per frame
        for tx in tt:
            n_par.append(sum(1 for tr in strong if tr["t"][0] <= tx <= tr["t"][-1]))
        rows.append({"f0_hz": f0, "dur_s": float(tt[-1] - tt[0]), "seg_dur_s": (b - a) / fs,
                     "jitter_cv": float(np.std(resid) / f0), "drift_rel": float(slope * (tt[-1] - tt[0]) / f0),
                     "harm2_db": harm_db[0], "harm3_db": harm_db[1], "harm4_db": harm_db[2],
                     "max_polyphony": int(max(n_par)), "start_in_seg": float(tt[0] / ((b - a) / fs)),
                     "end_in_seg": float(tt[-1] / ((b - a) / fs)), "n_tracks": len(tracks)})
    return {"rows": rows}


def summarize_wheeze(res: Dict) -> Dict:
    rows = res["rows"]
    if not rows:
        return {}

    def q(k):
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        return ({"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                 "n": int(len(v))} if len(v) else None)
    s = {k: q(k) for k in ("f0_hz", "dur_s", "jitter_cv", "drift_rel", "harm2_db", "harm3_db", "harm4_db",
                           "start_in_seg", "end_in_seg", "n_tracks")}
    s["polyphony_gt1_fraction"] = float(np.mean([r["max_polyphony"] > 1 for r in rows]))
    s["n_segments"] = len(rows)
    return s
