"""
extract_spectral_features.py
────────────────────────────
Extrae features espectrales / temporales / tonales por EVENTO anotado de SPRSound
(requiere haber corrido get_data.sh).

Lee data/raw_data/{train,test}/{wav,json} generado por get_data.sh
(train = 2022 + 2023 + 2024, test = 2025; los archivos llevan prefijo '<año>__').

Salida: eda/out/segments_spectral.csv

    python eda/extract_spectral_features.py            # todo
    python eda/extract_spectral_features.py --limit 20 # prueba rápida
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy import fft as sfft
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")

RAW = Path(__file__).resolve().parents[1] / "data" / "raw_data"
OUT = Path(__file__).resolve().parent / "out"

EVENT_TYPE_NORM = {
    "normal": "N", "rhonchi": "R", "wheeze": "W", "stridor": "S",
    "coarse crackle": "CC", "fine crackle": "FC",
    "wheeze&crackle": "WC", "wheeze+crackle": "WC",
}

SR, N_FFT, HOP = 4000, 512, 64            # mismo sr que la pipeline (4 kHz)
FMIN, FMAX = 50, 2000                      # misma banda que log_mel de pipeline.yaml
BAND_EDGES = [50, 100, 200, 300, 400, 600, 800, 1000, 1250, 1500, 2000]
EPS = 1e-12

SOS_HP = signal.butter(4, 200, "highpass", fs=SR, output="sos")
SOS_BP = signal.butter(4, [100, 1000], "bandpass", fs=SR, output="sos")
FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)
MASK = (FREQS >= FMIN) & (FREQS <= FMAX)
F = FREQS[MASK]


def _templates():
    """Crepitantes sintéticos amortiguados: A·exp(-t/τ)·sin(2πf0t), τ = dur/3."""
    out = []
    for f0 in (300, 400, 500, 650, 800):
        for dur_ms in (5, 10, 15, 20):
            n = max(int(dur_ms * SR / 1000), 8)
            t = np.arange(n) / SR
            w = np.exp(-t / (dur_ms / 1000 / 3)) * np.sin(2 * np.pi * f0 * t)
            out.append((f0, dur_ms, (w / np.linalg.norm(w))[::-1]))
    return out


TEMPLATES = _templates()


def _mel_fb(n_mels=40, fmin=FMIN, fmax=FMAX):
    """Banco mel triangular (HTK) sobre FREQS — evita librosa/numba."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(FREQS)))
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0, np.minimum((FREQS - l) / (c - l), (r - FREQS) / (r - c)))
    return fb


MEL_FB = _mel_fb()


def stft_power(x):
    """Potencia STFT (257, T), hop=HOP, ventana hann de N_FFT; ~1 frame cada HOP muestras."""
    _, _, Z = signal.stft(x, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann",
                          boundary="even", padded=True)
    return np.abs(Z) ** 2


def parse_stem(stem: str) -> dict:
    parts = stem.split("_")
    return {
        "patient_id": parts[0],
        "age": float(parts[1]) if re.fullmatch(r"[\d.]+", parts[1]) else np.nan,
        "gender": int(parts[2]) if parts[2].isdigit() else -1,
        "location": parts[3],
    }


def load_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        g = gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def band_names():
    return [f"band_{a}_{b}" for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]


def segment_features(x, S, xhp, env_hp, xbp, ifreq, rec, start_ms, end_ms):
    """Features de un evento. S = potencia STFT de toda la grabación."""
    n0, n1 = int(start_ms * SR / 1000), int(end_ms * SR / 1000)
    n0, n1 = max(n0, 0), min(n1, len(x))
    dur_s = (n1 - n0) / SR
    T = S.shape[1]
    t0 = min(int(n0 / HOP), T - 1)
    t1 = max(min(int(np.ceil(n1 / HOP)), T), t0 + 1)
    Sseg = S[:, t0:t1]
    f = {"duration_s": dur_s, "n_frames": t1 - t0}

    # ── espectro medio ────────────────────────────────────────────────────
    Pfull = Sseg.mean(1)
    Pm = Pfull[MASK]
    tot = Pm.sum() + EPS
    p = Pm / tot
    cum = np.cumsum(p)
    cen = float((F * p).sum())
    f["spec_centroid"] = cen
    f["spec_bandwidth"] = float(np.sqrt(((F - cen) ** 2 * p).sum()))
    f["spec_rolloff85"] = float(F[min(np.searchsorted(cum, 0.85), len(F) - 1)])
    f["spec_rolloff95"] = float(F[min(np.searchsorted(cum, 0.95), len(F) - 1)])
    f["spec_flatness"] = float(np.exp(np.mean(np.log(Pm + EPS))) / (Pm.mean() + EPS))
    f["spec_entropy"] = float(-(p * np.log(p + EPS)).sum() / np.log(len(p)))
    f["spec_crest"] = float(Pm.max() / (Pm.mean() + EPS))
    f["dom_freq"] = float(F[np.argmax(Pm)])
    f["spec_slope_db_khz"] = float(np.polyfit(F / 1000, 10 * np.log10(Pm + EPS), 1)[0])
    for name, (a, b) in zip(band_names(), zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
        m = (F >= a) & (F < b) if b < FMAX else (F >= a) & (F <= b)
        f[name] = float(10 * np.log10(Pm[m].sum() / tot + EPS))
    lo, hi = Pm[F < 300].sum(), Pm[F >= 600].sum()
    f["lf_hf_db"] = float(10 * np.log10((lo + EPS) / (hi + EPS)))

    # espectro medio normalizado en 32 bins de 62.5 Hz (para curvas por clase)
    spb = 10 * np.log10(Pfull[:256].reshape(32, 8).mean(1) / tot * len(Pm) + EPS)
    for k, v in enumerate(spb):
        f[f"sp_{k:02d}"] = float(v)

    peaks, props = signal.find_peaks(10 * np.log10(Pm + EPS), prominence=6)
    f["n_spec_peaks"] = len(peaks)
    f["max_peak_prom_db"] = float(props["prominences"].max()) if len(peaks) else 0.0

    # ── dinámica frame a frame (tonalidad / estabilidad) ──────────────────
    Pf = Sseg[MASK]
    e = Pf.sum(0)
    act = e >= 0.2 * e.max()
    if act.sum() >= 3:
        dom = F[np.argmax(Pf[:, act], 0)]
        f["dom_freq_std"] = float(dom.std())
        f["dom_persist"] = float((np.abs(dom - np.median(dom)) <= 31.25).mean())
    else:
        f["dom_freq_std"], f["dom_persist"] = np.nan, np.nan
    Pa = Pf[:, act]
    f["frame_peakiness_db"] = float(np.mean(10 * np.log10(Pa.max(0) / (np.median(Pa, 0) + EPS) + EPS)))
    if Pf.shape[1] >= 2:
        Pn = Pf / (Pf.sum(0, keepdims=True) + EPS)
        f["spec_flux"] = float(np.mean(np.sqrt((np.diff(Pn, axis=1) ** 2).sum(0))))
    else:
        f["spec_flux"] = np.nan
    f["energy_cv"] = float(e.std() / (e.mean() + EPS))
    f["temporal_flatness"] = float(np.exp(np.mean(np.log(e + EPS))) / (e.mean() + EPS))

    # ── dominio temporal ──────────────────────────────────────────────────
    xs = x[n0:n1]
    if len(xs) >= 64:
        rms = float(np.sqrt(np.mean(xs ** 2)) + EPS)
        f["rms_db"] = 20 * np.log10(rms)
        f["crest_factor"] = float(np.abs(xs).max() / rms)
        f["kurtosis"] = float(kurtosis(xs))
        f["skewness"] = float(skew(xs))
        f["zcr"] = float(np.mean(np.abs(np.diff(np.sign(xs))) > 0))

        es = uniform_filter1d(env_hp[n0:n1], 8)
        med = np.median(es)
        mad = 1.4826 * np.median(np.abs(es - med)) + EPS
        pk, _ = signal.find_peaks(es, height=med + 4 * mad, distance=int(0.01 * SR))
        f["n_transients"] = len(pk)
        f["transient_rate"] = len(pk) / max(dur_s, 1e-3)
        f["env_kurtosis"] = float(kurtosis(es))
        f["env_cv"] = float(es.std() / (es.mean() + EPS))
        f["env_peak_to_median"] = float(es.max() / (med + EPS))

        # frecuencia instantánea (banda 100–1000 Hz) ponderada por energía
        env2 = np.abs(signal.hilbert(xbp[n0:n1])) ** 2
        ifs = ifreq[n0:n1]
        if len(ifs) >= 32 and env2.sum() > 0:
            w = env2 * (env2 > np.median(env2))
            if w.sum() > 0:
                mu = (w * ifs).sum() / w.sum()
                f["if_mean"] = float(mu)
                f["if_std"] = float(np.sqrt((w * (ifs - mu) ** 2).sum() / w.sum()))
        # periodicidad (ACF) en banda 100–1000 Hz
        xb = xbp[n0:n1] - xbp[n0:n1].mean()
        if len(xb) >= 400 and xb.std() > 0:
            nfft = 1 << (2 * len(xb) - 1).bit_length()
            ac = np.fft.irfft(np.abs(np.fft.rfft(xb, nfft)) ** 2)[: len(xb)]
            ac /= ac[0] + EPS
            lags = np.arange(SR // 1000, SR // 100)
            seg = ac[lags]
            f["acf_peak"] = float(seg.max())
            f["f0_acf"] = float(SR / lags[np.argmax(seg)])

        # matched filter de crepitante amortiguado
        xh = xhp[n0:n1]
        best = (-1.0, 0, 0)
        for f0, dms, tpl in TEMPLATES:
            if len(xh) < len(tpl):
                continue
            r = np.abs(signal.fftconvolve(xh, tpl, mode="valid"))
            r = r / (np.median(r) + 1e-9)           # pico / mediana: cuánto resalta un crepitante
            if r.max() > best[0]:
                best = (float(r.max()), f0, dms)
                rbest = r
        if best[0] >= 0:
            f["mf_peak_ratio"], f["mf_f0"], f["mf_dur_ms"] = best
            pk2, _ = signal.find_peaks(rbest, height=6, distance=int(0.01 * SR))
            f["mf_rate"] = len(pk2) / max(dur_s, 1e-3)

    # ── MFCC (12 coef.) ───────────────────────────────────────────────────
    logmel = 10 * np.log10(np.maximum(MEL_FB @ Sseg, 1e-10))
    mf = sfft.dct(logmel, type=2, axis=0, norm="ortho")[:13].mean(1)
    for k in range(1, 13):
        f[f"mfcc_{k}_mean"] = float(mf[k])

    # ── relativas a la grabación (menos sensibles a dispositivo/ganancia) ─
    seg_e_db = 10 * np.log10(e.mean() + EPS)
    f["snr_vs_floor_db"] = float(seg_e_db - rec["floor_db"])
    f["level_vs_rec_db"] = float(seg_e_db - rec["median_db"])
    for nm, rb in zip(("low", "mid", "high"), rec["bands"]):
        bdb = 10 * np.log10(Sseg[rb["mask"]].sum(0).mean() + EPS)
        f[f"rel_{nm}_db"] = float(bdb - rb["median_db"])
    f["rel_tilt_db"] = f["rel_high_db"] - f["rel_low_db"]
    return f


def process_recording(wav_path, json_path, meta):
    try:
        ann = json.loads(Path(json_path).read_text(encoding="utf-8"))
        x = load_wav(wav_path)
    except Exception:
        return []
    events = []
    for k, ev in enumerate(ann.get("event_annotation", [])):
        cls = EVENT_TYPE_NORM.get(str(ev.get("type", "")).strip().lower())
        if cls is None:
            continue
        try:
            s, e = int(ev["start"]), int(ev["end"])
        except Exception:
            continue
        if e - s >= 20:
            events.append((k, cls, s, e))
    if not events or len(x) < N_FFT * 2:
        return []

    S = stft_power(x)
    xhp = signal.sosfiltfilt(SOS_HP, x)
    env_hp = np.abs(signal.hilbert(xhp))
    xbp = signal.sosfiltfilt(SOS_BP, x)
    ph = np.unwrap(np.angle(signal.hilbert(xbp)))
    ifreq = np.r_[0.0, np.diff(ph)] * SR / (2 * np.pi)

    erec = 10 * np.log10(S[MASK].sum(0) + EPS)
    bands = []
    for a, b in ((50, 300), (300, 600), (600, 2000)):
        m = (FREQS >= a) & (FREQS < b)
        bands.append({"mask": m, "median_db": float(np.median(10 * np.log10(S[m].sum(0) + EPS)))})
    rec = {"floor_db": float(np.percentile(erec, 10)), "median_db": float(np.median(erec)), "bands": bands}

    poor = "poor" in str(ann.get("record_annotation", "")).lower()
    rows = []
    for k, cls, s, e in events:
        try:
            feats = segment_features(x, S, xhp, env_hp, xbp, ifreq, rec, s, e)
        except Exception:
            continue
        rows.append({**meta, "event_idx": k, "class": cls, "start_ms": s, "end_ms": e,
                     "rec_duration_s": len(x) / SR, "n_events_rec": len(events),
                     "start_frac": s / 1000 / (len(x) / SR), "poor_quality": poor, **feats})
    return rows


def build_jobs():
    jobs = []
    for split in ("train", "test"):
        jmap = {p.stem: p for p in (RAW / split / "json").glob("*.json")}
        for wav in sorted((RAW / split / "wav").glob("*.wav")):
            if wav.stem in jmap:
                year, name = wav.stem.split("__", 1)
                jobs.append((wav, jmap[wav.stem], {"stem": wav.stem, "source": year,
                                                    "split": split, **parse_stem(name)}))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = build_jobs()
    if a.limit:
        jobs = jobs[:: max(len(jobs) // a.limit, 1)][: a.limit]
    print(f"{len(jobs)} grabaciones")
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(process_recording)(*j) for j in jobs)
    df = pd.DataFrame([r for rs in res for r in rs])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("segments_spectral_sample.csv" if a.limit else "segments_spectral.csv")
    df.to_csv(path, index=False)
    print(f"{len(df)} eventos, {df.shape[1]} columnas -> {path}")
    print(df.groupby(["split", "class"]).size().unstack(0))


if __name__ == "__main__":
    main()
