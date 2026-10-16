"""
physics_features.py
───────────────────
Features "physics-informed" de plan.md (sección 1) para los eventos W y FC:

  A. Envolvente + tasa de decaimiento τ   (Hilbert → regresión log-lineal tras cada transitorio)
  B. Frecuencia instantánea y su varianza (pasa-banda 100–1900 Hz → Hilbert → derivada de fase)
  C. Banco de matched filters de crepitante  y(t)=A·exp(-t/τ)·sin(2π f0 t): barrido (f0, τ) → máximo, (f0*, τ*)
  D. Matched filter de sibilancia (oscilador sostenido, ventana 100 ms, f0 100–1000 Hz): estabilidad y persistencia
  E. Persistencia espectral entre frames (espectro blanqueado por grabación) y pico blanqueado

Todas con prefijo `ph_`. Se calculan sobre el evento anotado (sin tile), con contexto de la grabación.

Salida: experiments/common/features/physics_features.csv   (stem, event_idx, ph_*)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_spectral_features as ex  # noqa: E402

SR = ex.SR
OUT = ex.OUT                       # experiments/common/features
EPS = 1e-12

SOS_HP = ex.SOS_HP
SOS_BP2 = signal.butter(4, [100, 1900], "bandpass", fs=SR, output="sos")

# C. banco de crepitantes
CR_F0 = (200, 300, 400, 500, 650, 800, 1000)
CR_TAU_MS = (1, 2, 4, 6, 8, 12)


def _cr_bank():
    bank = []
    for f0 in CR_F0:
        for tau in CR_TAU_MS:
            n = max(int(4 * tau * SR / 1000), 8)
            t = np.arange(n) / SR
            w = np.exp(-t / (tau / 1000)) * np.sin(2 * np.pi * f0 * t)
            bank.append((f0, tau, (w / np.linalg.norm(w))[::-1]))
    return bank


CR_BANK = _cr_bank()

# D. banco de sibilancias: exponenciales complejas con ventana Hann de 100 ms
WH_N = int(0.1 * SR)
WH_F0 = np.arange(100, 1001, 50)
_n = np.arange(WH_N)
_hann = np.hanning(WH_N)
WH_BANK = [np.conj(_hann * np.exp(2j * np.pi * f0 * _n / SR))[::-1] / np.linalg.norm(_hann) for f0 in WH_F0]


def _nan_row():
    names = ["tau_med", "tau_p25", "tau_p75", "n_decay_fits", "decay_frac", "env_slope_med", "env_slope_neg_frac",
             "if_mean", "if_std", "if_cv", "if_stable_frac", "if_std_best100",
             "mf_cr_max", "mf_cr_f0", "mf_cr_tau", "mf_cr_top5", "mf_cr_rate",
             "mf_wh_max", "mf_wh_f0", "mf_wh_frac50", "mf_wh_frac70", "mf_wh_f0_track_std",
             "persist_cos_mean", "persist_cos_p10", "persist_run_s", "wh_peak_db", "wh_peak_f", "dur_est_s"]
    return {f"ph_{n}": np.nan for n in names}


def event_features(x, S_white, xhp, env, xbp, ifreq, n0, n1):
    f = _nan_row()
    n0, n1 = max(n0, 0), min(n1, len(x))
    if n1 - n0 < 64:
        return f
    dur = (n1 - n0) / SR

    # ── A. envolvente y decaimiento τ ─────────────────────────────────────
    es = uniform_filter1d(env[n0:n1], 8)
    med = np.median(es)
    mad = 1.4826 * np.median(np.abs(es - med)) + EPS
    pk, _ = signal.find_peaks(es, height=med + 4 * mad, distance=int(0.01 * SR))
    taus = []
    for p in pk:
        seg = es[p:p + 60]                                    # hasta 15 ms tras el pico
        if len(seg) < 12:
            continue
        below = np.where(seg < 0.1 * seg[0])[0]
        seg = seg[: below[0]] if len(below) >= 1 and below[0] >= 8 else seg
        if len(seg) < 8:
            continue
        slope = np.polyfit(np.arange(len(seg)), np.log(seg + 1e-9), 1)[0]        # por muestra
        taus.append(100.0 if slope >= 0 else float(np.clip(-1000 / (slope * SR), 0.5, 100.0)))
    f["ph_n_decay_fits"] = len(taus)
    if taus:
        f["ph_tau_med"], f["ph_tau_p25"], f["ph_tau_p75"] = np.median(taus), *np.percentile(taus, [25, 75])
        f["ph_decay_frac"] = float(np.mean(np.array(taus) < 20))
    win = 80                                                  # 20 ms
    if len(es) >= 2 * win:
        starts = np.arange(0, len(es) - win, win // 2)
        lg = np.log(es + 1e-9) * 20 / np.log(10)               # dB
        sl = np.array([np.polyfit(np.arange(win), lg[s:s + win], 1)[0] * SR / 1000 for s in starts])  # dB/ms
        f["ph_env_slope_med"] = float(np.median(sl))
        f["ph_env_slope_neg_frac"] = float(np.mean(sl < -0.5))

    # ── B. frecuencia instantánea (100–1900 Hz) ───────────────────────────
    ib = median_filter(ifreq[n0:n1], size=21, mode="nearest")
    eb = np.abs(signal.hilbert(xbp[n0:n1])) ** 2
    if eb.sum() > 0 and len(ib) >= 64:
        w = eb * (eb > np.median(eb))
        if w.sum() > 0:
            mu = (w * ib).sum() / w.sum()
            sd = np.sqrt((w * (ib - mu) ** 2).sum() / w.sum())
            f["ph_if_mean"], f["ph_if_std"], f["ph_if_cv"] = mu, sd, sd / (abs(mu) + 1e-6)
            order = np.argsort(ib)
            cw = np.cumsum(w[order])
            wmed = ib[order][np.searchsorted(cw, cw[-1] / 2)]
            f["ph_if_stable_frac"] = float((w * (np.abs(ib - wmed) <= 50)).sum() / w.sum())
        L = int(0.1 * SR)
        if len(ib) >= L:
            best = []
            tot = [eb[s:s + L].sum() for s in range(0, len(ib) - L + 1, L // 4)]
            thr = 0.5 * max(tot)
            for k, s in enumerate(range(0, len(ib) - L + 1, L // 4)):
                if tot[k] >= thr:
                    ww = eb[s:s + L]
                    m = (ww * ib[s:s + L]).sum() / ww.sum()
                    best.append(np.sqrt((ww * (ib[s:s + L] - m) ** 2).sum() / ww.sum()))
            f["ph_if_std_best100"] = float(np.min(best))

    # ── C. banco de matched filters de crepitante ─────────────────────────
    xh = xhp[n0:n1]
    best = (-1.0, 0, 0, None)
    for f0, tau, tpl in CR_BANK:
        if len(xh) < len(tpl):
            continue
        r = np.abs(signal.fftconvolve(xh, tpl, mode="valid"))
        r = r / (np.median(r) + 1e-9)
        if r.max() > best[0]:
            best = (float(r.max()), f0, tau, r)
    if best[3] is not None:
        f["ph_mf_cr_max"], f["ph_mf_cr_f0"], f["ph_mf_cr_tau"] = best[:3]
        pk2, pr = signal.find_peaks(best[3], height=6, distance=int(0.01 * SR))
        top = np.sort(best[3][pk2])[::-1][:5] if len(pk2) else np.array([best[0]])
        f["ph_mf_cr_top5"] = float(top.mean())
        f["ph_mf_cr_rate"] = len(pk2) / max(dur, 1e-3)

    # ── D. matched filter de sibilancia ───────────────────────────────────
    xw = xbp[n0:n1]
    if len(xw) >= WH_N + 4:
        en = np.sqrt(np.maximum(np.convolve(xw ** 2, np.ones(WH_N), mode="valid"), 1e-18))
        # máximo teórico de |<x,w·e^{jωt}>|/(‖x‖‖w‖) para un tono puro con ventana Hann = 0,577 → se reescala a [0,1]
        C = np.stack([np.abs(signal.fftconvolve(xw, b, mode="valid")) / en for b in WH_BANK]) / 0.577   # (f0, t)
        cmax = C.max(0)
        am = WH_F0[C.argmax(0)]
        f["ph_mf_wh_max"] = float(cmax.max())
        f["ph_mf_wh_f0"] = float(am[np.argmax(cmax)])
        f["ph_mf_wh_frac50"] = float(np.mean(cmax >= 0.5))
        f["ph_mf_wh_frac70"] = float(np.mean(cmax >= 0.7))
        sel = am[cmax >= 0.5]
        if len(sel) >= 10:
            f["ph_mf_wh_f0_track_std"] = float(np.std(sel))

    # ── E. persistencia espectral (blanqueada) ────────────────────────────
    T = S_white.shape[1]
    t0 = min(int(n0 / ex.HOP), T - 1)
    t1 = max(min(int(np.ceil(n1 / ex.HOP)), T), t0 + 1)
    band = (ex.FREQS >= 250) & (ex.FREQS <= 1500)
    Wf = np.sqrt(S_white[band][:, t0:t1])
    if Wf.shape[1] >= 3:
        Wc = Wf - Wf.mean(0, keepdims=True)                          # centrado → correlación de Pearson entre frames (más contraste que el coseno)
        Wn = Wc / (np.linalg.norm(Wc, axis=0, keepdims=True) + EPS)
        cos = (Wn[:, 1:] * Wn[:, :-1]).sum(0)
        f["ph_persist_cos_mean"] = float(cos.mean())
        f["ph_persist_cos_p10"] = float(np.percentile(cos, 10))
        run = best_run = 0
        for c in cos >= 0.8:
            run = run + 1 if c else 0
            best_run = max(best_run, run)
        f["ph_persist_run_s"] = best_run * ex.HOP / SR
    mean_w = S_white[band][:, t0:t1].mean(1)
    f["ph_wh_peak_db"] = float(10 * np.log10(mean_w.max() / (np.median(mean_w) + EPS) + EPS))
    f["ph_wh_peak_f"] = float(ex.FREQS[band][np.argmax(mean_w)])
    ebnd = S_white[band][:, t0:t1].sum(0)
    f["ph_dur_est_s"] = float((ebnd >= 0.5 * ebnd.max()).sum() * ex.HOP / SR)
    return f


def process(stem, split, events):
    try:
        x = ex.load_wav(ex.RAW / split / "wav" / f"{stem}.wav")
    except Exception:
        return []
    return process_array(x, stem, events)


def process_array(x, stem, events):
    """Physics features of `events` [(idx, start_ms, end_ms)] on an in-memory 4 kHz waveform (also used for synthetic audio)."""
    if len(x) < ex.N_FFT * 2:
        return []
    S = ex.stft_power(x)
    S_white = S / (np.median(S, axis=1, keepdims=True) + EPS)          # blanqueo por grabación
    xhp = signal.sosfiltfilt(SOS_HP, x)
    env = np.abs(signal.hilbert(xhp))
    xbp = signal.sosfiltfilt(SOS_BP2, x)
    ph = np.unwrap(np.angle(signal.hilbert(xbp)))
    ifreq = np.r_[0.0, np.diff(ph)] * SR / (2 * np.pi)
    rows = []
    for k, s, e in events:
        try:
            feats = event_features(x, S_white, xhp, env, xbp, ifreq, int(s * SR / 1000), int(e * SR / 1000))
        except Exception:
            feats = _nan_row()
        rows.append({"stem": stem, "event_idx": k, **feats})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ex.OUT / "segments_spectral.csv", usecols=["stem", "event_idx", "class", "split", "start_ms", "end_ms"])
    df = df[df["class"].isin(["W", "FC"])]
    jobs = [(stem, g["split"].iloc[0], list(zip(g.event_idx, g.start_ms, g.end_ms))) for stem, g in df.groupby("stem")]
    print(f"{len(jobs)} grabaciones, {len(df)} eventos")
    res = Parallel(n_jobs=4, verbose=2)(delayed(process)(*j) for j in jobs)
    out = pd.DataFrame([r for rs in res for r in rs])
    out.to_csv(OUT / "physics_features.csv", index=False)
    print(out.shape, "->", OUT / "physics_features.csv")
    print(out.isna().mean().round(3).sort_values(ascending=False).head(8).to_string())


if __name__ == "__main__":
    main()
