"""
analyze_sprsound_events.py  (gate 2 · e04, step 2)
──────────────────────────────────────────────────
The reference clips come from another device, so their parameters are only a morphological prior. This script
measures the SAME crackle / wheeze statistics on real SPRSound events at the native 8 kHz (i.e. after the real
acquisition chain). These numbers are the calibration target for the synthetic generators, and the ratio between
the SPRSound and MEDZCOOL crackle spectra gives a first estimate of the chain's frequency response.

Also plots the median PSD per year: the high-frequency floor differs between years.

Input  : experiments/common/features/segments_spectral.csv, data/raw_data (native 8 kHz WAV)
Output : results/sprsound_event_stats.json, results/channel_estimate.json, results/sprsound_*.png, results/sprsound_event_stats.md
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402
from synthesis import FS  # noqa: E402
from synthesis import reference as ref  # noqa: E402
from synthesis.channel import ChannelModel  # noqa: E402

warnings.filterwarnings("ignore")
RES = paths.G2_E04 / "results"
N_PER_CLASS = 300
YEARS = ["2022", "2023", "2024", "2025"]
FC_CFG = {"k": 4.5, "hp_hz": 250.0}


def load_event(row, pad_s=0.1):
    x, sr = sf.read(str(paths.RAW / row.split / "wav" / f"{row.stem}.wav"), dtype="float64")
    assert sr == FS
    a = max(int((row.start_ms / 1000 - pad_s) * FS), 0)
    b = min(int((row.end_ms / 1000 + pad_s) * FS), len(x))
    return x, a, b


def trigger_spectrum(y, seg):
    """Mean spectrum of un-filtered ±8 ms windows around crackle peaks (peaks found on the high-passed envelope)."""
    hp, pk = ref.detect_crackles(y, seg, FS, FC_CFG["k"], hp_hz=150.0)
    out = []
    h = int(0.008 * FS)
    for p in pk:
        if p - h >= 0 and p + h <= len(y):
            w = y[p - h:p + h]
            w = w - w.mean()
            out.append(np.abs(np.fft.rfft(w * np.hanning(len(w)), 256)) ** 2)
    return out


def sample_events(df, cls, n, rng):
    d = df[df["class"] == cls]
    parts = [g.sample(min(len(g), max(n // 4, 1)), random_state=int(rng.integers(1e6))) for _, g in d.groupby("source")]
    return pd.concat(parts).reset_index(drop=True)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    df = pd.read_csv(paths.SEGMENTS_CSV, usecols=["stem", "event_idx", "class", "source", "split", "start_ms", "end_ms", "duration_s"])
    df["source"] = df["source"].astype(str)
    md = ["# Real SPRSound event statistics (native 8 kHz, after the real acquisition chain)\n",
          f"Up to {N_PER_CLASS} events per class, stratified by year. Same detectors and fitters as the reference analysis.\n"]
    out = {}

    # ── fine crackles ─────────────────────────────────────────────────────
    ev = sample_events(df, "FC", N_PER_CLASS, rng)
    cfgs = {"hp100": 100.0, "hp150": 150.0, "hp250": 250.0}
    fits_by = {k: [] for k in cfgs}
    ndet_by = {k: 0 for k in cfgs}
    ici, rate, amps, spectra = [], [], [], []
    for r in ev.itertuples():
        x, a, b = load_event(r)
        for name, hpz in cfgs.items():
            hp, pk = ref.detect_crackles(x, (a, b), FS, FC_CFG["k"], hp_hz=hpz)
            ndet_by[name] += len(pk)
            if name == "hp150":
                rate.append(len(pk) / max((b - a) / FS, 0.1))
                ici += [float(v) for v in np.diff(pk) / FS * 1000 if v < 150]
            n_fit = 0
            for i, p in enumerate(pk):
                iso = (i == 0 or (p - pk[i - 1]) / FS > 0.015) and (i == len(pk) - 1 or (pk[i + 1] - p) / FS > 0.015)
                if iso and n_fit < 8:                                   # at most 8 fits per event keeps the sample balanced
                    f = ref.fit_crackle(hp, p, FS)
                    if f:
                        f["year"] = r.source
                        fits_by[name].append(f)
                        n_fit += 1
                        if name == "hp150":
                            amps.append(float(abs(hp[p]) / (np.sqrt(np.mean(hp[max(p - 400, 0):p + 400] ** 2)) + 1e-12)))
        spectra += trigger_spectrum(x, (a, b))
    fits = fits_by["hp150"]
    res = {"fits": fits, "ici_ms": ici, "rate_hz_per_segment": rate, "amp_over_local_rms": amps,
           "position_in_segment": [], "n_detected": ndet_by["hp150"], "n_isolated_fits": len(fits)}
    s = ref.summarize_crackles(res)
    s["n_events"] = len(ev)
    s["f0_vs_highpass"] = {k: {"hp_hz": cfgs[k], "f0_median_hz": float(np.median([f["f0_hz"] for f in v])),
                               "share_within_60hz_of_cutoff": float(np.mean([f["f0_hz"] < cfgs[k] + 60 for f in v])),
                               "tau_median_ms": float(np.median([f["tau_ms"] for f in v])),
                               "twocd_median_ms": float(np.nanmedian([f["twocd_ms"] for f in v])), "n_fits": len(v)}
                           for k, v in fits_by.items()}
    s["tau_by_year"] = {y: float(np.median([f["tau_ms"] for f in fits if f["year"] == y])) for y in YEARS if any(f["year"] == y for f in fits)}
    out["FC"] = s
    fc_spec = np.mean(spectra, 0) if spectra else None
    md += ["## Fine crackles (FC) in SPRSound\n", f"- {len(ev)} events; {s['n_isolated_fits']} isolated transients fitted with the 150 Hz high-pass",
           "- **f0 is not identifiable at SPRSound's bandwidth**: the fitted f0 follows the high-pass cutoff "
           "(100 / 150 / 250 Hz → median f0 " + " / ".join(f"{v['f0_median_hz']:.0f}" for v in s["f0_vs_highpass"].values()) +
           " Hz; share of fits within 60 Hz of the cutoff " + " / ".join(f"{v['share_within_60hz_of_cutoff']:.0%}" for v in s["f0_vs_highpass"].values()) +
           "). The real crackle energy sits in a very low band, unlike the reference clip (f0 ≈ 610 Hz).",
           f"- tau {s['tau_ms']['median']:.1f} ms (p10–p90 {s['tau_ms']['p10']:.1f}–{s['tau_ms']['p90']:.1f}) and 2CD {s['twocd_ms']['median']:.1f} ms "
           f"are longer than in the reference (≈ 4 ms and 3 ms): the chain low-pass stretches the crackles; tau by year {s['tau_by_year']}",
           f"- transients per second inside events (150 Hz high-pass, k = 4.5; includes breath-noise bumps): median {s['rate_hz']['median']:.1f} /s "
           f"(p10–p90 {s['rate_hz']['p10']:.1f}–{s['rate_hz']['p90']:.1f})",
           f"- inter-transient interval median {s['ici_ms']['median']:.1f} ms, gamma shape {s.get('ici_gamma_shape', float('nan')):.2f}",
           f"- amplitude / local RMS log-sd {s['amp_over_local_rms']['log_sd']:.2f}"]

    # ── wheezes ───────────────────────────────────────────────────────────
    ev = sample_events(df, "W", N_PER_CLASS, rng)
    rows = []
    for r in ev.itertuples():
        if r.duration_s < 0.35:
            continue
        x, a, b = load_event(r, pad_s=0.0)
        rr = ref.analyze_wheeze(x, [(a, b)], FS)["rows"]
        if rr:
            rr[0]["year"] = r.source
            rr[0]["event_dur_s"] = float(r.duration_s)
            rows.append(rr[0])
    sw = ref.summarize_wheeze({"rows": rows})
    sw["n_events"] = len(rows)
    sw["f0_by_year"] = {y: float(np.median([r["f0_hz"] for r in rows if r["year"] == y])) for y in YEARS if any(r["year"] == y for r in rows)}
    sw["f0_values"] = [r["f0_hz"] for r in rows]
    out["W"] = sw
    md += ["\n## Wheezes (W) in SPRSound\n", f"- {len(rows)} events with a tracked tone",
           f"- f0 {sw['f0_hz']['median']:.0f} Hz (p10–p90 {sw['f0_hz']['p10']:.0f}–{sw['f0_hz']['p90']:.0f}); by year {sw['f0_by_year']}",
           f"- tone duration {sw['dur_s']['median']:.2f} s; jitter CV {sw['jitter_cv']['median']:.3f}; relative drift {sw['drift_rel']['median']:.3f}",
           f"- harmonics vs f0: 2f0 {sw['harm2_db']['median']:.1f} dB, 3f0 {sw['harm3_db']['median']:.1f} dB, 4f0 {sw['harm4_db']['median']:.1f} dB",
           f"- polyphonic in {sw['polyphony_gt1_fraction']:.0%} of the events"]

    # ── chain estimate: SPRSound / MEDZCOOL crackle spectra ───────────────
    rp = json.loads((RES / "reference_params.json").read_text())
    clip = next(p for p in sorted(paths.REFERENCE_AUDIO.glob("*.mp3")) if p.name.startswith("Fine"))
    c = ref.load_reference(clip)
    segs = ref.breath_segments(c["y"])
    reps, _, _ = ref.unique_segments(c["y"], segs)
    ref_spec = []
    for i in reps:
        ref_spec += trigger_spectrum(c["y"], segs[i])
    ref_spec = np.mean(ref_spec, 0)
    f = np.fft.rfftfreq(256, 1 / FS)
    m = (f >= 100) & (f <= 2000)
    ratio_db = 10 * np.log10(fc_spec / ref_spec)
    ratio_db = ratio_db - np.interp(300, f, ratio_db)
    best = None
    for lp in np.arange(300, 3001, 25):
        for order in (2, 4):
            h = ChannelModel(lp_hz=float(lp), lp_order=order, hp_hz=50.0).response_db(f[m])
            h = h - np.interp(300, f[m], h)
            sse = float(np.mean((ratio_db[m] - h) ** 2))
            if best is None or sse < best["rmse2"]:
                best = {"lp_hz": float(lp), "lp_order": order, "rmse2": sse}
    best["rmse_db"] = float(np.sqrt(best.pop("rmse2")))
    chain = {"hp_hz": 50.0, "hp_order": 2, **best,
             "method": "least-squares fit of a low-pass to the SPRSound/MEDZCOOL fine-crackle triggered-spectrum ratio (100–2000 Hz); "
                       "assumes similar source spectra, so it is a first estimate to be refined by calibrate_synthesis.py"}
    (RES / "channel_estimate.json").write_text(json.dumps(chain, indent=2), encoding="utf-8")
    md += ["\n## Channel estimate\n", f"- relative low-pass vs. the reference device: **{best['lp_hz']:.0f} Hz, order {best['lp_order']}** "
           f"(fit RMSE {best['rmse_db']:.1f} dB). Used only as the starting point of the calibration."]

    # ── per-year PSD ──────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
    allf = df.drop_duplicates("stem")
    psd = {}
    for y in YEARS:
        st = allf[allf.source == y].sample(min(200, (allf.source == y).sum()), random_state=0)
        P = []
        for r in st.itertuples():
            x, sr = sf.read(str(paths.RAW / r.split / "wav" / f"{r.stem}.wav"), dtype="float64")
            if len(x) >= 8000:
                fr, p = signal.welch(x, fs=sr, nperseg=1024, noverlap=512)
                P.append(10 * np.log10(p + 1e-14))
        psd[y] = (fr, np.median(P, 0), np.percentile(P, 90, 0))
        axs[0].plot(fr, psd[y][1], label=y)
        axs[1].plot(fr, psd[y][2], label=y)
    axs[0].set_title("median PSD per year (200 recordings each)")
    axs[1].set_title("p90 PSD per year (the noisier recordings)")
    for a in axs[:2]:
        a.set_xlabel("Hz"); a.set_ylabel("dB"); a.legend(); a.grid(alpha=.3)
    axs[2].semilogy(f, fc_spec / fc_spec.max(), label="SPRSound FC (real)")
    axs[2].semilogy(f, ref_spec / ref_spec.max(), label="MEDZCOOL FC")
    axs[2].set_title("crackle-triggered mean spectra (±8 ms)"); axs[2].set_xlabel("Hz"); axs[2].legend(); axs[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(RES / "sprsound_psd_and_trigger.png", dpi=90); plt.close()
    hf = {y: float(np.interp(1500, psd[y][0], psd[y][1])) for y in YEARS}
    md += ["\n## Per-year noise floor above the stethoscope band\n",
           "Median PSD at 1.5 kHz per year (dB; −135 dB is the 16-bit quantization floor): " + ", ".join(f"{y}: {v:.0f}" for y, v in hf.items())]
    out["psd_1500hz_by_year_db"] = hf

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    for name, col in (("hp100", "#4C78A8"), ("hp150", "#54A24B"), ("hp250", "#E45756")):
        axs[0].hist([f_["f0_hz"] for f_ in fits_by[name]], bins=np.linspace(100, 700, 40), alpha=.45, color=col, label=name)
    axs[0].set_title("real FC fitted f0 follows the high-pass"); axs[0].legend()
    axs[1].hist([f_["tau_ms"] for f_ in fits], bins=30, color="#E45756"); axs[1].set_title("real FC crackle tau (ms)")
    axs[2].hist(rate, bins=30, color="#E45756"); axs[2].set_title("real FC crackles per second inside events")
    axs[3].hist([r["f0_hz"] for r in rows], bins=30, color="#54A24B"); axs[3].set_title("real W tone f0 (Hz)")
    plt.tight_layout(); plt.savefig(RES / "sprsound_event_stats.png", dpi=90); plt.close()

    (RES / "sprsound_event_stats.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    (RES / "sprsound_event_stats.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
