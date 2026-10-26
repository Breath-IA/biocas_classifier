"""
analyze_references.py  (gate 2 · e04, step 1)
─────────────────────────────────────────────
First analysis of each reference label (all except normal) from the MEDZCOOL teaching clips:
fine crackles (FC), coarse crackles (CC, contrast only) and expiratory wheeze (W).

For each clip: native bandwidth, breath-cycle segmentation, duplicate-loop detection (the clips repeat cycles),
and source-morphology estimates on the UNIQUE cycles only:
  crackles : damped-sine fit (f0, tau, phase), IDW / 2CD, inter-crackle interval, rate, amplitude, position in cycle
  wheeze   : f0 track, harmonic ratios, f0 jitter and drift, polyphony, duration, position in cycle

Input  : Auscultaion_medzcool_audio/*.mp3
Output : results/reference_params.json, results/reference_analysis.md, results/ref_*.png
Usage  : python experiments/gate2_event_type/e04_synthetic_augmentation/analyze_references.py
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
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402
from synthesis import FS  # noqa: E402
from synthesis import reference as ref  # noqa: E402

warnings.filterwarnings("ignore")
RES = paths.G2_E04 / "results"
CLIPS = {"FC": "Fine Crackles", "CC": "Coarse Crackles", "W": "Wheezing"}
# Crackle detection settings per clip. The high-pass must stay well below the crackle f0:
# a 350 Hz high-pass biases the coarse-crackle f0 upward (see the sensitivity ranges in the report).
CRK_CFG = {"FC": {"k": 4.5, "hp_hz": 250.0}, "CC": {"k": 4.5, "hp_hz": 150.0}}


def find_clip(prefix: str) -> Path:
    return next(p for p in sorted(paths.REFERENCE_AUDIO.glob("*.mp3")) if p.name.startswith(prefix))


def spectrogram(ax, y, segs=None, uniq=None, title=""):
    f, t, Z = signal.stft(y, fs=FS, nperseg=512, noverlap=448)
    ax.pcolormesh(t, f, 20 * np.log10(np.abs(Z) + 1e-7), vmin=-100, vmax=-30, shading="auto")
    ax.set_ylim(0, 3000)
    ax.set_title(title, fontsize=9)
    if segs:
        for i, (a, b) in enumerate(segs):
            ax.axvspan(a / FS, b / FS, ymin=0.96, ymax=1.0, color="red" if uniq and i in uniq else "white", alpha=.9)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    out, md = {}, ["# Reference clip analysis (MEDZCOOL)\n",
                   "Analysis at the SPRSound native rate (8 kHz) on **unique** breath cycles only. Red marks in the "
                   "overview figure are the unique cycles used for the statistics.\n"]
    data = {}
    for key, prefix in CLIPS.items():
        c = ref.load_reference(find_clip(prefix))
        y = c["y"]
        segs = ref.breath_segments(y)
        reps, sizes, clusters = ref.unique_segments(y, segs)
        uniq = [segs[i] for i in reps]
        info = {"native_sr": c["sr"], "duration_s": c["duration_s"], "lr_corr": c["lr_corr"], "bw99_hz": c["bw99_hz"],
                "n_segments": len(segs), "n_unique_segments": len(reps), "cluster_sizes": sizes,
                "segment_duration_s": [float((b - a) / FS) for a, b in uniq]}
        if key in ("FC", "CC"):
            res = ref.analyze_crackles(y, uniq, **CRK_CFG[key])
            info["crackles"] = ref.summarize_crackles(res)
            info["crackle_cfg"] = CRK_CFG[key]
            sens = []                       # sensitivity of the medians to the detection / fit settings
            for hp in (CRK_CFG[key]["hp_hz"] - 50, CRK_CFG[key]["hp_hz"], CRK_CFG[key]["hp_hz"] + 50):
                for kk in (3.5, 4.5, 5.5):
                    for r2 in (0.4, 0.6):
                        r_ = ref.analyze_crackles(y, uniq, k=kk, hp_hz=hp, r2_min=r2)
                        if len(r_["fits"]) >= 5:
                            sens.append({"f0": float(np.median([f["f0_hz"] for f in r_["fits"]])),
                                         "tau": float(np.median([f["tau_ms"] for f in r_["fits"]])),
                                         "twocd": float(np.nanmedian([f["twocd_ms"] for f in r_["fits"]]))})
            info["sensitivity"] = {m: [min(x[m] for x in sens), max(x[m] for x in sens)] for m in ("f0", "tau", "twocd")}
            info["sensitivity"]["n_settings"] = len(sens)
            data[key] = (y, segs, set(reps), uniq, res)
        else:
            res = ref.analyze_wheeze(y, uniq)
            info["wheeze"] = ref.summarize_wheeze(res)
            data[key] = (y, segs, set(reps), uniq, res)
        out[key] = info
        md.append(f"## {key} — {prefix}\n")
        md.append(f"- native {c['sr']} Hz, stereo L/R correlation {c['lr_corr']:.3f}, 99 % of the energy below "
                  f"**{c['bw99_hz']:.0f} Hz** (SPRSound's median recording has no energy above ~1 kHz)")
        md.append(f"- {len(segs)} active breath segments, **{len(reps)} unique** (cluster sizes {sizes}): "
                  f"the clip is a loop of {len(reps)} distinct segment(s), so estimates rest on few independent examples")

    # ── summaries in markdown ─────────────────────────────────────────────
    def fmt(d, unit=""):
        return "n/a" if not d else f"{d['median']:.2f}{unit} (p10–p90 {d['p10']:.2f}–{d['p90']:.2f}, n={d['n']})"
    for key in ("FC", "CC"):
        s = out[key]["crackles"]
        md += [f"\n### Crackle morphology — {key}\n",
               f"- detected {s['n_detected']} transients, {s['n_isolated_fits']} isolated crackles fitted (R² ≥ 0.4)",
               f"- f0: {fmt(s['f0_hz'], ' Hz')}", f"- tau: {fmt(s['tau_ms'], ' ms')}",
               "- sensitivity of the medians across %d detection/fit settings: f0 %.0f–%.0f Hz, tau %.1f–%.1f ms, 2CD %.1f–%.1f ms" % (
                   out[key]["sensitivity"]["n_settings"], *out[key]["sensitivity"]["f0"], *out[key]["sensitivity"]["tau"],
                   *out[key]["sensitivity"]["twocd"]),
               f"- IDW: {fmt(s['idw_ms'], ' ms')} · 2CD: {fmt(s['twocd_ms'], ' ms')}",
               f"- rate inside active segments: {fmt(s['rate_hz'], ' /s')}",
               f"- inter-crackle interval (<150 ms): {fmt(s['ici_ms'], ' ms')}; gamma shape ≈ {s.get('ici_gamma_shape', float('nan')):.2f}",
               f"- amplitude / local RMS (log-normal): {fmt(s['amp_over_local_rms'])}",
               f"- timing in the segment (first/middle/last third): "
               + ", ".join(f"{v:.0%}" for v in s.get("position_thirds", []))]
    w = out["W"]["wheeze"]
    md += ["\n### Wheeze\n"]
    if w:
        md += [f"- f0 (dominant persistent spectral track, 120–1200 Hz): {fmt(w['f0_hz'], ' Hz')}", f"- track duration: {fmt(w['dur_s'], ' s')}",
               f"- f0 jitter (std of residual / f0): {fmt(w['jitter_cv'])}", f"- relative f0 drift over the event: {fmt(w['drift_rel'])}",
               f"- harmonics vs f0: 2f0 {fmt(w['harm2_db'], ' dB')}, 3f0 {fmt(w['harm3_db'], ' dB')}, 4f0 {fmt(w['harm4_db'], ' dB')}",
               f"- polyphonic (more than one simultaneous strong track) in {w['polyphony_gt1_fraction']:.0%} of unique segments",
               f"- track start/end inside the breath segment (fraction): {fmt(w['start_in_seg'])} / {fmt(w['end_in_seg'])}"]

    # ── generator parameters (consumed by build_synthetic.py) ─────────────
    fc = out["FC"]["crackles"]
    gen = {"FC": {"f0_median_hz": fc["f0_hz"]["median"], "f0_sigma": max(fc["f0_hz"]["log_sd"], 0.1),
                  "tau_median_ms": fc["tau_ms"]["median"], "tau_sigma": max(fc["tau_ms"]["log_sd"], 0.15),
                  "ici_shape": fc.get("ici_gamma_shape", 1.5),
                  "amp_sigma": float(np.clip(fc["amp_over_local_rms"]["log_sd"], 0.3, 0.9)),
                  "source": "MEDZCOOL fine crackles, unique cycles only (before the SPRSound channel)"},
           "CC": {"f0_median_hz": out["CC"]["crackles"]["f0_hz"]["median"], "tau_median_ms": out["CC"]["crackles"]["tau_ms"]["median"],
                  "note": "contrast only; coarse crackles are not generated in this stage"}}
    if w:
        gen["W"] = {"f0_hz": w["f0_hz"], "jitter_cv": w["jitter_cv"]["median"], "drift_rel": w["drift_rel"]["median"],
                    "harmonic_db": {"h2": w["harm2_db"]["median"] if w["harm2_db"] else -6.0,
                                    "h3": w["harm3_db"]["median"] if w["harm3_db"] else -12.0,
                                    "h4": w["harm4_db"]["median"] if w["harm4_db"] else -20.0},
                    "polyphony_fraction": w["polyphony_gt1_fraction"],
                    "source": "MEDZCOOL expiratory wheeze, unique cycles only"}
    out["generator_parameters"] = gen
    (RES / "reference_params.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    (RES / "reference_analysis.md").write_text("\n".join(md), encoding="utf-8")

    # ── figures ───────────────────────────────────────────────────────────
    fig, axs = plt.subplots(3, 1, figsize=(15, 9))
    for a, (k, (y, segs, reps, uniq, res)) in zip(axs, data.items()):
        spectrogram(a, y, segs, reps, f"{k}: {out[k]['n_segments']} active segments, {out[k]['n_unique_segments']} unique (red)")
    plt.tight_layout()
    plt.savefig(RES / "ref_overview.png", dpi=90)
    plt.close()

    fig, axs = plt.subplots(2, 4, figsize=(17, 7))
    for r, k in enumerate(("FC", "CC")):
        y, segs, reps, uniq, res = data[k]
        sos = signal.butter(4, 150, "highpass", fs=FS, output="sos")
        hp = signal.sosfiltfilt(sos, y)
        best = sorted(res["fits"], key=lambda f: -f["r2"])[:4]
        for c, f in enumerate(best):
            p = f["peak"]
            a0, b0 = p - int(.002 * FS), p + int(.016 * FS)
            w = hp[a0:b0] / (np.max(np.abs(hp[a0:b0])) + 1e-12)
            t = (np.arange(len(w)) - int(.002 * FS)) / FS * 1000
            axs[r, c].plot(t, w, "k", lw=1)
            on = int(np.argmax(np.abs(w) > .15))
            tt = np.arange(len(w) - on) / FS
            m = np.exp(-tt / (f["tau_ms"] / 1000)) * np.sin(2 * np.pi * f["f0_hz"] * tt + f["phi"])
            m = m * (np.max(np.abs(w[on:])) / (np.max(np.abs(m)) + 1e-12))
            axs[r, c].plot(t[on:], m, "r--", lw=1)
            axs[r, c].set_title(f"{k}: f0={f['f0_hz']:.0f} Hz τ={f['tau_ms']:.1f} ms R²={f['r2']:.2f}", fontsize=8)
            axs[r, c].set_xlabel("ms")
    plt.tight_layout()
    plt.savefig(RES / "ref_crackle_fits.png", dpi=90)
    plt.close()

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    for k, col in (("FC", "#E45756"), ("CC", "#4C78A8")):
        res = data[k][4]
        for a, (name, v, bins) in zip(axs[:3], (("f0 (Hz)", [f["f0_hz"] for f in res["fits"]], np.linspace(150, 3500, 30)),
                                                ("tau (ms)", [f["tau_ms"] for f in res["fits"]], np.linspace(0.3, 15, 30)),
                                                ("inter-crackle interval (ms)", res["ici_ms"], np.linspace(0, 150, 30)))):
            a.hist(v, bins=bins, alpha=.55, color=col, label=f"{k} (n={len(v)})", density=True)
            a.set_xlabel(name)
            a.legend()
        sp = res.get("trigger_spectrum")
        if sp:
            axs[3].semilogy(sp["freq_hz"], np.array(sp["power"]) / np.max(sp["power"]), color=col, label=k)
    axs[3].set_xlabel("Hz")
    axs[3].set_title("crackle-triggered mean spectrum (±8 ms)")
    axs[3].legend()
    plt.tight_layout()
    plt.savefig(RES / "ref_crackle_distributions.png", dpi=90)
    plt.close()

    y, segs, reps, uniq, res = data["W"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.5))
    if uniq:
        tracks, (f, t, S) = ref.track_wheeze(y, uniq[0])
        axs[0].pcolormesh(t, f, S, vmin=S.max() - 70, vmax=S.max(), shading="auto")
        for tr in tracks:
            axs[0].plot(tr["t"], tr["f"], "r.", ms=2)
        axs[0].set_ylim(0, 2000)
        axs[0].set_title("wheeze: first unique segment with tracked spectral peaks")
    rows = res["rows"]
    if rows:
        axs[1].bar(range(len(rows)), [r["f0_hz"] for r in rows])
        axs[1].set_xlabel("unique segment")
        axs[1].set_ylabel("f0 (Hz)")
        axs[1].set_title("f0 per unique segment")
    plt.tight_layout()
    plt.savefig(RES / "ref_wheeze.png", dpi=90)
    plt.close()
    print("\n".join(md))
    print("\nOK ->", RES)


if __name__ == "__main__":
    main()
