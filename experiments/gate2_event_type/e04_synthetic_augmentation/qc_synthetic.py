"""
qc_synthetic.py  (gate 2 · e04, step 5)
───────────────────────────────────────
Quality control of the synthetic set. No model is trained: realism is measured with univariate rank statistics.

1. Integrity: format (8 kHz mono PCM-16), JSON schema, no clipping, events inside the recording, no overlaps, balanced
   classes, and that the REAL host recording is bit-identical outside the inserted event window.
2. Realism: for N, FC and W, every feature of the real events (spectral, temporal, tonal and the physics features of
   plan.md) is compared with the synthetic events using AUC(real vs synthetic) (0.5 = indistinguishable), the median
   ratio and the Kolmogorov-Smirnov statistic.
3. Figures: feature gaps, distributions, median spectra, a spectrogram gallery and crackle waveform zooms.

Input : data/synthetic (build_synthetic.py), real feature tables
Output: results/qc_*.png, results/qc_feature_gaps.csv, results/qc_integrity.csv, results/report_auto.md
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
from joblib import Parallel, delayed
from scipy import signal
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402
from analysis_utils import META, auc  # noqa: E402
from synthesis import FS, builder as bd  # noqa: E402

warnings.filterwarnings("ignore")
RES = paths.G2_E04 / "results"
CCOL = {"real": "#4C78A8", "synthetic": "#E45756"}
KEY = {"N": ["spec_centroid", "spec_slope_db_khz", "band_100_200", "band_300_400", "env_kurtosis", "crest_factor", "snr_vs_floor_db", "spec_flux"],
       "FC": ["env_kurtosis", "crest_factor", "n_transients", "spec_centroid", "band_300_400", "snr_vs_floor_db", "ph_tau_p25", "ph_mf_cr_max"],
       "W": ["env_kurtosis", "crest_factor", "n_spec_peaks", "max_peak_prom_db", "band_200_300", "snr_vs_floor_db", "tonal_run_s", "ph_mf_wh_max"]}


def targets_of(row, js):
    ev = js["event_annotation"]
    if row["kind"] == "N":
        return [(k, "N", int(e["start"]), int(e["end"])) for k, e in enumerate(ev) if e["type"] == "Normal"][:4]
    return [(0, row["kind"], int(row["target_start_ms"]), int(row["target_end_ms"]))]


def features_one(row, base):
    x, sr = sf.read(str(base / "wav" / f"{row['name']}.wav"), dtype="int16")
    js = json.loads((base / "json" / f"{row['name']}.json").read_text(encoding="utf-8"))
    f = bd.extract_features(x, targets_of(row, js), stem=row["name"])
    if f.empty:
        return None
    f["kind"], f["year"], f["name"] = row["kind"], row["year"], row["name"]
    return f


def integrity(m, base, hosts_dir=paths.RAW, n_host_check=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    bad = {"format": 0, "schema": 0, "clip": 0, "outside": 0, "overlap": 0, "overlap_host": 0}
    for r in m.itertuples():
        i = sf.info(str(base / "wav" / f"{r.name}.wav"))
        if not (i.samplerate == FS and i.channels == 1 and i.subtype == "PCM_16"):
            bad["format"] += 1
        js = json.loads((base / "json" / f"{r.name}.json").read_text(encoding="utf-8"))
        ev = js["event_annotation"]
        if not (set(js) >= {"record_annotation", "event_annotation"} and all(set(e) == {"start", "end", "type"} for e in ev)):
            bad["schema"] += 1
        if any(int(e["start"]) < 0 or int(e["end"]) > i.duration * 1000 + 1 or int(e["end"]) <= int(e["start"]) for e in ev):
            bad["outside"] += 1
        se = sorted((int(e["start"]), int(e["end"])) for e in ev)
        if any(b[0] < a[1] for a, b in zip(se, se[1:])):
            bad["overlap"] += 1
            hj = json.loads((hosts_dir / r.host_split / "json" / f"{r.host_stem}.json").read_text(encoding="utf-8"))
            hs = sorted((int(e["start"]), int(e["end"])) for e in hj["event_annotation"])
            if any(b[0] < a[1] for a, b in zip(hs, hs[1:])):
                bad["overlap_host"] += 1                       # the overlap already exists in the real host annotation
    sub = m[m["kind"] != "N"].sample(min(n_host_check, (m["kind"] != "N").sum()), random_state=seed)
    ident, frac_clip = [], 0
    for r in sub.itertuples():
        syn, _ = sf.read(str(base / "wav" / f"{r.name}.wav"), dtype="int16")
        host, _ = sf.read(str(hosts_dir / r.host_split / "wav" / f"{r.host_stem}.wav"), dtype="int16")
        a, b = int(r.target_start_ms / 1000 * FS), int(r.target_end_ms / 1000 * FS)
        mask = np.ones(len(syn), bool)
        mask[max(a - int(0.1 * FS), 0):min(b + int(0.1 * FS), len(syn))] = False
        ident.append(float(np.mean(syn[mask] == host[mask])))
        frac_clip += int(np.max(np.abs(syn.astype(int))) >= 32767 and np.max(np.abs(host.astype(int))) < 32767)   # clipping introduced by us
    out = {"recordings": len(m), "format_errors": bad["format"], "schema_errors": bad["schema"], "events_outside": bad["outside"],
           "recordings_with_overlapping_events": bad["overlap"], "of_which_already_in_the_real_host": bad["overlap_host"],
           "clipping_introduced_by_synthesis(sampled)": frac_clip,
           "host_samples_identical_outside_event(min over sampled)": float(np.min(ident)) if ident else np.nan,
           "host_samples_identical_outside_event(mean)": float(np.mean(ident)) if ident else np.nan}
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    base = paths.SYNTH_DIR
    m = pd.read_csv(base / "manifest.csv", dtype={"year": str})
    md = ["# Synthetic set QC — automatic report\n"]
    md += ["## 1. Composition\n", m.groupby(["kind", "year"]).size().unstack(fill_value=0).to_markdown(), "\n",
           f"Host recordings used: {m['host_stem'].nunique()} (max reuse {m['host_stem'].value_counts().max()}). "
           f"All files are train-only (`train_only` = {bool(m['train_only'].all())}); synthetic recordings whose host is a 2025 test "
           f"recording: {int((m['host_split'] == 'test').sum())} (drop them for the official split).\n"]

    integ = integrity(m, base)
    pd.Series(integ).to_csv(RES / "qc_integrity.csv", header=["value"])
    md += ["## 2. Integrity checks\n", pd.Series(integ).to_frame("value").to_markdown(), "\n"]

    # ── features of synthetic events ──────────────────────────────────────
    res = Parallel(n_jobs=4, verbose=0)(delayed(features_one)(r, base) for _, r in m.iterrows())
    syn = pd.concat([r for r in res if r is not None], ignore_index=True)
    syn.to_csv(paths.G2_E04 / "cache" / "synthetic_features.csv", index=False)
    full = pd.read_csv(paths.SEGMENTS_CSV).merge(pd.read_csv(paths.PHYSICS_CSV), on=["stem", "event_idx"], how="left")
    full["source"] = full["source"].astype(str)
    feats = [c for c in syn.columns if c in full.columns and c not in META and not c.startswith("sp_") and pd.api.types.is_numeric_dtype(syn[c])]

    gaps = []
    for cls in ("N", "FC", "W"):
        r_, s_ = full[full["class"] == cls], syn[syn["kind"] == cls]
        for f in feats:
            x, y = r_[f].dropna().values, s_[f].dropna().values
            if len(x) < 30 or len(y) < 30:
                continue
            xs = np.random.default_rng(0).permutation(x)[:4000]          # random subsample (not the first rows: those are all 2022)
            a = auc(np.r_[xs, y], np.r_[np.zeros(len(xs)), np.ones(len(y))].astype(bool))
            gaps.append({"class": cls, "feature": f, "auc_real_vs_synth": a, "gap": abs(a - 0.5) * 2,
                         "median_real": float(np.median(x)), "median_synth": float(np.median(y)),
                         "ks": float(ks_2samp(x, y).statistic)})
    G = pd.DataFrame(gaps)
    G.to_csv(RES / "qc_feature_gaps.csv", index=False)
    summ = G.groupby("class").agg(n_features=("gap", "size"), median_gap=("gap", "median"), share_gap_below_0_2=("gap", lambda v: float(np.mean(v < 0.2))),
                                  share_gap_above_0_5=("gap", lambda v: float(np.mean(v > 0.5))))
    md += ["## 3. Realism: |2·AUC−1| between real and synthetic events (0 = indistinguishable, 1 = fully separable)\n",
           summ.round(3).to_markdown(), "\n"]
    for cls in ("N", "FC", "W"):
        g = G[G["class"] == cls].sort_values("gap", ascending=False).head(10)[["feature", "gap", "median_real", "median_synth", "ks"]]
        md += [f"### Largest gaps — {cls}\n", g.round(3).to_markdown(index=False), "\n"]
        k = G[(G["class"] == cls) & (G["feature"].isin(KEY[cls]))][["feature", "gap", "median_real", "median_synth"]]
        md += [f"Key features — {cls}:\n", k.round(3).to_markdown(index=False), "\n"]

    # ── per-year check of the key gap ─────────────────────────────────────
    py = []
    for cls in ("FC", "W"):
        for y in bd.YEARS:
            r_ = full[(full["class"] == cls) & (full["source"] == y)]
            s_ = syn[(syn["kind"] == cls) & (syn["year"] == y)]
            if len(r_) > 30 and len(s_) > 10:
                gg = [abs(auc(np.r_[r_[f].dropna().values, s_[f].dropna().values],
                              np.r_[np.zeros(r_[f].notna().sum()), np.ones(s_[f].notna().sum())].astype(bool)) - .5) * 2 for f in KEY[cls]]
                py.append({"class": cls, "year": y, "n_real": len(r_), "n_synth": len(s_), "median_gap_key_features": float(np.nanmedian(gg))})
    md += ["### Gap of the key features by year\n", pd.DataFrame(py).round(3).to_markdown(index=False), "\n"]

    # ── manifest-level checks ─────────────────────────────────────────────
    fcw = m[m["kind"] != "N"]
    md += ["## 4. Placement and level\n",
           f"- achieved vs target level (snr_vs_floor_db), median absolute error: {np.nanmedian(np.abs(fcw['info_achieved_snr_db'] - fcw['info_target_snr_db'])):.2f} dB",
           f"- events placed inside a detected breath burst: {np.mean(fcw['info_burst'] >= 0):.1%}; relative position in the burst (median): FC "
           f"{np.nanmedian(fcw[fcw.kind == 'FC']['info_burst_pos']):.2f}, W {np.nanmedian(fcw[fcw.kind == 'W']['info_burst_pos']):.2f}",
           f"- event duration (s), median synthetic vs real: " + ", ".join(
               f"{c} {np.median(fcw[fcw.kind == c]['info_duration_s']):.2f} vs {np.median(full[full['class'] == c]['duration_s']):.2f}" for c in ("FC", "W"))]

    # ── figures ───────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(19, 8))
    for a, cls in zip(axs, ("N", "FC", "W")):
        g = G[G["class"] == cls].sort_values("gap")
        a.barh(g["feature"], g["gap"], color=["#E45756" if v > .5 else "#4C78A8" for v in g["gap"]])
        a.axvline(0.2, color="k", ls="--", lw=.8)
        a.set_title(f"{cls}: |2·AUC−1| real vs synthetic"); a.tick_params(axis="y", labelsize=6)
    plt.tight_layout(); plt.savefig(RES / "qc_feature_gaps.png", dpi=85); plt.close()

    fig, axs = plt.subplots(3, 8, figsize=(26, 9))
    for r, cls in enumerate(("N", "FC", "W")):
        for c, f in enumerate(KEY[cls]):
            a = axs[r, c]
            vals = [full[full["class"] == cls][f].dropna().values, syn[syn["kind"] == cls][f].dropna().values]
            bp = a.boxplot(vals, showfliers=False, patch_artist=True, widths=.6)
            for b, col in zip(bp["boxes"], (CCOL["real"], CCOL["synthetic"])):
                b.set_facecolor(col); b.set_alpha(.8)
            a.set_xticklabels(["real", "synth"]); a.set_title(f"{cls}: {f}", fontsize=8)
    plt.tight_layout(); plt.savefig(RES / "qc_distributions.png", dpi=75); plt.close()

    sp = [f"sp_{k:02d}" for k in range(32)]
    fx = np.arange(32) * 62.5 + 31.25
    sfull = syn.copy()
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
    for a, cls in zip(axs, ("N", "FC", "W")):
        a.plot(fx, full[full["class"] == cls][sp].median(), color=CCOL["real"], label="real")
        a.plot(fx, sfull[sfull["kind"] == cls][sp].median(), color=CCOL["synthetic"], label="synthetic")
        a.set_title(f"{cls}: median event spectrum"); a.set_xlabel("Hz"); a.set_ylabel("dB re total"); a.legend(); a.grid(alpha=.3); a.set_xlim(0, 1500)
    plt.tight_layout(); plt.savefig(RES / "qc_spectra.png", dpi=90); plt.close()

    # spectrogram gallery and crackle zooms
    rng = np.random.default_rng(1)
    real_ev = full[full["class"].isin(["FC", "W"])].sample(400, random_state=2)
    fig, axs = plt.subplots(4, 4, figsize=(18, 11))
    for r, (cls, src) in enumerate((("FC", "real"), ("FC", "synthetic"), ("W", "real"), ("W", "synthetic"))):
        for c in range(4):
            a = axs[r, c]
            if src == "real":
                q = real_ev[real_ev["class"] == cls].iloc[c]
                x, _ = sf.read(str(paths.RAW / q["split"] / "wav" / f"{q['stem']}.wav"), dtype="float64")
                s0, e0 = q["start_ms"] / 1000, q["end_ms"] / 1000
            else:
                q = m[m["kind"] == cls].iloc[c * 7]
                x, _ = sf.read(str(base / "wav" / f"{q['name']}.wav"), dtype="float64")
                s0, e0 = q["target_start_ms"] / 1000, q["target_end_ms"] / 1000
            lo, hi = max(s0 - 0.5, 0), min(e0 + 0.5, len(x) / FS)
            f_, t_, Z = signal.stft(x[int(lo * FS):int(hi * FS)], fs=FS, nperseg=512, noverlap=480)
            a.pcolormesh(t_ + lo, f_, 20 * np.log10(np.abs(Z) + 1e-7), vmin=-110, vmax=-45, shading="auto")
            a.axvline(s0, color="w", lw=1); a.axvline(e0, color="w", lw=1); a.set_ylim(0, 1200)
            a.set_title(f"{src} {cls}", fontsize=8)
    plt.tight_layout(); plt.savefig(RES / "qc_gallery.png", dpi=75); plt.close()

    fig, axs = plt.subplots(2, 4, figsize=(18, 6))
    for r, src in enumerate(("real", "synthetic")):
        for c in range(4):
            if src == "real":
                q = real_ev[real_ev["class"] == "FC"].iloc[10 + c]
                x, _ = sf.read(str(paths.RAW / q["split"] / "wav" / f"{q['stem']}.wav"), dtype="float64")
                a0, b0 = int(q["start_ms"] / 1000 * FS), int(q["end_ms"] / 1000 * FS)
            else:
                q = m[m["kind"] == "FC"].iloc[10 + c * 5]
                x, _ = sf.read(str(base / "wav" / f"{q['name']}.wav"), dtype="float64")
                a0, b0 = int(q["target_start_ms"] / 1000 * FS), int(q["target_end_ms"] / 1000 * FS)
            sos = signal.butter(2, 120, "highpass", fs=FS, output="sos")
            h = signal.sosfiltfilt(sos, x[a0:b0])
            p = int(np.argmax(np.abs(h)))
            w = h[max(p - 200, 0):p + 200]
            axs[r, c].plot((np.arange(len(w)) - min(p, 200)) / FS * 1000, w, "k", lw=.8)
            axs[r, c].set_title(f"{src} FC: 50 ms around the strongest transient", fontsize=8); axs[r, c].set_xlabel("ms")
    plt.tight_layout(); plt.savefig(RES / "qc_crackle_zoom.png", dpi=80); plt.close()

    (RES / "report_auto.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
