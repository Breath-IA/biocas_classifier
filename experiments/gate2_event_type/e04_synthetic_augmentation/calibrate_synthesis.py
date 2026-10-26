"""
calibrate_synthesis.py  (gate 2 · e04, step 3)
──────────────────────────────────────────────
Calibrate the synthesis so that synthetic events look like real SPRSound events AFTER the dataset's own acquisition
chain. Crackle f0 is not identifiable at SPRSound's bandwidth (see sprsound_event_stats.md), so the targets are
filter-independent features measured with the SAME extractors used for real events: spectral shape, impulsiveness
(kurtosis, crest factor), transient counts and the physics features of plan.md (decay tau, matched-filter statistics).

Distance D = mean over features of |median(synthetic) - median(real)| / IQR(real).

Stages (no model is trained; this is a small parameter search on generator hyper-parameters):
  1. chain low-pass cutoff and order, from fine-crackle events          → ChannelModel.lp_hz / lp_order
  2. chain high-pass corner                                             → ChannelModel.hp_hz
  3. crackle rate scale and level offset                                → rate_scale, snr_offset_db
  4. wheeze parameters (polyphony, van der Pol share, f0 jitter)        → validated on W, searched on W only
  5. independent check: fitted tau / 2CD of synthetic vs real crackles, which were NOT used as targets

Input : results/reference_params.json, results/channel_estimate.json (first estimate), real feature tables
Output: results/calibration.json, results/calibration_search.csv, results/calibration.png, results/calibration.md
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402
from synthesis import FS, builder as bd  # noqa: E402
from synthesis import reference as ref  # noqa: E402

warnings.filterwarnings("ignore")
RES = paths.G2_E04 / "results"
YEARS = bd.YEARS
N_CAL = 90
FEATS = ["spec_centroid", "spec_rolloff85", "spec_rolloff95", "spec_slope_db_khz", "band_50_100", "band_100_200", "band_200_300",
         "band_300_400", "band_400_600", "band_600_800", "spec_crest", "env_kurtosis", "crest_factor", "kurtosis", "env_cv",
         "env_peak_to_median", "n_transients", "transient_rate", "spec_flux", "n_spec_peaks", "mf_peak_ratio", "mf_rate",
         "snr_vs_floor_db", "rel_tilt_db", "ph_tau_p25", "ph_tau_med", "ph_decay_frac", "ph_env_slope_neg_frac",
         "ph_mf_cr_max", "ph_mf_cr_tau", "ph_mf_cr_top5", "rel_low_db", "rel_mid_db", "rel_high_db", "level_vs_rec_db"]
W_FEATS = ["spec_centroid", "spec_rolloff85", "spec_slope_db_khz", "band_100_200", "band_200_300", "band_300_400", "band_400_600",
           "spec_crest", "env_kurtosis", "crest_factor", "env_cv", "n_spec_peaks", "max_peak_prom_db", "tonal_run_s",
           "harm_ratio_db", "snr_vs_floor_db", "ph_mf_wh_max", "ph_mf_wh_frac50", "ph_mf_wh_f0_track_std", "ph_if_std_best100",
           "rel_low_db", "rel_mid_db", "rel_high_db", "rel_tilt_db", "level_vs_rec_db"]


def one(job, cal):
    r = bd.run_job(job, cal, keep_audio=True)
    if r is None:
        return None
    f = bd.extract_features(r["_pcm"], [(0, job["kind"], r["target_start_ms"], r["target_end_ms"])], stem=r["name"])
    if f.empty:
        return None
    out = f.iloc[0].to_dict()
    out["_pcm"] = r["_pcm"] if job.get("keep_pcm") else None
    out["_target"] = (r["target_start_ms"], r["target_end_ms"])
    return out


def synth_table(jobs, cal, workers=4):
    res = Parallel(n_jobs=workers)(delayed(one)(j, cal) for j in jobs)
    return pd.DataFrame([r for r in res if r is not None])


def distance(s, real, feats):
    d, med = {}, {}
    for f in feats:
        if f not in s or f not in real:
            continue
        iqr = np.nanpercentile(real[f], 75) - np.nanpercentile(real[f], 25)
        iqr = max(iqr, 1e-6 + 0.05 * abs(np.nanmedian(real[f])))
        d[f] = abs(np.nanmedian(s[f]) - np.nanmedian(real[f])) / iqr
        med[f] = (float(np.nanmedian(s[f])), float(np.nanmedian(real[f])))
    return float(np.nanmean(list(d.values()))), d, med


def with_chain(cal, **kw):
    c = json.loads(json.dumps(cal))
    c["chain"].update(kw)
    return c


def main():
    RES.mkdir(parents=True, exist_ok=True)
    reference = json.loads((RES / "reference_params.json").read_text())
    chain0 = json.loads((RES / "channel_estimate.json").read_text())
    cal = bd.default_calibration(reference, chain0)
    resume = "--resume" in sys.argv and (RES / "calibration.json").exists()
    if resume:                                     # keep the chain found by stages 1-2 and redo the rest
        cal = json.loads((RES / "calibration.json").read_text())
        cal.pop("chain_W", None)
        cal["wheeze"] = bd.default_calibration()["wheeze"]          # new burst-train wheeze model
        cal["vdp_fraction"] = 0.15
        cal["snr_offset_db"], cal["rate_scale"] = 0.0, 1.0
        lp_grid, ord_grid, hp_grid = (cal["chain"]["lp_hz"],), (cal["chain"]["lp_order"],), (cal["chain"]["hp_hz"],)
        old_log = pd.read_csv(RES / "calibration_search.csv")
        log = old_log[old_log["stage"].isin(["1_lowpass", "2_highpass"])].to_dict("records")
    else:
        lp_grid, ord_grid, hp_grid = (120, 150, 175, 200, 225, 250, 300, 400), (2, 4, 6), (20.0, 35.0, 50.0, 70.0, 100.0, 140.0)
        log = []
    hosts, profiles, seg = bd.load_tables()
    full = pd.read_csv(paths.SEGMENTS_CSV).merge(pd.read_csv(paths.PHYSICS_CSV), on=["stem", "event_idx"], how="left")
    full["source"] = full["source"].astype(str)
    real = {c: full[full["class"] == c] for c in ("FC", "W")}
    md = ["# Calibration of the synthesis\n", f"{N_CAL} synthetic recordings per candidate; same hosts, profiles and seeds for every "
          "candidate (paired comparison). Distance D = mean |median(synthetic) − median(real)| / IQR(real).\n"]

    jobs_fc = bd.plan_jobs(hosts, profiles, N_CAL, YEARS, seed=11, kinds=("FC",))
    jobs_w = bd.plan_jobs(hosts, profiles, N_CAL, YEARS, seed=12, kinds=("W",))

    # ── 1. chain low-pass ─────────────────────────────────────────────────
    best = None
    for lp, order in itertools.product(lp_grid, ord_grid):
        c = with_chain(cal, lp_hz=float(lp), lp_order=order, hp_hz=cal["chain"]["hp_hz"] if resume else 50.0)
        s = synth_table(jobs_fc, c)
        D, _, _ = distance(s, real["FC"], FEATS)
        log.append({"stage": "1_lowpass", "lp_hz": lp, "lp_order": order, "hp_hz": 50.0, "D_FC": D})
        print(f"  lp {float(lp):5.0f} order {order}: D_FC = {D:.3f}", flush=True)
        if best is None or D < best[0]:
            best = (D, lp, order)
    cal["chain"].update(lp_hz=float(best[1]), lp_order=int(best[2]), hp_hz=cal["chain"]["hp_hz"] if resume else 50.0)

    # ── 2. high-pass corner ───────────────────────────────────────────────
    bhp = None
    for hp in hp_grid:
        c = with_chain(cal, hp_hz=hp)
        D, _, _ = distance(synth_table(jobs_fc, c), real["FC"], FEATS)
        log.append({"stage": "2_highpass", "lp_hz": cal["chain"]["lp_hz"], "lp_order": cal["chain"]["lp_order"], "hp_hz": hp, "D_FC": D})
        print(f"  hp {hp:5.0f}: D_FC = {D:.3f}", flush=True)
        if bhp is None or D < bhp[0]:
            bhp = (D, hp)
    cal["chain"]["hp_hz"] = float(bhp[1])

    # ── 3. rate scale and level offset ────────────────────────────────────
    for it in range(3):
        s = synth_table(jobs_fc, cal)
        r_med, s_med = np.nanmedian(real["FC"]["mf_rate"]), np.nanmedian(s["mf_rate"])
        cal["rate_scale"] = float(np.clip(cal["rate_scale"] * r_med / max(s_med, 1e-6), 0.2, 5.0))
        gap = np.nanmedian(real["FC"]["snr_vs_floor_db"]) - np.nanmedian(s["snr_vs_floor_db"])
        # damped and clipped: the level can never fall below the host's own level, so a larger correction has no effect
        cal["snr_offset_db"] = float(np.clip(cal["snr_offset_db"] + 0.5 * gap, -3.0, 3.0))
        D, _, _ = distance(synth_table(jobs_fc, cal), real["FC"], FEATS)
        log.append({"stage": "3_rate_level", "iter": it, "rate_scale": cal["rate_scale"], "snr_offset_db": cal["snr_offset_db"], "D_FC": D})
        print(f"  iter {it}: rate_scale {cal['rate_scale']:.2f}, snr_offset {cal['snr_offset_db']:.2f} dB, D_FC {D:.3f}", flush=True)
    s_fc = synth_table([dict(j, keep_pcm=True) for j in jobs_fc[:40]], cal)
    D_fc, d_fc, med_fc = distance(synth_table(jobs_fc, cal), real["FC"], FEATS)

    # ── 4a. wheeze effective chain (crackles are transients filtered by lung tissue; wheezes are sustained airway tones) ──
    bwc = None
    for lp, order in itertools.product((250, 300, 400, 500, 650, 800), (2, 4, 6)):
        c = json.loads(json.dumps(cal))
        c["chain_W"] = {**cal["chain"], "lp_hz": float(lp), "lp_order": order}
        D, _, _ = distance(synth_table(jobs_w, c), real["W"], W_FEATS)
        log.append({"stage": "4a_wheeze_chain", "lp_hz": lp, "lp_order": order, "D_W": D})
        print(f"  wheeze chain lp {lp:4d} order {order}: D_W = {D:.3f}", flush=True)
        if bwc is None or D < bwc[0]:
            bwc = (D, lp, order)
    cal["chain_W"] = {**cal["chain"], "lp_hz": float(bwc[1]), "lp_order": int(bwc[2])}

    # ── 4b. wheeze parameters (searched on W only) ────────────────────────
    bw = None
    for pp, bm, gl in itertools.product((0.2, 0.4), (0.15, 0.25, 0.4), (0.10, 0.25)):
        c = json.loads(json.dumps(cal))
        c["wheeze"].update(poly_prob=pp, burst_median_s=bm, glide_sd=gl)
        D, _, _ = distance(synth_table(jobs_w, c), real["W"], W_FEATS)
        log.append({"stage": "4b_wheeze", "poly_prob": pp, "burst_median_s": bm, "glide_sd": gl, "D_W": D})
        print(f"  wheeze poly {pp} burst {bm} glide {gl}: D_W = {D:.3f}", flush=True)
        if bw is None or D < bw[0]:
            bw = (D, pp, bm, gl)
    cal["wheeze"].update(poly_prob=bw[1], burst_median_s=bw[2], glide_sd=bw[3])
    bh = None
    for harm in ([1.0, 0.2, 0.04], [1.0, 0.1, 0.02], [1.0, 0.05, 0.0], [1.0, 0.0, 0.0]):
        c = json.loads(json.dumps(cal))
        c["wheeze"]["harmonics"] = harm
        D, _, _ = distance(synth_table(jobs_w, c), real["W"], W_FEATS)
        log.append({"stage": "4c_wheeze_harmonics", "harmonics": str(harm), "D_W": D})
        print(f"  wheeze harmonics {harm}: D_W = {D:.3f}", flush=True)
        if bh is None or D < bh[0]:
            bh = (D, harm)
    cal["wheeze"]["harmonics"] = bh[1]
    D_w, d_w, med_w = distance(synth_table(jobs_w, cal), real["W"], W_FEATS)
    print(f"  wheeze: poly {bw[1]}, burst median {bw[2]} s, glide sd {bw[3]} -> D_W {D_w:.3f}", flush=True)

    # ── 5. independent check on fitted crackle tau / 2CD (not used as targets) ──
    s40 = synth_table([dict(j, keep_pcm=True) for j in jobs_fc[:60]], cal)
    fits = []
    for _, r in s40.iterrows():
        pcm = r["_pcm"]
        if pcm is None:
            continue
        a, b = r["_target"]
        x = pcm.astype(np.float64) / 32768
        hp, pk = ref.detect_crackles(x, (int(a / 1000 * FS), int(b / 1000 * FS)), FS, 4.5, hp_hz=150.0)
        n = 0
        for i, p in enumerate(pk):
            if (i == 0 or (p - pk[i - 1]) / FS > .015) and (i == len(pk) - 1 or (pk[i + 1] - p) / FS > .015):
                f = ref.fit_crackle(hp, p, FS)
                if f and n < 8:
                    fits.append(f); n += 1
    real_stats = json.loads((RES / "sprsound_event_stats.json").read_text())["FC"]
    check = {"synthetic_tau_ms": float(np.median([f["tau_ms"] for f in fits])), "synthetic_twocd_ms": float(np.nanmedian([f["twocd_ms"] for f in fits])),
             "real_tau_ms": real_stats["tau_ms"]["median"], "real_twocd_ms": real_stats["twocd_ms"]["median"], "n_fits": len(fits)}

    cal["calibration_summary"] = {"D_FC": D_fc, "D_W": D_w, "independent_check": check, "n_per_candidate": N_CAL}
    (RES / "calibration.json").write_text(json.dumps(cal, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(log).to_csv(RES / "calibration_search.csv", index=False)

    # ── report + figure ───────────────────────────────────────────────────
    s1 = pd.DataFrame([r for r in log if r["stage"] == "1_lowpass"]).drop_duplicates(["lp_hz", "lp_order"], keep="first")
    md += ["## Result\n", f"- chain: high-pass {cal['chain']['hp_hz']:.0f} Hz (order {cal['chain']['hp_order']}), "
           f"**low-pass {cal['chain']['lp_hz']:.0f} Hz (order {cal['chain']['lp_order']})**",
           f"- wheeze effective chain: low-pass {cal['chain_W']['lp_hz']:.0f} Hz (order {cal['chain_W']['lp_order']})",
           f"- crackle rate scale {cal['rate_scale']:.2f}, level offset {cal['snr_offset_db']:+.2f} dB (clipped to ±3 dB)",
           f"- wheeze harmonic amplitudes (1f0, 2f0, 3f0): {cal['wheeze']['harmonics']}",
           f"- wheeze (gliding-burst model): parallel-tone probability {cal['wheeze']['poly_prob']}, burst median {cal['wheeze']['burst_median_s']} s, glide sd {cal['wheeze']['glide_sd']}",
           f"- final distance: **D_FC = {D_fc:.2f}**, D_W = {D_w:.2f} (0 = identical medians; values below ≈ 0.3 mean the medians fall within a third of the real IQR)",
           "\n## Independent check (not used as a target)\n",
           f"Fitted crackle decay on synthetic fine-crackle events: tau {check['synthetic_tau_ms']:.1f} ms, 2CD {check['synthetic_twocd_ms']:.1f} ms "
           f"({check['n_fits']} fits) vs. real SPRSound fine crackles: tau {check['real_tau_ms']:.1f} ms, 2CD {check['real_twocd_ms']:.1f} ms. "
           "The source tau (≈ 4 ms) and 2CD (≈ 3 ms) of the reference clip were stretched by the calibrated low-pass; if the chain is right they land near the real values.",
           "\n## Search tables\n", "### Stage 1: low-pass (hp = 50 Hz)\n", s1.pivot(index="lp_hz", columns="lp_order", values="D_FC").round(3).to_markdown(),
           "\n### Stage 2: high-pass\n", pd.DataFrame([r for r in log if r["stage"] == "2_highpass"])[["hp_hz", "D_FC"]].round(3).to_markdown(index=False),
           "\n### Stage 4a: wheeze low-pass (D_W)\n",
           pd.DataFrame([r for r in log if r["stage"] == "4a_wheeze_chain"]).pivot(index="lp_hz", columns="lp_order", values="D_W").round(3).to_markdown(),
           "\n### Medians after calibration (synthetic / real)\n"]
    tab = pd.DataFrame({"FC synth": {k: v[0] for k, v in med_fc.items()}, "FC real": {k: v[1] for k, v in med_fc.items()},
                        "FC |Δ|/IQR": d_fc})
    md += [tab.round(3).to_markdown(), "\n### Wheeze (searched only on W-specific parameters)\n"]
    tabw = pd.DataFrame({"W synth": {k: v[0] for k, v in med_w.items()}, "W real": {k: v[1] for k, v in med_w.items()}, "W |Δ|/IQR": d_w})
    md += [tabw.round(3).to_markdown()]
    (RES / "calibration.md").write_text("\n".join(md), encoding="utf-8")

    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    for order, g in s1.groupby("lp_order"):
        axs[0].plot(g["lp_hz"], g["D_FC"], "o-", label=f"order {order}")
    axs[0].set_xscale("log"); axs[0].set_xlabel("channel low-pass (Hz)"); axs[0].set_ylabel("D_FC"); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[0].set_title("calibration of the chain low-pass (fine crackles)")
    for a, (t, nm) in zip(axs[1:], ((d_fc, "FC"), (d_w, "W"))):
        ser = pd.Series(t).sort_values()
        a.barh(ser.index, ser.values, color="#4C78A8")
        a.axvline(0.3, color="r", ls="--", lw=1)
        a.set_title(f"{nm}: |median diff| / real IQR after calibration"); a.tick_params(axis="y", labelsize=7)
    plt.tight_layout(); plt.savefig(RES / "calibration.png", dpi=90); plt.close()
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
