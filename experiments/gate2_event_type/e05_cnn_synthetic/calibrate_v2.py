"""
calibrate_v2.py  (gate 2 · e05, stage 2)
────────────────────────────────────────
Refinement of e04's pooled calibration, run separately for each DEVICE STATE (2022 vs 2023–2025: the median spectrum of 2022
peaks ~9 dB lower and has a different shape), and extended with the suggestions of the e04 report:

  S1  crackle chain: low-pass (order 6) and high-pass, around the pooled values
  S2  crackle source decay tau  (the physics tau of synthetic crackles was too short; the reference value 4.1 ms is now searched)
  S3  crackle rate and level offset
  S4  wheeze chain low-pass
  S5  wheeze irregularity: gap between bursts, transient clicks, slow envelope
  S6  final distances and the independent check on fitted crackle tau / 2CD (not a target)

Targets are the filter-independent features of e04 (same extractors for real and synthetic events); hosts are matched to the
real event's chest location and age. No classifier is trained here.

Usage : python calibrate_v2.py --tag 2022 --years 2022
        python calibrate_v2.py --tag 2023_25 --years 2023 2024 2025
Output: results/calibration_v2_<tag>.json, results/calibration_v2_<tag>.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import paths  # noqa: E402

sys.path.insert(0, str(paths.G2_E04))
import calibrate_synthesis as cs  # noqa: E402  (synth_table, distance, with_chain, FEATS, W_FEATS)
from synthesis import FS, builder as bd  # noqa: E402
from synthesis import reference as ref  # noqa: E402

warnings.filterwarnings("ignore")
RES = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--years", nargs="+", required=True)
    ap.add_argument("--base", type=Path, default=RES / "calibration_v1_e04.json")
    ap.add_argument("--n-cal", type=int, default=90)
    a = ap.parse_args()
    years = a.years

    cal = json.loads(a.base.read_text())
    cal["noise_chain"] = bd.default_calibration()["noise_chain"]
    hosts, profiles, _ = bd.load_tables()
    full = pd.read_csv(paths.SEGMENTS_CSV).merge(pd.read_csv(paths.PHYSICS_CSV), on=["stem", "event_idx"], how="left")
    full["source"] = full["source"].astype(str)
    real = {c: full[(full["class"] == c) & full["source"].isin(years)] for c in ("FC", "W")}
    jobs_fc = bd.plan_jobs(hosts, profiles, a.n_cal, years, 11, kinds=("FC",), match_hosts=True)
    jobs_w = bd.plan_jobs(hosts, profiles, a.n_cal, years, 12, kinds=("W",), match_hosts=True)
    log = []
    md = [f"# Calibration v2 — {a.tag} (years {', '.join(years)})\n", f"{a.n_cal} synthetic events per candidate; paired seeds; hosts matched to the real event's location/age.\n"]

    def dfc(c):
        return cs.distance(cs.synth_table(jobs_fc, c), real["FC"], cs.FEATS)[0]

    def dw(c):
        return cs.distance(cs.synth_table(jobs_w, c), real["W"], cs.W_FEATS)[0]

    D0 = (dfc(cal), dw(cal))
    print(f"[{a.tag}] pooled e04 calibration on this group: D_FC {D0[0]:.3f}, D_W {D0[1]:.3f}", flush=True)
    md.append(f"Pooled e04 calibration applied to this group: D_FC = {D0[0]:.3f}, D_W = {D0[1]:.3f}\n")

    # S1 crackle chain
    best = None
    for lp in (180, 225, 250, 300, 350):
        D = dfc(cs.with_chain(cal, lp_hz=float(lp), lp_order=6))
        log.append({"stage": "S1_lp", "value": lp, "D": D}); print(f"  S1 lp {lp}: {D:.3f}", flush=True)
        best = min(best, (D, lp)) if best else (D, lp)
    cal["chain"].update(lp_hz=float(best[1]), lp_order=6)
    bhp = None
    for hp in (50.0, 70.0, 100.0):
        D = dfc(cs.with_chain(cal, hp_hz=hp))
        log.append({"stage": "S1_hp", "value": hp, "D": D}); print(f"  S1 hp {hp}: {D:.3f}", flush=True)
        bhp = min(bhp, (D, hp)) if bhp else (D, hp)
    cal["chain"]["hp_hz"] = float(bhp[1])

    # S2 crackle source tau
    bt = None
    for tau in (4.1, 5.0, 5.7, 7.0):
        c = json.loads(json.dumps(cal)); c["crackle"]["tau_median_ms"] = tau
        D = dfc(c)
        log.append({"stage": "S2_tau", "value": tau, "D": D}); print(f"  S2 tau {tau}: {D:.3f}", flush=True)
        bt = min(bt, (D, tau)) if bt else (D, tau)
    cal["crackle"]["tau_median_ms"] = float(bt[1])

    # S3 rate / level
    cal["rate_scale"], cal["snr_offset_db"] = 1.0, 0.0
    for it in range(3):
        s = cs.synth_table(jobs_fc, cal)
        cal["rate_scale"] = float(np.clip(cal["rate_scale"] * np.nanmedian(real["FC"]["mf_rate"]) / max(np.nanmedian(s["mf_rate"]), 1e-6), 0.2, 5.0))
        gap = np.nanmedian(real["FC"]["snr_vs_floor_db"]) - np.nanmedian(s["snr_vs_floor_db"])
        cal["snr_offset_db"] = float(np.clip(cal["snr_offset_db"] + 0.5 * gap, -3.0, 3.0))
        print(f"  S3 iter {it}: rate_scale {cal['rate_scale']:.2f}, offset {cal['snr_offset_db']:+.2f}", flush=True)

    # S4 wheeze chain
    bw = None
    for lp in (400, 500, 650, 800):
        c = json.loads(json.dumps(cal)); c["chain_W"] = {**cal["chain"], "lp_hz": float(lp), "lp_order": 6}
        D = dw(c)
        log.append({"stage": "S4_wheeze_lp", "value": lp, "D": D}); print(f"  S4 wheeze lp {lp}: {D:.3f}", flush=True)
        bw = min(bw, (D, lp)) if bw else (D, lp)
    cal["chain_W"] = {**cal["chain"], "lp_hz": float(bw[1]), "lp_order": 6}

    # S5 wheeze irregularity
    bi = None
    for gap_m, clk, dep in itertools.product((0.15, 0.30), (0.0, 3.0, 8.0), (0.0, 0.3, 0.6)):
        c = json.loads(json.dumps(cal)); c["wheeze"].update(gap_median_s=gap_m, click_rate_hz=clk, slow_env_depth=dep)
        D = dw(c)
        log.append({"stage": "S5_wheeze_irreg", "gap": gap_m, "clicks": clk, "slow_env": dep, "D": D})
        print(f"  S5 gap {gap_m} clicks {clk} slow {dep}: {D:.3f}", flush=True)
        bi = min(bi, (D, gap_m, clk, dep)) if bi else (D, gap_m, clk, dep)
    cal["wheeze"].update(gap_median_s=bi[1], click_rate_hz=bi[2], slow_env_depth=bi[3])

    # S6 final distances + independent check
    D_fc, d_fc, med_fc = cs.distance(cs.synth_table(jobs_fc, cal), real["FC"], cs.FEATS)
    D_w, d_w, med_w = cs.distance(cs.synth_table(jobs_w, cal), real["W"], cs.W_FEATS)
    s40 = cs.synth_table([dict(j, keep_pcm=True) for j in jobs_fc[:60]], cal)
    fits = []
    for _, r in s40.iterrows():
        a0, b0 = r["_target"]
        x = r["_pcm"].astype(np.float64) / 32768
        hp, pk = ref.detect_crackles(x, (int(a0 / 1000 * FS), int(b0 / 1000 * FS)), FS, 4.5, hp_hz=150.0)
        n = 0
        for i, p in enumerate(pk):
            if (i == 0 or (p - pk[i - 1]) / FS > .015) and (i == len(pk) - 1 or (pk[i + 1] - p) / FS > .015):
                f = ref.fit_crackle(hp, p, FS)
                if f and n < 8:
                    fits.append(f); n += 1
    rs = json.loads((paths.G2_E04 / "results" / "sprsound_event_stats.json").read_text())["FC"]
    check = {"synthetic_tau_ms": float(np.median([f["tau_ms"] for f in fits])), "synthetic_twocd_ms": float(np.nanmedian([f["twocd_ms"] for f in fits])),
             "real_tau_ms": rs["tau_ms"]["median"], "real_twocd_ms": rs["twocd_ms"]["median"], "n_fits": len(fits)}
    cal["calibration_summary"] = {"tag": a.tag, "years": years, "D_FC_pooled_e04": D0[0], "D_W_pooled_e04": D0[1], "D_FC": D_fc, "D_W": D_w,
                                  "independent_check": check}
    (RES / f"calibration_v2_{a.tag}.json").write_text(json.dumps(cal, indent=2, default=float), encoding="utf-8")
    md += ["## Result\n", f"- crackle chain: high-pass {cal['chain']['hp_hz']:.0f} Hz, low-pass {cal['chain']['lp_hz']:.0f} Hz (order 6)",
           f"- crackle source tau {cal['crackle']['tau_median_ms']} ms, rate scale {cal['rate_scale']:.2f}, level offset {cal['snr_offset_db']:+.2f} dB",
           f"- wheeze chain low-pass {cal['chain_W']['lp_hz']:.0f} Hz; burst gap {bi[1]} s, clicks {bi[2]}/s, slow envelope depth {bi[3]}",
           f"- distances: D_FC {D0[0]:.3f} → **{D_fc:.3f}**, D_W {D0[1]:.3f} → **{D_w:.3f}** (pooled e04 calibration → refined)",
           f"- independent check (fitted, not a target): synthetic tau {check['synthetic_tau_ms']:.1f} ms / 2CD {check['synthetic_twocd_ms']:.1f} ms "
           f"vs real {check['real_tau_ms']:.1f} / {check['real_twocd_ms']:.1f} ms ({check['n_fits']} fits)\n", "## Search log\n",
           pd.DataFrame(log).round(3).to_markdown(index=False)]
    (RES / f"calibration_v2_{a.tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
