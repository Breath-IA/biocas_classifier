"""
qc_probe.py  (gate 2 · e05)
───────────────────────────
The plan's real-vs-synthetic probe: a classifier (random forest and logistic regression, grouped cross-validation by patient)
is trained to tell REAL events of a class from SYNTHETIC ones using the same features as everywhere else
(spectral, temporal, tonal and the physics features). AUC = 0.5 means the classifier cannot tell them apart; the closer to 1, the
more the synthetic data betray themselves. The most informative features show what to fix next.

Compares synthetic sets generated with the e04 generator (v1) and with the improved one (v2) on identical settings
(same number of events, same seeds).

Usage : python qc_probe.py --v1 results/calibration_v1_e04.json --v2 results/calibration_map_v2.json --n 400
Output: results/probe.csv, results/probe_top_features.csv, results/probe.png, results/probe.md
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
from analysis_utils import META, auc  # noqa: E402
from build_synth_mels import load_cal_map  # noqa: E402
from synthesis import builder as bd  # noqa: E402

warnings.filterwarnings("ignore")
RES = HERE / "results"


def feat_one(job, cal_map):
    cal = cal_map.get(job["year"], cal_map["default"])
    r = bd.run_job(job, cal, keep_audio=True)
    if r is None:
        return None
    f = bd.extract_features(r["_pcm"], [(0, job["kind"], r["target_start_ms"], r["target_end_ms"])], stem=r["name"])
    if f.empty:
        return None
    f["kind"] = job["kind"]
    f["host_patient"] = job["patient_id"]
    f["year"] = job["year"]
    return f.iloc[[0]]


def synth_features(cal_path, n, seed, match):
    cal_map = load_cal_map(Path(cal_path))
    hosts, profiles, _ = bd.load_tables()
    jobs = bd.plan_jobs(hosts, profiles, n, bd.YEARS, seed, kinds=("FC", "W"), match_hosts=match)
    res = Parallel(n_jobs=4)(delayed(feat_one)(j, cal_map) for j in jobs)
    return pd.concat([r for r in res if r is not None], ignore_index=True)


def probe(real, syn, feats, seed=0):
    """Out-of-fold AUC of real-vs-synthetic classifiers (grouped by patient)."""
    rs = real.sample(min(len(real), 2000), random_state=seed)
    d = pd.concat([rs.assign(_y=0, _g=rs["patient_id"].astype(str)), syn.assign(_y=1, _g="syn_" + syn["host_patient"].astype(str))], ignore_index=True)
    X = SimpleImputer(strategy="median").fit_transform(d[feats].replace([np.inf, -np.inf], np.nan))
    y = d["_y"].values
    out = {}
    imp = np.zeros(len(feats))
    for name, mk in (("RF", lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=4, random_state=seed)),
                     ("LR", lambda: make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed),
                                                  LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000)))):
        oof = np.zeros(len(d))
        for tr, te in GroupKFold(5).split(X, y, d["_g"]):
            m = mk().fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
            if name == "RF":
                imp += m.feature_importances_ / 5
        out[name] = roc_auc_score(y, oof)
    return out, pd.Series(imp, index=feats).sort_values(ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, default=RES / "calibration_v1_e04.json")
    ap.add_argument("--v2", type=Path, default=RES / "calibration_map_v2.json")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    full = pd.read_csv(paths.SEGMENTS_CSV).merge(pd.read_csv(paths.PHYSICS_CSV), on=["stem", "event_idx"], how="left")
    full["source"] = full["source"].astype(str)
    rows, tops, gaps = [], [], {}
    sets = {"v1 (e04 generator)": (a.v1, False), "v2 (improved)": (a.v2, True)}
    md = ["# Real-vs-synthetic probe\n", f"{a.n} synthetic events per class and set (same seed); real events: up to 2000 per class, random. "
          "Out-of-fold AUC, 5 folds grouped by patient (real) / host patient (synthetic). 0.5 = indistinguishable.\n"]
    for sname, (cal_path, match) in sets.items():
        syn = synth_features(cal_path, a.n, a.seed, match)
        feats = [c for c in syn.columns if c in full.columns and c not in META and not c.startswith("sp_") and pd.api.types.is_numeric_dtype(syn[c])]
        for cls in ("FC", "W"):
            s_ = syn[syn["kind"] == cls]
            r_ = full[full["class"] == cls]
            res, imp = probe(r_, s_, feats)
            gap = [abs(auc(np.r_[r_[f].dropna().values[:4000], s_[f].dropna().values],
                           np.r_[np.zeros(min(r_[f].notna().sum(), 4000)), np.ones(s_[f].notna().sum())].astype(bool)) - .5) * 2 for f in feats]
            rows.append({"set": sname, "class": cls, "probe_auc_RF": res["RF"], "probe_auc_LR": res["LR"], "median_univariate_gap": float(np.nanmedian(gap)),
                         "n_synth": len(s_), "n_real": min(len(r_), 2000), "n_features": len(feats)})
            tops.append(imp.head(8).rename("importance").reset_index().rename(columns={"index": "feature"}).assign(set=sname, **{"class": cls}))
            print(f"  {sname} {cls}: probe AUC RF {res['RF']:.3f}, LR {res['LR']:.3f}, median univariate gap {rows[-1]['median_univariate_gap']:.3f}", flush=True)
    R = pd.DataFrame(rows)
    T = pd.concat(tops)
    R.to_csv(RES / "probe.csv", index=False)
    T.to_csv(RES / "probe_top_features.csv", index=False)
    md += ["## Probe AUC and univariate gap\n", R.round(3).to_markdown(index=False), "\n## Features that betray the synthetic events (RF importance)\n"]
    for (s, c), g in T.groupby(["set", "class"]):
        md.append(f"- **{s} · {c}**: " + ", ".join(f"{r.feature} ({r.importance:.2f})" for r in g.itertuples()))
    (RES / "probe.md").write_text("\n".join(md), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4))
    piv = R.pivot(index="class", columns="set", values="probe_auc_RF")
    piv.plot.bar(ax=ax, color=["#E45756", "#4C78A8"])
    ax.axhline(0.5, color="k", lw=.8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("real-vs-synthetic probe AUC (RF)")
    ax.set_title("lower = more realistic")
    plt.tight_layout()
    plt.savefig(RES / "probe.png", dpi=100)
    print("\n".join(md[:8]))


if __name__ == "__main__":
    main()
