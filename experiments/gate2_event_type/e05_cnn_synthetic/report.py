"""
report.py  (gate 2 · e05)
─────────────────────────
Tables and figures comparing every condition trained by train_cnn_synth.py (results/<name>_results.csv):
mean AUC per held-out year, seed-ensemble AUC, and PAIRED differences against the real-only run with the same input band
(same folds, same seeds, same real validation split).

Output: results/report_auto.md, results/fig_auc_by_fold.png, results/fig_delta_auc.png, results/summary.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
warnings.filterwarnings("ignore")
RES = HERE / "results"
YEARS = ["2022", "2023", "2024", "2025"]


def md(df, nd=3):
    return df.round(nd).to_markdown()


def main():
    files = sorted(RES.glob("*_results.csv"))
    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    res["fold"] = res["fold"].astype(str)
    res = res.drop_duplicates(["name", "fold", "seed"])
    pref = ["real_b2000", "v1_r1_b2000", "v2_r1_b2000", "real_b900", "v2_r1_b900"]
    names = [n for n in pref if n in set(res["name"])] + sorted(set(res["name"]) - set(pref))
    rep = ["# e05 — CNN with synthetic data: automatic tables\n", "Conditions: " + ", ".join(f"`{n}`" for n in names) +
           ". `real_*` = real data only (e03 recipe); `v1`/`v2` = real + synthetic (generator version); `r1` = synthetic events per class = 1.0 × the larger real class; "
           "`b2000`/`b900` = mel band 50–2000 Hz (64 mels) / 60–900 Hz (40 mels).\n"]

    auc = res.pivot_table(index="name", columns="fold", values="auc", aggfunc="mean").reindex(names)[YEARS]
    auc["mean"] = auc.mean(1)
    sd = res.pivot_table(index="name", columns="fold", values="auc", aggfunc="std").reindex(names)[YEARS]
    nseeds = res.groupby("name")["seed"].nunique().reindex(names)
    rep += ["## ROC AUC per held-out year (mean of seeds)\n", md(auc), "\n", f"Seeds per condition: {nseeds.to_dict()}. Std across seeds per year:\n", md(sd), "\n"]

    # seed-ensemble AUC
    ens = {}
    for n in names:
        pr = pd.read_csv(RES / f"{n}_predictions.csv")
        pr["fold"] = pr["fold"].astype(str)
        g = pr.groupby(["fold", "stem", "event_idx"]).agg(p=("p", "mean"), y=("y", "first")).reset_index()
        ens[n] = {f: roc_auc_score(d["y"], d["p"]) for f, d in g.groupby("fold")}
    E = pd.DataFrame(ens).T.reindex(names)[YEARS]
    E["mean"] = E.mean(1)
    rep += ["## Seed-ensemble ROC AUC (probabilities averaged over seeds)\n", md(E), "\n"]

    # threshold metrics
    rep += ["## Balanced accuracy and recalls at the fixed 0.5 threshold (mean of seeds; unstable across seeds, see std)\n",
            md(res.pivot_table(index="name", columns="fold", values="bal_acc", aggfunc="mean").reindex(names)[YEARS].assign(
                mean=lambda d: d.mean(1))), "\n", "Std of the balanced accuracy across seeds, mean over years: " +
            ", ".join(f"{n}: {res[res.name == n].groupby('fold')['bal_acc'].std().mean():.3f}" for n in names) + "\n"]

    # paired differences
    base_of = lambda n: "real_b900" if n.endswith("b900") else "real_b2000"  # noqa: E731
    rows, deltas = [], {}
    for n in names:
        b = base_of(n)
        if n == b or b not in names:
            continue
        m = res[res.name == n].merge(res[res.name == b], on=["fold", "seed"], suffixes=("", "_b"))
        m["d"] = m["auc"] - m["auc_b"]
        deltas[n] = m
        row = {"condition": n, "baseline": b, "n_pairs": len(m), "mean_delta_auc": m["d"].mean(), "std_pairs": m["d"].std(),
               "share_positive": float((m["d"] > 0).mean()), "n_synth_mean": m["n_synth"].mean()}
        for y in YEARS:
            row[y] = m[m.fold == y]["d"].mean()
        rows.append(row)
    if rows:
        D = pd.DataFrame(rows).set_index("condition")
        D.to_csv(RES / "summary.csv")
        rep += ["## Paired ΔAUC against the real-only run with the same band (mean over seeds per year; `std_pairs` over all fold×seed pairs)\n", md(D), "\n"]
        rep += ["Per-year Δ with its seed std:\n"]
        for n, m in deltas.items():
            t = m.groupby("fold")["d"].agg(["mean", "std"]).reindex(YEARS)
            rep.append(f"- `{n}`: " + ", ".join(f"{y} {t.loc[y, 'mean']:+.3f} (±{t.loc[y, 'std']:.3f})" for y in YEARS))
    # attribution: pairs of conditions that differ in one factor
    from scipy.stats import wilcoxon
    extra = [("v2_pre_b2000", "real_lr3e4_b2000", "synthetic pretraining vs. real-only at the same fine-tuning learning rate"),
             ("v2_pre_b2000", "real_b2000", "synthetic pretraining (v2) vs. the e03 baseline"),
             ("v1_pre_b2000", "real_lr3e4_b2000", "synthetic pretraining (v1 generator) vs. same-lr real-only"),
             ("v2_pre_b2000", "v1_pre_b2000", "pretraining: v2 generator vs. v1 generator"),
             ("v2_r1_b2000", "v1_r1_b2000", "mixed 1:1: v2 vs. v1"),
             ("v2_r025_b2000", "v1_r025_b2000", "mixed 25 %: v2 vs. v1"),
             ("real_lr3e4_b2000", "real_b2000", "learning rate 3e-4 vs. 1e-3, real-only"),
             ("real_b900", "real_b2000", "mel band 60–900 Hz vs. 50–2000 Hz, real-only")]
    rows2 = []
    for a_, b_, why in extra:
        if a_ in names and b_ in names:
            m = res[res.name == a_].merge(res[res.name == b_], on=["fold", "seed"], suffixes=("", "_b"))
            m["d"] = m["auc"] - m["auc_b"]
            pv = wilcoxon(m["d"]).pvalue if len(m) >= 6 and m["d"].abs().sum() > 0 else float("nan")
            row = {"comparison": f"{a_} − {b_}", "what": why, "n_pairs": len(m), "mean_delta_auc": m["d"].mean(), "share_positive": float((m["d"] > 0).mean()),
                   "wilcoxon_p": pv}
            for y in YEARS:
                row[y] = m[m.fold == y]["d"].mean()
            rows2.append(row)
    if rows2:
        rep += ["\n## Attribution: paired comparisons that differ in one factor\n",
                "`wilcoxon_p` is a paired signed-rank test over the fold×seed pairs; the 3 seeds within a year are not independent samples of the data, so treat it as a rough guide, not as a confirmation.\n",
                md(pd.DataFrame(rows2).set_index("comparison"))]
    rep += ["\n## Training details\n", md(res.groupby("name").agg(epochs=("epochs", "mean"), n_real_train=("n_real_train", "mean"), n_synth=("n_synth", "mean"),
                                                             secs=("secs", "mean")).reindex(names), 1)]
    (RES / "report_auto.md").write_text("\n".join(rep), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(np.arange(4) + i * w, [auc.loc[n, y] for y in YEARS], w, label=n)
    ax.set_xticks(np.arange(4) + 0.4 - w / 2)
    ax.set_xticklabels(YEARS)
    ax.set_ylim(0.8, None)
    ax.set_ylabel("ROC AUC (mean of seeds)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.3, axis="y")
    ax.set_title("W vs FC, leave-one-year-out")
    plt.tight_layout()
    plt.savefig(RES / "fig_auc_by_fold.png", dpi=100)
    plt.close()
    if deltas:
        ncol = 4
        nrow = -(-len(deltas) // ncol)
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)
        for a in axs.ravel()[len(deltas):]:
            a.axis("off")
        for a, (n, m) in zip(axs.ravel(), deltas.items()):
            for y, c in zip(YEARS, ["#4C78A8", "#54A24B", "#F58518", "#E45756"]):
                d = m[m.fold == y]["d"]
                a.scatter([y] * len(d), d, color=c, s=25)
                a.hlines(d.mean(), YEARS.index(y) - .3, YEARS.index(y) + .3, color=c)
            a.axhline(0, color="k", lw=.8)
            a.set_title(f"{n} − {base_of(n)}", fontsize=9)
            a.set_ylabel("ΔAUC per seed")
        plt.tight_layout()
        plt.savefig(RES / "fig_delta_auc.png", dpi=100)
    print("\n".join(rep[:30]))


if __name__ == "__main__":
    main()
