"""
confusion_matrix.py  (gate 2 · e05)
───────────────────────────────────
Confusion matrix of the best e05 model: pretraining on the v2 synthetic events, then fine-tuning on real data only
(`v2_pre_b2000`). Uses the saved held-out predictions of the leave-one-year-out runs (no retraining): for every held-out year the
probabilities of the 3 seeds are averaged (seed ensemble) and thresholded at 0.5. W = class 0, FC = class 1.

Usage : python confusion_matrix.py [--name v2_pre_b2000] [--threshold 0.5]
Output: results/best_model/confusion_matrix.png, confusion_matrix.csv (counts), confusion_matrix.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "best_model"
LABELS = ["W", "FC"]
YEARS = ["2022", "2023", "2024", "2025"]


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return cm, {"n": len(y), "auc": roc_auc_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, pred),
                "recall_W": cm[0, 0] / cm[0].sum(), "recall_FC": cm[1, 1] / cm[1].sum(),
                "precision_W": cm[0, 0] / max(cm[:, 0].sum(), 1), "precision_FC": cm[1, 1] / max(cm[:, 1].sum(), 1)}


def report_table(y, p, thr):
    """Per-class precision / recall / F1 / support plus accuracy, macro and weighted averages."""
    r = classification_report(y, (p >= thr).astype(int), labels=[0, 1], target_names=LABELS, output_dict=True, zero_division=0)
    t = pd.DataFrame(r).T.rename(columns={"f1-score": "f1"})
    t.loc["accuracy", ["precision", "recall"]] = np.nan                       # sklearn stores the accuracy in the f1 column
    t.loc["accuracy", "support"] = r["macro avg"]["support"]
    return t[["precision", "recall", "f1", "support"]]


def draw(ax, cm, title):
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.1%}", ha="center", va="center", fontsize=10, color="white" if cmn[i, j] > .5 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v2_pre_b2000")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pr = pd.read_csv(RES / f"{a.name}_predictions.csv")
    pr["fold"] = pr["fold"].astype(str)
    ens = pr.groupby(["fold", "stem", "event_idx"]).agg(p=("p", "mean"), y=("y", "first")).reset_index()

    rows, cms = [], {}
    for y in YEARS:
        g = ens[ens.fold == y]
        cms[y], m = metrics(g["y"].values, g["p"].values, a.threshold)
        rows.append({"held_out_year": y, **m})
    cms["all years"], m = metrics(ens["y"].values, ens["p"].values, a.threshold)
    rows.append({"held_out_year": "all years (pooled)", **m})
    M = pd.DataFrame(rows).set_index("held_out_year")
    seed_rows = []
    for s, g in pr[pr.fold == "2025"].groupby("seed"):
        cm, m = metrics(g["y"].values, g["p"].values, a.threshold)
        seed_rows.append({"seed": s, "TN(W→W)": cm[0, 0], "FP(W→FC)": cm[0, 1], "FN(FC→W)": cm[1, 0], "TP(FC→FC)": cm[1, 1], "auc": m["auc"],
                          "balanced_accuracy": m["balanced_accuracy"]})
    S = pd.DataFrame(seed_rows).set_index("seed")
    tables = {y: report_table(ens[ens.fold == y]["y"].values, ens[ens.fold == y]["p"].values, a.threshold) for y in YEARS}
    tables["all years (pooled)"] = report_table(ens["y"].values, ens["p"].values, a.threshold)
    pd.concat(tables, names=["held_out_year", "class"]).to_csv(OUT / "classification_report.csv")

    pd.DataFrame({k: v.ravel() for k, v in cms.items()}, index=["W→W", "W→FC", "FC→W", "FC→FC"]).T.to_csv(OUT / "confusion_matrix.csv")
    fig, axs = plt.subplots(1, 5, figsize=(21, 4.2))
    for ax, k in zip(axs, list(cms)):
        draw(ax, cms[k], f"{a.name} · held-out {k}\nAUC {M.loc[k if k != 'all years' else 'all years (pooled)', 'auc']:.3f}")
    plt.tight_layout()
    fig.subplots_adjust(top=0.80)
    plt.savefig(OUT / "confusion_matrix.png", dpi=110)
    (OUT / "confusion_matrix.md").write_text(
        "\n".join([f"# Confusion matrix of the best e05 model (`{a.name}`, seed ensemble, threshold {a.threshold})\n",
                   "Rows = true class, columns = predicted class (W = wheeze, FC = fine crackle). Each year was held out from training.\n",
                   "## Official split (2022–2024 train, **2025 test**)\n", "| true \\ predicted | W | FC |", "|---|---:|---:|",
                   f"| W | {cms['2025'][0, 0]} | {cms['2025'][0, 1]} |", f"| FC | {cms['2025'][1, 0]} | {cms['2025'][1, 1]} |", "",
                   "## Precision, recall and F1 — official split (2025 test)\n", tables["2025"].round(3).to_markdown(),
                   "\n`macro avg` = unweighted mean of the two classes; `weighted avg` = mean weighted by class support (2025 is 86 % FC); "
                   "`accuracy` is the share of correct predictions.\n",
                   "## Metrics per held-out year\n", M.round(3).to_markdown(), "",
                   "## Precision, recall and F1 for every held-out year\n",
                   *[f"### {k}\n\n{v.round(3).to_markdown()}\n" for k, v in tables.items() if k != "2025"],
                   "## 2025 per seed (single networks, before ensembling)\n", S.round(3).to_markdown()]), encoding="utf-8")
    print((OUT / "confusion_matrix.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
