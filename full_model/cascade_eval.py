"""
cascade_eval.py — score every valid (gate-1, gate-2) finalist pair end to end on the out-of-fold predictions.

Rule: predict N if p1 < t1; else FC if p2 >= t2; else W. Thresholds maximise the class-balanced macro-F1 on pooled OOF predictions (never the test set).
A pair is valid when params(g1) + params(g2) < --max-params. Writes results/cascade_pairs.md and results/cascade.json (the chosen pair).
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

import fm_common as fc

GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def cascade_pred(p1, p2, t1, t2):
    return np.where(p1 < t1, 0, np.where(p2 >= t2, 2, 1))


def macro_f1(y, pred, w):
    f = []
    for c in (0, 1, 2):
        tp = w[(y == c) & (pred == c)].sum()
        fp = w[(y != c) & (pred == c)].sum()
        fn = w[(y == c) & (pred != c)].sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def best_thresholds(y, p1, p2, w):
    best = (-1, 0.5, 0.5)
    for t1, t2 in itertools.product(GRID, GRID):
        f = macro_f1(y, cascade_pred(p1, p2, t1, t2), w)
        if f > best[0]:
            best = (f, float(t1), float(t2))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="full")
    ap.add_argument("--max-params", type=int, default=1_000_000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--size-tolerance-se", type=float, default=1.0)
    a = ap.parse_args()

    r1, r2 = (pd.read_csv(fc.RESULTS / f"refit_g{g}.csv").head(a.top_k) for g in (1, 2))
    pool = fc.read_split("train_pool.csv").merge(fc.read_split("cv_folds.csv")[["eid", "fold"]], on="eid", how="left")
    oof = lambda r: np.load(fc.RESULTS / "oof" / r["oof_file"])  # noqa: E731
    y = pool["cls4"].to_numpy()
    base = (y < 3)
    rows = []
    for (_, g1), (_, g2) in itertools.product(r1.iterrows(), r2.iterrows()):
        total = int(g1.params + g2.params)
        p1, p2 = oof(g1), oof(g2)
        ok = base & np.isfinite(p1) & np.isfinite(p2)
        yy, a1, a2, folds = y[ok], p1[ok], p2[ok], pool["fold"].to_numpy()[ok]
        w = np.zeros(len(yy))
        for c in (0, 1, 2):
            w[yy == c] = 1.0 / max((yy == c).sum(), 1)
        w /= w.sum()
        f1, t1, t2 = best_thresholds(yy, a1, a2, w)
        per_fold = []
        for k in sorted(set(folds)):
            m = folds == k
            wk = w[m] / w[m].sum()
            per_fold.append(macro_f1(yy[m], cascade_pred(a1[m], a2[m], t1, t2), wk))
        pred = cascade_pred(a1, a2, t1, t2)
        ba = float(np.mean([w[(yy == c) & (pred == c)].sum() / w[yy == c].sum() for c in (0, 1, 2)]))
        rows.append(dict(g1_trial=int(g1.trial), g2_trial=int(g2.trial), params_g1=int(g1.params), params_g2=int(g2.params), params_total=total,
                         valid=total < a.max_params, t1=t1, t2=t2, macro_f1=f1, bal_acc=ba, f1_fold_mean=float(np.mean(per_fold)),
                         f1_fold_se=float(np.std(per_fold) / np.sqrt(len(per_fold)))))
    df = pd.DataFrame(rows)
    valid = df[df.valid].copy()
    if valid.empty:
        raise SystemExit(f"no (g1, g2) pair fits under {a.max_params:,} parameters")
    valid = valid.sort_values("params_total")
    valid["pareto"] = valid["macro_f1"] > valid["macro_f1"].cummax().shift(fill_value=-1)
    best = valid.loc[valid.macro_f1.idxmax()]
    near = valid[valid.macro_f1 >= best.macro_f1 - a.size_tolerance_se * best.f1_fold_se]
    pick = near.sort_values(["params_total", "macro_f1"], ascending=[True, False]).iloc[0]
    assert pick.params_total < a.max_params

    g1r = r1[r1.trial == pick.g1_trial].iloc[0]
    g2r = r2[r2.trial == pick.g2_trial].iloc[0]
    chosen = dict(max_params=a.max_params, params_total=int(pick.params_total), t1=float(pick.t1), t2=float(pick.t2),
                  oof_macro_f1=float(pick.macro_f1), oof_bal_acc=float(pick.bal_acc),
                  gate1=dict(study=g1r.study, trial=int(g1r.trial), params=int(g1r.params), config=json.loads(g1r.config)),
                  gate2=dict(study=g2r.study, trial=int(g2r.trial), params=int(g2r.params), config=json.loads(g2r.config)))
    fc.jdump(chosen, fc.RESULTS / "cascade.json")
    df.sort_values("macro_f1", ascending=False).to_csv(fc.RESULTS / "cascade_pairs.csv", index=False)
    md = ["# Cascade pairs (out-of-fold, class-balanced macro-F1)", "",
          f"Rule: N if p1 < t1; else FC if p2 >= t2; else W. Pairs must satisfy params(g1)+params(g2) < {a.max_params:,}. {len(df)} pairs, {len(valid)} valid.", "",
          f"**Chosen** (smallest within {a.size_tolerance_se:g} SE of the best): g1 trial {pick.g1_trial} + g2 trial {pick.g2_trial}, {int(pick.params_total):,} params, "
          f"t1={pick.t1:.2f}, t2={pick.t2:.2f}, macro-F1 {pick.macro_f1:.3f} (best {best.macro_f1:.3f}, SE {best.f1_fold_se:.3f})", "",
          "| g1 | g2 | params | t1 | t2 | macro-F1 | bal-acc | fold SE | valid | Pareto |", "|---|---|---|---|---|---|---|---|---|---|"]
    pv = valid.set_index(["g1_trial", "g2_trial"])["pareto"].to_dict()
    for _, r in df.sort_values("macro_f1", ascending=False).iterrows():
        md.append(f"| {r.g1_trial} | {r.g2_trial} | {r.params_total:,} | {r.t1:.2f} | {r.t2:.2f} | {r.macro_f1:.3f} | {r.bal_acc:.3f} | {r.f1_fold_se:.3f} | "
                  f"{'yes' if r.valid else 'no'} | {'yes' if pv.get((r.g1_trial, r.g2_trial), False) else ''} |")
    (fc.RESULTS / "cascade_pairs.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:6]))


if __name__ == "__main__":
    main()
