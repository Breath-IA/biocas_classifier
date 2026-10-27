"""
make_splits.py — frozen class-balanced test set (K events of N, W and FC) + patient-disjoint train pool, validation split and CV folds.

Writes to splits/ (or $FM_SPLITS_DIR):  balanced_test.csv  train_pool.csv (column `role` = train|val)  val_split.csv  cv_folds.csv  splits_report.md
Whole patients go to test; their leftover events are dropped (never used for training). All assertions run before anything is written.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import fm_common as fc

KEEP = ["eid", "stem", "split", "event_idx", "start_ms", "end_ms", "year", "pid", "class", "cls4"]


def pick_test(ev: pd.DataFrame, k: int, cap: int, rng, keep_poor: bool):
    base = ev if keep_poor else ev[~ev["poor_quality"].astype(bool)]
    cand = base[base["cls4"] < 3]
    pats = {c: cand[cand["cls4"] == c].groupby("pid").size() for c in (0, 1, 2)}
    chosen, avail = [], {0: 0, 1: 0, 2: 0}
    seen = set()
    for c in (1, 2, 0):                                    # scarce class first: wheeze, then fine crackle, then normal
        order = list(rng.permutation(pats[c].index.to_numpy()))
        for p in order:
            if avail[c] >= k:
                break
            if p in seen:
                continue
            seen.add(p)
            chosen.append(p)
            for cc in (0, 1, 2):
                avail[cc] += int(min(cap, pats[cc].get(p, 0)))
    if min(avail.values()) < k:
        raise SystemExit(f"cannot build a balanced test set with K={k}, max_per_patient={cap}: candidates {avail}")
    sel = cand[cand["pid"].isin(chosen)]
    rows = []
    for c in (0, 1, 2):
        g = sel[sel["cls4"] == c]
        g = g.groupby("pid", group_keys=False).apply(lambda x: x.sample(min(len(x), cap), random_state=int(rng.integers(1 << 30))))
        rows.append(g.sample(k, random_state=int(rng.integers(1 << 30))))
    return pd.concat(rows), set(chosen)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k-per-class", type=int, default=250)
    ap.add_argument("--max-per-patient", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--keep-poor-quality", action="store_true", help="also allow poor-quality recordings in the test pool")
    ap.add_argument("--out", default=None, help="output folder (default: splits/ or $FM_SPLITS_DIR)")
    a = ap.parse_args()
    out = fc.SPLITS if a.out is None else __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = fc.load_events()

    # ── balanced test: retry the seed until every class comes from >= 3 of the 4 years
    for attempt in range(50):
        rng = np.random.default_rng(a.seed + attempt)
        test, test_pats = pick_test(ev, a.k_per_class, a.max_per_patient, rng, a.keep_poor_quality)
        if all(test[test.cls4 == c]["year"].nunique() >= 3 for c in (0, 1, 2)):
            break
    else:
        raise SystemExit("no seed gave >= 3 years per class; lower --k-per-class or relax --max-per-patient")
    pool = ev[~ev["pid"].isin(test_pats)].copy()
    assert not set(test["pid"]) & set(pool["pid"]), "patient overlap between test and pool"
    assert test.groupby("cls4").size().eq(a.k_per_class).all(), "test set is not exactly K per class"

    # ── validation: a fixed patient-disjoint share of the pool, with enough W and FC to be informative
    pids = np.array(sorted(pool["pid"].unique()))
    for attempt in range(200):
        r2 = np.random.default_rng(a.seed + 1000 + attempt)
        val_p = set(r2.choice(pids, size=max(1, int(round(a.val_fraction * len(pids)))), replace=False))
        v = pool[pool["pid"].isin(val_p)]
        need = max(10, int(0.1 * a.k_per_class))
        if all((v["cls4"] == c).sum() >= need for c in (0, 1, 2)):
            break
    else:
        raise SystemExit("could not find a validation split with enough W/FC events")
    pool["role"] = np.where(pool["pid"].isin(val_p), "val", "train")
    assert not set(pool[pool.role == "val"]["pid"]) & set(pool[pool.role == "train"]["pid"])

    # ── CV folds over the whole pool (patient-grouped, stratified on the 4-way class)
    sgkf = StratifiedGroupKFold(n_splits=a.cv_folds, shuffle=True, random_state=a.seed)
    fold = np.full(len(pool), -1)
    for i, (_, te) in enumerate(sgkf.split(pool, pool["cls4"], pool["pid"])):
        fold[te] = i
    pool["fold"] = fold
    for i in range(a.cv_folds):
        tr_p, te_p = set(pool[pool.fold != i]["pid"]), set(pool[pool.fold == i]["pid"])
        assert not tr_p & te_p, "CV folds share patients"

    test[KEEP].to_csv(out / "balanced_test.csv", index=False)
    pool[KEEP + ["role"]].to_csv(out / "train_pool.csv", index=False)
    pool[pool.role == "val"][KEEP].to_csv(out / "val_split.csv", index=False)
    pool[["eid", "pid", "fold"]].to_csv(out / "cv_folds.csv", index=False)

    names = {0: "N", 1: "W", 2: "FC", 3: "other"}
    lines = ["# Splits report", "", f"K={a.k_per_class} per class · max {a.max_per_patient}/patient/class · seed {a.seed} (+{attempt} retries for years) · val fraction {a.val_fraction} · {a.cv_folds} folds", "",
             f"Patients: test {len(test_pats)} · pool {pool['pid'].nunique()} (val {len(val_p)})", "",
             "## Test set (events, patients) per class and year", "", "| class | " + " | ".join(sorted(ev.year.unique())) + " | patients |", "|---|" + "---|" * (ev.year.nunique() + 1)]
    for c in (0, 1, 2):
        g = test[test.cls4 == c]
        lines.append(f"| {names[c]} | " + " | ".join(str(int((g.year == y).sum())) for y in sorted(ev.year.unique())) + f" | {g.pid.nunique()} |")
    lines += ["", "## Pool events per class and role", "", "| class | train | val |", "|---|---|---|"]
    for c in (0, 1, 2, 3):
        lines.append(f"| {names[c]} | {int(((pool.cls4 == c) & (pool.role == 'train')).sum())} | {int(((pool.cls4 == c) & (pool.role == 'val')).sum())} |")
    lines += ["", f"Dropped (events of test patients not in the test set): {int(ev['pid'].isin(test_pats).sum() - len(test))}",
              "Assertions passed: test = exactly K per class, no patient shared between test / pool, train / val, or across the folds of a CV split."]
    (out / "splits_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
