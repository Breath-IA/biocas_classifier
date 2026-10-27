"""
refit_top.py — refit the TOP_K distinct configs of one gate with patient-grouped CV x seeds on the training pool (train + val merged).

For each finalist and fold: train on the other folds (a patient-grouped 12 % of them is used for early stopping), score the held-out fold, and
store out-of-fold probabilities for EVERY pool event (all classes: the cascade needs gate-1 and gate-2 outputs on the same events).
Outputs: results/refit_g<gate>.csv  and  results/oof/<study>_t<trial>.npy (aligned with train_pool.csv).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd

import fm_common as fc


def top_configs(study, k: int) -> list[dict]:
    done = sorted([t for t in study.trials if t.state.name == "COMPLETE"], key=lambda t: -t.value)
    seen, out = set(), []
    for t in done:
        cfg = t.user_attrs["config"]
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(trial=t.number, value=t.value, config=cfg, params=t.user_attrs["params"], params_folded=t.user_attrs["params_folded"],
                        macs=t.user_attrs["macs"]))
        if len(out) == k:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", type=int, choices=(1, 2), required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--storage", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs-max", type=int, default=60)
    ap.add_argument("--n-per-epoch", type=int, default=6000)
    ap.add_argument("--trial-max-min", type=float, default=15)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cache-on-cpu", action="store_true")
    a = ap.parse_args()

    torch = fc.import_torch()
    import train_lib as tl
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = fc.resolve_device(torch, a.device)
    study = optuna.load_study(study_name=a.study, storage=fc.storage_url(a.storage, a.study))
    finalists = top_configs(study, a.top_k)
    if not finalists:
        raise SystemExit(f"study {a.study} has no completed trial: run the search first")
    pool = fc.read_split("train_pool.csv").merge(fc.read_split("cv_folds.csv")[["eid", "fold"]], on="eid", how="left")
    assert pool["fold"].notna().all(), "cv_folds.csv does not cover the pool: rerun make_splits.py"
    n_folds = min(a.cv_folds, int(pool["fold"].max()) + 1)
    bundle = tl.Bundle(torch, device, a.cache_on_cpu)
    (fc.RESULTS / "oof").mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, f in enumerate(finalists):
        cfg, band = f["config"], f["config"]["band"]
        cache = bundle.cache(band)
        synth = bundle.synth(band) if (a.gate == 2 and cfg.get("synth_pretrain")) else None
        oof = np.full((a.seeds, len(pool)), np.nan, dtype=np.float32)
        in_cache = pool["eid"].isin(cache.row).to_numpy()
        all_rows = np.array([cache.row.get(e, 0) for e in pool["eid"]], dtype=np.int64)
        fold_scores, fold_metrics = [], []
        t0 = time.time()
        for fold in range(n_folds):
            tr_df, te_df = pool[pool.fold != fold], pool[pool.fold == fold]
            tr_all = tl.make_subset(tr_df, a.gate, cache)
            te_sub = tl.make_subset(te_df, a.gate, cache)
            ms = []
            for s in range(a.seeds):
                tr, va = tl.split_inner(tr_all, 0.12, seed=1000 * fold + s)
                excl = set(te_df["pid"]) | set(va.pid)
                res = tl.train(torch, cfg, a.gate, tr, va, cache, device, seed=1000 * fold + s, amp=a.amp, synth=synth, exclude_pids=excl,
                               epochs_max=a.epochs_max, n_per_epoch=a.n_per_epoch, max_seconds=a.trial_max_min * 60)
                te_idx = np.flatnonzero((pool.fold == fold).to_numpy() & in_cache)
                oof[s, te_idx] = tl.predict(torch, res.model, cache.X, all_rows[te_idx], device)
                ms.append(tl.metrics(te_sub.y, oof[s, te_df.index.to_numpy()[te_sub.pos]], te_sub.w))
            m = {k: float(np.nanmean([x[k] for x in ms])) for k in ms[0]}
            fold_metrics.append(m)
            fold_scores.append(tl.score_of(m))
            print(f"[g{a.gate} #{rank} trial {f['trial']}] fold {fold}: bal_acc {m['bal_acc']:.3f} auc {m['auc']:.3f}", flush=True)
        oof_path = fc.RESULTS / "oof" / f"{a.study}_t{f['trial']}.npy"
        np.save(oof_path, np.nanmean(oof, axis=0))
        rows.append(dict(rank=rank, study=a.study, trial=f["trial"], search_value=f["value"], params=f["params"], params_folded=f["params_folded"], macs=f["macs"],
                         cv_score_mean=float(np.mean(fold_scores)), cv_score_std=float(np.std(fold_scores)),
                         cv_bal_acc=float(np.mean([m["bal_acc"] for m in fold_metrics])), cv_auc=float(np.nanmean([m["auc"] for m in fold_metrics])),
                         seconds=round(time.time() - t0), oof_file=oof_path.name, config=json.dumps(cfg)))
        print(f"[g{a.gate} #{rank}] CV score {rows[-1]['cv_score_mean']:.4f} ± {rows[-1]['cv_score_std']:.4f} (search value {f['value']:.4f})", flush=True)
    out = pd.DataFrame(rows).sort_values("cv_score_mean", ascending=False)
    out.to_csv(fc.RESULTS / f"refit_g{a.gate}.csv", index=False)
    print(out.drop(columns=["config"]).to_string(index=False))


if __name__ == "__main__":
    main()
