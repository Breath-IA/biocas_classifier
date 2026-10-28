"""
final_test.py — the ONE evaluation of the chosen cascade on the class-balanced test set.

Retrains the chosen gate-1 and gate-2 configs on the whole training pool (seeds 0..S-1; seed 0 is the deployable pair, the others only give
variance), evaluates the cascade on balanced_test.csv, and writes: final_report.md, final_metrics.json, confusion_matrix.csv/.png, gate1.onnx,
gate2.onnx (+ the deployment block in cascade.json). Also: patient-clustered bootstrap CIs, per-year table, leave-one-year-out table on the
pool (test events are not used there), and the synthetic-pretraining ablation of gate 2. A lock file makes a second run fail unless --allow-retest.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import fm_common as fc
from cascade_eval import cascade_pred, macro_f1

NAMES = ["N", "W", "FC"]


def class_weights(y):
    w = np.zeros(len(y))
    for c in (0, 1, 2):
        if (y == c).any():
            w[y == c] = 1.0 / (y == c).sum()
    return w / max(w.sum(), 1e-12)


def summarize(y, pred, w=None):
    w = class_weights(y) if w is None else w
    cm = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cm[i, j] = w[(y == i) & (pred == j)].sum()
    rec = np.array([cm[i, i] / max(cm[i].sum(), 1e-12) for i in range(3)])
    prec = np.array([cm[i, i] / max(cm[:, i].sum(), 1e-12) for i in range(3)])
    f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-12), 0)
    return dict(macro_f1=float(f1.mean()), bal_acc=float(rec.mean()), sens=dict(zip(NAMES, rec.tolist())), precision=dict(zip(NAMES, prec.tolist())))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="full")
    ap.add_argument("--max-params", type=int, default=1_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--confirm", action="store_true", help="required: the test set is opened once")
    ap.add_argument("--allow-retest", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42, help="base seed for the nets (42 = the original sequence 0, 1, 2)")
    ap.add_argument("--epochs-max", type=int, default=60)
    ap.add_argument("--n-per-epoch", type=int, default=6000)
    ap.add_argument("--trial-max-min", type=float, default=15)
    ap.add_argument("--no-loyo", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cache-on-cpu", action="store_true")
    a = ap.parse_args()
    if not a.confirm:
        raise SystemExit("final_test.py opens the test set once: pass --confirm")
    lock = fc.RESULTS / "final_test.lock"
    if lock.exists() and not a.allow_retest:
        raise SystemExit(f"{lock} exists: the test set was already opened ({lock.read_text().strip()}). Use --allow-retest to repeat.")
    cas = json.loads((fc.RESULTS / "cascade.json").read_text())
    assert cas["gate1"]["params"] + cas["gate2"]["params"] < a.max_params, "chosen pair is over the parameter budget"

    torch = fc.import_torch()
    import train_lib as tl
    device = fc.resolve_device(torch, a.device)
    pool = fc.read_split("train_pool.csv")
    test = fc.read_split("balanced_test.csv")
    assert not set(pool["pid"]) & set(test["pid"])
    bundle = tl.Bundle(torch, device, a.cache_on_cpu)
    cfgs = {1: cas["gate1"]["config"], 2: cas["gate2"]["config"]}
    t1, t2 = cas["t1"], cas["t2"]

    def fit(gate, cfg, df, seed):
        cache = bundle.cache(cfg["band"])
        sub = tl.make_subset(df, gate, cache)
        tr, va = tl.split_inner(sub, 0.12, seed=seed)
        synth = bundle.synth(cfg["band"]) if (gate == 2 and cfg.get("synth_pretrain")) else None
        res = tl.train(torch, cfg, gate, tr, va, cache, device, seed=seed, amp=a.amp, synth=synth, exclude_pids=set(va.pid) | set(test["pid"]),
                       epochs_max=a.epochs_max, n_per_epoch=a.n_per_epoch, max_seconds=a.trial_max_min * 60, deterministic=True)
        return res.model

    def probs(model, cfg, df):
        cache = bundle.cache(cfg["band"])
        rows = np.array([cache.row.get(e, -1) for e in df["eid"]])
        out = np.full(len(df), np.nan, dtype=np.float32)
        ok = rows >= 0
        out[ok] = tl.predict(torch, model, cache.X, rows[ok], device)
        return out

    off = fc.seed_offset(a.seed)
    ytest = test["cls4"].to_numpy()
    per_seed, models0 = [], {}
    for s in range(a.seeds):
        m1, m2 = fit(1, cfgs[1], pool, s + off), fit(2, cfgs[2], pool, s + off)
        p1, p2 = probs(m1, cfgs[1], test), probs(m2, cfgs[2], test)
        ok = np.isfinite(p1) & np.isfinite(p2)
        pred = cascade_pred(p1, p2, t1, t2)
        r = summarize(ytest[ok], pred[ok])
        w12 = tl.eval_weights(ytest[ok], 1)
        r["gate1"] = tl.metrics((ytest[ok] != 0).astype(int), p1[ok], w12)
        g2 = ok & (ytest > 0)
        r["gate2"] = tl.metrics((ytest[g2] == 2).astype(int), p2[g2], tl.eval_weights(ytest[g2], 2))
        per_seed.append(dict(seed=s, **r, p1=p1, p2=p2, ok=ok))
        if s == 0:
            models0 = {1: m1, 2: m2}
        print(f"seed {s}: macro-F1 {r['macro_f1']:.3f} bal-acc {r['bal_acc']:.3f} | gate 1 AUC {r['gate1']['auc']:.3f} | gate 2 AUC {r['gate2']['auc']:.3f}", flush=True)

    head = per_seed[0]
    ok, pred = head["ok"], cascade_pred(head["p1"], head["p2"], t1, t2)
    yt, pt = ytest[ok], pred[ok]
    cm = np.array([[int(((yt == i) & (pt == j)).sum()) for j in range(3)] for i in range(3)])
    pd.DataFrame(cm, index=[f"true_{n}" for n in NAMES], columns=[f"pred_{n}" for n in NAMES]).to_csv(fc.RESULTS / "confusion_matrix.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.4, 4))
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(NAMES); ax.set_yticklabels(NAMES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Balanced test set")
        plt.tight_layout(); plt.savefig(fc.RESULTS / "confusion_matrix.png", dpi=120); plt.close()
    except Exception as e:  # noqa: BLE001
        print(f"confusion_matrix.png skipped: {e}")

    # ── patient-clustered bootstrap (resample test patients, class-balanced weights recomputed on each resample)
    tdf = test[ok].reset_index(drop=True)
    by_pid = {p: g.index.to_numpy() for p, g in tdf.groupby("pid")}
    pids = np.array(list(by_pid))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.bootstrap):
        idx = np.concatenate([by_pid[p] for p in rng.choice(pids, size=len(pids), replace=True)])
        if len(set(yt[idx])) < 3:
            continue
        r = summarize(yt[idx], pt[idx])
        boots.append([r["macro_f1"], r["bal_acc"], *r["sens"].values()])
    ci = np.percentile(np.array(boots), [2.5, 97.5], axis=0) if boots else np.full((2, 5), np.nan)
    labels = ["macro_f1", "bal_acc", "sens_N", "sens_W", "sens_FC"]
    cis = {k: [float(ci[0, i]), float(ci[1, i])] for i, k in enumerate(labels)}

    # ── per-year breakdown
    yrs = []
    for y in sorted(tdf["year"].unique()):
        m = (tdf["year"] == y).to_numpy()
        yrs.append(dict(year=y, n=int(m.sum()), **{k: v for k, v in summarize(yt[m], pt[m]).items() if k in ("macro_f1", "bal_acc")}))

    # ── leave-one-year-out on the pool (the held-out year is never trained on; test events are not used)
    loyo = []
    if not a.no_loyo:
        for y in sorted(pool["year"].unique()):
            tr_df, ho = pool[pool.year != y], pool[(pool.year == y) & (pool.cls4 < 3)]
            if ho.cls4.nunique() < 3:
                continue
            m1, m2 = fit(1, cfgs[1], tr_df, off), fit(2, cfgs[2], tr_df, off)
            q1, q2 = probs(m1, cfgs[1], ho), probs(m2, cfgs[2], ho)
            k = np.isfinite(q1) & np.isfinite(q2)
            r = summarize(ho.cls4.to_numpy()[k], cascade_pred(q1, q2, t1, t2)[k])
            loyo.append(dict(held_out=y, n=int(k.sum()), macro_f1=r["macro_f1"], bal_acc=r["bal_acc"]))
            print(f"LOYO {y}: macro-F1 {r['macro_f1']:.3f}", flush=True)

    # ── ablation: gate 2 with the synthetic pretraining flipped, same seeds, W vs FC test events
    abl = []
    if not a.no_ablation:
        g2m = test[test.cls4 > 0]
        for flag in (False, True):
            aucs, bas = [], []
            for s in range(a.seeds):
                if bool(cfgs[2].get("synth_pretrain")) == flag:
                    r = per_seed[s]["gate2"]
                else:
                    cfg = {**cfgs[2], "synth_pretrain": flag, "synth_ratio": cfgs[2].get("synth_ratio", 1.0)}
                    mdl = fit(2, cfg, pool, s + off)
                    p = probs(mdl, cfg, g2m)
                    r = tl.metrics((g2m.cls4.to_numpy() == 2).astype(int), np.nan_to_num(p, nan=0.5), tl.eval_weights(g2m.cls4.to_numpy(), 2))
                aucs.append(r["auc"]); bas.append(r["bal_acc"])
            abl.append(dict(synth_pretrain=flag, auc=float(np.mean(aucs)), bal_acc=float(np.mean(bas))))

    # ── deployable artefacts: seed-0 nets → ONNX, parameter budget re-asserted on the files
    deploy = {}
    for g in (1, 2):
        path = fc.RESULTS / f"gate{g}.onnx"
        n_mels = fc.BANDS[cfgs[g]["band"]]["n_mels"]
        fc.export_onnx(torch, models0[g], n_mels, path)
        deploy[f"gate{g}"] = dict(onnx=path.name, band=cfgs[g]["band"], input=[1, 1, n_mels, 95], onnx_params=fc.onnx_param_count(path))
    total = deploy["gate1"]["onnx_params"] + deploy["gate2"]["onnx_params"]
    assert total < a.max_params, f"exported ONNX files hold {total:,} parameters (cap {a.max_params:,})"
    cas["deployment"] = dict(**deploy, onnx_params_total=total, rule="N if p1 < t1; else FC if p2 >= t2; else W")
    fc.jdump(cas, fc.RESULTS / "cascade.json")

    seeds_tab = {k: [float(s[k]) for s in per_seed] for k in ("macro_f1", "bal_acc")}
    metrics = dict(n_test=int(ok.sum()), confusion_matrix_rows_true_cols_pred=cm.tolist(), classes=NAMES, headline_seed=0,
                   macro_f1=head["macro_f1"], bal_acc=head["bal_acc"], sensitivity=head["sens"], precision=head["precision"], ci95=cis,
                   gate1=head["gate1"], gate2=head["gate2"], seeds=seeds_tab, per_year=yrs, loyo=loyo, ablation_gate2_synth=abl,
                   thresholds=dict(t1=t1, t2=t2), onnx_params_total=total)
    fc.jdump(metrics, fc.RESULTS / "final_metrics.json")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))      # the test set counts as opened once the metrics exist

    f = lambda v: f"{v:.3f}"  # noqa: E731
    md = ["# Final report — balanced test set (opened once)", "",
          f"Cascade: gate 1 trial {cas['gate1']['trial']} + gate 2 trial {cas['gate2']['trial']} · {total:,} parameters in the ONNX files (cap {a.max_params:,}) · t1={t1:.2f}, t2={t2:.2f}", "",
          f"Test events: {int(ok.sum())} ({', '.join(f'{n} {int((yt == i).sum())}' for i, n in enumerate(NAMES))}) from {tdf.pid.nunique()} patients. Headline = seed 0 (deployable nets).", "",
          "## Confusion matrix (rows true, columns predicted)", "", "| | pred N | pred W | pred FC | sensitivity | precision |", "|---|---|---|---|---|---|"]
    for i, n in enumerate(NAMES):
        md.append(f"| true {n} | {cm[i, 0]} | {cm[i, 1]} | {cm[i, 2]} | {f(head['sens'][n])} | {f(head['precision'][n])} |")
    md += ["", f"**Macro-F1 {f(head['macro_f1'])}** [{f(cis['macro_f1'][0])}, {f(cis['macro_f1'][1])}] · **balanced accuracy {f(head['bal_acc'])}** [{f(cis['bal_acc'][0])}, {f(cis['bal_acc'][1])}] "
           f"(patient-clustered bootstrap, 95 % CI, {len(boots)} resamples)", "",
           f"Per-class sensitivity 95 % CI: " + ", ".join(f"{n} [{f(cis['sens_' + n][0])}, {f(cis['sens_' + n][1])}]" for n in NAMES), "",
           f"Across {a.seeds} seeds: macro-F1 {np.mean(seeds_tab['macro_f1']):.3f} ± {np.std(seeds_tab['macro_f1']):.3f}", "",
           f"Gate 1 (N vs abnormal): AUC {f(head['gate1']['auc'])}, balanced accuracy {f(head['gate1']['bal_acc'])} · Gate 2 (W vs FC): AUC {f(head['gate2']['auc'])}, balanced accuracy {f(head['gate2']['bal_acc'])}", "",
           "## Per year (test events)", "", "| year | n | macro-F1 | bal-acc |", "|---|---|---|---|"]
    md += [f"| {r['year']} | {r['n']} | {f(r['macro_f1'])} | {f(r['bal_acc'])} |" for r in yrs]
    md += ["", "## Leave-one-year-out (pool events of the held-out year; chosen configs, seed 0)", "", "| held-out | n | macro-F1 | bal-acc |", "|---|---|---|---|"]
    md += [f"| {r['held_out']} | {r['n']} | {f(r['macro_f1'])} | {f(r['bal_acc'])} |" for r in loyo] or ["| (skipped) | | | |"]
    md += ["", "## Ablation: gate 2 with / without synthetic pretraining (test W vs FC, mean over seeds)", "", "| synthetic pretraining | AUC | bal-acc |", "|---|---|---|"]
    md += [f"| {'yes' if r['synth_pretrain'] else 'no'} | {f(r['auc'])} | {f(r['bal_acc'])} |" for r in abl] or ["| (skipped) | | |"]
    (fc.RESULTS / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
