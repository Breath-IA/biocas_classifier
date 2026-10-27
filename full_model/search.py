"""
search.py — resumable Optuna search for one gate (1: N vs abnormal, 2: W vs FC) under a parameter cap and a wall-clock budget.

Objective: weighted balanced accuracy (+ 1e-3 AUC) on the validation split. Configs above --max-params-net are pruned before training.
Stops on --hours, --max-trials, or the file full_model/STOP. Every finished trial appends a row to results/<study>_trials.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time

import optuna

import fm_common as fc
import search_space as ss

DEFAULTS = dict(epochs_max=40, n_per_epoch=6000)


def trial_row(t) -> dict:
    return dict(number=t.number, state=t.state.name, value=t.value, params=t.user_attrs.get("params"), params_folded=t.user_attrs.get("params_folded"),
                macs=t.user_attrs.get("macs"), seconds=t.user_attrs.get("seconds"), epochs=t.user_attrs.get("epochs"),
                metrics=json.dumps(t.user_attrs.get("metrics", {})), config=json.dumps(t.user_attrs.get("config", {})))


def append_csv(path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def make_study(a):
    url = fc.storage_url(a.storage, a.study)
    try:                                                  # resumed study: do not replay the random sequence of the first run
        resumed = len(optuna.load_study(study_name=a.study, storage=url).trials)
    except KeyError:
        resumed = 0
    seed = a.seed + resumed
    sampler = optuna.samplers.RandomSampler(seed=seed) if a.sampler == "random" else optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    pruner = {"hyperband": lambda: optuna.pruners.HyperbandPruner(min_resource=3, max_resource=a.epochs_max, reduction_factor=3),
              "median": lambda: optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3),
              "none": lambda: optuna.pruners.NopPruner()}[a.pruner]()
    return optuna.create_study(study_name=a.study, storage=url, sampler=sampler, pruner=pruner,
                               direction="maximize", load_if_exists=True)


def status(a) -> None:
    try:
        study = optuna.load_study(study_name=a.study, storage=fc.storage_url(a.storage, a.study))
    except Exception as e:  # noqa: BLE001
        print(f"{a.study}: no study yet ({type(e).__name__})")
        return
    ts = study.trials
    n = {s: sum(t.state.name == s for t in ts) for s in ("COMPLETE", "PRUNED", "RUNNING", "FAIL")}
    done = [t for t in ts if t.state.name == "COMPLETE"]
    secs = [t.user_attrs.get("seconds", 0) for t in done]
    print(f"{a.study}: {len(ts)} trials {n}" + (f" · best {study.best_value:.4f} (trial {study.best_trial.number}, "
          f"{study.best_trial.user_attrs.get('params'):,} params) · mean {sum(secs) / max(len(secs), 1):.0f} s/complete trial" if done else ""))


def run(a) -> None:
    torch = fc.import_torch()
    import train_lib as tl
    from models_small import build_model, count_params, profile
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = fc.resolve_device(torch, a.device)
    study = make_study(a)
    pool = fc.read_split("train_pool.csv")
    bundle = tl.Bundle(torch, device, a.cache_on_cpu)
    csv_path = fc.RESULTS / f"{a.study}_trials.csv"
    wb = None
    if a.wandb:
        import wandb
        wb = wandb

    t_start, deadline = time.time(), time.time() + a.hours * 3600
    durations, n_done, fails = [], 0, 0
    print(f"[{a.study}] gate {a.gate} · device {device} · {a.hours} h · cap {a.max_params_net:,} params · pool {len(pool)} events", flush=True)

    last, prev_n = {}, -1

    def objective(trial):
        last["n"] = trial.number
        cfg = ss.sample_config(trial, a.gate, no_se=a.no_se, synth=a.synth_pretrain)
        trial.set_user_attr("config", cfg)
        n_mels = fc.BANDS[cfg["band"]]["n_mels"]
        raw, folded = count_params(build_model(cfg, n_mels))
        prof = profile(build_model(cfg, n_mels), (n_mels, 95))
        trial.set_user_attr("params", raw)
        trial.set_user_attr("params_folded", folded)
        trial.set_user_attr("macs", prof["macs"])
        if raw > a.max_params_net:
            trial.set_user_attr("pruned_reason", "over parameter cap")
            raise optuna.TrialPruned()
        cache = bundle.cache(cfg["band"])
        tr = tl.make_subset(pool[pool.role == "train"], a.gate, cache)
        va = tl.make_subset(pool[pool.role == "val"], a.gate, cache)
        excl = set(pool[pool.role == "val"]["pid"])                       # test patients are not in the pool at all
        synth = bundle.synth(cfg["band"]) if (a.gate == 2 and cfg.get("synth_pretrain")) else None

        def report(ep, score):
            trial.report(score, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        remaining = deadline - time.time()
        res = tl.train(torch, cfg, a.gate, tr, va, cache, device, seed=a.seed + trial.number, amp=a.amp, report=report, synth=synth,
                       exclude_pids=excl, epochs_max=a.epochs_max, n_per_epoch=a.n_per_epoch,
                       max_seconds=min(a.trial_max_min * 60, max(remaining, 30)))
        trial.set_user_attr("seconds", round(res.seconds, 1))
        trial.set_user_attr("epochs", res.epochs)
        trial.set_user_attr("metrics", res.metrics)
        if wb is not None:
            wb.init(project="biocas-full-model", name=f"{a.study}-{trial.number}", config=cfg, reinit=True)
            wb.log({**res.metrics, "params": raw})
            wb.finish()
        return res.score

    while True:
        if fc.STOP_FILE.exists():
            print(f"STOP file found: stopping after {n_done} trials (delete {fc.STOP_FILE} before the next search)", flush=True)
            break
        remaining = deadline - time.time()
        est = max(durations[-5:]) if durations else 0.0
        if remaining <= 0 or remaining < min(est, a.trial_max_min * 60) * 0.5:
            print("time budget used up", flush=True)
            break
        if a.max_trials and len(study.trials) >= a.max_trials:
            print(f"--max-trials {a.max_trials} reached", flush=True)
            break
        t0 = time.time()
        before = len(study.trials)
        try:
            study.optimize(objective, n_trials=1, catch=())
            fails = 0
        except KeyboardInterrupt:
            print("interrupted: the study is saved; run the same command to resume", flush=True)
            break
        except Exception as e:  # noqa: BLE001 — a failing trial must not end a 12 h run, but a broken setup must
            fails += 1
            print(f"trial failed: {type(e).__name__}: {e}", flush=True)
            if fails >= 5:
                raise SystemExit("5 consecutive failed trials: aborting (see the error above)")
        if "n" not in last or last["n"] == prev_n:
            continue
        prev_n = last["n"]
        t = study.get_trials(deepcopy=False)[last["n"]]
        n_done += 1
        dt = time.time() - t0
        if t.state.name == "COMPLETE":
            durations.append(dt)
        append_csv(csv_path, trial_row(t))
        best = f"{study.best_value:.4f}" if any(x.state.name == "COMPLETE" for x in study.trials) else "-"
        print(f"[{a.study}] trial {t.number} {t.state.name:8s} value {t.value if t.value is not None else float('nan'):.4f} params {t.user_attrs.get('params', 0):>8,} "
              f"{dt:5.0f}s | best {best} | {len(study.trials)} trials | {max(deadline - time.time(), 0) / 3600:.2f} h left", flush=True)
    secs = [d for d in durations]
    if secs:
        print(f"seconds per complete trial: mean {sum(secs) / len(secs):.1f} (use trials ~= 3600 x hours / s)", flush=True)
    print(f"total {time.time() - t_start:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", type=int, choices=(1, 2))
    ap.add_argument("--study")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--hours", type=float, default=5)
    ap.add_argument("--max-trials", type=int, default=0)
    ap.add_argument("--trial-max-min", type=float, default=15)
    ap.add_argument("--max-params-net", type=int, default=500_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--pruner", choices=("hyperband", "median", "none"), default="hyperband")
    ap.add_argument("--sampler", choices=("tpe", "random"), default="tpe")
    ap.add_argument("--synth-pretrain", choices=("auto", "0", "1"), default="auto")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-se", action="store_true")
    ap.add_argument("--cache-on-cpu", action="store_true")
    ap.add_argument("--wandb", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs-max", type=int, default=DEFAULTS["epochs_max"])
    ap.add_argument("--n-per-epoch", type=int, default=DEFAULTS["n_per_epoch"], help="samples drawn per epoch (fixed, so trial time is predictable)")
    a = ap.parse_args()
    if not a.study:
        ap.error("--study is required")
    if a.status:
        status(a)
        return
    if a.gate is None:
        ap.error("--gate is required")
    if a.n_jobs > 1:                                              # extra worker processes share the SQLite study
        worker_args = [x for x in sys.argv[1:]]
        i = worker_args.index("--n-jobs")
        worker_args[i + 1] = "1"
        procs = [subprocess.Popen([sys.executable, __file__, *worker_args, "--seed", str(a.seed + 1000 * k)]) for k in range(1, a.n_jobs)]
        try:
            run(a)
        finally:
            for p in procs:
                p.wait()
        return
    run(a)


if __name__ == "__main__":
    main()
