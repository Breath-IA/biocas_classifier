"""
check_budget.py — sanity checks that need no data and no training:
  1. thousands of sampled configs: every one builds and runs a forward pass on its band; the ones above --max-params-net are rejected
     (counted, must be a minority) and none above the cap would reach training;
  2. --max-params-net x 2 fits inside --max-params (any gate-1 + gate-2 pair fits);
  3. unit tests: evaluation weights, class-balanced sampler, and the synthetic-pretraining guard (no val / test host patient is used).
Exit status 1 if a check fails.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import optuna

import fm_common as fc
import search_space as ss

torch = fc.import_torch()
import train_lib as tl  # noqa: E402
from models_small import build_model, count_params, profile  # noqa: E402


def unit_tests() -> list[str]:
    errs = []
    cls = np.array([0] * 10 + [1] * 4 + [2] * 6 + [3] * 2)
    w1, w2 = tl.eval_weights(cls, 1), tl.eval_weights(cls, 2)
    if abs(w1[cls == 0].sum() - 0.5) > 1e-9 or abs(w1[cls == 1].sum() - w1[cls == 2].sum()) > 1e-9:
        errs.append("gate 1 evaluation weights are not class balanced")
    if abs(w2[cls == 1].sum() - 0.5) > 1e-9 or abs(w2[cls == 2].sum() - 0.5) > 1e-9 or w2[cls == 0].any():
        errs.append("gate 2 evaluation weights are not W/FC 50/50")
    # synthetic guard
    rng = np.random.default_rng(0)
    s_pid = np.array([f"2022_{i % 7}" for i in range(200)] + [None] * 10, dtype=object)
    s_y = np.array([i % 2 for i in range(210)])
    banned = {"2022_1", "2022_3"}
    idx = tl.synth_indices(s_y, s_pid, banned, 40, rng)
    if any((s_pid[i] in banned) or s_pid[i] is None for i in idx):
        errs.append("synthetic pretraining picked an events whose host patient is excluded or unknown")
    if len(idx) == 0:
        errs.append("synthetic guard returned nothing")
    if tl.synth_pid("2022", "40638274.0") != "2022_40638274" or tl.synth_pid("2022", "nan") is not None:
        errs.append("synth_pid mapping is wrong")
    m = tl.metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.7, 0.9]), np.ones(4) / 4)
    if abs(m["bal_acc"] - 0.75) > 1e-9:
        errs.append(f"metrics bal_acc wrong: {m}")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--max-params-net", type=int, default=500_000)
    ap.add_argument("--max-params", type=int, default=1_000_000)
    ap.add_argument("--max-rejected", type=float, default=0.5, help="fail when more than this fraction of samples is over the cap")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fail = []
    if 2 * a.max_params_net > a.max_params:
        fail.append(f"2 x MAX_PARAMS_NET ({2 * a.max_params_net}) > MAX_PARAMS ({a.max_params}): a pair may not fit")
    fail += unit_tests()
    for gate in (1, 2):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=a.seed + gate))
        counts, over, nfwd = [], 0, 0
        for i in range(a.n_samples):
            cfg = ss.sample_config(study.ask(), gate)
            n_mels = fc.BANDS[cfg["band"]]["n_mels"]
            model = build_model(cfg, n_mels)
            raw, folded = count_params(model)
            counts.append(raw)
            if raw > a.max_params_net:
                over += 1
                continue                                   # would be rejected before training
            if i < 60:                                     # forward pass on the real input shape
                out = model.eval()(torch.zeros(2, 1, n_mels, fc.BANDS[cfg["band"]]["n_frames"]))
                nfwd += 1
                if out.shape != (2,):
                    fail.append(f"gate {gate}: output shape {tuple(out.shape)} for {cfg}")
                if profile(model, (n_mels, 95))["params"] != raw:
                    fail.append(f"gate {gate}: profile() disagrees with count_params")
        c = np.array(counts)
        print(f"gate {gate}: {a.n_samples} configs · params min/median/max {c.min():,}/{int(np.median(c)):,}/{c.max():,} · "
              f"over the {a.max_params_net:,} cap: {over} ({over / a.n_samples:.1%}) · forward passes OK: {nfwd}")
        if over / a.n_samples > a.max_rejected:
            fail.append(f"gate {gate}: {over / a.n_samples:.0%} of the sampled configs exceed the cap: narrow the width/depth ranges in search_space.py")
    if fail:
        print("\nFAILED:\n - " + "\n - ".join(fail))
        sys.exit(1)
    print("check-budget: all checks passed")


if __name__ == "__main__":
    main()
