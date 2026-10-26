"""
train_cnn_synth.py  (gate 2 · e05)
──────────────────────────────────
W contra FC with the e03 CNN (4 conv blocks on log-mel, SpecAugment) trained with and without SYNTHETIC events,
leave-one-year-out, several seeds. The CNN, the augmentation, the optimiser and the early-stopping rule are those of
e03 (imported), so the real-only condition is the e03 `cnn` baseline and every difference comes from the data.

Leakage guards
  * the held-out year is never used: synthetic events whose host recording is from that year are dropped;
  * early stopping uses REAL validation events only (15 % of the real training patients, as in e03); synthetic events
    whose host patient is in that validation set are dropped as well;
  * synthetic events are only ever added to the training part; the test year is always real and untouched.
Synthetic events per class = ratio × (size of the larger real training class), the same number for W and FC
(so that "synthetic" carries no class information), drawn at random from what is left after the guards.

Usage : python train_cnn_synth.py --name real_b2000                              (real only)
        python train_cnn_synth.py --name v1_r1_b2000 --synth v1 --ratio 1.0      (real + synthetic set v1)
        add --band 900 for the 60–900 Hz / 40-mel input
Output: results/<name>_results.csv, results/<name>_predictions.csv   (resumes finished runs)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

sys.path.insert(0, str(paths.G2_E03))
import train_cnn as e03  # noqa: E402  (CNN, augment, load_data: the e03 definitions)
import cnn_data as cd  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
RES, CACHE = HERE / "results", HERE / "cache"
YEARS = ["2022", "2023", "2024", "2025"]


def load_real(band: int):
    ev, X2000, _ = e03.load_data()                       # same rows, same order as e03
    y = (ev["class"] == "FC").values.astype(int)
    if band == 2000:
        return ev, X2000, y
    f = CACHE / "real_mel900.npz"
    if f.exists():
        X = np.load(f)["X"]
    else:
        X = cd.real_mels(ev, 900)
        np.savez(f, X=X)
    return ev, X, y


def load_synth(name: str, band: int):
    z = np.load(CACHE / f"synth_{name}.npz", allow_pickle=True)
    return {"X": z[f"mel{band}"], "y": z["y"].astype(int), "year": z["year"].astype(str), "patient": z["patient"].astype(str),
            "host_split": z["host_split"].astype(str)}


def run_one(ev, X, y, synth, H, seed, ratio, epochs, patience, mode="mix", pre_epochs=5, ft_lr=3e-4, lr0=1e-3, return_model=False):
    tr = (ev["source"] != H).values
    te = ~tr
    Xtr, ytr, gtr = X[tr], y[tr], ev["patient_id"].astype(str).values[tr]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr, ytr, gtr))
    X_train, y_train = Xtr[tr_i], ytr[tr_i]
    n_syn = 0
    X_syn = y_syn = None
    if synth is not None and ratio > 0:
        ok = (synth["year"] != H) & ~np.isin(synth["patient"], np.unique(gtr[va_i]))
        n_per = int(round(ratio * max((y_train == 0).sum(), (y_train == 1).sum())))
        srng = np.random.default_rng(seed + 1000)          # separate stream: the real-only runs are unchanged
        pick = []
        for c in (0, 1):
            ids = np.flatnonzero(ok & (synth["y"] == c))
            pick.append(srng.choice(ids, size=min(n_per, len(ids)), replace=False))
        pick = np.concatenate(pick)
        n_syn = len(pick)
        if mode == "pretrain":                             # synthetic events are used only to pretrain; fine-tuning sees real data only
            X_syn, y_syn = synth["X"][pick], synth["y"][pick]
        else:                                              # "mix": synthetic and real events in the same batches
            X_train = np.concatenate([X_train, synth["X"][pick]])
            y_train = np.concatenate([y_train, synth["y"][pick]])
    Xt = lambda a: torch.from_numpy(a)[:, None]  # noqa: E731
    Xa, Xv, Xe = Xt(X_train), Xt(Xtr[va_i]), Xt(X[te])
    ya = torch.from_numpy(y_train.astype(np.float32))
    pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = e03.CNN(0)
    lr = lr0
    if mode == "pretrain" and X_syn is not None:           # stage 1: synthetic only (balanced classes), no model selection on it
        prng = np.random.default_rng(seed + 2000)
        Xs_t, ys_t = Xt(X_syn), torch.from_numpy(y_syn.astype(np.float32))
        opt_p = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        loss_p = nn.BCEWithLogitsLoss()
        for _ in range(pre_epochs):
            model.train()
            perm_p = prng.permutation(len(Xs_t))
            for i in range(0, len(perm_p), 64):
                b = perm_p[i:i + 64]
                if len(b) < 4:
                    continue
                loss = loss_p(model(e03.augment(Xs_t[b], prng)), ys_t[b])
                opt_p.zero_grad()
                loss.backward()
                opt_p.step()
        lr = ft_lr                                          # stage 2: fine-tune on real data with a lower learning rate
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pw)))

    def predict(Xs):
        model.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(model(Xs[i:i + 256])).numpy() for i in range(0, len(Xs), 256)])

    best, best_state, bad, ep_done = -1, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(Xa))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            if len(b) < 4:
                continue
            loss = lossf(model(e03.augment(Xa[b], rng)), ya[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        va = roc_auc_score(ytr[va_i], predict(Xv))           # real validation events only
        ep_done = ep + 1
        if va > best + 1e-4:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    if return_model:                                        # used by export_best_onnx.py
        return predict(Xe), best, ep_done, n_syn, int(len(tr_i)), model
    return predict(Xe), best, ep_done, n_syn, int(len(tr_i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--synth", default=None, help="name of cache/synth_<name>.npz")
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--band", type=int, default=2000, choices=[2000, 900])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", default=YEARS)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--mode", default="mix", choices=["mix", "pretrain"], help="mix: synthetic in the training batches; pretrain: synthetic pretraining then real fine-tuning")
    ap.add_argument("--pre-epochs", type=int, default=5)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate of the standard (non-pretrain) training")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    ev, X, y = load_real(a.band)
    synth = load_synth(a.synth, a.band) if a.synth else None
    print(f"{a.name}: {len(ev)} real events, mel {X.shape[1:]}, synthetic: {None if synth is None else len(synth['y'])}", flush=True)

    rp, pp = RES / f"{a.name}_results.csv", RES / f"{a.name}_predictions.csv"
    results = pd.read_csv(rp).to_dict("records") if rp.exists() else []
    preds = [pd.read_csv(pp)] if pp.exists() else []
    done = {(str(r["fold"]), int(r["seed"])) for r in results}
    for H in a.folds:
        te = (ev["source"] == H).values
        for seed in a.seeds:
            if (H, seed) in done:
                continue
            t0 = time.time()
            p, best, ep, n_syn, n_real = run_one(ev, X, y, synth, H, seed, a.ratio, a.epochs, a.patience, a.mode, a.pre_epochs, a.ft_lr, a.lr)
            pred = (p >= .5).astype(int)
            r = {"name": a.name, "fold": H, "seed": seed, "epochs": ep, "best_val_auc": best, "auc": roc_auc_score(y[te], p),
                 "bal_acc": balanced_accuracy_score(y[te], pred), "macro_f1": f1_score(y[te], pred, average="macro"),
                 "rec_W": float(((pred == 0) & (y[te] == 0)).sum() / (y[te] == 0).sum()),
                 "rec_FC": float(((pred == 1) & (y[te] == 1)).sum() / (y[te] == 1).sum()),
                 "n_real_train": n_real, "n_synth": n_syn, "n_test": int(te.sum()), "secs": round(time.time() - t0, 1)}
            results.append(r)
            preds.append(pd.DataFrame({"stem": ev["stem"].values[te], "event_idx": ev["event_idx"].values[te], "fold": H, "seed": seed,
                                       "p": p, "y": y[te]}))
            pd.DataFrame(results).to_csv(rp, index=False)
            pd.concat(preds).to_csv(pp, index=False)
            print(f"[{a.name} {H} s{seed}] ep={ep} val={best:.3f} AUC={r['auc']:.3f} bal={r['bal_acc']:.3f} synth={n_syn} ({r['secs']:.0f}s)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
