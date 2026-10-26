"""
train_cnn.py  (gate 2 · e03)
────────────────────────────
W contra FC con una CNN de 4 bloques convolucionales sobre log-mel, validación dejando un año fuera (igual que e02):

  variante `cnn`       : solo log-mel (64 × 94, 3 s, tile como en pipeline.yaml) + SpecAugment      → baseline del plan (paso 1)
  variante `cnn+phys`  : CNN + rama MLP con las features físicas de plan.md (fusión por concatenación, paso 2/4)

Entradas : experiments/common/features/{segments_spectral.csv, physics_features.csv}  (common/physics_features.py)
Salidas  : results/{cnn_results.csv, cnn_history.json, cnn_predictions.csv, ph_importance.csv}; cache/mel_cache.npz

Early stopping con el 15 % de los PACIENTES de los años de entrenamiento (nunca se mira el año retenido).
Uso:  python experiments/gate2_event_type/e03_W_vs_FC_cnn_physics/train_cnn.py [--seeds 0 1 2] [--epochs 20] [--folds 2022 2023 2024 2025]   (reanuda lo ya hecho)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import extract_spectral_features as ex  # noqa: E402
import paths  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
RES = paths.G2_E03 / "results"
CACHE = paths.G2_E03 / "cache"       # mel_cache.npz (ignorado por git)
SR, DUR, N_FFT, HOP, N_MELS = ex.SR, 3.0, 512, 128, 64
MEL_FB = ex._mel_fb(n_mels=N_MELS)
YEARS = ["2022", "2023", "2024", "2025"]


# ── datos ─────────────────────────────────────────────────────────────────────
def to_mel(seg: np.ndarray) -> np.ndarray:
    L = int(DUR * SR)
    w = seg[(len(seg) - L) // 2:(len(seg) - L) // 2 + L] if len(seg) >= L else np.tile(seg, int(np.ceil(L / len(seg))))[:L]
    _, _, Z = signal.stft(w, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann", boundary="even", padded=True)
    D = 10 * np.log10(np.maximum(MEL_FB @ (np.abs(Z) ** 2), 1e-10))
    return ((D - D.mean()) / (D.std() + 1e-6)).astype(np.float32)               # normalización por instancia


def mel_job(stem, split, events):
    try:
        x = ex.load_wav(ex.RAW / split / "wav" / f"{stem}.wav")
    except Exception:
        return []
    return [(k, to_mel(x[int(s * SR / 1000):int(e * SR / 1000)])) for k, s, e in events
            if int(e * SR / 1000) - int(s * SR / 1000) >= 64]


def load_data():
    ev = pd.read_csv(paths.SEGMENTS_CSV,
                     usecols=["stem", "event_idx", "class", "source", "split", "patient_id", "start_ms", "end_ms", "duration_s"])
    ev = ev[ev["class"].isin(["W", "FC"])].reset_index(drop=True)
    ev["source"] = ev["source"].astype(str)
    ev["patient_id"] = ev["patient_id"].astype(str)
    cache = CACHE / "mel_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        keys, X = list(z["keys"]), z["X"]
    else:
        jobs = [(s, g["split"].iloc[0], list(zip(g.event_idx, g.start_ms, g.end_ms))) for s, g in ev.groupby("stem")]
        res = Parallel(n_jobs=4, verbose=1)(delayed(mel_job)(*j) for j in jobs)
        keys, arrs = [], []
        for (s, _, _), r in zip(jobs, res):
            for k, m in r:
                keys.append(f"{s}|{k}")
                arrs.append(m)
        X = np.stack(arrs)
        np.savez(cache, keys=np.array(keys), X=X)
    pos = {k: i for i, k in enumerate(keys)}
    ev["key"] = ev["stem"] + "|" + ev["event_idx"].astype(str)
    ev = ev[ev["key"].isin(pos)].reset_index(drop=True)
    X = X[[pos[k] for k in ev["key"]]]
    ph = pd.read_csv(paths.PHYSICS_CSV)
    ev = ev.merge(ph, on=["stem", "event_idx"], how="left")
    pcols = [c for c in ev.columns if c.startswith("ph_")]
    return ev, X, pcols


# ── modelo ────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    """4 bloques Conv3×3-BN-ReLU-MaxPool → pooling global → (opcional) concatena rama de features físicas → MLP."""

    def __init__(self, n_ph=0, ch=(16, 32, 64, 128), drop=0.3):
        super().__init__()
        blocks, c_in = [], 1
        for c in ch:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.conv, self.pool, self.n_ph = nn.Sequential(*blocks), nn.AdaptiveAvgPool2d(1), n_ph
        if n_ph:
            self.ph = nn.Sequential(nn.Linear(n_ph, 32), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(ch[-1] + (32 if n_ph else 0), 64), nn.ReLU(), nn.Linear(64, 1))

    def embed(self, x):
        return self.pool(self.conv(x)).flatten(1)

    def classify(self, e, ph=None):
        if self.n_ph:
            e = torch.cat([e, self.ph(ph)], 1)
        return self.head(e).squeeze(1)

    def forward(self, x, ph=None):
        return self.classify(self.embed(x), ph)


def augment(x, rng):
    """Desplazamiento circular en el tiempo (el evento está tileado) + 1 máscara de frecuencia + 1 de tiempo (SpecAugment)."""
    out = x.clone()
    _, _, F, T = out.shape
    for i in range(out.shape[0]):
        out[i] = torch.roll(out[i], int(rng.integers(0, T)), dims=-1)
        f = int(rng.integers(0, 9)); f0 = int(rng.integers(0, F - f))
        out[i, :, f0:f0 + f] = 0
        t = int(rng.integers(0, 11)); t0 = int(rng.integers(0, T - t))
        out[i, :, :, t0:t0 + t] = 0
    return out


def prep_ph(Ptr, others):
    imp = SimpleImputer(strategy="median").fit(Ptr)
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit(imp.transform(Ptr))
    f = lambda P: np.clip(qt.transform(imp.transform(P)), -4, 4).astype(np.float32)  # noqa: E731
    return f(Ptr), [f(P) for P in others]


def run_one(Xtr, ytr, Ptr, gtr, Xte, yte, Pte, use_ph, seed, epochs, patience, want_imp):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr, ytr, gtr))
    if use_ph:
        Ptr_n, (Pte_n,) = prep_ph(Ptr[tr_i], [Pte])
        Pva_n = prep_ph(Ptr[tr_i], [Ptr[va_i]])[1][0]
        Pt_tr, Pt_va, Pt_te = map(torch.from_numpy, (Ptr_n, Pva_n, Pte_n))
    else:
        Pt_tr = Pt_va = Pt_te = None
    Xt = lambda a: torch.from_numpy(a)[:, None]  # noqa: E731
    Xa, Xv, Xe = Xt(Xtr[tr_i]), Xt(Xtr[va_i]), Xt(Xte)
    ya = torch.from_numpy(ytr[tr_i].astype(np.float32))
    pw = (ytr[tr_i] == 0).sum() / max((ytr[tr_i] == 1).sum(), 1)
    model = CNN(Ptr.shape[1] if use_ph else 0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pw)))

    def predict(X, P):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                out.append(torch.sigmoid(model(X[i:i + 256], None if P is None else P[i:i + 256])).numpy())
        return np.concatenate(out)

    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(Xa))
        tl = 0.0
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            if len(b) < 4:
                continue
            xb = augment(Xa[b], rng)
            loss = lossf(model(xb, None if Pt_tr is None else Pt_tr[b]), ya[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(b)
        va = roc_auc_score(ytr[va_i], predict(Xv, Pt_va))
        hist.append({"epoch": ep + 1, "train_loss": tl / len(perm), "val_auc": float(va)})
        if va > best + 1e-4:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    p = predict(Xe, Pt_te)
    imp_rows = None
    if want_imp and use_ph:                       # permutación de cada feature física sobre el embebido congelado
        model.eval()
        with torch.no_grad():
            emb = torch.cat([model.embed(Xe[i:i + 256]) for i in range(0, len(Xe), 256)])
        base = roc_auc_score(yte, p)
        imp_rows = []
        for j in range(Pt_te.shape[1]):
            d = []
            for r in range(3):
                Pp = Pt_te.clone()
                Pp[:, j] = Pp[rng.permutation(len(Pp)), j]
                with torch.no_grad():
                    d.append(base - roc_auc_score(yte, torch.sigmoid(model.classify(emb, Pp)).numpy()))
            imp_rows.append(float(np.mean(d)))
    return p, hist, best, imp_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--folds", nargs="+", default=YEARS)
    ap.add_argument("--variants", nargs="+", default=["cnn", "cnn+phys"])
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    ev, X, pcols = load_data()
    y = (ev["class"] == "FC").values.astype(int)
    P = ev[pcols].replace([np.inf, -np.inf], np.nan).values.astype(np.float64)
    print(f"{len(ev)} eventos | mel {X.shape} | {len(pcols)} features físicas", flush=True)

    rp, hp, ip, pp = RES / "cnn_results.csv", RES / "cnn_history.json", RES / "ph_importance.csv", RES / "cnn_predictions.csv"
    results = pd.read_csv(rp).to_dict("records") if rp.exists() else []
    hists = json.loads(hp.read_text()) if hp.exists() else {}
    imps = pd.read_csv(ip).to_dict("records") if ip.exists() else []
    preds = [pd.read_csv(pp)] if pp.exists() else []
    done = {(str(r["fold"]), r["variant"], int(r["seed"])) for r in results}   # fold se lee como int del CSV

    for H in a.folds:
        te = (ev["source"] == H).values
        tr = ~te
        for seed in a.seeds:
            for var in a.variants:
                if (H, var, seed) in done:
                    continue
                t0 = time.time()
                p, hist, best, imp = run_one(X[tr], y[tr], P[tr], ev["patient_id"].values[tr], X[te], y[te], P[te],
                                             var == "cnn+phys", seed, a.epochs, a.patience, seed == a.seeds[0])
                pred = (p >= .5).astype(int)
                r = {"fold": H, "variant": var, "seed": seed, "epochs": len(hist), "best_val_auc": best,
                     "auc": roc_auc_score(y[te], p), "bal_acc": balanced_accuracy_score(y[te], pred),
                     "macro_f1": f1_score(y[te], pred, average="macro"),
                     "rec_W": float(((pred == 0) & (y[te] == 0)).sum() / (y[te] == 0).sum()),
                     "rec_FC": float(((pred == 1) & (y[te] == 1)).sum() / (y[te] == 1).sum()),
                     "n_train": int(tr.sum()), "n_test": int(te.sum()), "secs": round(time.time() - t0, 1)}
                results.append(r)
                hists[f"{H}|{var}|{seed}"] = hist
                if imp is not None:
                    imps += [{"fold": H, "feature": c, "drop_auc": v} for c, v in zip(pcols, imp)]
                preds.append(pd.DataFrame({"stem": ev["stem"].values[te], "event_idx": ev["event_idx"].values[te],
                                           "fold": H, "variant": var, "seed": seed, "p": p, "y": y[te]}))
                pd.DataFrame(results).to_csv(rp, index=False)
                hp.write_text(json.dumps(hists))
                pd.DataFrame(imps).to_csv(ip, index=False)
                pd.concat(preds).to_csv(pp, index=False)
                print(f"[{H} {var} s{seed}] ep={len(hist)} val={best:.3f} AUC={r['auc']:.3f} bal={r['bal_acc']:.3f} "
                      f"recW={r['rec_W']:.2f} recFC={r['rec_FC']:.2f} ({r['secs']:.0f}s)", flush=True)
    print("TERMINADO")


if __name__ == "__main__":
    main()
