"""
CNNs pequenas para Normal vs anormal sobre espectrograma y sobre k-space (eda_kspace_binary.ipynb).

Entradas por rama ("kind"):
  spec         log-mel z-score (1, 64, 94)
  spec_demean  log-mel menos la media temporal por banda (= notch ku=0 en el dominio imagen)
  kmag         log(1+|rfft2|) con kv centrado (fftshift) y sin ku=0 (1, 64, 46): la imagen de k-space

La seleccion de epoca usa un 10% de las GRABACIONES del entrenamiento; el anio retenido nunca se mira.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import kspace_lib as kl

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AUG_SIGMA = 0.3          # amplitud del canal sintetico usado como augmentation (el de evaluacion es 0.5)


def make_inputs(X):
    """Entradas de cada tipo, forma (n, 1, H, W) float32."""
    kmag = np.fft.fftshift(kl.klogmag(X, True)[:, :, 1:], axes=1)
    return {"spec": X[:, None], "spec_demean": (X - X.mean(-1, keepdims=True))[:, None], "kmag": kmag[:, None].astype(np.float32)}


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class Branch(nn.Module):
    def __init__(self, widths=(16, 32, 64, 96)):
        super().__init__()
        layers, cin = [], 1
        for w in widths:
            layers.append(_block(cin, w)); cin = w
        self.net = nn.Sequential(*layers)
        self.out_dim = 2 * cin

    def forward(self, x):
        h = self.net(x)
        return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)       # avg + max pooling global


class Net(nn.Module):
    def __init__(self, n_branches):
        super().__init__()
        self.branches = nn.ModuleList([Branch() for _ in range(n_branches)])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(sum(b.out_dim for b in self.branches), 1)

    def forward(self, xs):
        return self.fc(self.drop(torch.cat([b(x) for b, x in zip(self.branches, xs)], 1))).squeeze(1)


def _augment(x, kind):
    # desplazamiento circular en tiempo + canal suave en frecuencia; |K| es invariante a ambos, asi que kmag no se aumenta
    if kind == "kmag":
        return x
    B, _, F, T = x.shape
    shift = torch.randint(0, T, (B,), device=x.device)
    idx = (torch.arange(T, device=x.device)[None, :] + shift[:, None]) % T
    x = x.gather(3, idx[:, None, None, :].expand(B, 1, F, T))
    if kind == "spec":
        f = torch.linspace(0, 1, F, device=x.device)[None, :]
        c = sum(torch.randn(B, 1, device=x.device) * AUG_SIGMA / math.sqrt(3)
                * torch.cos(2 * math.pi * q * f + torch.rand(B, 1, device=x.device) * 2 * math.pi) for q in (1, 2, 3))
        x = x + c[:, None, :, None]
        x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-6)
    return x


@torch.no_grad()
def _predict(model, xs, stats, bs=512):
    model.eval()
    out = []
    for i in range(0, len(xs[0]), bs):
        batch = [(torch.from_numpy(a[i:i + bs]).to(DEV) - m) / s for a, (m, s) in zip(xs, stats)]
        out.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(out)


def _wandb_run(project, group, name, config):
    """Abre una run de WandB; si falla (sin red / sin login) devuelve None y el entrenamiento sigue sin logging."""
    if not project:
        return None
    try:
        import wandb
        return wandb.init(project=project, group=group, name=name, config=config, reinit=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"[wandb] desactivado: {type(e).__name__}: {e}")
        return None


def fit_predict(kinds, train_xs, y, groups, test_sets, seed=0, epochs=30, bs=128, lr=2e-3, wd=1e-2, patience=8, run=None):
    """Entrena una CNN de len(kinds) ramas; devuelve {nombre: P(anormal)} para cada conjunto de test.
    Si `run` (wandb) viene dado, registra por epoca train/loss, val/auc y lr. Solo se mira el AUC de validacion interna."""
    torch.manual_seed(seed); np.random.seed(seed)
    itr, iva = next(GroupShuffleSplit(1, test_size=0.1, random_state=seed).split(y, y, groups))
    stats = [(float(a[itr[:2000]].mean()), float(a[itr[:2000]].std()) + 1e-6) for a in train_xs]
    gx = [torch.from_numpy(a[itr]).to(DEV) for a in train_xs]
    gy = torch.from_numpy(y[itr]).float().to(DEV)
    model = Net(len(kinds)).to(DEV)
    pos_w = torch.tensor((y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1), device=DEV, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(itr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    val_xs = [a[iva] for a in train_xs]
    best, best_auc, bad = None, -1, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(itr), device=DEV)
        ep_loss, n_b = 0.0, 0
        for i in range(0, len(itr), bs):
            b = perm[i:i + bs]
            if len(b) < 8:
                continue
            xb = [(_augment(x[b], k) - m) / s for x, k, (m, s) in zip(gx, kinds, stats)]
            loss = lossf(model(xb), gy[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            ep_loss += float(loss); n_b += 1
        auc = roc_auc_score(y[iva], _predict(model, val_xs, stats))
        if run is not None:
            run.log({"epoch": ep, "train/loss": ep_loss / max(n_b, 1), "val/auc": auc, "lr": sched.get_last_lr()[0]})
        if auc > best_auc:
            best_auc, best, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    return {name: _predict(model, xs, stats) for name, xs in test_sets.items()}, best_auc


def eval_cnn(d: kl.Data, inp, inp_chan, kinds, seeds=(0, 1), name=None, wandb_project=None, wandb_group=None, **kw):
    """
    Protocolo de kl.protocol_folds: 3 folds de desarrollo dentro de 2022-2024 (sin 2025) + una prueba final 2022-24 -> 2025.
    Devuelve (filas por fold/condicion con AUC promediado entre semillas, scores limpios promediados por muestra, mejor AUC de validacion interna).
    Con wandb_project: una run por (fold, semilla) con curvas por epoca y el AUC final, y una run resumen del modelo.
    """
    name = name or "+".join(kinds)
    cfg = dict(model=name, kinds=list(kinds), aug_sigma=AUG_SIGMA, device=DEV, **kw,
               protocol="dev: leave-one-train-year-out en 2022-24 | test: train 2022-24 -> 2025")
    rows, scores, vals = [], np.full(len(d.y), np.nan), []
    for split, hold, tr, te in kl.protocol_folds(d):
        per_seed = []
        for sd in seeds:
            run = _wandb_run(wandb_project, wandb_group, f"{name}|{split}{hold}|s{sd}", {**cfg, "split": split, "eval_year": hold, "seed": sd})
            out, vauc = fit_predict(kinds, [inp[k][tr] for k in kinds], d.yb[tr], d.groups[tr],
                                    {"limpio": [inp[k][te] for k in kinds], "canal": [inp_chan[k][te] for k in kinds]},
                                    seed=sd, run=run, **kw)
            per_seed.append(out); vals.append(vauc)
            if run is not None:                              # el AUC del anio evaluado se registra una sola vez, al final
                run.summary.update({"best_val_auc": vauc, f"{split}/auc_limpio": roc_auc_score(d.yb[te], out["limpio"]),
                                    f"{split}/auc_canal": roc_auc_score(d.yb[te], out["canal"])})
                run.finish()
        yte, yb = d.y[te], d.yb[te]
        scores[te] = np.mean([o["limpio"] for o in per_seed], 0)
        for cond in ("limpio", "canal"):
            rs = []
            for o in per_seed:
                s = o[cond]
                row = dict(auc=roc_auc_score(yb, s), bal_acc=balanced_accuracy_score(yb, s > 0.5))
                for k, c in enumerate(kl.CLASSES[1:], start=1):
                    m = (yte == 0) | (yte == k)
                    row[f"auc_{c}"] = roc_auc_score(yb[m], s[m]) if (yte == k).sum() > 5 else np.nan
                rs.append(row)
            rows.append(dict(split=split, hold=hold, cond=cond, **pd.DataFrame(rs).mean().to_dict()))
    res = pd.DataFrame(rows)
    run = _wandb_run(wandb_project, wandb_group, f"{name}|resumen", {**cfg, "seeds": list(seeds)})
    if run is not None:
        dev, test = res[(res.split == "dev") & (res.cond == "limpio")], res[(res.split == "test") & (res.cond == "limpio")].iloc[0]
        run.summary.update({"auc_dev_mean": dev.auc.mean(), "auc_dev_canal_mean": res[(res.split == "dev") & (res.cond == "canal")].auc.mean(),
                            **{f"auc_dev{h}": a for h, a in zip(dev.hold, dev.auc)}, "auc_2025": test.auc,
                            "auc_2025_N-FC": test.auc_FC, "auc_2025_N-W": test.auc_W, "auc_2025_N-WC": test.auc_WC})
        run.finish()
    return res, scores, float(np.mean(vals))
