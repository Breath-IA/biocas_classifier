"""
train_lib.py — GPU-resident data, batched augmentation, class-balanced sampling, one training routine and the balanced metrics.

Gate 1: y = 1 for every abnormal event (W, FC, other), 0 for N.   Gate 2: W = 0, FC = 1 (events of any other class are ignored).
Validation / OOF / test metrics are *weighted* so that the classes count equally (the same quantity that is reported on the balanced test):
gate 1 → N 50 %, abnormal 50 % split evenly over the abnormal classes present; gate 2 → W 50 %, FC 50 %.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import fm_common as fc
from models_small import build_model, count_params


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────
class MelCache:
    """cache/events_mel_<band>.npz on the device (float16)."""

    def __init__(self, torch, band: int, device: str, on_cpu: bool = False):
        p = fc.CACHE / f"events_mel_{band}.npz"
        if not p.exists():
            raise SystemExit(f"{p} not found: run `make cache` first.")
        z = np.load(p, allow_pickle=False)
        self.band, self.eid = band, z["eid"]
        self.row = {e: i for i, e in enumerate(self.eid)}
        self.n_mels = z["mel"].shape[1]
        dev = "cpu" if on_cpu else device
        self.X = torch.from_numpy(z["mel"]).to(dev)


class Bundle:
    """Lazy per-band caches plus the synthetic set."""

    def __init__(self, torch, device: str, on_cpu: bool = False):
        self.torch, self.device, self.on_cpu, self._c, self._s = torch, device, on_cpu, {}, {}

    def cache(self, band: int) -> MelCache:
        if band not in self._c:
            self._c[band] = MelCache(self.torch, band, self.device, self.on_cpu)
        return self._c[band]

    def synth(self, band: int) -> dict:
        if band not in self._s:
            if not fc.SYNTH_NPZ.exists():
                raise SystemExit(f"{fc.SYNTH_NPZ} not found: run `make synth` (or use --synth-pretrain 0).")
            z = np.load(fc.SYNTH_NPZ, allow_pickle=True)
            years, pats = z["year"].astype(str), z["patient"].astype(str)
            self._s[band] = dict(X=self.torch.from_numpy(z[f"mel{band}"]).to("cpu" if self.on_cpu else self.device).half(),
                                 y=z["y"].astype(int), pid=np.array([synth_pid(y, p) for y, p in zip(years, pats)], dtype=object),
                                 year=years)
        return self._s[band]


def synth_pid(year: str, patient: str):
    """Host patient of a synthetic event in the pid format of fm_common.patient_key; None when the host id is unknown."""
    try:
        return f"{year}_{int(float(patient))}"
    except ValueError:
        return None


def synth_indices(s_y, s_pid, exclude_pids: set, n_per_class: int, rng) -> np.ndarray:
    """Synthetic rows allowed for pretraining: host patient known and NOT in `exclude_pids` (validation / test / held-out patients)."""
    ok = np.array([(p is not None) and (p not in exclude_pids) for p in s_pid])
    pick = []
    for c in (0, 1):
        idx = np.flatnonzero(ok & (s_y == c))
        if len(idx):
            pick.append(rng.choice(idx, size=min(n_per_class, len(idx)), replace=False))
    return np.concatenate(pick) if pick else np.zeros(0, dtype=int)


@dataclass
class Subset:
    rows: np.ndarray      # row in the mel cache
    y: np.ndarray         # float32 binary label of the gate
    cls: np.ndarray       # cls4 (0 N, 1 W, 2 FC, 3 other)
    pid: np.ndarray
    pos: np.ndarray       # position in the source DataFrame (to map predictions back)
    w: np.ndarray         # evaluation weights (classes count equally), sum 1
    gate: int = 1

    def __len__(self):
        return len(self.rows)


def eval_weights(cls: np.ndarray, gate: int) -> np.ndarray:
    w = np.zeros(len(cls))
    if gate == 1:
        groups = {0: 0.5}
        ab = [c for c in (1, 2, 3) if (cls == c).any()]
        groups.update({c: 0.5 / len(ab) for c in ab})
    else:
        groups = {1: 0.5, 2: 0.5}
    for c, share in groups.items():
        m = cls == c
        if m.any():
            w[m] = share / m.sum()
    return w / max(w.sum(), 1e-12)


def make_subset(df: pd.DataFrame, gate: int, cache: MelCache) -> Subset:
    d = df.reset_index(drop=True)
    keep = d["eid"].isin(cache.row).to_numpy()
    if gate == 2:
        keep &= d["cls4"].isin([1, 2]).to_numpy()
    pos = np.flatnonzero(keep)
    d = d.iloc[pos]
    cls = d["cls4"].to_numpy()
    y = (cls != 0).astype(np.float32) if gate == 1 else (cls == 2).astype(np.float32)
    return Subset(rows=np.array([cache.row[e] for e in d["eid"]], dtype=np.int64), y=y, cls=cls, pid=d["pid"].to_numpy(), pos=pos,
                  w=eval_weights(cls, gate), gate=gate)


def sampling_probs(sub: Subset, gate: int) -> np.ndarray:
    """Class-balanced sampler (gate 1: 50 % N, 50 % abnormal split over its classes; gate 2: 50/50); N events are damped by 1/sqrt(events of
    their patient) so that a few long recordings do not dominate."""
    p = eval_weights(sub.cls, gate)
    nm = sub.cls == 0
    if nm.any():
        counts = pd.Series(sub.pid[nm]).map(pd.Series(sub.pid[nm]).value_counts()).to_numpy()
        q = p[nm] / np.sqrt(counts)
        p[nm] = q / q.sum() * p[nm].sum()
    return p / p.sum()


def split_inner(sub: Subset, frac: float, seed: int):
    """Patient-grouped split of a subset into (train, early-stop) subsets."""
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(set(sub.pid)))
    for _ in range(50):
        v = set(rng.choice(pids, size=max(1, int(round(frac * len(pids)))), replace=False))
        m = np.array([p in v for p in sub.pid])
        if len(set(sub.y[m])) == 2 and len(set(sub.y[~m])) == 2:
            break
    return take(sub, ~m), take(sub, m)


def take(sub: Subset, mask) -> Subset:
    cls = sub.cls[mask]
    return Subset(sub.rows[mask], sub.y[mask], cls, sub.pid[mask], sub.pos[mask], eval_weights(cls, sub.gate), sub.gate)


# ── metrics ─────────────────────────────────────────────────────────────────────────────────────────────
def metrics(y, p, w, thr: float = 0.5) -> dict:
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w)
    pred = (p >= thr).astype(int)
    pos, neg = w[y == 1].sum(), w[y == 0].sum()
    sens = float(w[(y == 1) & (pred == 1)].sum() / pos) if pos > 0 else float("nan")
    spec = float(w[(y == 0) & (pred == 0)].sum() / neg) if neg > 0 else float("nan")
    auc = float(roc_auc_score(y, p, sample_weight=w)) if len(set(y.tolist())) == 2 else float("nan")
    f1 = float(f1_score(y, pred, average="macro", sample_weight=w, zero_division=0))
    return dict(bal_acc=float(np.nanmean([sens, spec])), auc=auc, macro_f1=f1, sens=sens, spec=spec)


def score_of(m: dict) -> float:
    return m["bal_acc"] + 1e-3 * (0.0 if np.isnan(m["auc"]) else m["auc"])


# ── GPU augmentation ────────────────────────────────────────────────────────────────────────────────────
def augment(torch, x, cfg: dict, gen=None):
    """x: (B, F, T) float → (B, 1, F, T). Circular time shift (events are tiled to 3 s) + frequency / time masks."""
    B, F, T = x.shape
    dev = x.device
    shift = torch.randint(0, T, (B, 1), device=dev)
    x = x.gather(2, ((torch.arange(T, device=dev)[None] + shift) % T)[:, None, :].expand(B, F, T))[:, None]
    for key, size, dim_n, shape in (("specaug_f", F, F, (1, 1, F, 1)), ("specaug_t", T, T, (1, 1, 1, T))):
        mx = int(cfg[key])
        if mx > 0:
            width = torch.randint(0, mx + 1, (B, 1, 1, 1), device=dev)
            start = (torch.rand(B, 1, 1, 1, device=dev) * (dim_n - width).clamp(min=1)).long()
            ar = torch.arange(dim_n, device=dev).view(shape)
            x = x.masked_fill((ar >= start) & (ar < start + width), 0.0)
    return x


def _loss_fn(torch, cfg):
    kind, g, ls = cfg["loss"], float(cfg["focal_gamma"]), float(cfg["label_smooth"])

    def fn(logit, t):
        if kind == "smooth":
            t = t * (1 - ls) + 0.5 * ls
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, t, reduction="none")
        if kind == "focal":
            pt = torch.exp(-bce)
            bce = (1 - pt) ** g * bce
        return bce.mean()
    return fn


@dataclass
class Result:
    model: object
    score: float
    metrics: dict
    epochs: int
    seconds: float
    history: list


def predict(torch, model, X, rows: np.ndarray, device: str, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            xb = X[torch.from_numpy(rows[i:i + bs]).to(X.device)].to(device).float()[:, None]
            out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train(torch, cfg: dict, gate: int, tr: Subset, va: Subset, cache: MelCache, device: str, *, seed: int, max_seconds: float = 900,
          epochs_max: int = 40, n_per_epoch: int = 6000, patience: int = 6, amp: bool = True, report=None,
          synth: dict | None = None, exclude_pids: set | None = None, pre_epochs: int = 3, deterministic: bool = False) -> Result:
    """Train one network. `report(epoch, score)` may raise optuna.TrialPruned. The returned model carries the best-validation weights."""
    fc.seed_all(torch, seed, deterministic)
    t0 = time.time()
    model = build_model(cfg, cache.n_mels).to(device)
    use_amp = amp and device.startswith("cuda")
    if device.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    lossf = _loss_fn(torch, cfg)
    X = cache.X
    rng = np.random.default_rng(seed)

    def fit_step(opt, sched, xb, tb):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = lossf(model(xb).float(), tb)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()

    def batches(rows_np, y_np, prob, n, bsz):
        pick = torch.from_numpy(rng.choice(len(rows_np), size=n, replace=True, p=prob))
        for i in range(0, n - bsz + 1, bsz):
            b = pick[i:i + bsz].numpy()
            yield torch.from_numpy(rows_np[b]), torch.from_numpy(y_np[b])

    bsz = int(cfg["batch"])

    # ── optional pretraining on synthetic events (gate 2), never from excluded (val / test / held-out) patients
    if gate == 2 and cfg.get("synth_pretrain") and synth is not None:
        n_real = max(int((tr.y == 0).sum()), int((tr.y == 1).sum()))
        idx = synth_indices(synth["y"], synth["pid"], exclude_pids or set(), int(round(float(cfg["synth_ratio"]) * n_real)), rng)
        if len(idx) >= 2 * bsz:
            Xs, ys = synth["X"][torch.from_numpy(idx).to(synth["X"].device)], torch.from_numpy(synth["y"][idx].astype(np.float32))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            for _ in range(pre_epochs):
                model.train()
                perm = torch.randperm(len(idx))
                for i in range(0, len(idx) - bsz + 1, bsz):
                    b = perm[i:i + bsz]
                    xb = augment(torch, Xs[b.to(Xs.device)].to(device).float(), cfg)
                    fit_step(opt, None, xb, ys[b].to(device))

    # ── main training on real events
    steps = max(1, n_per_epoch // bsz)
    total = steps * epochs_max
    params = model.parameters()
    if cfg["opt"] == "sgd":
        opt = torch.optim.SGD(params, lr=float(cfg["lr"]), momentum=0.9, nesterov=True, weight_decay=float(cfg["wd"]))
    else:
        opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    if cfg["sched"] == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg["lr"]), total_steps=total + 1)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    prob = sampling_probs(tr, gate)
    best, best_state, best_m, bad, hist, ep = -1.0, None, {}, 0, [], 0
    for ep in range(1, epochs_max + 1):
        model.train()
        for rows, yb in batches(tr.rows, tr.y, prob, steps * bsz, bsz):
            xb = X[rows.to(X.device)].to(device).float()
            yb = yb.to(device)
            xb = augment(torch, xb, cfg)
            if float(cfg["mixup"]) > 0:
                lam = float(rng.beta(float(cfg["mixup"]), float(cfg["mixup"])))
                perm = torch.randperm(len(xb), device=device)
                xb, yb = lam * xb + (1 - lam) * xb[perm], lam * yb + (1 - lam) * yb[perm]
            fit_step(opt, sched, xb, yb)
        m = metrics(va.y, predict(torch, model, X, va.rows, device), va.w)
        s = score_of(m)
        hist.append(s)
        if s > best + 1e-4:
            best, best_m, bad = s, m, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        if report is not None:
            report(ep, s)
        if bad >= patience or time.time() - t0 > max_seconds:
            break
    model.load_state_dict(best_state)
    return Result(model.eval(), best, best_m, ep, time.time() - t0, hist)


def n_params(cfg: dict, n_mels: int) -> tuple[int, int]:
    return count_params(build_model(cfg, n_mels))
