"""
Utilidades de k-space (FFT 2D de log-mel) para eda_kspace_binary.ipynb.

eda_kspace.ipynb define estas mismas piezas en linea; aqui se reunen para el notebook binario.
Las caracteristicas son sin estado (no aprenden nada de las etiquetas), asi que se calculan una vez
sobre todo X y la validacion leave-one-year-out solo reparte filas.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import rfft2, rfftfreq, fftfreq
from scipy.signal.windows import hann, tukey
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Parametros del pipeline (workflow/config/pipeline.yaml -> log_mel)
SR, HOP, N_FFT, N_MELS, FMIN, FMAX = 4000, 128, 512, 64, 50, 2000
FRAME_RATE = SR / HOP
T = 94                                          # frames de 3 s
CLASSES = ["N", "FC", "W", "WC"]
YEARS = [2022, 2023, 2024, 2025]
TRAIN_YEARS = [2022, 2023, 2024]

KU_HZ = rfftfreq(T, d=1 / FRAME_RATE)
KU_IDX = np.arange(len(KU_HZ))
KV_SIGNED = np.rint(fftfreq(N_MELS) * N_MELS).astype(int)     # 0..31, -32..-1 (orden FFT)
KV_ABS = np.abs(KV_SIGNED)
WIN = np.outer(tukey(N_MELS, 0.25), tukey(T, 0.25)).astype(np.float32)


# ─── datos ────────────────────────────────────────────────────────────────────
@dataclass
class Data:
    X: np.ndarray          # (n, 64, 94) z-score por muestra
    y: np.ndarray          # 0=N, 1=FC, 2=W, 3=WC
    year: np.ndarray
    groups: np.ndarray     # grabacion de origen (raw_stem)
    meta: pd.DataFrame

    @property
    def yb(self):          # 0 = normal, 1 = anormal
        return (self.y > 0).astype(int)


def detect_specaugment(X):
    return X.std(axis=1) < 1e-6, X.std(axis=2) < 1e-6      # (n, T) frames constantes, (n, F) bins constantes


def _interp_fill(x, bad):
    L = bad.size
    if not bad.any() or bad.all():
        return x
    idx = np.arange(L)
    left = np.maximum.accumulate(np.where(~bad, idx, -1))
    right = np.minimum.accumulate(np.where(~bad, idx, L)[::-1])[::-1]
    j = np.where(bad)[0]
    l, r = left[j], right[j]
    l = np.where(l < 0, r, l)
    r = np.where(r >= L, l, r)
    w = np.where(r > l, (j - l) / np.maximum(r - l, 1), 0.0)
    out = x.copy()
    out[..., j] = x[..., l] * (1 - w) + x[..., r] * w
    return out


def inpaint_specaugment(X, t_mask, f_mask):
    Xo = X.copy()
    for i in np.where(t_mask.any(1) | f_mask.any(1))[0]:
        x = _interp_fill(Xo[i], t_mask[i])
        x = _interp_fill(x.T, f_mask[i]).T
        Xo[i] = (x - x.mean()) / (x.std() + 1e-8)
    return Xo


def load_data(proc_root, max_per_year_class=1200, seed=42, mask_policy="inpaint") -> Data:
    proc_root = Path(proc_root)
    frames = []
    for split in ("train", "test"):
        d = pd.read_csv(proc_root / split / "index.csv")
        d["split"] = split
        frames.append(d)
    index = pd.concat(frames, ignore_index=True)
    index["year"] = index["raw_stem"].str[:4].astype(int)       # patient_id/gender de index.csv estan desplazados
    index = index[index["label_str"].isin(CLASSES)]
    parts = []
    for _, g in index.groupby(["year", "label_str"]):
        n = len(g) if max_per_year_class is None else min(len(g), max_per_year_class)
        parts.append(g.sample(n, random_state=seed))
    meta = pd.concat(parts, ignore_index=True)
    X = np.stack([np.load(proc_root / r.split / r.path) for r in meta.itertuples()]).astype(np.float32)
    assert X.shape[1:] == (N_MELS, T), X.shape
    y = meta["label_str"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    year = meta["year"].to_numpy()
    groups = meta["raw_stem"].to_numpy()
    t_mask, f_mask = detect_specaugment(X)
    masked = t_mask.any(1) | f_mask.any(1)
    if mask_policy == "inpaint":
        X = inpaint_specaugment(X, t_mask, f_mask)
    elif mask_policy == "drop":
        k = ~masked
        X, y, year, groups, meta = X[k], y[k], year[k], groups[k], meta[k].reset_index(drop=True)
    return Data(X, y, year, groups, meta)


def channel_perturb(X, sigma, seed=0):
    # offset aditivo suave en frecuencia (canal fijo, 1-3 ciclos por espectro) + re-normalizacion por muestra
    rng = np.random.default_rng(seed)
    f = np.linspace(0, 1, X.shape[1])[None, :]
    c = np.zeros((len(X), X.shape[1]), np.float32)
    for q in (1, 2, 3):
        c += rng.normal(0, sigma / np.sqrt(3), (len(X), 1)) * np.cos(2 * np.pi * q * f + rng.uniform(0, 2 * np.pi, (len(X), 1)))
    Xp = X + c[:, :, None]
    return (Xp - Xp.mean((1, 2), keepdims=True)) / (Xp.std((1, 2), keepdims=True) + 1e-8)


# ─── k-space global ───────────────────────────────────────────────────────────
def klogmag(X, demean_time=True):
    """log(1+|rfft2|) con ventana. demean_time resta la media temporal por banda antes de ventanear (canal fijo -> 0 exacto)."""
    Z = X - X.mean(-1, keepdims=True) if demean_time else X
    return np.log1p(np.abs(rfft2(Z * WIN, axes=(-2, -1))))        # (n, F, T//2+1)


def feat_logk(X, demean_time=True):
    L = klogmag(X, demean_time)
    return (L[:, :, 1:] if demean_time else L).reshape(len(X), -1)   # sin ku=0 si hay demean


def log_cell_labels(ku_edges, kv_edges):
    """Etiqueta de celda (F, T//2+1) con bins log-espaciados en ku y en |kv| (kv con signo = 2 mitades). -1 = ignorado."""
    ku_bin = np.digitize(KU_IDX, ku_edges) - 1
    ku_bin = np.where((KU_IDX < ku_edges[0]) | (KU_IDX >= ku_edges[-1]), -1, ku_bin)
    nkv = len(kv_edges) - 1
    kv_bin = np.clip(np.digitize(KV_ABS, kv_edges) - 1, 0, nkv - 1) + nkv * (KV_SIGNED < 0)
    lab = ku_bin[None, :] * (2 * nkv) + kv_bin[:, None]
    return np.where(ku_bin[None, :] < 0, -1, lab)


def cell_pool_matrix(labels):
    flat = labels.ravel()
    ids = np.unique(flat[flat >= 0])
    P = np.zeros((len(ids), flat.size), np.float32)
    for r, i in enumerate(ids):
        m = flat == i
        P[r, m] = 1 / m.sum()
    return P


KU_EDGES_LOG = [1, 2, 3, 4, 6, 8, 12, 17, 25, len(KU_IDX)]      # ~0.33 .. 15.6 Hz
KV_EDGES_LOG = [0, 1, 2, 3, 5, 8, 12, 18, 33]                    # ciclos/espectro


def feat_logpool(X, ku_edges=KU_EDGES_LOG, kv_edges=KV_EDGES_LOG):
    """log|K| promediado en celdas log-espaciadas (modulation spectrum en escala log): pocos numeros, robusto."""
    P = cell_pool_matrix(log_cell_labels(ku_edges, kv_edges))
    return klogmag(X, True).reshape(len(X), -1) @ P.T


# ─── k-space local (parches) ──────────────────────────────────────────────────
def feat_localk(X, pf=16, pt=16, sf=8, st=8, chunk=300):
    """
    FFT 2D de parches pf x pt (mel x frames) con paso (sf, st), demean temporal por parche.
    Un crepitante aislado en 3 s se diluye en la FFT global; por parche se ve y luego se resume.
    Devuelve dict con pooling sobre el tiempo (mantiene la posicion en frecuencia) y pooling total, cada uno con mean y max.
    """
    w = (hann(pf, sym=False)[:, None] * hann(pt, sym=False)[None, :]).astype(np.float32)
    keys = ("mean_t", "max_t", "mean_all", "max_all")
    out = {k: [] for k in keys}
    for s in range(0, len(X), chunk):
        W = sliding_window_view(X[s:s + chunk], (pf, pt), axis=(1, 2))[:, ::sf, ::st]       # (n, nf, nt, pf, pt)
        W = W - W.mean(-1, keepdims=True)
        K = np.log1p(np.abs(rfft2(W * w, axes=(-2, -1))))[..., 1:]                           # sin ku=0
        n, nf = K.shape[:2]
        out["mean_t"].append(K.mean(2).reshape(n, -1))
        out["max_t"].append(K.max(2).reshape(n, -1))
        out["mean_all"].append(K.mean((1, 2)).reshape(n, -1))
        out["max_all"].append(K.max((1, 2)).reshape(n, -1))
    return {k: np.concatenate(v) for k, v in out.items()}


# ─── clasificadores ───────────────────────────────────────────────────────────
class BlockScaler(BaseEstimator, TransformerMixin):
    """Tras StandardScaler: cada bloque de columnas se divide por sqrt(dim) para que todos pesen igual al concatenar."""
    def __init__(self, dims=None):
        self.dims = dims

    def fit(self, X, y=None):
        self.w_ = np.concatenate([np.full(d, 1 / np.sqrt(d)) for d in self.dims]).astype(np.float32)
        return self

    def transform(self, X): return X * self.w_


def clf_lr_pca(dim=64, C=1.0):
    return make_pipeline(StandardScaler(), PCA(dim, svd_solver="randomized", random_state=0),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=500))


def clf_lr(C=0.01, dims=None):
    steps = [StandardScaler()] + ([BlockScaler(dims)] if dims else [])
    return make_pipeline(*steps, LogisticRegression(C=C, class_weight="balanced", max_iter=1000))


def clf_lda(pca=128):
    return make_pipeline(StandardScaler(), PCA(pca, svd_solver="randomized", random_state=0),
                         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def clf_hgb():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_leaf_nodes=15, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


# ─── protocolo: 2022-2024 = desarrollo, 2025 = prueba ─────────────────────────
TEST_YEAR = 2025


def protocol_folds(d: Data):
    """
    2025 es SOLO la prueba (train 2022-24 -> test 2025, el split oficial); nunca entra al entrenamiento ni a la seleccion.
    Para elegir disenos se usa "leave-one-train-year-out" dentro de 2022-2024 (entrena con 2 anios, evalua el tercero).
    Rinde (split, anio_evaluado, mascara_train, mascara_test).
    """
    in_train = np.isin(d.year, TRAIN_YEARS)
    for h in TRAIN_YEARS:
        yield "dev", h, in_train & (d.year != h), d.year == h
    yield "test", TEST_YEAR, in_train, d.year == TEST_YEAR


def _scores(clf, F):
    return clf.predict_proba(F)[:, 1]                        # P(anormal)


def eval_binary(d: Data, F, F_chan, make_clf):
    """Un modelo por fold del protocolo (limpio y con canal sintetico). AUC N-vs-anormal + AUC N-vs-cada subclase."""
    rows = []
    for split, hold, tr, te in protocol_folds(d):
        clf = make_clf().fit(F[tr], d.yb[tr])
        for cond, Fte in (("limpio", F[te]), ("canal", F_chan[te])):
            s = _scores(clf, Fte)
            yte, yb = d.y[te], d.yb[te]
            row = dict(split=split, hold=hold, cond=cond, auc=roc_auc_score(yb, s), bal_acc=balanced_accuracy_score(yb, s > 0.5))
            for k, c in enumerate(CLASSES[1:], start=1):
                m = (yte == 0) | (yte == k)
                row[f"auc_{c}"] = roc_auc_score(yb[m], s[m]) if (yte == k).sum() > 5 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def domain_indices(d: Data, cap=300, seed=42):
    rng = np.random.default_rng(seed)
    n = min(min(((d.year == yy) & (d.y == c)).sum() for yy in TRAIN_YEARS for c in (0, 1, 2)), cap)
    return np.concatenate([rng.choice(np.where((d.year == yy) & (d.y == c))[0], n, replace=False)
                           for yy in TRAIN_YEARS for c in (0, 1, 2)])


def eval_domain(d: Data, F, idx, n_splits=4):
    """Precision balanceada para adivinar el anio (2022/23/24) con celdas (anio, clase) balanceadas. Menor = mas invariante."""
    Fd, yrd, gd = F[idx], d.year[idx], d.groups[idx]
    accs = []
    for tr, te in GroupKFold(n_splits).split(Fd, yrd, gd):
        clf = make_pipeline(StandardScaler(), PCA(64, svd_solver="randomized", random_state=0),
                            LogisticRegression(class_weight="balanced", max_iter=500)).fit(Fd[tr], yrd[tr])
        accs.append(balanced_accuracy_score(yrd[te], clf.predict(Fd[te])))
    return float(np.mean(accs))
