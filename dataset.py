"""
dataset.py
──────────
PyTorch Dataset que carga features bajo demanda desde archivos .npy individuales.

Cada __getitem__ lee un solo archivo .npy — los workers del DataLoader operan
de forma completamente independiente sin compartir memoria ni file descriptors.
Funciona correctamente en Windows con num_workers > 0.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from stages.base import Batch


class BioCASDataset(Dataset):
    """
    Dataset que lee cada feature de su propio .npy en __getitem__.

    Dos formas de construirlo:

        # Desde un batch recién procesado (después de pipeline.run())
        ds = BioCASDataset(batch)

        # Desde disco (después de reiniciar el kernel)
        ds = BioCASDataset.from_index(index_df, base_dir=Path("data/processed/train"))

    Cada item devuelto es (feature_tensor, label_tensor):
        feature_tensor : float32, shape (1, n_mels, T)
        label_tensor   : int64, scalar
    """

    def __init__(self, batch: Batch) -> None:
        """Construye el dataset desde un batch en memoria (features ya cargados)."""
        valid = [s for s in batch if s.feature is not None]
        if len(valid) < len(batch):
            import warnings
            warnings.warn(f"{len(batch) - len(valid)} samples sin feature — descartados.")

        # En este modo guardamos los features en memoria (solo durante la sesión activa)
        self._features_mem: Optional[np.ndarray] = np.stack(
            [s.feature for s in valid], axis=0
        )
        self._labels      = np.array([s.label_int for s in valid], dtype=np.int64)
        self._ids         = [s.sample_id for s in valid]
        self._label_strs  = [s.label_str  for s in valid]
        self._metas       = [s.meta       for s in valid]
        self._paths: Optional[List[Path]] = None   # None = usar _features_mem
        self._base_dir: Optional[Path]    = None

        n_unlabelled = int((self._labels == -1).sum())
        if n_unlabelled > 0:
            import warnings
            warnings.warn(
                f"{n_unlabelled}/{len(self._labels)} samples tienen label_int=-1. "
                f"Revisá EVENT_TYPE_NORM en stages.py."
            )

    @classmethod
    def from_index(cls, index: pd.DataFrame, base_dir: Path) -> "BioCASDataset":
        """
        Construye el dataset desde un CSV de índice (modo lazy — no carga features).
        Cada __getitem__ lee el .npy correspondiente del disco.
        """
        obj = object.__new__(cls)
        obj._features_mem = None   # lazy: leer de disco en __getitem__
        obj._labels       = index["label_int"].to_numpy(dtype=np.int64)
        obj._ids          = index["sample_id"].tolist()
        obj._label_strs   = index["label_str"].tolist()
        obj._metas        = index.to_dict("records")
        obj._paths        = [base_dir / p for p in index["path"]]
        obj._base_dir     = base_dir
        return obj

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        if self._features_mem is not None:
            # Modo en memoria: copia del slice para que torch no se queje
            feat_np = np.array(self._features_mem[idx], dtype=np.float32)
        else:
            # Modo lazy: leer el .npy individual — seguro con num_workers > 0
            feat_np = np.load(self._paths[idx]).astype(np.float32)

        feature = torch.from_numpy(feat_np).unsqueeze(0)   # (1, n_mels, T)
        label   = torch.tensor(self._labels[idx], dtype=torch.long)
        return feature, label

    # ── Propiedades útiles ─────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        valid = self._labels[self._labels >= 0]
        return 0 if len(valid) == 0 else int(valid.max()) + 1

    @property
    def class_weights(self) -> torch.Tensor:
        """Pesos inverso-frecuencia para nn.CrossEntropyLoss(weight=...)."""
        counts  = np.bincount(self._labels[self._labels >= 0], minlength=self.num_classes).astype(float)
        weights = 1.0 / (counts + 1e-9)
        weights /= weights.sum()
        return torch.from_numpy(weights).float()

    @property
    def feature_shape(self):
        """Shape de un feature individual: (1, n_mels, T)."""
        feat = self[0][0]
        return tuple(feat.shape)

    def __repr__(self) -> str:
        mode = "in-memory" if self._features_mem is not None else "lazy (per-file)"
        return (
            f"BioCASDataset(n={len(self)}, n_classes={self.num_classes}, "
            f"feature_shape={self.feature_shape}, mode={mode})"
        )