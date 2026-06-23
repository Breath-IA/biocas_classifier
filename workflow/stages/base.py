"""
stages/base.py
──────────────
Protocol (structural typing) for all pipeline stages.
Every stage is a stateless callable:  batch_out = stage(batch_in, cfg)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
from tqdm.auto import tqdm

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# Domain types
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP_1_1 = {"N": 0, "R": 1, "W": 1, "S": 1, "CC": 1, "FC": 1, "WC": 1}
LABEL_MAP_1_2 = {"N": 0, "R": 1, "W": 2, "S": 3, "CC": 4, "FC": 5, "WC": 6}


@dataclass
class AudioSample:
    """
    Unit of data flowing through the pipeline.

    waveform = np.array([]) significa que el audio no fue cargado todavía
    (modo lazy — se carga en WaveformLoadingStage antes de procesarlo).
    """
    sample_id:  str
    waveform:   np.ndarray
    sr:         int
    label_str:  str
    label_int:  int                   = -1
    meta:       Dict[str, Any]        = field(default_factory=dict)
    feature:    Optional[np.ndarray]  = None


Batch = List[AudioSample]


# ─────────────────────────────────────────────────────────────────────────────
# Stage Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class Stage(Protocol):
    name: str
    def __call__(self, batch: Batch, cfg: Any) -> Batch: ...


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseStage:
    """
    Base con logging, tqdm y guard de enabled.

    Parámetro opcional `device`: stages que usan GPU lo reciben en el
    constructor. Los demás lo ignoran. Siempre es un torch.device.
    """

    name: str = "base"

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.log    = logging.getLogger(f"pipeline.{self.name}")
        self.device = device or torch.device("cpu")

    def progress(
        self,
        iterable: Iterable[T],
        *,
        desc:  str  = "",
        total: Optional[int] = None,
        unit:  str  = "sample",
        leave: bool = False,
    ) -> Iterable[T]:
        return tqdm(
            iterable,
            desc          = f"  {self.name}" if not desc else f"  {desc}",
            total         = total or (len(iterable) if hasattr(iterable, "__len__") else None),
            unit          = unit,
            leave         = leave,
            dynamic_ncols = True,
        )

    def __call__(self, batch: Batch, cfg: Any) -> Batch:
        stage_cfg = getattr(cfg.stages, self.name, None)
        if stage_cfg is not None and not stage_cfg.enabled:
            self.log.info("Stage '%s' disabled – skipping.", self.name)
            return batch
        self.log.info("Stage '%s' | in=%d", self.name, len(batch))
        batch_out = self._run(batch, cfg)
        self.log.info("Stage '%s' | out=%d", self.name, len(batch_out))
        return batch_out

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        raise NotImplementedError