"""
pipeline.py
───────────
Misma API de siempre:
    pipeline = PreprocessingPipeline.from_config(cfg)
    batch    = pipeline.run()

Internamente ahora tiene dos fases para no colapsar la RAM:

  Fase 1 — liviana (sin audio en memoria)
  ─────────────────────────────────────────
  LoadingStage(load_audio=False)  → lee solo JSONs + paths
  EventExtractionStage            → explota recordings → eventos (lazy, sin audio)
  BalancedSamplingStage           → filtra/repite filas de metadata

  Resultado: lista de AudioSamples con waveform=[] pero con todos los
  metadatos necesarios. ~1 KB/muestra → 20k eventos = 20 MB.

  Fase 2 — procesamiento en chunks con GPU
  ─────────────────────────────────────────
  Para cada chunk de `chunk_size` eventos:
    WaveformLoadingStage  → carga WAVs del chunk (libera el anterior)
    SliceEventsStage      → corta los eventos del WAV completo
    ResamplingStage       → GPU
    ConcatenationStage    → CPU (opera sobre numpy)
    PaddingStage          → CPU
    WaveformAugmentation  → GPU
    LogMelStage           → GPU
    SpecAugmentStage      → GPU
    NormalizationStage    → GPU
    → acumula solo features (float32, ~64 KB/muestra)
    → libera waveforms del chunk

  Sin cache en disco.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from stages.base import AudioSample, Batch
from stages.stages import (
    LoadingStage,
    WaveformLoadingStage,
    ResamplingStage,
    EventExtractionStage,
    SliceEventsStage,
    BalancedSamplingStage,
    ConcatenationStage,
    PaddingStage,
    WaveformAugmentationStage,
    LogMelStage,
    SpecAugmentStage,
    NormalizationStage,
)

log = logging.getLogger("pipeline")


class PreprocessingPipeline:

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg    = cfg
        self.device = self._resolve_device()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "PreprocessingPipeline":
        return cls(cfg)

    def _resolve_device(self) -> torch.device:
        requested = getattr(self.cfg.runner, "device", "auto")
        if requested == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(requested)
        log.info("Device: %s", dev)
        return dev

    # ── Fase 1: metadata liviana ───────────────────────────────────────────

    def _phase1(self) -> Batch:
        """
        Corre las stages que solo necesitan metadata (sin cargar audio).
        Devuelve un batch de AudioSamples con waveform=[] y sr=-1.
        """
        print("\n── Fase 1: Metadata (sin audio) ──────────────────────────")

        # LoadingStage en modo lazy — NO carga waveforms
        stage1_batch = LoadingStage(load_audio=False)([], self.cfg)
        print(f"  Recordings: {len(stage1_batch):,}")

        # EventExtraction detecta sr==-1 y guarda timestamps en meta
        stage2_batch = EventExtractionStage()(stage1_batch, self.cfg)
        n_valid = sum(1 for s in stage2_batch if s.label_int >= 0)
        print(f"  Events:     {len(stage2_batch):,}  (valid labels: {n_valid:,})")

        # BalancedSampling opera solo sobre metadatos — es instantáneo
        stage3_batch = BalancedSamplingStage()(stage2_batch, self.cfg)
        dist = Counter(s.label_str for s in stage3_batch)
        print(f"  Balanced:   {len(stage3_batch):,}  {dict(sorted(dist.items()))}")

        return stage3_batch

    # ── Fase 2: procesamiento en chunks ───────────────────────────────────

    def _build_phase2_stages(self):
        """Construye las stages de la Fase 2, todas con device inyectado."""
        d = self.device
        return [
            WaveformLoadingStage(device=d),   # carga WAVs del chunk
            SliceEventsStage(device=d),        # corta eventos del WAV completo
            ResamplingStage(device=d),         # GPU
            ConcatenationStage(device=d),      # CPU (numpy)
            PaddingStage(device=d),            # CPU
            WaveformAugmentationStage(device=d),  # GPU
            LogMelStage(device=d),             # GPU
            SpecAugmentStage(device=d),        # GPU
            NormalizationStage(device=d),      # GPU
        ]

    def _phase2(self, event_batch: Batch, augment: bool = True) -> Batch:
        """
        Procesa event_batch en chunks de cfg.runner.chunk_size.
        Al final de cada chunk libera los waveforms — solo acumula features.
        """
        chunk_size = getattr(self.cfg.runner, "chunk_size", 256)
        stages     = self._build_phase2_stages()

        # Desactivar augmentation si se pidió (ej. para el test set)
        if not augment:
            for s in stages:
                if s.name in ("augmentation", "spec_augment"):
                    s.name = s.name   # duck-type: el enabled guard lo saltea

        n_chunks  = (len(event_batch) + chunk_size - 1) // chunk_size
        all_out: Batch = []

        print(f"\n── Fase 2: Procesamiento en {n_chunks} chunks de {chunk_size} ──")
        chunk_bar = tqdm(range(n_chunks), desc="Chunks", unit="chunk", leave=True)

        for i in chunk_bar:
            chunk = event_batch[i * chunk_size : (i + 1) * chunk_size]

            for stage in stages:
                # Saltear WaveformAugmentationStage y SpecAugmentStage si augment=False
                if not augment and stage.name in ("augmentation", "spec_augment"):
                    continue
                try:
                    chunk = stage(chunk, self.cfg)
                except Exception as exc:
                    log.error("Stage '%s' failed on chunk %d: %s", stage.name, i, exc)
                    if self.cfg.runner.fail_fast:
                        raise
                    break

            # Acumular solo features; descartar waveforms para liberar RAM
            for s in chunk:
                s.waveform = np.array([], dtype=np.float32)
            all_out.extend(chunk)

            chunk_bar.set_postfix({"done": len(all_out), "chunk": f"{i+1}/{n_chunks}"})

            # Liberar caché GPU entre chunks
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return all_out

    # ── API pública ────────────────────────────────────────────────────────

    def run(self, augment: bool = True) -> Batch:
        """
        Corre las dos fases y devuelve List[AudioSample] con .feature poblado.

        Parameters
        ----------
        augment : False para el test set (salta WaveformAug y SpecAugment).
        """
        t0 = time.perf_counter()

        event_batch = self._phase1()
        final_batch = self._phase2(event_batch, augment=augment)

        elapsed = time.perf_counter() - t0
        valid   = [s for s in final_batch if s.feature is not None]
        print(f"\n── Done: {len(valid):,} samples en {elapsed:.1f}s ──")
        return final_batch

    # ── Introspección ──────────────────────────────────────────────────────

    def summary(self) -> str:
        cfg        = self.cfg
        device_str = str(self.device)
        if self.device.type == "cuda":
            device_str += f" ({torch.cuda.get_device_name(self.device)})"

        lines = [
            "PreprocessingPipeline",
            "=" * 42,
            f"  Device      : {device_str}",
            f"  Chunk size  : {getattr(cfg.runner, 'chunk_size', 256)}",
            f"  Task        : {cfg.data.task}",
            f"  Target SR   : {cfg.loading.target_sr} Hz",
            f"  Duration    : {cfg.padding.target_duration_s}s",
            f"  Mel bins    : {cfg.log_mel.n_mels}",
            "",
            "  Fase 1 — sin audio (CPU)",
            f"    Loading(lazy) → EventExtraction → BalancedSampling",
            f"    Sampling    : {cfg.balanced_sampling.strategy}",
            "",
            "  Fase 2 — chunks con GPU",
            f"    WavLoad → Slice → Resample → Concat → Pad",
            f"    → WavAug → LogMel → SpecAugment → Norm",
            f"    Augment     : shift={cfg.augmentation.time_shift.enabled} "
                             f"noise={cfg.augmentation.add_noise.enabled} "
                             f"stretch={cfg.augmentation.time_stretch.enabled}",
            f"    SpecAugment : {cfg.augmentation.spec_augment.enabled}",
            f"    Normalize   : {cfg.normalization.strategy}",
            "=" * 42,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"PreprocessingPipeline("
                f"device={self.device}, "
                f"chunk_size={getattr(self.cfg.runner, 'chunk_size', 256)})")