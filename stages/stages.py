"""
stages/stages.py
────────────────
Los 10 stages de la pipeline. Mismo contrato que antes:
  batch_out = stage(batch_in, cfg)

Cambios respecto a la versión anterior:
  • LoadingStage admite load_audio=False → guarda solo paths/metadata (liviano)
  • WaveformLoadingStage → carga el audio real para un chunk de eventos
  • ResamplingStage, WaveformAugmentationStage, LogMelStage,
    SpecAugmentStage, NormalizationStage → operan en GPU si se les pasa device
"""
from __future__ import annotations

import collections
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as TAF
import torchaudio.transforms as TAT

from .base import AudioSample, BaseStage, Batch, LABEL_MAP_1_1, LABEL_MAP_1_2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers compartidos
# ─────────────────────────────────────────────────────────────────────────────

GENDER_MAP   = {"0": "male", "1": "female"}
LOCATION_MAP = {
    "p1": "left_posterior", "p2": "right_posterior",
    "p3": "left_anterior",  "p4": "right_anterior",
    "p5": "left_lateral",   "p6": "right_lateral",
}
EVENT_TYPE_NORM = {
    "normal":         "N",
    "rhonchi":        "R",
    "wheeze":         "W",
    "stridor":        "S",
    "coarse crackle": "CC",
    "fine crackle":   "FC",
    "wheeze&crackle": "WC",
    "wheeze+crackle": "WC",
}


def _parse_filename(stem: str) -> Dict[str, str]:
    parts = stem.split("_")
    meta: Dict[str, str] = {"raw_stem": stem}
    if len(parts) >= 5:
        meta["patient_id"] = parts[0]
        meta["age"]        = parts[1]
        meta["gender"]     = GENDER_MAP.get(parts[2], parts[2])
        meta["location"]   = LOCATION_MAP.get(parts[3], parts[3])
        meta["rec_num"]    = parts[4]
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOADING
# ─────────────────────────────────────────────────────────────────────────────

class LoadingStage(BaseStage):
    """
    Carga WAVs + JSONs del disco.

    load_audio=True  (default) → comportamiento original, carga el waveform.
    load_audio=False           → solo guarda path y annotation en meta,
                                 waveform queda vacío (np.array([])).
                                 Úsalo en la Fase 1 para no colapsar la RAM.
    """

    name = "loading"

    def __init__(self, load_audio: bool = True, device=None) -> None:
        super().__init__(device)
        self.load_audio = load_audio

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        wav_dir   = Path(cfg.data.raw_wav_dir)
        json_dir  = Path(cfg.data.raw_json_dir)
        keep_poor = getattr(cfg.loading, "keep_poor_quality", False)

        wav_files        = sorted(wav_dir.glob("*.wav"))
        samples: Batch   = []
        n_skip_quality   = 0

        for wav_path in self.progress(wav_files, desc="loading", unit="file"):
            json_path = json_dir / wav_path.with_suffix(".json").name
            if not json_path.exists():
                continue

            try:
                annotation = json.loads(json_path.read_text())
            except Exception:
                continue

            record_label = annotation.get("record_annotation", "?")
            if record_label == "Poor Quality" and not keep_poor:
                n_skip_quality += 1
                continue

            file_meta = _parse_filename(wav_path.stem)

            if self.load_audio:
                waveform_t, sr = torchaudio.load(wav_path)
                waveform_np    = waveform_t.mean(dim=0).numpy().astype(np.float32)
            else:
                waveform_np = np.array([], dtype=np.float32)
                sr          = -1   # señal de "no cargado"

            samples.append(AudioSample(
                sample_id = wav_path.stem,
                waveform  = waveform_np,
                sr        = sr,
                label_str = record_label,
                label_int = -1,
                meta      = {
                    **file_meta,
                    "source_path": str(wav_path),
                    "annotation":  annotation,
                    "n_events":    len(annotation.get("event_annotation", [])),
                },
            ))

        if n_skip_quality:
            self.log.info("Skipped %d Poor Quality recordings.", n_skip_quality)
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 1b. WAVEFORM LOADING  (nueva — solo para uso interno de pipeline.py)
#     Carga el audio para un batch de AudioSamples que tienen sr == -1.
# ─────────────────────────────────────────────────────────────────────────────

class WaveformLoadingStage(BaseStage):
    """
    Carga los waveforms de los samples que llegaron sin audio (sr == -1).
    Se usa en la Fase 2 del pipeline, sobre cada chunk de eventos.
    """

    name = "waveform_loading"

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        out: Batch = []
        # Cachear WAVs ya leídos dentro del chunk (varios eventos → mismo archivo)
        wav_cache: Dict[str, Tuple[np.ndarray, int]] = {}

        for s in self.progress(batch, desc="loading audio", unit="file"):
            if s.sr != -1:          # ya tiene audio (no debería pasar, pero seguro)
                out.append(s)
                continue

            src = s.meta.get("source_path", "")
            if not src:
                self.log.warning("Sample '%s' has no source_path – skipping.", s.sample_id)
                continue

            if src not in wav_cache:
                try:
                    t, sr         = torchaudio.load(src)
                    wav_cache[src] = (t.mean(dim=0).numpy().astype(np.float32), sr)
                except Exception as e:
                    self.log.warning("Could not load %s: %s", src, e)
                    continue

            waveform_np, sr = wav_cache[src]
            out.append(AudioSample(**{**s.__dict__, "waveform": waveform_np, "sr": sr}))

        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. RESAMPLING  (GPU)
# ─────────────────────────────────────────────────────────────────────────────

class ResamplingStage(BaseStage):
    """Resamplea a cfg.loading.target_sr. Mueve el tensor a GPU si está disponible."""

    name = "resampling"

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        target_sr = cfg.loading.target_sr
        out: Batch = []
        for s in self.progress(batch, desc="resampling"):
            if s.sr == target_sr:
                out.append(s)
                continue
            wav_t = torch.from_numpy(s.waveform).to(self.device)
            wav_t = TAF.resample(wav_t, s.sr, target_sr)
            out.append(AudioSample(
                **{**s.__dict__,
                   "waveform": wav_t.cpu().numpy(),
                   "sr":       target_sr}
            ))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVENT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

class EventExtractionStage(BaseStage):
    """
    Usa los timestamps de event_annotation para cortar cada recording en eventos.
    Funciona tanto si el waveform está cargado como si sr == -1 (modo lazy):
    en modo lazy guarda start/end en meta y deja waveform vacío.
    """

    name = "event_extraction"

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        task      = cfg.data.task
        label_map = LABEL_MAP_1_1 if task == "1-1" else LABEL_MAP_1_2
        ecfg      = cfg.event_extraction
        lazy      = False   # se detecta automáticamente por sr == -1

        out: Batch = []
        n_short = n_unknown = 0
        unknown_types: set = set()

        for recording in self.progress(batch, desc="extracting events", unit="recording"):
            annotation = recording.meta.get("annotation", {})
            event_list = annotation.get("event_annotation", [])
            lazy       = recording.sr == -1

            if not event_list:
                out.append(recording)
                continue

            for idx, event in enumerate(event_list):
                raw_type  = event.get("type", "")
                norm_type = EVENT_TYPE_NORM.get(raw_type.lower().strip())
                if norm_type is None:
                    n_unknown += 1
                    unknown_types.add(repr(raw_type))
                    if not ecfg.keep_unlabelled:
                        continue
                    norm_type = raw_type

                start_ms   = int(event["start"])
                end_ms     = int(event["end"])
                duration_s = (end_ms - start_ms) / 1000.0

                if duration_s < ecfg.min_duration_s:
                    n_short += 1
                    continue

                label_int = label_map.get(norm_type, -1)

                if lazy:
                    # No cortamos el waveform todavía — guardamos timestamps en meta
                    segment = np.array([], dtype=np.float32)
                    sr      = -1
                else:
                    sr          = recording.sr
                    start_samp  = int(start_ms / 1000 * sr)
                    end_samp    = min(int(end_ms / 1000 * sr), len(recording.waveform))
                    segment     = recording.waveform[start_samp:end_samp].copy()

                out.append(AudioSample(
                    sample_id = f"{recording.sample_id}_ev{idx:03d}",
                    waveform  = segment,
                    sr        = sr,
                    label_str = norm_type,
                    label_int = label_int,
                    meta      = {
                        **{k: v for k, v in recording.meta.items() if k != "annotation"},
                        "event_idx":    idx,
                        "start_ms":     start_ms,
                        "end_ms":       end_ms,
                        "duration_s":   duration_s,
                        "record_label": recording.label_str,
                    },
                ))

        if n_short:
            self.log.info("Dropped %d events shorter than %.2fs.", n_short, ecfg.min_duration_s)
        if n_unknown:
            self.log.warning(
                "Dropped %d events with unrecognised types: %s\n"
                "  → Add to EVENT_TYPE_NORM in stages.py if valid.",
                n_unknown, unknown_types,
            )
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3b. SLICE EVENTS  (nueva — par de EventExtraction en modo lazy)
#     Cuando los eventos llegaron sin waveform (lazy), los corta del WAV ya cargado.
# ─────────────────────────────────────────────────────────────────────────────

class SliceEventsStage(BaseStage):
    """
    Complemento de EventExtractionStage en modo lazy.
    Después de WaveformLoadingStage, el sample tiene el WAV completo en .waveform
    pero todavía hay que cortarlo con start_ms/end_ms guardados en meta.
    """

    name = "slice_events"

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        out: Batch = []
        for s in batch:
            start_ms = s.meta.get("start_ms")
            end_ms   = s.meta.get("end_ms")
            if start_ms is None or s.sr <= 0 or len(s.waveform) == 0:
                out.append(s)
                continue
            start = int(start_ms / 1000 * s.sr)
            end   = min(int(end_ms   / 1000 * s.sr), len(s.waveform))
            out.append(AudioSample(**{**s.__dict__, "waveform": s.waveform[start:end].copy()}))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. BALANCED SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

class BalancedSamplingStage(BaseStage):
    """
    Balanceo proporcional (√count), igual, o custom.
    Opera sobre AudioSamples — no necesita waveforms ni features.
    """

    name = "balanced_sampling"

    @staticmethod
    def _targets_proportional(counts: Dict[int, int]) -> Dict[int, int]:
        sqrt_total = sum(math.sqrt(n) for n in counts.values())
        total_orig = sum(counts.values())
        return {
            c: max(1, round(math.sqrt(n) / sqrt_total * total_orig))
            for c, n in counts.items()
        }

    @staticmethod
    def _targets_equal(counts: Dict[int, int]) -> Dict[int, int]:
        return {c: max(counts.values()) for c in counts}

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        scfg     = cfg.balanced_sampling
        strategy = scfg.strategy
        rng      = random.Random(scfg.random_seed)

        by_class: Dict[int, List[AudioSample]] = collections.defaultdict(list)
        for s in batch:
            by_class[s.label_int].append(s)

        counts = {c: len(v) for c, v in by_class.items()}
        self.log.info("Before sampling: %s", counts)

        if strategy in ("proportional", "sqrt"):
            targets = self._targets_proportional(counts)
        elif strategy == "equal":
            targets = self._targets_equal(counts)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        if getattr(scfg, "target_samples_per_class", None):
            targets = {c: scfg.target_samples_per_class for c in counts}

        self.log.info("After  sampling: %s", targets)

        out: Batch = []
        for cls, samples in by_class.items():
            n = targets[cls]
            if n <= len(samples):
                out.extend(rng.sample(samples, n))
            else:
                repeats  = n // len(samples)
                leftover = n  % len(samples)
                out.extend(samples * repeats)
                out.extend(rng.sample(samples, leftover))

        rng.shuffle(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONCATENATION
# ─────────────────────────────────────────────────────────────────────────────

class ConcatenationStage(BaseStage):
    """Pega eventos cortos del mismo label hasta alcanzar min_duration_s."""

    name = "concatenation"

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        ccfg    = cfg.concatenation
        sr      = cfg.loading.target_sr
        min_len = int(ccfg.min_duration_s * sr)
        max_len = int(ccfg.max_duration_s * sr)

        pool: Dict[int, List[AudioSample]] = collections.defaultdict(list)
        for s in batch:
            pool[s.label_int].append(s)

        out: Batch = []
        for s in self.progress(batch, desc="concatenation"):
            if len(s.waveform) >= min_len:
                out.append(s)
                continue

            parts   = [s.waveform]
            current = len(s.waveform)
            for donor in random.sample(pool[s.label_int],
                                       min(len(pool[s.label_int]), 10)):
                if current >= min_len:
                    break
                chunk    = donor.waveform[:max_len - current]
                parts.append(chunk)
                current += len(chunk)

            out.append(AudioSample(
                **{**s.__dict__,
                   "waveform": np.concatenate(parts)[:max_len],
                   "meta":     {**s.meta, "concatenated": True}}
            ))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. PADDING
# ─────────────────────────────────────────────────────────────────────────────

class PaddingStage(BaseStage):
    """Lleva todos los waveforms a exactamente target_duration_s."""

    name = "padding"

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        pcfg       = cfg.padding
        target_len = int(pcfg.target_duration_s * cfg.loading.target_sr)
        mode       = pcfg.mode

        out: Batch = []
        for s in self.progress(batch, desc="padding"):
            wav = s.waveform
            n   = len(wav)
            if n >= target_len:
                wav = wav[:target_len]
            else:
                deficit = target_len - n
                if mode == "zero":
                    wav = np.pad(wav, (0, deficit))
                elif mode == "reflect":
                    if deficit > n:
                        wav = np.tile(wav, math.ceil(target_len / n))[:target_len]
                    else:
                        wav = np.pad(wav, (0, deficit), mode="reflect")
                elif mode == "tile":
                    wav = np.tile(wav, math.ceil(target_len / n))[:target_len]
                else:
                    raise ValueError(f"Unknown padding mode: {mode!r}")
            out.append(AudioSample(**{**s.__dict__, "waveform": wav}))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. WAVEFORM AUGMENTATION  (GPU)
# ─────────────────────────────────────────────────────────────────────────────

class WaveformAugmentationStage(BaseStage):
    """Time-shift, noise, time-stretch. Mueve el tensor a GPU durante el procesamiento."""

    name = "augmentation"

    def _time_shift(self, wav: torch.Tensor, cfg: Any) -> torch.Tensor:
        shift = random.randint(0, int(cfg.max_shift_s * wav.shape[-1]))
        return torch.roll(wav, shift)

    def _add_noise(self, wav: torch.Tensor, cfg: Any) -> torch.Tensor:
        snr_db  = random.uniform(*list(cfg.snr_db_range))
        sig_pwr = wav.pow(2).mean() + 1e-9
        snr_lin = 10 ** (snr_db / 10)
        noise   = torch.randn_like(wav) * (sig_pwr / snr_lin).sqrt()
        return wav + noise

    def _time_stretch(self, wav: torch.Tensor, sr: int, cfg: Any) -> torch.Tensor:
        rate     = random.uniform(*list(cfg.rate_range))
        fake_sr  = int(sr * rate)
        orig_len = wav.shape[-1]
        wav      = TAF.resample(wav, fake_sr, sr)
        if wav.shape[-1] < orig_len:
            wav = torch.nn.functional.pad(wav, (0, orig_len - wav.shape[-1]))
        return wav[:orig_len]

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        acfg = cfg.augmentation
        out: Batch = []
        for s in self.progress(batch, desc="waveform aug"):
            wav = torch.from_numpy(s.waveform).to(self.device)

            if acfg.time_shift.enabled:
                wav = self._time_shift(wav, acfg.time_shift)
            if acfg.add_noise.enabled:
                wav = self._add_noise(wav, acfg.add_noise)
            if acfg.time_stretch.enabled:
                wav = self._time_stretch(wav, s.sr, acfg.time_stretch)

            out.append(AudioSample(
                **{**s.__dict__, "waveform": wav.cpu().numpy().astype(np.float32)}
            ))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOG-MEL  (GPU)
# ─────────────────────────────────────────────────────────────────────────────

class LogMelStage(BaseStage):
    """
    Log-mel spectrogram en GPU.
    El transform se construye una sola vez y se reutiliza para todo el batch.
    """

    name = "log_mel"

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        lm  = cfg.log_mel
        mel = TAT.MelSpectrogram(
            sample_rate = cfg.loading.target_sr,
            n_fft       = lm.n_fft,
            hop_length  = lm.hop_length,
            n_mels      = lm.n_mels,
            f_min       = lm.fmin,
            f_max       = lm.fmax,
        ).to(self.device)
        to_db = TAT.AmplitudeToDB(stype="power", top_db=80).to(self.device)

        out: Batch = []
        for s in self.progress(batch, desc="log-mel"):
            wav_t   = torch.from_numpy(s.waveform).to(self.device)
            log_mel = to_db(mel(wav_t)).cpu().numpy()
            out.append(AudioSample(**{**s.__dict__, "feature": log_mel}))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 9. SPECAUGMENT  (GPU)
# ─────────────────────────────────────────────────────────────────────────────

class SpecAugmentStage(BaseStage):
    """Frequency y time masking sobre el espectrograma. Corre en GPU."""

    name = "spec_augment"

    def _apply(self, spec: torch.Tensor, cfg: Any) -> torch.Tensor:
        n_mel, n_time = spec.shape
        fill = cfg.mask_value
        for _ in range(cfg.num_freq_masks):
            f  = random.randint(0, cfg.freq_mask_param)
            f0 = random.randint(0, max(0, n_mel - f))
            spec[f0:f0 + f, :] = fill
        for _ in range(cfg.num_time_masks):
            t  = random.randint(0, cfg.time_mask_param)
            t0 = random.randint(0, max(0, n_time - t))
            spec[:, t0:t0 + t] = fill
        return spec

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        sacfg = cfg.augmentation.spec_augment
        if not sacfg.enabled:
            return batch

        out: Batch = []
        for s in self.progress(batch, desc="SpecAugment"):
            if s.feature is None:
                out.append(s)
                continue
            spec = torch.from_numpy(s.feature).to(self.device)
            spec = self._apply(spec, sacfg)
            out.append(AudioSample(**{**s.__dict__, "feature": spec.cpu().numpy()}))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 10. NORMALIZATION  (GPU)
# ─────────────────────────────────────────────────────────────────────────────

class NormalizationStage(BaseStage):
    """Normalización instance o global. Corre en GPU."""

    name = "normalization"

    def __init__(self, device=None) -> None:
        super().__init__(device)
        self._global_mean: Optional[float] = None
        self._global_std:  Optional[float] = None

    def _load_global_stats(self, path: str) -> Tuple[float, float]:
        import json
        with open(path) as fh:
            stats = json.load(fh)
        return stats["mean"], stats["std"]

    def _run(self, batch: Batch, cfg: Any) -> Batch:
        ncfg     = cfg.normalization
        strategy = ncfg.strategy

        if strategy == "none":
            return batch

        if strategy == "global" and self._global_mean is None:
            self._global_mean, self._global_std = self._load_global_stats(
                ncfg.global_stats_path
            )

        out: Batch = []
        for s in self.progress(batch, desc="normalizing"):
            if s.feature is None:
                out.append(s)
                continue

            feat = torch.from_numpy(s.feature).to(self.device)
            if strategy == "instance":
                mu, sigma = feat.mean(), feat.std() + 1e-9
            else:
                mu    = self._global_mean
                sigma = self._global_std + 1e-9

            feat = (feat - mu) / sigma
            out.append(AudioSample(**{**s.__dict__, "feature": feat.cpu().numpy()}))
        return out