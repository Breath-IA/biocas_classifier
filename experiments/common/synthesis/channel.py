"""
channel.py — SPRSound acquisition chain.

SPRSound recordings are 8 kHz, mono, PCM-16 from an electronic stethoscope, and their median spectrum is
band-limited: it falls to the 16-bit quantization floor above ~1 kHz. A synthetic event therefore has to go
through the same kind of response before being mixed into a real recording.

The chain is applied with a *causal* filter (like a real device). Zero-phase filtering would make crackle onsets
symmetric, which real recordings do not show.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd

import numpy as np
from scipy import signal

from . import FS


@dataclass
class ChannelModel:
    """Band-pass chain (high-pass sensor/DC removal + low-pass lung-mode / chest-wall response)."""

    hp_hz: float = 50.0
    hp_order: int = 2
    lp_hz: float = 800.0
    lp_order: int = 4
    fs: int = FS

    def sos(self) -> np.ndarray:
        hp = signal.butter(self.hp_order, self.hp_hz, "highpass", fs=self.fs, output="sos")
        lp = signal.butter(self.lp_order, self.lp_hz, "lowpass", fs=self.fs, output="sos")
        return np.vstack([hp, lp])

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Causal filtering of a synthetic source event."""
        return signal.sosfilt(self.sos(), np.asarray(x, dtype=np.float64))

    def response_db(self, freqs_hz: np.ndarray) -> np.ndarray:
        _, h = signal.sosfreqz(self.sos(), worN=freqs_hz, fs=self.fs)
        return 20 * np.log10(np.abs(h) + 1e-12)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ChannelModel":
        return ChannelModel(**{k: d[k] for k in ("hp_hz", "hp_order", "lp_hz", "lp_order", "fs") if k in d})


def to_native_rate(x: np.ndarray, fs_in: int, fs_out: int = FS) -> np.ndarray:
    """Polyphase resampling with anti-aliasing (used for the 44.1 kHz reference clips)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    g = gcd(int(fs_in), int(fs_out))
    return signal.resample_poly(np.asarray(x, dtype=np.float64), fs_out // g, fs_in // g)


def quantize_pcm16(x: np.ndarray) -> np.ndarray:
    """Float in [-1, 1) → int16, exactly like the dataset's PCM-16 grid (no dither, no normalization)."""
    return np.clip(np.round(np.asarray(x) * 32768.0), -32768, 32767).astype(np.int16)
