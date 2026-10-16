"""
generators.py — parametric sources from plan.md ("Physical reference models").

Crackle  : y(t) = A · exp(-t/τ) · sin(2π f0 t + φ)            (damped oscillation; fine: short τ, higher f0)
Wheeze   : self-sustained oscillation. Real SPRSound wheezes are NOT long steady harmonic stacks: they are weak,
           intermittent, short tonal bursts (≈ 0.1–0.4 s) that often glide in frequency, with one or two partials.
           The default model is therefore a train of gliding bursts ("burst"); the steady model of the first
           version ("steady") is kept for reference. Each tone is either a frequency-modulated harmonic series
           ("fm") or a van der Pol oscillator ("vdp").

All sources are generated at the dataset's native 8 kHz and are *unit-peak, pre-channel*: the amplitude is set
later by the insertion step so that the event reaches a target level relative to the host recording.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal

from . import FS


# ── crackles ──────────────────────────────────────────────────────────────────
@dataclass
class CrackleParams:
    """Distribution of single-crackle morphology and of the crackle train."""

    f0_median_hz: float = 500.0          # damped-sine frequency (log-normal)
    f0_sigma: float = 0.25               # log-sd
    tau_median_ms: float = 3.0           # decay constant (log-normal)
    tau_sigma: float = 0.35
    phase_range: Tuple[float, float] = (-0.3, 0.6)    # rad; small phases keep a sharp initial deflection
    amp_sigma: float = 0.5               # log-normal sd of crackle amplitudes inside a train
    ici_min_ms: float = 8.0              # refractory time between crackles
    ici_shape: float = 1.5               # gamma shape of the inter-crackle interval (1 = Poisson)
    n_tau: float = 6.0                   # waveform length in decay constants
    random_polarity: bool = True         # real crackle polarity depends on sensor orientation: ± with equal probability


def crackle_waveform(fs: int, f0: float, tau_s: float, phase: float = 0.0, n_tau: float = 6.0) -> np.ndarray:
    n = int(max(8, round(n_tau * tau_s * fs)))
    t = np.arange(n) / fs
    w = np.exp(-t / tau_s) * np.sin(2 * np.pi * f0 * t + phase)
    return w / (np.max(np.abs(w)) + 1e-12)


def synth_crackle_train(duration_s: float, rate_hz: float, rng: np.random.Generator,
                        p: Optional[CrackleParams] = None, fs: int = FS):
    """
    Crackle train of `duration_s`: gamma-distributed intervals (with refractory time), log-normal amplitudes,
    per-crackle (f0, τ) and random polarity. Returns (signal, info) with the onsets, amplitudes, f0 and τ of every crackle.
    """
    p = p or CrackleParams()
    n = int(round(duration_s * fs))
    y = np.zeros(n + int(0.05 * fs))
    mean_ici = max(1.0 / max(rate_hz, 0.5), (p.ici_min_ms + 1.0) / 1000.0)
    scale = max(mean_ici - p.ici_min_ms / 1000.0, 1e-3) / p.ici_shape
    t = rng.uniform(0.0, mean_ici * 0.5)
    onsets, amps, f0s, taus = [], [], [], []
    while t < duration_s - 0.004:
        f0 = float(np.exp(rng.normal(np.log(p.f0_median_hz), p.f0_sigma)))
        tau = float(np.exp(rng.normal(np.log(p.tau_median_ms / 1000.0), p.tau_sigma)))
        a = float(np.exp(rng.normal(0.0, p.amp_sigma))) * (float(rng.choice([-1.0, 1.0])) if p.random_polarity else 1.0)
        w = crackle_waveform(fs, f0, tau, float(rng.uniform(*p.phase_range)), p.n_tau) * a
        i = int(round(t * fs))
        y[i:i + len(w)] += w[: len(y) - i]
        onsets.append(t); amps.append(abs(a)); f0s.append(f0); taus.append(tau)
        t += p.ici_min_ms / 1000.0 + rng.gamma(p.ici_shape, scale)
    if len(onsets) < 3:                                  # an annotated crackle event always holds a few crackles
        return synth_crackle_train(duration_s, rate_hz * 2.0, rng, p, fs)
    y = y[:n]
    return y / (np.max(np.abs(y)) + 1e-12), {"onsets_s": np.array(onsets), "amps": np.array(amps),
                                             "f0_hz": np.array(f0s), "tau_s": np.array(taus)}


# ── wheezes ───────────────────────────────────────────────────────────────────
@dataclass
class WheezeParams:
    f0_hz: float = 300.0
    mode: str = "burst"                  # "burst" (gliding tonal bursts, default) | "steady" (one continuous tone)
    jitter_cv: float = 0.012             # std of the fast f0 fluctuation, relative to f0
    drift: float = 0.03                  # steady mode: total relative f0 drift over the event
    harmonics: Tuple[float, ...] = (1.0, 0.2, 0.04)   # linear amplitudes of harmonics 1..K (real: 2f0 ≈ -14..-24 dB, 3f0 ≈ -28..-48 dB)
    fade_s: float = 0.03                 # raised-cosine attack / release of every burst
    model: str = "fm"                    # tone model: "fm" | "vdp"
    vdp_mu: float = 1.5
    poly_prob: float = 0.35              # probability of a second, parallel tone (real: ≈ 1/3 of the events)
    poly_ratio_range: Tuple[float, float] = (1.25, 2.0)
    # burst mode
    burst_median_s: float = 0.22         # tonal burst duration (log-normal)
    burst_sigma: float = 0.5
    gap_median_s: float = 0.15           # silence between consecutive bursts (log-normal)
    f0_sd: float = 0.10                  # log-sd of the burst-to-burst f0 change
    glide_sd: float = 0.18               # relative f0 change across a burst (linear glide)
    amp_sigma: float = 0.5               # log-normal sd of burst amplitudes
    tremolo_hz: float = 0.0              # steady mode only
    tremolo_depth: float = 0.0
    # irregularity of real wheeze windows (v2): slow random envelope and low-level transient clicks
    slow_env_depth: float = 0.0          # depth of a slow (slow_env_hz) random amplitude modulation
    slow_env_hz: float = 3.0
    click_rate_hz: float = 0.0           # Poisson rate of crackle-like clicks (breath / heart-sound / contact transients)
    click_level: float = 0.15            # click peak relative to the tone peak


def _jitter(n: int, fs: int, cutoff_hz: float, rng: np.random.Generator) -> np.ndarray:
    sos = signal.butter(2, cutoff_hz, "low", fs=fs, output="sos")
    j = signal.sosfilt(sos, rng.standard_normal(n))
    return j / (j.std() + 1e-12)


def _vdp(phase: np.ndarray, mu: float) -> np.ndarray:
    """Van der Pol oscillator x'' − μ(1−x²)x' + x = 0 integrated in the instantaneous phase (RK4)."""
    dphi = np.diff(phase, prepend=phase[0])            # radians per sample
    x, v = 0.1, 0.0
    out = np.empty(len(phase))
    for i, h in enumerate(dphi):
        def f(x_, v_):
            return v_, mu * (1 - x_ * x_) * v_ - x_
        k1 = f(x, v); k2 = f(x + .5 * h * k1[0], v + .5 * h * k1[1])
        k3 = f(x + .5 * h * k2[0], v + .5 * h * k2[1]); k4 = f(x + h * k3[0], v + h * k3[1])
        x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        out[i] = x
    return out / (np.max(np.abs(out)) + 1e-12)


def _tone_from_f(f_inst: np.ndarray, p: WheezeParams, fs: int, rng: np.random.Generator) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    if p.model == "vdp":
        return _vdp(phase, p.vdp_mu)
    f_max = float(np.max(f_inst))
    return sum(a * np.sin((k + 1) * phase + rng.uniform(0, 2 * np.pi)) for k, a in enumerate(p.harmonics) if (k + 1) * f_max < fs / 2)


def _tone(n: int, fs: int, f0: float, p: WheezeParams, rng: np.random.Generator) -> np.ndarray:
    """One continuous tone of n samples (steady mode)."""
    t = np.arange(n) / fs
    T = max(n / fs, 1e-3)
    f_inst = f0 * (1 + p.drift * (t / T - 0.5) * 2 * rng.choice([-1, 1]) + p.jitter_cv * _jitter(n, fs, 6.0, rng))
    y = _tone_from_f(f_inst, p, fs, rng)
    if p.tremolo_depth > 0 and p.tremolo_hz > 0:
        y = y * (1 + p.tremolo_depth * np.sin(2 * np.pi * p.tremolo_hz * t + rng.uniform(0, 2 * np.pi)))
    return y


def _fade(n: int, fade_s: float, fs: int) -> np.ndarray:
    nf = int(min(fade_s * fs, n // 2))
    env = np.ones(n)
    if nf > 1:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
        env[:nf] = ramp
        env[-nf:] = ramp[::-1]
    return env


def _burst_train(n: int, fs: int, f0: float, p: WheezeParams, rng: np.random.Generator, lock=None):
    """
    A train of short gliding tonal bursts over n samples. `lock` (starts, durations, amps, glides, f0 changes) makes a
    second, parallel tone follow the same burst timing.
    """
    y = np.zeros(n)
    plan = []
    t = float(rng.uniform(0.0, p.gap_median_s))
    while t < n / fs - 0.08:
        d = float(np.clip(np.exp(rng.normal(np.log(p.burst_median_s), p.burst_sigma)), 0.08, 0.7))
        d = min(d, n / fs - t)
        plan.append((t, d, float(np.exp(rng.normal(0.0, p.amp_sigma))), float(rng.normal(0.0, p.glide_sd)),
                     float(np.exp(rng.normal(0.0, p.f0_sd)))))
        t += d + float(np.exp(rng.normal(np.log(p.gap_median_s), 0.5)))
    if not plan:
        plan = [(0.0, n / fs, 1.0, 0.0, 1.0)]
    if lock is not None:
        plan = lock
    for (t0, d, a, g, fr) in plan:
        i0, m = int(t0 * fs), int(d * fs)
        if m < 16 or i0 + m > n:
            m = min(m, n - i0)
            if m < 16:
                continue
        fb = f0 * fr * (1 + g * (np.arange(m) / m - 0.5)) * (1 + p.jitter_cv * _jitter(m, fs, 8.0, rng))
        y[i0:i0 + m] += a * _tone_from_f(fb, p, fs, rng) * _fade(m, p.fade_s, fs)
    return y, plan


def synth_wheeze(duration_s: float, rng: np.random.Generator, p: Optional[WheezeParams] = None, fs: int = FS):
    """Wheeze of `duration_s` (optionally with a parallel second tone). Returns (signal, info)."""
    p = p or WheezeParams()
    n = int(round(duration_s * fs))
    f0s = [p.f0_hz]
    if p.mode == "steady":
        y = _tone(n, fs, p.f0_hz, p, rng)
        if rng.random() < p.poly_prob:
            f2 = p.f0_hz * float(rng.uniform(*p.poly_ratio_range))
            y = y + float(rng.uniform(0.4, 0.9)) * _tone(n, fs, f2, p, rng)
            f0s.append(f2)
        y = y * _fade(n, 0.06, fs)
        n_bursts = 1
    else:
        y, plan = _burst_train(n, fs, p.f0_hz, p, rng)
        n_bursts = len(plan)
        if rng.random() < p.poly_prob:
            f2 = p.f0_hz * float(rng.uniform(*p.poly_ratio_range))
            y2, _ = _burst_train(n, fs, f2, p, rng, lock=plan)
            y = y + float(rng.uniform(0.4, 0.9)) * y2
            f0s.append(f2)
    if p.slow_env_depth > 0:
        y = y * np.clip(1 + p.slow_env_depth * _jitter(n, fs, p.slow_env_hz, rng), 0.05, None)
    y = y / (np.max(np.abs(y)) + 1e-12)
    if p.click_rate_hz > 0:
        yc = np.zeros(n + 400)
        t = float(rng.exponential(1.0 / p.click_rate_hz))
        while t < duration_s - 0.01:
            w = crackle_waveform(fs, float(rng.uniform(250, 600)), float(rng.uniform(0.002, 0.005)), float(rng.uniform(-0.3, 0.6)))
            w = w * float(rng.choice([-1.0, 1.0])) * float(np.exp(rng.normal(0.0, 0.5)))
            i = int(t * fs)
            yc[i:i + len(w)] += w[: len(yc) - i]
            t += float(rng.exponential(1.0 / p.click_rate_hz))
        if np.max(np.abs(yc)) > 0:
            y = y + p.click_level * yc[:n] / np.max(np.abs(yc))
    return y / (np.max(np.abs(y)) + 1e-12), {"f0_hz": np.array(f0s), "model": p.model, "n_tones": len(f0s), "n_bursts": n_bursts}
