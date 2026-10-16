"""
insertion.py — put a synthetic event into a REAL SPRSound recording, the way the dataset itself is built.

Steps for a crackle / wheeze recording (`insert_event`):
  1. find the host's breath-sound bursts (events are placed inside bursts, never in the silent gaps);
  2. generate the source (generators.py) and pass it through the SPRSound channel model (channel.py);
  3. choose the position inside a burst (crackles early-to-mid burst, wheezes late: plan.md phase-aware rule,
     without claiming to know inspiration vs. expiration);
  4. scale the event so that the *total* level of the annotated window reaches a target `snr_vs_floor_db`
     drawn from a real event of the same class and year (same definition as the feature extractor);
  5. quantize to PCM-16 (no normalization) and write the event-level annotation in the dataset's style.

`normal_variant` applies the same final steps to a real normal recording with a natural variation (small gain,
real ambient noise from another normal recording of the same year, annotation boundary jitter), so that N, FC and
W recordings are produced by the same pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

import extract_spectral_features as ex  # experiments/common (on sys.path)

from . import FS
from .channel import ChannelModel, quantize_pcm16
from .generators import CrackleParams, WheezeParams, synth_crackle_train, synth_wheeze
from .reference import breath_segments

TYPE_NAME = {"N": "Normal", "FC": "Fine Crackle", "W": "Wheeze"}
RECORD_TAG = {"N": "Normal", "FC": "DAS", "W": "CAS"}      # DAS = discontinuous (crackles), CAS = continuous (wheezes)


# ── level measurement (same definition as extract_spectral_features.snr_vs_floor_db) ──────────────
def frame_energy_db(x8k: np.ndarray) -> np.ndarray:
    S = ex.stft_power(signal.resample_poly(x8k, 1, 2))
    return 10 * np.log10(S[ex.MASK].sum(0) + 1e-12)


def floor_db(x8k: np.ndarray) -> float:
    """10th percentile of the per-frame energy (dB) of the whole recording, like the extractor."""
    return float(np.percentile(frame_energy_db(x8k), 10))


def event_snr_db(x8k: np.ndarray, start_s: float, end_s: float, floor: float) -> float:
    """Mean frame energy inside [start, end] (dB) minus the recording floor; computed on a local excerpt."""
    i0 = max(int((start_s - 0.25) * FS) // 128 * 128, 0)
    i1 = min(int((end_s + 0.25) * FS), len(x8k))
    S = ex.stft_power(signal.resample_poly(x8k[i0:i1], 1, 2))
    e = S[ex.MASK].sum(0)
    t = i0 / FS + np.arange(S.shape[1]) * ex.HOP / ex.SR
    sel = (t >= start_s) & (t <= end_s)
    if not sel.any():
        sel[np.argmin(np.abs(t - 0.5 * (start_s + end_s)))] = True
    return float(10 * np.log10(e[sel].mean() + 1e-12) - floor)


def solve_gain_db(host: np.ndarray, ev_full: np.ndarray, start_s: float, end_s: float, target_db: float,
                  floor: float, iters: int = 14) -> float:
    """Bisection on the event gain (dB re unit peak) so that the window reaches `target_db` above the floor."""
    lo, hi = -70.0, 25.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if event_snr_db(host + 10 ** (mid / 20) * ev_full, start_s, end_s, floor) < target_db:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── placement ────────────────────────────────────────────────────────────────────────────────
def breath_bursts(host: np.ndarray) -> List[Tuple[int, int]]:
    return breath_segments(host, FS, band=(60, 600), min_len_s=0.3, rel_thr=0.15)


def choose_start(rng: np.random.Generator, bursts, dur_s: float, n: int, kind: str) -> Tuple[int, Dict]:
    """Start sample of the event. Crackles prefer the first part of a burst, wheezes the last part."""
    dur = int(dur_s * FS)
    margin = int(0.2 * FS)
    lo_all, hi_all = margin, n - dur - margin
    if hi_all <= lo_all:
        return lo_all, {"burst": -1, "burst_pos": np.nan}
    fit = [i for i, (a, b) in enumerate(bursts) if (b - a) >= 0.5 * dur and a < hi_all]
    if not fit:
        return int(rng.integers(lo_all, hi_all)), {"burst": -1, "burst_pos": np.nan}
    w = np.array([bursts[i][1] - bursts[i][0] for i in fit], float)
    i = int(rng.choice(fit, p=w / w.sum()))
    a, b = bursts[i]
    span = max(b - a - dur, 0)
    frac = float(rng.beta(1.5, 2.5) if kind == "FC" else rng.beta(2.5, 1.5))
    start = int(np.clip(a + frac * span - (0 if span > 0 else 0.15 * dur), lo_all, hi_all))
    return start, {"burst": i, "burst_pos": float((start - a) / max(b - a, 1))}


# ── annotations ──────────────────────────────────────────────────────────────────────────────
def merge_annotations(host_events: List[Dict], new: Dict) -> List[Dict]:
    """Keep the host events outside the new event; trim partial overlaps, drop what would be < 0.25 s."""
    out = []
    for e in host_events:
        s, t = int(e["start"]), int(e["end"])
        if t <= new["start"] or s >= new["end"]:
            out.append(e)
            continue
        pieces = []
        if s < new["start"]:
            pieces.append((s, new["start"]))
        if t > new["end"]:
            pieces.append((new["end"], t))
        for a, b in pieces:
            if b - a >= 250:
                out.append({"start": a, "end": b, "type": e["type"]})
    out.append(new)
    return sorted(out, key=lambda e: int(e["start"]))


def _norm_events(events) -> List[Dict]:
    return [{"start": int(e["start"]), "end": int(e["end"]), "type": e["type"]} for e in events]


# ── recording builders ───────────────────────────────────────────────────────────────────────
def insert_event(host: np.ndarray, host_events: List[Dict], kind: str, rng: np.random.Generator,
                 profile: Dict, chain: ChannelModel, cal: Dict) -> Dict:
    """
    kind: "FC" or "W". `profile` holds the real-event statistics to imitate:
        duration_s, snr_vs_floor_db, and rate_hz (FC) or f0_hz (W).
    `cal`: calibration (rate_scale, snr_offset_db, crackle / wheeze parameter dicts).
    """
    n = len(host)
    dur = float(np.clip(profile["duration_s"], 0.4 if kind == "FC" else 0.3, min(3.5, n / FS - 0.8)))
    if kind == "FC":
        cp = CrackleParams(**cal["crackle"])
        src, info = synth_crackle_train(dur, max(profile["rate_hz"] * cal.get("rate_scale", 1.0), 3.0), rng, cp)
        extra = {"n_crackles": int(len(info["onsets_s"])), "f0_med_hz": float(np.median(info["f0_hz"])),
                 "tau_med_ms": float(np.median(info["tau_s"]) * 1000)}
    else:
        wp = WheezeParams(**{**cal["wheeze"], "f0_hz": float(profile["f0_hz"]),
                             "model": "vdp" if rng.random() < cal.get("vdp_fraction", 0.3) else "fm"})
        src, info = synth_wheeze(dur, rng, wp)
        extra = {"f0_hz": float(info["f0_hz"][0]), "n_tones": int(info["n_tones"]), "model": info["model"]}
    ev = chain.apply(src)
    ev = ev / (np.max(np.abs(ev)) + 1e-12)

    bursts = breath_bursts(host)
    fl = floor_db(host)
    want = profile["snr_vs_floor_db"] + cal.get("snr_offset_db", 0.0)
    best = None
    for _ in range(8):          # the event can only ADD energy, so prefer a host window quiet enough for the real level
        st, pl = choose_start(rng, bursts, dur, n, kind)
        hs = event_snr_db(host, st / FS, (st + len(ev)) / FS, fl)
        if best is None or hs < best[2]:
            best = (st, pl, hs)
        if hs <= want - 0.8:
            break
    start, place, host_snr = best
    ev_full = np.zeros(n)
    m = min(len(ev), n - start)
    ev_full[start:start + m] = ev[:m]
    s_s, e_s = start / FS, (start + len(ev)) / FS
    target = max(want, host_snr + float(rng.uniform(0.3, 2.0)))
    g_db = solve_gain_db(host, ev_full, s_s, e_s, target, fl)
    mix = host + 10 ** (g_db / 20) * ev_full
    peak = np.max(np.abs(mix))
    if peak > 0.98:                                   # never clip: reduce the event only
        g_db = solve_gain_db(host, ev_full, s_s, e_s, target - 3.0, fl)
        mix = host + 10 ** (g_db / 20) * ev_full
    pcm = quantize_pcm16(mix)
    final = pcm.astype(np.float64) / 32768.0
    new = {"start": int(round(s_s * 1000)), "end": int(round(e_s * 1000)), "type": TYPE_NAME[kind]}
    return {"pcm": pcm, "events": merge_annotations(_norm_events(host_events), new), "record_annotation": RECORD_TAG[kind],
            "info": {"kind": kind, "duration_s": dur, "start_s": s_s, "target_snr_db": float(target), "host_snr_db": float(host_snr),
                     "achieved_snr_db": event_snr_db(final, s_s, e_s, fl), "gain_db": float(g_db), "floor_db": fl,
                     "n_bursts": len(bursts), **place, **extra}}


def quiet_noise(x: np.ndarray, rng: np.random.Generator, length: int, win_s: float = 1.0) -> np.ndarray:
    """A real ambient-noise excerpt: the quietest `win_s` window of a recording, tiled to `length` samples."""
    w = int(win_s * FS)
    if len(x) <= w:
        seg = x
    else:
        e = np.convolve(x ** 2, np.ones(w) / w, mode="valid")
        cand = np.flatnonzero(e <= np.percentile(e, 15))
        i = int(rng.choice(cand))
        seg = x[i:i + w]
    seg = seg - seg.mean()
    return np.resize(seg, length)


def normal_variant(host: np.ndarray, host_events: List[Dict], rng: np.random.Generator, noise_src: Optional[np.ndarray],
                   cal: Dict) -> Dict:
    """Natural variation of a real normal recording: ±gain, real ambient noise at the host's floor, boundary jitter."""
    n = len(host)
    gain_db = float(np.clip(rng.normal(0.0, cal.get("n_gain_sd_db", 1.5)), -3.0, 3.0))
    mix = host * 10 ** (gain_db / 20)
    noise_db = np.nan
    if noise_src is not None:
        fl = floor_db(host)
        noise = quiet_noise(noise_src, rng, n)
        if cal.get("noise_chain"):                                              # same device band-limit as the rest of the recording
            noise = ChannelModel.from_dict(cal["noise_chain"]).apply(noise)
        target_floor = fl + float(rng.uniform(-3.0, 3.0))                    # noise level around the host's own floor
        lo, hi = -90.0, 0.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if floor_db(mix + 10 ** (mid / 20) * noise / (np.max(np.abs(noise)) + 1e-12)) < target_floor:
                lo = mid
            else:
                hi = mid
        noise_db = 0.5 * (lo + hi)
        mix = mix + 10 ** (noise_db / 20) * noise / (np.max(np.abs(noise)) + 1e-12)
    pcm = quantize_pcm16(np.clip(mix, -1, 32767 / 32768))
    ev = []
    for e in _norm_events(host_events):
        d = e["end"] - e["start"]
        a = int(np.clip(e["start"] + rng.uniform(-0.1, 0.1) * d, 0, n / FS * 1000 - 300))
        b = int(np.clip(e["end"] + rng.uniform(-0.1, 0.1) * d, a + 250, n / FS * 1000))
        ev.append({"start": a, "end": b, "type": e["type"]})
    ev = sorted(ev, key=lambda e: e["start"])             # keep the events ordered and non-overlapping after the jitter
    for i in range(1, len(ev)):
        if ev[i]["start"] < ev[i - 1]["end"]:
            ev[i]["start"] = ev[i - 1]["end"]
    ev = [e for e in ev if e["end"] - e["start"] >= 250]
    return {"pcm": pcm, "events": ev, "record_annotation": "Normal",
            "info": {"kind": "N", "gain_db": gain_db, "noise_gain_db": float(noise_db)}}
