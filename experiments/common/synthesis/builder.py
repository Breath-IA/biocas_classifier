"""
builder.py — plan and run synthetic recordings (shared by calibrate_synthesis.py, build_synthetic.py, qc_synthetic.py).

Hosts   : real SPRSound recordings whose annotated events are all Normal, longer than 6 s, not "Poor Quality".
Profiles: for FC / W, the duration, level (snr_vs_floor_db), crackle rate (mf_rate) and wheeze f0 (ph_mf_wh_f0) of a
          real event of the same class AND year, so the annotation style and level distribution stay natural.
Jobs    : every class (N, FC, W) goes through the same planner and the same final steps (symmetry: "synthetic"
          must not become a label cue).
"""
from __future__ import annotations

import collections
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

import extract_spectral_features as ex
import paths
import physics_features as pf

from . import FS
from .channel import ChannelModel
from .insertion import insert_event, normal_variant
from .sprsound_io import synthetic_name, write_recording

YEARS = ["2022", "2023", "2024", "2025"]
TYPE_TO_CODE = {"Normal": "N", "Fine Crackle": "FC", "Wheeze": "W", "Coarse Crackle": "CC", "Wheeze+Crackle": "WC",
                "Rhonchi": "R", "Stridor": "S"}


def default_calibration(reference: Optional[dict] = None, chain: Optional[dict] = None) -> dict:
    """Starting calibration: source morphology from the reference clips, chain from the first channel estimate."""
    fc = (reference or {}).get("generator_parameters", {}).get("FC", {})
    return {
        "chain": chain or ChannelModel().to_dict(),
        "crackle": {"f0_median_hz": fc.get("f0_median_hz", 600.0), "f0_sigma": fc.get("f0_sigma", 0.2),
                    "tau_median_ms": fc.get("tau_median_ms", 4.0), "tau_sigma": fc.get("tau_sigma", 0.45),
                    "ici_shape": fc.get("ici_shape", 1.5), "amp_sigma": fc.get("amp_sigma", 0.6), "ici_min_ms": 8.0},
        "wheeze": {"mode": "burst", "jitter_cv": 0.012, "harmonics": [1.0, 0.2, 0.04], "fade_s": 0.03, "poly_prob": 0.35,
                   "poly_ratio_range": [1.25, 2.0], "vdp_mu": 1.5, "burst_median_s": 0.22, "burst_sigma": 0.5,
                   "gap_median_s": 0.15, "f0_sd": 0.10, "glide_sd": 0.18, "amp_sigma": 0.5},
        "rate_scale": 1.0, "snr_offset_db": 0.0, "vdp_fraction": 0.15, "n_gain_sd_db": 1.5,
        "noise_chain": {"hp_hz": 50.0, "hp_order": 2, "lp_hz": 800.0, "lp_order": 4},
    }


def load_tables() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    cols = ["stem", "event_idx", "class", "source", "split", "patient_id", "age", "gender", "location", "start_ms", "end_ms",
            "duration_s", "rec_duration_s", "snr_vs_floor_db", "mf_rate"]
    seg = pd.read_csv(paths.SEGMENTS_CSV, usecols=cols)
    seg["source"] = seg["source"].astype(str)
    seg["patient_id"] = seg["patient_id"].astype(str)
    ph = pd.read_csv(paths.PHYSICS_CSV, usecols=["stem", "event_idx", "ph_mf_wh_f0"])
    seg = seg.merge(ph, on=["stem", "event_idx"], how="left")
    only_n = seg.groupby("stem")["class"].agg(lambda s: set(s) == {"N"})
    hosts = seg[seg["stem"].isin(only_n[only_n].index)].drop_duplicates("stem")
    hosts = hosts[hosts["rec_duration_s"] >= 6.0][["stem", "source", "split", "patient_id", "age", "gender", "location",
                                                   "rec_duration_s"]].reset_index(drop=True)
    return hosts, {c: seg[seg["class"] == c] for c in ("FC", "W")}, seg


def _profile(kind: str, year: str, profiles, rng) -> Dict:
    p = profiles[kind]
    q = p[p["source"] == year]
    r = q.iloc[int(rng.integers(len(q)))]
    out = {"duration_s": float(r["duration_s"]), "snr_vs_floor_db": float(r["snr_vs_floor_db"]),
           "real_stem": r["stem"], "real_event_idx": int(r["event_idx"]), "location": str(r["location"]),
           "age": float(r["age"]) if np.isfinite(r["age"]) else float("nan")}
    if kind == "FC":
        rate = r["mf_rate"]
        out["rate_hz"] = float(rate) if np.isfinite(rate) and rate > 1 else float(np.nanmedian(q["mf_rate"]))
    else:
        f0 = r["ph_mf_wh_f0"]
        f0 = float(f0) if np.isfinite(f0) else float(np.nanmedian(q["ph_mf_wh_f0"]))
        out["f0_hz"] = float(np.clip(f0 + rng.uniform(-25, 25), 100, 700))
    return out


def plan_jobs(hosts: pd.DataFrame, profiles, n_per_class: int, years: List[str], seed: int,
              kinds=("N", "FC", "W"), max_reuse: int = 2, start_idx: int = 0, match_hosts: bool = False) -> List[Dict]:
    rng = np.random.default_rng(seed)
    hosts = hosts[hosts["source"].isin(years)].reset_index(drop=True)
    w = hosts["source"].value_counts(normalize=True)
    used = collections.Counter()
    jobs = []
    idx = start_idx
    for _ in range(n_per_class):
        for kind in kinds:
            year = str(rng.choice(w.index.tolist(), p=w.values))
            pool = hosts[hosts["source"] == year]
            # the profile is drawn early only when hosts are matched; otherwise the original draw order is kept so that
            # e04's datasets stay reproducible with their seeds
            prof = _profile(kind, year, profiles, rng) if (match_hosts and kind != "N") else None
            if match_hosts and prof is not None:       # same chest location and similar age as the real event being imitated
                cand = pool[(pool["location"] == prof["location"]) & ((pool["age"] - prof["age"]).abs() <= 3)]
                if len(cand) < 5:
                    cand = pool[pool["location"] == prof["location"]]
                pool = cand if len(cand) >= 5 else pool
            h = pool.iloc[int(rng.integers(len(pool)))]
            for _ in range(20):                                     # cap the reuse of a host recording
                if used[h["stem"]] < max_reuse:
                    break
                h = pool.iloc[int(rng.integers(len(pool)))]
            used[h["stem"]] += 1
            job = {"idx": idx, "kind": kind, "year": year, "seed": int(rng.integers(2 ** 31 - 1)), "host_stem": h["stem"],
                   "host_split": h["split"], "patient_id": h["patient_id"], "age": h["age"], "gender": int(h["gender"]),
                   "location": h["location"]}
            if kind == "N":
                job["noise_stem"] = pool.iloc[int(rng.integers(len(pool)))]["stem"]
                job["noise_split"] = pool[pool["stem"] == job["noise_stem"]]["split"].iloc[0]
            else:
                job["profile"] = prof if prof is not None else _profile(kind, year, profiles, rng)
            jobs.append(job)
            idx += 1
    return jobs


def _read_host(stem: str, split: str):
    x, sr = sf.read(str(paths.RAW / split / "wav" / f"{stem}.wav"), dtype="float64")
    assert sr == FS, f"{stem}: expected {FS} Hz"
    d = json.loads((paths.RAW / split / "json" / f"{stem}.json").read_text(encoding="utf-8"))
    ev = [{"start": int(e["start"]), "end": int(e["end"]), "type": e["type"]} for e in d.get("event_annotation", [])]
    return x, ev, str(d.get("record_annotation"))


def run_job(job: Dict, cal: Dict, out_dir=None, keep_audio: bool = False) -> Optional[Dict]:
    """Generate one synthetic recording. With `out_dir` the WAV/JSON are written; always returns the manifest row."""
    rng = np.random.default_rng(job["seed"])
    x, events, tag = _read_host(job["host_stem"], job["host_split"])
    if "poor" in tag.lower():
        return None
    if job["kind"] == "N":
        noise = None
        if job.get("noise_stem"):
            noise = _read_host(job["noise_stem"], job["noise_split"])[0]
        res = normal_variant(x, events, rng, noise, cal)
    else:
        chain = cal["chain_W"] if (job["kind"] == "W" and "chain_W" in cal) else cal["chain"]   # effective response per source type
        res = insert_event(x, events, job["kind"], rng, job["profile"], ChannelModel.from_dict(chain), cal)
    name = synthetic_name(job["year"], job["idx"], job["age"], job["gender"], job["location"])
    prov = {"kind": job["kind"], "host": job["host_stem"], "year": job["year"], "train_only": True,
            "profile_from": job.get("profile", {}).get("real_stem"), **{k: v for k, v in res["info"].items()}}
    if out_dir is not None:
        write_recording(out_dir, name, res["pcm"], res["record_annotation"], res["events"], prov)
    row = {"name": name, "kind": job["kind"], "year": job["year"], "host_stem": job["host_stem"], "host_split": job["host_split"],
           "patient_id": job["patient_id"], "age": job["age"], "gender": job["gender"], "location": job["location"],
           "train_only": True, "record_annotation": res["record_annotation"], "n_events": len(res["events"]),
           **{f"info_{k}": v for k, v in res["info"].items() if np.isscalar(v)}}
    if job["kind"] != "N":
        row["profile_real_stem"] = job["profile"]["real_stem"]
        row["target_start_ms"] = int(round(res["info"]["start_s"] * 1000))
        row["target_end_ms"] = int(round((res["info"]["start_s"] + res["info"]["duration_s"]) * 1000))
    if keep_audio:
        row["_pcm"], row["_events"] = res["pcm"], res["events"]
    return row


def extract_features(pcm: np.ndarray, targets: List[Tuple[int, str, int, int]], stem: str = "syn") -> pd.DataFrame:
    """
    The same extractors as for real events (spectral/temporal features + physics features) on an int16 8 kHz
    recording. `targets` = [(event_idx, class_code, start_ms, end_ms)].
    """
    x = signal.resample_poly(pcm.astype(np.float64) / 32768.0, 1, 2).astype(np.float32)    # as ex.load_wav does
    rows = ex.process_waveform(x, [(k, c, s, e) for k, c, s, e in targets], {"stem": stem}, False)
    ph = pf.process_array(x, stem, [(k, s, e) for k, _, s, e in targets])
    a = pd.DataFrame(rows)
    b = pd.DataFrame(ph).drop(columns=["stem"], errors="ignore")
    return a.merge(b, on="event_idx", how="left") if len(a) and len(b) else a
