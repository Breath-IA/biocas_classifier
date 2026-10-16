"""
build_synth_mels.py  (gate 2 · e05)
───────────────────────────────────
Generate a LARGE synthetic W / FC set for CNN training and store, for each synthetic event, its log-mel inputs (both bands)
and its provenance, instead of 5 000 WAV files. The recordings are produced by exactly the same code as e04's
`build_synthetic.py` (same hosts, same insertion, same calibration), so a few of them can be written as WAV for inspection
with e04's builder using the same seed.

Usage : python build_synth_mels.py --name v1 --calibration <calibration.json | calibration_map.json> --n-per-class 2500 [--seed 21]
Output: cache/synth_<name>.npz (mels + metadata), results/synth_<name>_manifest.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
import cnn_data as cd  # noqa: E402
from synthesis import builder as bd  # noqa: E402

warnings.filterwarnings("ignore")


def one(job, cal_map):
    cal = cal_map.get(job["year"], cal_map["default"])
    row = bd.run_job(job, cal, keep_audio=True)
    if row is None:
        return None
    pcm = row.pop("_pcm")
    row.pop("_events", None)
    x4 = signal.resample_poly(pcm.astype(np.float64) / 32768.0, 1, 2)
    seg = x4[int(row["target_start_ms"] * cd.SR / 1000):int(row["target_end_ms"] * cd.SR / 1000)]
    if len(seg) < 64:
        return None
    return row, cd.to_mel(seg, 2000), cd.to_mel(seg, 900)


def load_cal_map(path: Path) -> dict:
    """A single calibration.json, or a JSON {"default": {...}, "2022": {...}, ...} / {"files": {...}} map of calibrations."""
    d = json.loads(path.read_text())
    if "chain" in d:
        return {"default": d}
    out = {}
    for k, v in d["files"].items():
        out[k] = json.loads((path.parent / v).read_text())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--n-per-class", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--match-hosts", action="store_true", help="choose hosts with the location/age of the real profile event")
    a = ap.parse_args()

    cal_map = load_cal_map(a.calibration)
    hosts, profiles, _ = bd.load_tables()
    kw = {"match_hosts": True} if a.match_hosts else {}
    jobs = bd.plan_jobs(hosts, profiles, a.n_per_class, bd.YEARS, a.seed, kinds=("FC", "W"), max_reuse=3, **kw)
    t0 = time.time()
    res = Parallel(n_jobs=a.workers, verbose=1)(delayed(one)(j, cal_map) for j in jobs)
    res = [r for r in res if r is not None]
    m = pd.DataFrame([r[0] for r in res])
    out = HERE / "cache" / f"synth_{a.name}.npz"
    np.savez(out, y=(m["kind"] == "FC").values.astype(np.int8), year=m["year"].astype(str).values, patient=m["patient_id"].astype(str).values,
             host_split=m["host_split"].values, mel2000=np.stack([r[1] for r in res]), mel900=np.stack([r[2] for r in res]),
             name=m["name"].values)
    m.to_csv(HERE / "results" / f"synth_{a.name}_manifest.csv", index=False)
    print(f"{len(m)} synthetic events in {time.time() - t0:.0f} s -> {out}")
    print(m.groupby(["kind", "year"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
