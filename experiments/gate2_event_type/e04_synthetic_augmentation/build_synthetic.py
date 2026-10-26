"""
build_synthetic.py  (gate 2 · e04, step 4)
──────────────────────────────────────────
Build synthetic recordings for N, FC and W in the SPRSound format (8 kHz mono PCM-16 WAV + event-level JSON).

Every recording is a REAL SPRSound normal recording ("host"):
  N  : the host with a natural variation (±gain, real ambient noise at its own floor, annotation boundary jitter)
  FC : the host + a parametric crackle train inside a breath burst, annotated "Fine Crackle"   (record tag DAS)
  W  : the host + a parametric wheeze inside a breath burst, annotated "Wheeze"                (record tag CAS)
The three classes use the same planner and the same final steps. Nothing is trained here.

Synthetic data are TRAIN-ONLY. The manifest keeps the host year / patient / split so that a leave-one-year-out
experiment can drop the held-out year (synthesis.sprsound_io.load_manifest(exclude_years=[...])). For the official
split (train 2022-24, test 2025) drop `host_split == "test"` as well.

Usage : python experiments/gate2_event_type/e04_synthetic_augmentation/build_synthetic.py --n-per-class 300
Output: data/synthetic/{wav,json}/..., data/synthetic/manifest.csv, results/manifest.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402
from synthesis import builder as bd  # noqa: E402

warnings.filterwarnings("ignore")
RES = paths.G2_E04 / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--years", nargs="+", default=bd.YEARS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--calibration", type=Path, default=RES / "calibration.json")
    ap.add_argument("--out", type=Path, default=paths.SYNTH_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if a.out.exists() and any(a.out.iterdir()) and not a.overwrite:
        sys.exit(f"{a.out} is not empty; pass --overwrite to rebuild it.")
    if a.out.exists() and a.overwrite:
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.calibration.exists():
        cal = json.loads(a.calibration.read_text())
        print("calibration:", a.calibration)
    else:
        reference = json.loads((RES / "reference_params.json").read_text())
        chain = json.loads((RES / "channel_estimate.json").read_text())
        cal = bd.default_calibration(reference, chain)
        print("WARNING: calibration.json not found, using the first-pass parameters")
    hosts, profiles, _ = bd.load_tables()
    jobs = bd.plan_jobs(hosts, profiles, a.n_per_class, a.years, a.seed)
    print(f"{len(jobs)} recordings ({a.n_per_class} per class) from {len(hosts)} candidate host recordings")
    rows = Parallel(n_jobs=a.workers, verbose=2)(delayed(bd.run_job)(j, cal, a.out) for j in jobs)
    m = pd.DataFrame([r for r in rows if r is not None])
    m.to_csv(a.out / "manifest.csv", index=False)
    m.to_csv(RES / "manifest.csv", index=False)
    (a.out / "calibration_used.json").write_text(json.dumps(cal, indent=2, default=float), encoding="utf-8")
    print(m.groupby(["kind", "year"]).size().unstack(fill_value=0).to_string())
    print("host reuse (max):", m["host_stem"].value_counts().max(), "| written to", a.out)


if __name__ == "__main__":
    main()
