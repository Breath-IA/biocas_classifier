"""
build_event_cache.py — log-mel of every annotated event, one npz per band: cache/events_mel_<band>.npz  (eid, mel float16 (N, F, 95)).

Uses e05's `cnn_data.to_mel` (same recipe as every earlier experiment). float16 on disk: values are per-event standardised, so the
rounding error is < 5e-3. `--check N` recomputes N random events in float32 and fails if the cache differs by more than 1e-2.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from joblib import Parallel, delayed

import fm_common as fc

sys.path.insert(0, str(fc.E5))
sys.path.insert(0, str(fc.REPO / "experiments" / "common"))
import cnn_data as cd  # noqa: E402


def build(band: int, jobs: int, limit: int = 0):
    ev = fc.load_events()
    if limit:
        ev = ev[ev["stem"].isin(ev["stem"].drop_duplicates().iloc[:limit])]
    work = [(s, g["split"].iloc[0], list(zip(g.eid, g.start_ms, g.end_ms)), band) for s, g in ev.groupby("stem")]
    t0 = time.time()
    res = Parallel(n_jobs=jobs, verbose=0)(delayed(cd._job)(*w) for w in work)
    eids, mels = [], []
    for r in res:
        for k, m in r:
            eids.append(k)
            mels.append(m.astype(np.float16))
    print(f"band {band}: {len(eids)}/{len(ev)} events ({len(ev) - len(eids)} too short or unreadable), {time.time() - t0:.0f} s", flush=True)
    return ev, np.array(eids), np.stack(mels)


def check(ev, eids, mel, band: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    row = {e: i for i, e in enumerate(eids)}
    sub = ev[ev["eid"].isin(row)].sample(min(n, len(row)), random_state=seed)
    worst = 0.0
    for stem, g in sub.groupby("stem"):
        got = cd._job(stem, g["split"].iloc[0], list(zip(g.eid, g.start_ms, g.end_ms)), band)
        for k, m in got:
            worst = max(worst, float(np.abs(m - mel[row[k]].astype(np.float32)).max()))
    print(f"check: max |cache - cnn_data.to_mel| over {len(sub)} events = {worst:.2e}")
    assert worst < 1e-2, "cache does not reproduce cnn_data.to_mel"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band", type=int, choices=sorted(fc.BANDS), required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--check", type=int, default=200)
    ap.add_argument("--limit-recordings", type=int, default=0, help="debug: only the first N recordings")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = fc.CACHE / f"events_mel_{a.band}.npz"
    if out.exists() and not a.force and not a.limit_recordings:
        print(f"{out} exists (use --force to rebuild)")
        return
    ev, eids, mel = build(a.band, a.jobs, a.limit_recordings)
    assert mel.shape[1:] == (fc.BANDS[a.band]["n_mels"], fc.BANDS[a.band]["n_frames"]), mel.shape
    if a.check:
        check(ev, eids, mel, a.band, a.check)
    if a.limit_recordings:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, eid=eids, mel=mel)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
