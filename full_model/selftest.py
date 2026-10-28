"""
selftest.py — runs the WHOLE full_model process end to end with tiny settings (CPU, a few minutes) to prove that every stage works on this machine.

Used by run_full_model_test.bat (Windows) and `make selftest` (Linux/macOS). It never touches the real splits or results: it sets
FM_SPLITS_DIR=full_model/splits_test and FM_RESULTS_DIR=full_model/results_test (both git-ignored); only the mel cache is shared (it does not depend on splits).

Stages: files -> cache -> splits -> check-budget -> search g1/g2 -> status -> resume -> STOP file -> refit -> cascade -> final -> fpga -> outputs.
Exit status 0 only when every stage succeeded.  Options: --device cpu|cuda  --skip-cache  --keep (do not wipe results_test first)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

FM = Path(__file__).resolve().parent
REPO = FM.parent
RES, SPL = FM / "results_test", FM / "splits_test"     # suffixed with _s<seed> when --seed is not 42


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--k-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42, help="seed of splits, search, refit and final test; a seed other than 42 uses its own results_test_s<seed>/ folders")
    a = ap.parse_args()
    global RES, SPL
    if a.seed != 42:
        RES, SPL = FM / f"results_test_s{a.seed}", FM / f"splits_test_s{a.seed}"

    env = {**os.environ, "FM_RESULTS_DIR": str(RES), "FM_SPLITS_DIR": str(SPL), "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    if not a.keep:
        shutil.rmtree(RES, ignore_errors=True)
        shutil.rmtree(SPL, ignore_errors=True)
    (FM / "logs").mkdir(exist_ok=True)
    log = open(FM / "logs" / f"selftest_{time.strftime('%Y%m%d_%H%M%S')}.log", "w", encoding="utf-8")
    results = []

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    def step(name, args, expect_ok=True):
        say(f"\n=== {name}\n$ {' '.join(map(str, args))}")
        t0 = time.time()
        p = subprocess.Popen([sys.executable, *map(str, args)], cwd=FM, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        out = []
        for line in p.stdout:
            out.append(line)
            log.write(line)
            if len(out) <= 400 or line.startswith(("[", "seed", "gate", "STOP", "check", "band")):
                print(line, end="", flush=True)
        p.wait()
        ok = (p.returncode == 0) == expect_ok
        results.append((name, ok, time.time() - t0))
        say(f"--> {'OK' if ok else 'FAILED'} ({time.time() - t0:.0f} s, exit {p.returncode})")
        if not ok:
            say("\nSelftest stopped at the first failing stage. Log: " + log.name)
            summary()
            sys.exit(1)
        return "".join(out)

    def summary():
        say("\n" + "-" * 60)
        for n, ok, s in results:
            say(f"{'PASS' if ok else 'FAIL'}  {n:34s} {s:6.0f} s")

    def need(path: Path, hint: str):
        if not path.exists():
            say(f"missing {path}: {hint}")
            sys.exit(2)

    need(REPO / "data" / "raw_data" / "train" / "wav", "run `make data` (bash get_data.sh)")
    need(REPO / "experiments" / "common" / "features" / "segments_spectral.csv", "run `make features`")
    synth_ok = (REPO / "experiments" / "gate2_event_type" / "e05_cnn_synthetic" / "cache" / "synth_v2.npz").exists()
    say(f"python {sys.version.split()[0]} · seed {a.seed} · device {a.device} · synthetic set {'found' if synth_ok else 'NOT found (gate 2 runs without pretraining; `make synth` builds it)'}")

    if not a.skip_cache:
        for band in (2000, 900):
            if (FM / "cache" / f"events_mel_{band}.npz").exists():
                say(f"cache {band}: exists")
            else:
                step(f"cache band {band}", ["build_event_cache.py", "--band", band, "--jobs", 4, "--check", 100])
    step("make_splits", ["make_splits.py", "--k-per-class", a.k_per_class, "--max-per-patient", 4, "--cv-folds", 3, "--seed", a.seed])
    step("check_budget", ["check_budget.py", "--n-samples", 300])

    amp = [] if a.device.startswith("cuda") else ["--no-amp"]          # mixed precision is exercised on the GPU, as in a real run
    tiny = ["--device", a.device, "--epochs-max", 4, "--n-per-epoch", 1500, "--pruner", "none", "--sampler", "random", "--hours", 0.2, *amp, "--seed", a.seed]
    synth = ["--synth-pretrain", "auto" if synth_ok else "0"]
    step("search gate 1 (4 trials)", ["search.py", "--gate", 1, "--study", "selftest_g1", "--max-trials", 4, *tiny])
    step("search gate 2 (4 trials)", ["search.py", "--gate", 2, "--study", "selftest_g2", "--max-trials", 4, *tiny, *synth])
    out = step("status", ["search.py", "--status", "--study", "selftest_g1"])
    step("resume: same study, 6 trials", ["search.py", "--gate", 1, "--study", "selftest_g1", "--max-trials", 6, *tiny])
    out2 = step("status after resume", ["search.py", "--status", "--study", "selftest_g1"])
    if "6 trials" not in out2:
        say("resume did not continue the same study (expected 6 trials)")
        results.append(("resume continues study", False, 0))
        summary()
        sys.exit(1)
    (FM / "STOP").write_text("selftest")
    try:
        o = step("STOP file honoured", ["search.py", "--gate", 1, "--study", "selftest_g1", "--max-trials", 50, *tiny])
        if "STOP file found" not in o:
            say("STOP file was not honoured")
            results.append(("STOP honoured", False, 0))
            summary()
            sys.exit(1)
    finally:
        (FM / "STOP").unlink(missing_ok=True)

    fit = ["--device", a.device, "--top-k", 2, "--cv-folds", 3, "--epochs-max", 3, "--n-per-epoch", 1000, *amp, "--seed", a.seed]
    step("refit gate 1", ["refit_top.py", "--gate", 1, "--study", "selftest_g1", "--seeds", 1, *fit])
    step("refit gate 2", ["refit_top.py", "--gate", 2, "--study", "selftest_g2", "--seeds", 1, *fit])
    step("cascade_eval", ["cascade_eval.py", "--study", "selftest", "--top-k", 2])
    step("final_test", ["final_test.py", "--study", "selftest", "--confirm", "--device", a.device, "--seeds", 2, "--bootstrap", 200,
                        "--epochs-max", 3, "--n-per-epoch", 1000, *amp, "--seed", a.seed])
    step("final_test refuses a 2nd run", ["final_test.py", "--study", "selftest", "--confirm", "--device", a.device], expect_ok=False)
    step("fpga_report (int8)", ["fpga_report.py", "--study", "selftest", "--int8"])

    for f in ("cascade_pairs.md", "cascade.json", "final_report.md", "final_metrics.json", "confusion_matrix.csv", "gate1.onnx", "gate2.onnx",
              "fpga_report.md", "fpga_report.csv", "selftest_g1_trials.csv", "selftest_g2_trials.csv"):
        ok = (RES / f).exists()
        results.append((f"output {f}", ok, 0))
        if not ok:
            say(f"missing output {RES / f}")
    summary()
    bad = [r for r in results if not r[1]]
    say("\nALL STAGES PASSED: the full_model process runs on this machine." if not bad else f"\n{len(bad)} stage(s) failed.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
