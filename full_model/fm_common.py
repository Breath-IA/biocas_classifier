"""
fm_common.py — paths, event table, labels and small helpers shared by every full_model script.

Directories can be redirected with environment variables (the .bat / `make selftest` use this so that a quick test
never touches the committed splits or the real results):
    FM_SPLITS_DIR   default full_model/splits
    FM_RESULTS_DIR  default full_model/results
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FM = Path(__file__).resolve().parent
REPO = FM.parent


def _dir(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, FM / default))
    return p if p.is_absolute() else (REPO / p).resolve()


SPLITS = _dir("FM_SPLITS_DIR", "splits")
RESULTS = _dir("FM_RESULTS_DIR", "results")
CACHE = FM / "cache"
LOGS = FM / "logs"
STOP_FILE = FM / "STOP"
SEGMENTS_CSV = REPO / "experiments" / "common" / "features" / "segments_spectral.csv"
SYNTH_NPZ = REPO / "experiments" / "gate2_event_type" / "e05_cnn_synthetic" / "cache" / "synth_v2.npz"
E5 = REPO / "experiments" / "gate2_event_type" / "e05_cnn_synthetic"

BANDS = {2000: dict(n_mels=64, n_frames=95), 900: dict(n_mels=40, n_frames=95)}
CLS3 = {"N": 0, "W": 1, "FC": 2}          # 3-class test labels; every other class is "other abnormal" (3)
EVENT_COLS = ["eid", "stem", "split", "event_idx", "start_ms", "end_ms", "year", "pid", "class", "poor_quality"]


def patient_key(stem: str, year) -> str:
    """Patient id from the recording name '<year>__<patient>_<age>_<sex>_<loc>_<n>'; some recordings have no id
    ('2025___10.5_0_p1_32851'): those are grouped by (age, sex), which is the best available proxy."""
    rest = stem.split("__", 1)[1] if "__" in stem else stem
    parts = rest.split("_")
    if parts[0]:
        return f"{year}_{parts[0]}"
    return f"{year}_u{parts[1] if len(parts) > 1 else '?'}_{parts[2] if len(parts) > 2 else '?'}"


def load_events() -> pd.DataFrame:
    """All annotated events (segments_spectral.csv) with eid, year, pid and the 4-way label `cls4` (N, W, FC, other=3)."""
    if not SEGMENTS_CSV.exists():
        raise SystemExit(f"{SEGMENTS_CSV} not found: run `make features` first.")
    d = pd.read_csv(SEGMENTS_CSV, usecols=["stem", "source", "split", "event_idx", "start_ms", "end_ms", "class", "poor_quality"])
    d["year"] = d["source"].astype(str)
    d["eid"] = d["stem"] + "|" + d["event_idx"].astype(str)
    d["pid"] = [patient_key(s, y) for s, y in zip(d["stem"], d["year"])]
    d["cls4"] = d["class"].map(CLS3).fillna(3).astype(int)
    return d[EVENT_COLS + ["cls4"]].reset_index(drop=True)


def read_split(name: str) -> pd.DataFrame:
    p = SPLITS / name
    if not p.exists():
        raise SystemExit(f"{p} not found: run `make splits` first.")
    return pd.read_csv(p)


def import_torch(retries: int = 4):
    """`import torch` can fail intermittently on Windows (App Control, torch_python.dll): retry before giving up."""
    last = None
    for i in range(retries):
        try:
            import torch
            return torch
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"import torch failed ({type(e).__name__}: {e}); retry {i + 1}/{retries}", file=sys.stderr, flush=True)
            for m in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
                sys.modules.pop(m, None)
            time.sleep(3)
    raise SystemExit(f"torch could not be imported: {last}")


def resolve_device(torch, device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: {device} requested but CUDA is not available: using cpu (much slower).", file=sys.stderr, flush=True)
        return "cpu"
    return device


def storage_url(storage: str | None, study: str) -> str:
    """Resolve a relative sqlite path against the repo root, and create its folder."""
    url = storage or f"sqlite:///{(RESULTS / (study + '.db')).as_posix()}"
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////") and not Path(url[10:]).is_absolute():
        path = (REPO / url[10:]).resolve()
        url = f"sqlite:///{path.as_posix()}"
    if url.startswith("sqlite:///"):
        Path(url[10:]).parent.mkdir(parents=True, exist_ok=True)
    return url


def study_base(study: str) -> str:
    """'full_g1' -> 'full' (the Makefile passes STUDY=full and the scripts add _g1/_g2)."""
    return study[:-3] if study.endswith(("_g1", "_g2")) else study


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def onnx_param_count(path) -> int:
    """Float initializer elements of an ONNX file (what the FPGA stores as weights; shape constants are integer tensors and are skipped)."""
    import onnx
    from onnx import TensorProto
    m = onnx.load(str(path))
    return int(sum(int(np.prod(i.dims)) for i in m.graph.initializer if i.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE)))


def export_onnx(torch, model, n_mels: int, path) -> None:
    """model (logit) + sigmoid -> ONNX `mel (batch, 1, n_mels, 95) -> p (batch,)`, checked against PyTorch with onnxruntime."""
    import onnx
    import onnxruntime as ort
    net = torch.nn.Sequential(model.cpu().eval(), torch.nn.Sigmoid()).eval()
    x = torch.randn(8, 1, n_mels, 95)
    torch.onnx.export(net, x[:1], str(path), input_names=["mel"], output_names=["p"], dynamic_axes={"mel": {0: "batch"}, "p": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = net(x).numpy()
    diff = float(np.abs(sess.run(None, {"mel": x.numpy()})[0] - ref).max())
    assert diff < 1e-2, f"ONNX differs from PyTorch by {diff}"


def seed_offset(seed: int) -> int:
    """Non-negative offset added to the fixed per-run seeds (fold / repetition / LOYO): 0 for the default seed 42, so its sequence is unchanged."""
    return 0 if seed == 42 else (abs(int(seed)) * 1009 + 1) % (2**31 - 1)


def seed_all(torch, seed: int, deterministic: bool = False) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
