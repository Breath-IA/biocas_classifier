"""
fpga_report.py — FPGA readiness of the exported cascade (results/gate1.onnx, gate2.onnx written by final_test.py).

Per gate: trainable parameters (raw / BN-folded / counted in the ONNX file), MACs, int8 weight memory, peak activation memory, operator check,
and (--int8) static post-training quantisation calibrated on training events with the change in validation metrics. A drop in balanced accuracy
above --qat-tolerance marks the model "needs QAT". Writes results/fpga_report.csv and .md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import fm_common as fc

ALLOWED = {"Conv", "Relu", "Clip", "LeakyRelu", "MaxPool", "AveragePool", "GlobalAveragePool", "Gemm", "Add", "Flatten", "Sigmoid"}
BENIGN = {"Squeeze", "Reshape", "Identity", "Constant", "Shape"}


def ops_of(path) -> list[str]:
    import onnx
    return sorted({n.op_type for n in onnx.load(str(path)).graph.node})


def quantize(path, out, X_cal):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(X_cal)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"mel": x[None, None].astype(np.float32)}

    quantize_static(str(path), str(out), Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="full")
    ap.add_argument("--max-params", type=int, default=1_000_000)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--qat-tolerance", type=float, default=0.02)
    ap.add_argument("--n-calib", type=int, default=128)
    a = ap.parse_args()
    cas = json.loads((fc.RESULTS / "cascade.json").read_text())
    if "deployment" not in cas:
        raise SystemExit("cascade.json has no deployment block: run final_test.py first")
    torch = fc.import_torch()
    import onnxruntime as ort
    import train_lib as tl
    from models_small import build_model, profile
    pool = fc.read_split("train_pool.csv")
    rows, tot = [], 0
    for g in (1, 2):
        cfg, dep = cas[f"gate{g}"]["config"], cas["deployment"][f"gate{g}"]
        path = fc.RESULTS / dep["onnx"]
        n_mels = fc.BANDS[cfg["band"]]["n_mels"]
        prof = profile(build_model(cfg, n_mels), (n_mels, 95))
        onnx_n = fc.onnx_param_count(path)
        tot += onnx_n
        ops = ops_of(path)
        blockers = [o for o in ops if o not in ALLOWED and o not in BENIGN]
        r = dict(gate=g, trial=cas[f"gate{g}"]["trial"], band=cfg["band"], params_trainable=prof["params"], params_bn_folded=prof["params_folded"], params_onnx=onnx_n,
                 macs=prof["macs"], weights_int8_kb=round(onnx_n / 1024, 1), peak_activation_int8_kb=prof["peak_act_kb"], operators=" ".join(ops),
                 blockers=" ".join(blockers) or "none")
        if a.int8:
            cache = tl.MelCache(torch, cfg["band"], "cpu")
            tr = tl.make_subset(pool[pool.role == "train"], g, cache)
            va = tl.make_subset(pool[pool.role == "val"], g, cache)
            rng = np.random.default_rng(0)
            cal = cache.X[torch.from_numpy(tr.rows[rng.choice(len(tr), min(a.n_calib, len(tr)), replace=False)])].float().numpy()
            q = fc.RESULTS / f"gate{g}_int8.onnx"
            quantize(path, q, cal)
            xv = cache.X[torch.from_numpy(va.rows)].float().numpy()[:, None]

            def run(p):
                s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                return np.concatenate([s.run(None, {"mel": xv[i:i + 256]})[0] for i in range(0, len(xv), 256)])
            m32, m8 = tl.metrics(va.y, run(path), va.w), tl.metrics(va.y, run(q), va.w)
            r.update(val_bal_acc_fp32=m32["bal_acc"], val_bal_acc_int8=m8["bal_acc"], val_auc_fp32=m32["auc"], val_auc_int8=m8["auc"],
                     delta_bal_acc=m8["bal_acc"] - m32["bal_acc"], int8_file_kb=round(q.stat().st_size / 1024, 1),
                     needs_qat=bool(m32["bal_acc"] - m8["bal_acc"] > a.qat_tolerance))
        rows.append(r)
    assert tot < a.max_params, f"ONNX parameters {tot:,} >= cap {a.max_params:,}"
    df = pd.DataFrame(rows)
    df.to_csv(fc.RESULTS / "fpga_report.csv", index=False)
    md = ["# FPGA report", "", f"Total parameters in the ONNX files: **{tot:,}** (cap {a.max_params:,}) · front-end (STFT, mel, log, per-event normalisation) is NOT in the ONNX files:",
          "4 kHz audio, 3 s window (tiled / centre-cropped), n_fft 512, hop 128, hann, mel band per gate, 10·log10(power), standardise per event.", "",
          "| | " + " | ".join(f"gate {r['gate']}" for r in rows) + " |", "|---|" + "---|" * len(rows)]
    for k in df.columns:
        if k != "gate":
            md.append(f"| {k} | " + " | ".join(str(round(v, 4) if isinstance(v, float) else v) for v in df[k]) + " |")
    (fc.RESULTS / "fpga_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
