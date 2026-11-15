"""Cut the e09 CRNN (crnn_w_vs_fc.onnx) after its last 3x3 convolution, save it as a new ONNX and save the outputs for the FPGA comparison.

The new ONNX is the SAME graph and the SAME weights as e09 (onnx.utils.extract_model), just truncated, so nothing is retrained.
BatchNorm is already folded into the convolutions in the exported graph, so the last conv already includes it.
Outputs (all float32, batch first):
  conv3_out   (B,128,16,23)  output of the last Conv3x3 (+folded BN, bias included), before the ReLU   <- main comparison point
  conv3_relu  (B,128,16,23)  after ReLU
  conv3_pool  (B,128, 8,23)  after the (2,1) max-pool = what the GRU receives (before the transpose)
Runs on CPU. Run from anywhere:  python export_last_conv.py
"""
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
from onnx import utils

HERE = Path(__file__).parent
SRC = HERE.parent / "e09_crnn_optuna_colab" / "crnn_w_vs_fc.onnx"
REAL = HERE.parent / "e06_cnn_zoo_colab" / "real_events.npz"
OUT_NAMES = {"conv3_out": "/m/conv/conv.8/Conv_output_0", "conv3_relu": "/m/conv/conv.10/Relu_output_0", "conv3_pool": "/m/conv/conv.11/MaxPool_output_0"}

new = HERE / "crnn_last_conv3x3.onnx"
utils.extract_model(str(SRC), str(new), ["log_mel"], list(OUT_NAMES.values()))
m = onnx.load(str(new))
for o, name in zip(m.graph.output, OUT_NAMES):                       # friendly output names
    for n in m.graph.node:
        n.output[:] = [name if t == o.name else t for t in n.output]
        n.input[:] = [name if t == o.name else t for t in n.input]
    o.name = name
onnx.checker.check_model(m)
onnx.save(m, str(new))
print("saved", new.name, f"({new.stat().st_size / 1024:.0f} KB), nodes:", [n.op_type for n in m.graph.node])

# ---- samples: 8 real 2025 events (W/FC) + 1 fixed random tile
z = np.load(REAL, allow_pickle=True)
idx = np.where(z["year"] == "2025")[0]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(idx, 8, replace=False))
x = np.concatenate([z["X"][idx], rng.normal(size=(1, 64, 95)).astype(np.float32)])[:, None].astype(np.float32)

sess = ort.InferenceSession(str(new), providers=["CPUExecutionProvider"])
outs = dict(zip(OUT_NAMES, sess.run(None, {"log_mel": x})))

# ---- check against the ORIGINAL e09 network: same tensors taken from inside the full graph, and the full model still runs
full = onnx.load(str(SRC))
inter = {v: v for v in OUT_NAMES.values()}
for v in inter:
    full.graph.output.append(onnx.ValueInfoProto(name=v))
sf = ort.InferenceSession(full.SerializeToString(), providers=["CPUExecutionProvider"])
res = sf.run(None, {"log_mel": x})
for (name, v), r in zip(OUT_NAMES.items(), res[1:]):
    d = np.abs(r - outs[name]).max()
    print(f"{name:11s} shape {outs[name].shape}  max|new - e09 inside| = {d:.1e}")
    assert d == 0.0
print("e09 p_fine_crackle of the samples:", np.round(res[0], 3))

np.save(HERE / "input_log_mel.npy", x)
for name, a in outs.items():
    np.save(HERE / f"output_{name}.npy", a)
np.savez(HERE / "fpga_reference.npz", input_log_mel=x, event_index=np.append(idx, -1), **{k: v for k, v in outs.items()})
# plain text of the first sample's conv output, channel-major (c, h, w), one value per line, for $readmemh-style / testbench loading
np.savetxt(HERE / "sample0_input.txt", x[0, 0].ravel(), fmt="%.9e")
np.savetxt(HERE / "sample0_conv3_out.txt", outs["conv3_out"][0].ravel(), fmt="%.9e")
print("saved: input_log_mel.npy, output_conv3_out/relu/pool.npy, fpga_reference.npz, sample0_*.txt")
