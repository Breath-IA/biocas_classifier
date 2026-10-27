"""
mcp_server.py — read-mostly Model Context Protocol server (stdio) over the full_model search results.  Needs: pip install mcp

Tools: run_status, leaderboard, trial_detail, check_budget, classify_event (never for events of the frozen test CSV), and, only with the flags,
enqueue_trial (--allow-enqueue) and request_stop (--allow-stop). Paths are resolved inside full_model/ and data/ only.
"""
from __future__ import annotations

import argparse
import json
import sys

import fm_common as fc

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise SystemExit("the `mcp` package is missing: pip install mcp")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="full")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--allow-enqueue", action="store_true")
    ap.add_argument("--allow-stop", action="store_true")
    a = ap.parse_args()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = fc.study_base(a.study)
    mcp = FastMCP("biocas-full-model")

    def load(gate: int):
        name = f"{base}_g{gate}"
        return optuna.load_study(study_name=name, storage=fc.storage_url(a.storage, base))

    @mcp.tool()
    def run_status() -> str:
        """Trials done / running / pruned and the best value of both searches, plus whether a STOP file exists."""
        out = {}
        for g in (1, 2):
            try:
                s = load(g)
                states = [t.state.name for t in s.trials]
                out[f"gate{g}"] = dict(trials=len(states), **{k: states.count(k) for k in set(states)}, best=s.best_value if "COMPLETE" in states else None)
            except Exception as e:  # noqa: BLE001
                out[f"gate{g}"] = f"no study ({type(e).__name__})"
        out["stop_file"] = fc.STOP_FILE.exists()
        return json.dumps(out)

    @mcp.tool()
    def leaderboard(gate: int, top_k: int = 5) -> str:
        """Best completed trials of a gate with parameters, MACs, validation metrics and config."""
        done = sorted([t for t in load(gate).trials if t.state.name == "COMPLETE"], key=lambda t: -t.value)[:max(1, min(top_k, 50))]
        return json.dumps([dict(number=t.number, value=t.value, params=t.user_attrs.get("params"), macs=t.user_attrs.get("macs"),
                                metrics=t.user_attrs.get("metrics"), config=t.user_attrs.get("config")) for t in done])

    @mcp.tool()
    def trial_detail(gate: int, number: int) -> str:
        """Full config, user attributes and learning curve of one trial."""
        t = load(gate).trials[number]
        return json.dumps(dict(state=t.state.name, value=t.value, attrs=t.user_attrs, curve=t.intermediate_values))

    @mcp.tool()
    def check_budget(config: dict, max_params_net: int = 500_000) -> str:
        """Parameter count of a config against the per-net cap, without training."""
        import search_space as ss
        from models_small import build_model, profile
        torch = fc.import_torch()  # noqa: F841
        cfg = ss.validate_config(config)
        n_mels = fc.BANDS[cfg["band"]]["n_mels"]
        p = profile(build_model(cfg, n_mels), (n_mels, 95))
        return json.dumps(dict(**p, cap=max_params_net, fits=p["params"] <= max_params_net))

    @mcp.tool()
    def classify_event(wav: str, start_ms: int, end_ms: int) -> str:
        """p(N), p(W), p(FC) of one event of a WAV under data/raw_data, with the exported cascade (results/cascade.json)."""
        import numpy as np
        import onnxruntime as ort
        p = (fc.REPO / "data" / "raw_data" / wav).resolve()
        if not str(p).startswith(str((fc.REPO / "data").resolve())) or p.suffix != ".wav" or not p.exists():
            return "refused: the file must be a .wav under data/"
        test = fc.read_split("balanced_test.csv")
        if ((test["stem"] == p.stem) & (test["start_ms"] == start_ms) & (test["end_ms"] == end_ms)).any():
            return "refused: this event belongs to the frozen test set"
        cas = json.loads((fc.RESULTS / "cascade.json").read_text())
        sys.path.insert(0, str(fc.E5))
        sys.path.insert(0, str(fc.REPO / "experiments" / "common"))
        import cnn_data as cd
        import extract_spectral_features as ex
        x = ex.load_wav(p)
        seg = x[int(start_ms * cd.SR / 1000):int(end_ms * cd.SR / 1000)]
        ps = []
        for g in (1, 2):
            dep = cas["deployment"][f"gate{g}"]
            mel = cd.to_mel(seg, dep["band"])[None, None]
            ps.append(float(ort.InferenceSession(str(fc.RESULTS / dep["onnx"]), providers=["CPUExecutionProvider"]).run(None, {"mel": mel})[0][0]))
        p1, p2 = ps
        return json.dumps(dict(p_gate1_abnormal=p1, p_gate2_fc=p2, prediction="N" if p1 < cas["t1"] else ("FC" if p2 >= cas["t2"] else "W")))

    if a.allow_stop:
        @mcp.tool()
        def request_stop() -> str:
            """Create the STOP file: the running search ends after its current trial."""
            fc.STOP_FILE.touch()
            return "STOP file created"

    if a.allow_enqueue:
        @mcp.tool()
        def enqueue_trial(gate: int, config: dict, max_params_net: int = 500_000) -> str:
            """Queue a config (validated against the parameter cap) as the next trial of a gate's study."""
            import search_space as ss
            from models_small import build_model, count_params
            fc.import_torch()
            cfg = ss.validate_config(config)
            n = count_params(build_model(cfg, fc.BANDS[cfg["band"]]["n_mels"]))[0]
            if n > max_params_net:
                return f"refused: {n:,} parameters > {max_params_net:,}"
            load(gate).enqueue_trial({k: v for k, v in cfg.items() if k in ss.DEFAULT}, skip_if_exists=True)
            return f"queued ({n:,} parameters)"

    @mcp.resource("report://{name}")
    def report(name: str) -> str:
        """results/<name>.md (cascade_pairs, final_report, fpga_report, splits report)."""
        p = (fc.RESULTS / f"{name}.md").resolve()
        if fc.RESULTS.resolve() not in p.parents or not p.exists():
            return "not found"
        return p.read_text(encoding="utf-8")

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
