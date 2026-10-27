"""
search_space.py — Optuna search space (model + training recipe) shared by search.py, check_budget.py and mcp_server.py.

`sample_config(trial, gate, ...)` returns a plain dict; `validate_config` fills defaults for configs written by hand (MCP enqueue).
"""
from __future__ import annotations

DEFAULT = dict(
    band=2000, n_stages=4, base_width=16, growth=2.0, block="plain", se=False, kernel_first=3, act="relu", pool="max",
    head_hidden=32, dropout=0.3, opt="adamw", lr=1e-3, wd=1e-3, batch=128, sched="cosine", loss="bce", focal_gamma=2.0,
    label_smooth=0.05, specaug_f=8, specaug_t=10, mixup=0.0, synth_pretrain=False, synth_ratio=1.0,
)


def sample_config(trial, gate: int, no_se: bool = False, synth: str = "auto") -> dict:
    s = trial.suggest_categorical
    cfg = dict(
        band=s("band", [2000, 900]),
        n_stages=trial.suggest_int("n_stages", 3, 6),
        base_width=s("base_width", [8, 12, 16, 24, 32]),
        growth=s("growth", [1.5, 2.0]),
        block=s("block", ["plain", "dws", "res"]),
        se=False if no_se else s("se", [False, True]),
        kernel_first=s("kernel_first", [3, 5]),
        act=s("act", ["relu", "relu6", "leaky"]),
        pool=s("pool", ["max", "avg"]),
        head_hidden=s("head_hidden", [0, 16, 32, 64]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        opt=s("opt", ["adamw", "sgd"]),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        batch=s("batch", [64, 128, 256]),
        sched=s("sched", ["cosine", "onecycle"]),
        loss=s("loss", ["bce", "focal", "smooth"]),
        focal_gamma=trial.suggest_float("focal_gamma", 1.0, 3.0),
        label_smooth=trial.suggest_float("label_smooth", 0.0, 0.1),
        specaug_f=trial.suggest_int("specaug_f", 0, 12),
        specaug_t=trial.suggest_int("specaug_t", 0, 20),
        mixup=s("mixup", [0.0, 0.2, 0.4]),
    )
    if gate == 2:
        cfg["synth_pretrain"] = {"auto": s("synth_pretrain", [False, True]), "1": True, "0": False}[str(synth)]
        cfg["synth_ratio"] = s("synth_ratio", [0.5, 1.0]) if cfg["synth_pretrain"] else 1.0
    return cfg


def validate_config(cfg: dict) -> dict:
    out = {**DEFAULT, **cfg}
    assert out["band"] in (2000, 900) and 3 <= int(out["n_stages"]) <= 6 and out["block"] in ("plain", "dws", "res")
    assert out["act"] in ("relu", "relu6", "leaky") and out["pool"] in ("max", "avg") and out["loss"] in ("bce", "focal", "smooth")
    assert out["opt"] in ("adamw", "sgd") and out["sched"] in ("cosine", "onecycle")
    return out
