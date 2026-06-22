"""
run.py
──────
Entry point.  Run with:

    python run.py                                   # default config
    python run.py data.task=1-1                     # binary task
    python run.py stages.augmentation.enabled=false # skip augmentation
    python run.py runner.n_workers=8
"""
from __future__ import annotations

import logging
import os
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

from pipeline import PreprocessingPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_processed_dataset(batch, output_dir: str) -> None:
    """
    Serialises the processed batch to disk as:
        features.npy  – float32 array (N, n_mels, T)
        labels.npy    – int32   array (N,)
        metadata.json – list of {sample_id, label_str, meta} dicts
    """
    import json
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    features  = np.stack([s.feature for s in batch], axis=0).astype(np.float32)
    labels    = np.array([s.label_int for s in batch], dtype=np.int32)
    metadata  = [
        {"sample_id": s.sample_id, "label_str": s.label_str, "meta": s.meta}
        for s in batch
    ]

    np.save(out / "features.npy", features)
    np.save(out / "labels.npy",   labels)
    with open(out / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    log = logging.getLogger("run")
    log.info("Saved %d samples to %s", len(batch), out)
    log.info("  features.npy : %s", features.shape)
    log.info("  labels.npy   : %s  unique=%s", labels.shape, sorted(set(labels.tolist())))


# ─────────────────────────────────────────────────────────────────────────────
# Hydra entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="config", config_name="pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level   = cfg.runner.log_level,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%H:%M:%S",
    )
    log = logging.getLogger("run")

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Build and inspect pipeline
    pipeline = PreprocessingPipeline.from_config(cfg)
    log.info("\n%s", pipeline.summary())

    # Run
    batch = pipeline.run()

    # Persist
    save_processed_dataset(batch, cfg.data.output_dir)


if __name__ == "__main__":
    main()
