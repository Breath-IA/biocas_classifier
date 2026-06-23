"""
pipeline_api.py
───────────────
API programática para usar la pipeline desde notebooks o scripts de research.

El decorador @hydra.main NO funciona en Jupyter porque el intérprete maneja
sys.argv de forma distinta. Este módulo expone la misma pipeline a través de
hydra.initialize() + hydra.compose(), que sí funciona en cualquier contexto.

Uso básico en notebook:
    from pipeline_api import build_pipeline, load_config

    cfg      = load_config()                      # config por defecto
    pipeline = build_pipeline(cfg)
    batch    = pipeline.run()

Uso con datos ya procesados (después de reiniciar kernel):
    from pipeline_api import load_dataset

    dataset = load_dataset("data/processed")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from pipeline import PreprocessingPipeline
from dataset import BioCASDataset
from stages.base import AudioSample


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(
    config_dir:  str            = "config",
    config_name: str            = "pipeline",
    overrides:   Optional[list] = None,
) -> DictConfig:
    """
    Carga el YAML de configuración y aplica overrides opcionales.

    Parameters
    ----------
    config_dir  : Ruta al directorio de configs (relativa al cwd del notebook).
    config_name : Nombre del archivo YAML sin extensión.
    overrides   : Lista de strings estilo Hydra, ej. ["data.task=1-1", "log_mel.n_mels=128"].

    Returns
    -------
    cfg : DictConfig navegable con punto (cfg.log_mel.n_mels).

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg = load_config(overrides=["data.task=1-1", "log_mel.n_mels=128"])
    """
    GlobalHydra.instance().clear()

    abs_config_dir = str(Path(config_dir).resolve())
    with initialize_config_dir(config_dir=abs_config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    return cfg


def show_config(cfg: DictConfig) -> None:
    """Pretty-print el config activo."""
    print(OmegaConf.to_yaml(cfg))


def patch_config(cfg: DictConfig, updates: Dict[str, Any]) -> DictConfig:
    """
    Aplica overrides en forma de dict sobre un config existente.
    Útil para exploración rápida en notebooks sin recargar desde disco.

    Parameters
    ----------
    updates : dict con keys en notación de punto, ej.
              {"log_mel.n_mels": 128, "balanced_sampling.strategy": "equal"}

    Returns
    -------
    Nuevo DictConfig con los cambios aplicados (el original no se muta).

    Examples
    --------
    >>> cfg2 = patch_config(cfg, {"log_mel.n_mels": 128, "runner.n_workers": 2})
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    for dotted_key, value in updates.items():
        keys = dotted_key.split(".")
        node = cfg_dict
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    return OmegaConf.create(cfg_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia — guardar y cargar datos procesados
# ─────────────────────────────────────────────────────────────────────────────

def save_dataset(dataset: "BioCASDataset", output_dir: str) -> None:
    """
    Guarda el dataset como un archivo .npy por muestra + un CSV de índice.

    Estructura en disco:
        output_dir/
          index.csv          ← una fila por muestra (sample_id, label_int, label_str, path, ...)
          features/
            00000.npy        ← feature de la muestra 0, shape (n_mels, T)
            00001.npy
            ...

    Ventaja sobre un solo features.npy:
      DataLoader con num_workers>0 carga cada muestra de forma independiente
      sin necesidad de mmap ni de duplicar el array completo en cada worker.
    """
    import pandas as pd

    out      = Path(output_dir)
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (feat, label, sid, lstr, meta) in enumerate(zip(
        dataset._features_mem, dataset._labels, dataset._ids,
        dataset._label_strs, dataset._metas,
    )):
        fname = f"{i:05d}.npy"
        np.save(feat_dir / fname, feat)
        rows.append({
            "idx":        i,
            "path":       f"features/{fname}",
            "sample_id":  sid,
            "label_int":  int(label),
            "label_str":  lstr,
            **{k: v for k, v in meta.items()
               if isinstance(v, (str, int, float, bool))},  # solo escalares al CSV
        })

    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)

    print(f"✓ Dataset guardado en '{out}'")


def load_dataset(output_dir: str) -> "BioCASDataset":
    """
    Carga el dataset guardado por save_dataset().
    Solo lee el CSV — los .npy individuales se leen bajo demanda en __getitem__.

    Uso típico después de reiniciar el kernel:
        dataset = load_dataset("data/processed/train")
        loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    """
    import pandas as pd

    out = Path(output_dir)

    if not (out / "index.csv").exists():
        raise FileNotFoundError(
            f"No se encontró 'index.csv' en '{out}'.\n"
            f"Primero corré run_pipeline() y guardá con save_dataset()."
        )

    index = pd.read_csv(out / "index.csv")
    dataset = BioCASDataset.from_index(index, base_dir=out)

    print(f"✓ Dataset cargado desde '{out}'")
    print(f"  {dataset}")
    return dataset


def processed_exists(output_dir: str) -> bool:
    """Devuelve True si ya hay datos procesados guardados en output_dir."""
    out = Path(output_dir)
    return (out / "index.csv").exists() and (out / "features").is_dir()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(cfg: DictConfig) -> PreprocessingPipeline:
    """Construye la pipeline a partir de un config ya cargado."""
    return PreprocessingPipeline.from_config(cfg)


def run_pipeline(cfg: DictConfig, save: bool = True) -> BioCASDataset:
    """
    Construye la pipeline, la corre, y opcionalmente guarda el resultado.

    Parameters
    ----------
    cfg  : Config cargado con load_config().
    save : Si True (default), guarda automáticamente en cfg.data.output_dir.

    Returns
    -------
    BioCASDataset listo para DataLoader.
    """
    pipeline = build_pipeline(cfg)
    print(pipeline.summary())
    batch   = pipeline.run()
    dataset = BioCASDataset(batch)

    if save:
        save_dataset(dataset, cfg.data.output_dir)

    return dataset


def load_or_run(cfg: DictConfig) -> BioCASDataset:
    """
    Carga los datos procesados si ya existen; si no, corre la pipeline completa.

    Es el punto de entrada recomendado para notebooks: la primera vez tarda,
    las siguientes es instantáneo.

        dataset = load_or_run(cfg)  # ← siempre funciona, sin pensar

    Parameters
    ----------
    cfg : Config cargado con load_config().
    """
    output_dir = cfg.data.output_dir

    if processed_exists(output_dir):
        print(f"Datos procesados encontrados en '{output_dir}' — cargando desde disco.")
        return load_dataset(output_dir)
    else:
        print(f"No hay datos procesados en '{output_dir}' — corriendo pipeline completa.")
        return run_pipeline(cfg, save=True)



# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(
    config_dir:  str            = "config",
    config_name: str            = "pipeline",
    overrides:   Optional[list] = None,
) -> DictConfig:
    """
    Carga el YAML de configuración y aplica overrides opcionales.

    Parameters
    ----------
    config_dir  : Ruta al directorio de configs (relativa al cwd del notebook).
    config_name : Nombre del archivo YAML sin extensión.
    overrides   : Lista de strings estilo Hydra, ej. ["data.task=1-1", "log_mel.n_mels=128"].

    Returns
    -------
    cfg : DictConfig navegable con punto (cfg.log_mel.n_mels).

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg = load_config(overrides=["data.task=1-1", "log_mel.n_mels=128"])
    """
    # GlobalHydra es un singleton – hay que limpiar entre llamadas en notebooks
    GlobalHydra.instance().clear()

    abs_config_dir = str(Path(config_dir).resolve())
    with initialize_config_dir(config_dir=abs_config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    return cfg


def show_config(cfg: DictConfig) -> None:
    """Pretty-print el config activo."""
    print(OmegaConf.to_yaml(cfg))


def patch_config(cfg: DictConfig, updates: Dict[str, Any]) -> DictConfig:
    """
    Aplica overrides en forma de dict sobre un config existente.
    Útil para exploración rápida en notebooks sin recargar desde disco.

    Parameters
    ----------
    updates : dict con keys en notación de punto, ej.
              {"log_mel.n_mels": 128, "balanced_sampling.strategy": "equal"}

    Returns
    -------
    Nuevo DictConfig con los cambios aplicados (el original no se muta).

    Examples
    --------
    >>> cfg2 = patch_config(cfg, {"log_mel.n_mels": 128, "runner.n_workers": 2})
    """
    # OmegaConf no permite mutación directa en structs — convertimos a dict, parchamos, reconstruimos
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    for dotted_key, value in updates.items():
        keys = dotted_key.split(".")
        node = cfg_dict
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    return OmegaConf.create(cfg_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(cfg: DictConfig) -> PreprocessingPipeline:
    """Construye la pipeline a partir de un config ya cargado."""
    return PreprocessingPipeline.from_config(cfg)


def run_pipeline(cfg: DictConfig) -> BioCASDataset:
    """
    Shortcut: construye la pipeline, la corre y devuelve un Dataset listo para DataLoader.

    Returns
    -------
    BioCASDataset con .feature y .label_int por muestra.
    """
    pipeline = build_pipeline(cfg)
    print(pipeline.summary())
    batch    = pipeline.run()
    return BioCASDataset(batch)