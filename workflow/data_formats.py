"""
data_formats.py
───────────────
Capa de compatibilidad entre datasets (SPRSound/BioCAS e ICBHI 2017).

No importa torch: lo usan tanto la pipeline (stages/) como data_config.py.

Todos los datasets se normalizan al formato de anotación de SPRSound:

    {"record_annotation": str,
     "event_annotation":  [{"start": ms, "end": ms, "type": str}, ...]}

de modo que el resto de la pipeline (EventExtractionStage, etc.) no cambia.

  sprsound : <stem>.json con record_annotation + event_annotation (start/end en ms).
  icbhi    : <stem>.txt, una fila por ciclo respiratorio:  inicio(s) fin(s) crackles wheezes
             El ciclo se convierte en un "evento" cuyo tipo sale de las dos banderas:
               (0,0) normal | (1,0) crackle | (0,1) wheeze | (1,1) wheeze&crackle
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATASETS = ("sprsound", "icbhi")
DEFAULT_DATASET = "sprsound"

# Extensión del archivo de anotación junto a cada .wav
ANNOTATION_EXT: Dict[str, str] = {"sprsound": ".json", "icbhi": ".txt"}

# ── Etiquetas ────────────────────────────────────────────────────────────────
# Código corto canónico de clase (compartido por ambos datasets).
EVENT_TYPE_NORM: Dict[str, str] = {
    "normal":         "N",
    "rhonchi":        "R",
    "wheeze":         "W",
    "stridor":        "S",
    "coarse crackle": "CC",
    "fine crackle":   "FC",
    "wheeze&crackle": "WC",
    "wheeze+crackle": "WC",
    "crackle":        "C",    # ICBHI: crepitante sin distinguir fino/grueso
}

CLASS_NAMES: Dict[str, str] = {
    "N":  "Normal",
    "R":  "Rhonchi",
    "W":  "Wheeze",
    "S":  "Stridor",
    "CC": "Coarse Crackle",
    "FC": "Fine Crackle",
    "C":  "Crackle",
    "WC": "Wheeze+Crackle",
}

# Orden de clases por dataset (para tablas/resúmenes) y default de allowed_classes.
CLASS_ORDER: Dict[str, List[str]] = {
    "sprsound": ["N", "R", "W", "S", "CC", "FC", "WC"],
    "icbhi":    ["N", "C", "W", "WC"],
}
DEFAULT_ALLOWED: Dict[str, List[str]] = {
    "sprsound": ["N", "W", "FC", "WC"],
    "icbhi":    ["N", "C", "W", "WC"],
}

# data.task → mapa clase→int, usado cuando allowed_classes está vacío.
LABEL_MAPS: Dict[str, Dict[str, Dict[str, int]]] = {
    "sprsound": {
        "1-1": {"N": 0, "R": 1, "W": 1, "S": 1, "CC": 1, "FC": 1, "WC": 1},
        "1-2": {"N": 0, "R": 1, "W": 2, "S": 3, "CC": 4, "FC": 5, "WC": 6},
    },
    "icbhi": {
        "1-1": {"N": 0, "C": 1, "W": 1, "WC": 1},
        "1-2": {"N": 0, "C": 1, "W": 2, "WC": 3},
    },
}
LABEL_MAP_1_1 = LABEL_MAPS["sprsound"]["1-1"]   # compat con stages.base
LABEL_MAP_1_2 = LABEL_MAPS["sprsound"]["1-2"]


def check_dataset(dataset: str) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"data.dataset={dataset!r} no válido; opciones: {DATASETS}")
    return dataset


def get_dataset_name(cfg: Any) -> str:
    """Lee cfg.data.dataset (default 'sprsound' para configs antiguos)."""
    data = cfg.data if hasattr(cfg, "data") else cfg
    name = data.get("dataset", DEFAULT_DATASET) if hasattr(data, "get") else DEFAULT_DATASET
    return check_dataset(name or DEFAULT_DATASET)


def get_label_map(dataset: str, task: str) -> Dict[str, int]:
    maps = LABEL_MAPS[check_dataset(dataset)]
    if task not in maps:
        raise ValueError(f"data.task={task!r} no válido para {dataset}; opciones: {list(maps)}")
    return maps[task]


# ── Nombres de archivo ───────────────────────────────────────────────────────

_GENDER_MAP   = {"0": "male", "1": "female"}
_LOCATION_MAP = {
    "p1": "left_posterior", "p2": "right_posterior",
    "p3": "left_anterior",  "p4": "right_anterior",
    "p5": "left_lateral",   "p6": "right_lateral",
}
_ICBHI_LOCATION = {   # foco de auscultación ICBHI
    "Tc": "trachea", "Al": "anterior_left", "Ar": "anterior_right",
    "Pl": "posterior_left", "Pr": "posterior_right",
    "Ll": "lateral_left", "Lr": "lateral_right",
}


def parse_filename(stem: str, dataset: str = DEFAULT_DATASET) -> Dict[str, str]:
    """
    sprsound: [<año>__]<paciente>_<edad>_<género>_<ubicación>_<rec>
    icbhi   : <paciente>_<índice>_<foco>_<modo sc/mc>_<equipo>   (ej. 101_1b1_Al_sc_Meditron)
    """
    parts = stem.split("_")
    meta: Dict[str, str] = {"raw_stem": stem}
    if dataset == "icbhi":
        if len(parts) >= 5:
            meta["patient_id"] = parts[0]
            meta["rec_idx"]    = parts[1]
            meta["location"]   = _ICBHI_LOCATION.get(parts[2], parts[2])
            meta["mode"]       = parts[3]
            meta["device"]     = parts[4]
        return meta
    if len(parts) >= 5:
        meta["patient_id"] = parts[0]
        meta["age"]        = parts[1]
        meta["gender"]     = _GENDER_MAP.get(parts[2], parts[2])
        meta["location"]   = _LOCATION_MAP.get(parts[3], parts[3])
        meta["rec_num"]    = parts[4]
    return meta


# ── Lectores de anotaciones ──────────────────────────────────────────────────

def _read_sprsound(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _read_icbhi(path: Path) -> Dict[str, Any]:
    events = []
    for line in path.read_text().splitlines():
        cols = line.split()
        if len(cols) < 4:
            continue
        start_s, end_s = float(cols[0]), float(cols[1])
        crackle, wheeze = int(cols[2]), int(cols[3])
        etype = {(0, 0): "normal", (1, 0): "crackle",
                 (0, 1): "wheeze", (1, 1): "wheeze&crackle"}[(crackle, wheeze)]
        events.append({
            "start": int(round(start_s * 1000)),
            "end":   int(round(end_s * 1000)),
            "type":  etype,
        })
    # ICBHI no tiene etiqueta de calidad por grabación
    return {"record_annotation": "ICBHI", "event_annotation": events}


_READERS = {"sprsound": _read_sprsound, "icbhi": _read_icbhi}


def read_annotation(path: Path, dataset: str = DEFAULT_DATASET) -> Dict[str, Any]:
    """Lee el archivo de anotación y lo devuelve en formato SPRSound."""
    return _READERS[check_dataset(dataset)](Path(path))


def annotation_path(wav_path: Path, ann_dir: Path, dataset: str = DEFAULT_DATASET) -> Path:
    """Ruta del archivo de anotación que corresponde a un .wav."""
    return Path(ann_dir) / (Path(wav_path).stem + ANNOTATION_EXT[check_dataset(dataset)])
