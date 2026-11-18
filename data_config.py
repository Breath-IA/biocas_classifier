import argparse
import sys

if hasattr(sys.stdout, 'reconfigure'):  # consolas Windows cp1252
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path

# data_formats.py vive en workflow/ (sin dependencia de torch)
sys.path.insert(0, str(Path(__file__).resolve().parent / "workflow"))
from data_formats import (
    ANNOTATION_EXT, CLASS_NAMES, CLASS_ORDER, DATASETS, DEFAULT_ALLOWED,
    EVENT_TYPE_NORM, read_annotation,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Escanea anotaciones BioCAS y actualiza config/pipeline.yaml"
    )
    p.add_argument(
        "--dataset",
        choices=DATASETS,
        default="sprsound",
        help="Dataset: sprsound (BioCAS, anotaciones .json) o icbhi (anotaciones .txt)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Carpeta con train/{wav,<ann>} y test/{wav,<ann>}, donde <ann> es json (sprsound) "
             "o txt (icbhi). Default: ./data/raw_data (sprsound) | ./data/icbhi/raw_data (icbhi)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta de salida de los datos procesados. "
             "Default: ./data/processed (sprsound) | ./data/icbhi/processed (icbhi)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("workflow/config/pipeline.yaml"),
        help="Ruta del yaml a actualizar (default: config/pipeline.yaml)",
    )
    return p.parse_args()


args     = parse_args()
DATASET  = args.dataset
ANN_EXT  = ANNOTATION_EXT[DATASET]
ANN_DIR  = ANN_EXT.lstrip(".")           # nombre de la subcarpeta: json | txt
_default_root = Path("./data") if DATASET == "sprsound" else Path("./data/icbhi")
DATA_DIR   = args.data_dir or _default_root / "raw_data"
OUTPUT_DIR = (args.output_dir or _default_root / "processed").resolve()


def scan_events(ann_dir: Path):
    """Lee todas las anotaciones de un directorio y devuelve una lista de dicts por evento."""
    rows = []
    for jpath in tqdm(sorted(ann_dir.glob(f"*{ANN_EXT}"))):
        try:
            ann = read_annotation(jpath, DATASET)
        except Exception:
            continue

        record_label = ann.get("record_annotation", "?")
        for ev in ann.get("event_annotation", []):
            raw   = ev.get("type", "").lower().strip()
            label = EVENT_TYPE_NORM.get(raw, raw.upper())
            dur_s = (int(ev.get("end", 0)) - int(ev.get("start", 0))) / 1000
            rows.append({
                "file":         jpath.stem,
                "record_label": record_label,
                "label":        label,
                "duration_s":   dur_s,
            })
    return rows

# Escanear train y test por separado
train_rows = scan_events(DATA_DIR / "train" / ANN_DIR)
test_rows  = scan_events(DATA_DIR / "test"  / ANN_DIR)

df_train = pd.DataFrame(train_rows)
df_test  = pd.DataFrame(test_rows)

from ruamel.yaml import YAML as _YAML
_yaml = _YAML()
_cfg  = _yaml.load(args.config) if args.config.exists() else {}
allowed = list((_cfg.get("data") or {}).get("allowed_classes") or [])
# Si el yaml trae clases de otro dataset (p. ej. FC en ICBHI) se reemplazan por el default
if any(c not in CLASS_ORDER[DATASET] for c in allowed):
    allowed = list(DEFAULT_ALLOWED[DATASET])
    print(f"allowed_classes no válidas para {DATASET}; usando {allowed}")

if allowed:
    df_train = df_train[df_train["label"].isin(allowed)].reset_index(drop=True)
    df_test  = df_test[df_test["label"].isin(allowed)].reset_index(drop=True)
    print(f"Filtrando a clases: {allowed}")

print(f"Train events : {len(df_train):,}")
print(f"Test  events : {len(df_test):,}")
print()
print(df_train["label"].value_counts().rename("train").to_frame()
      .join(df_test["label"].value_counts().rename("test"), how="outer")
      .fillna(0).astype(int))


# ── Resumen: cuánto augmentation necesita cada clase ─────────────────────────
# Esto es lo que hay que mirar ANTES de fijar los parámetros de la pipeline.

ORDER  = CLASS_ORDER[DATASET]

train_counts = df_train["label"].value_counts().reindex(ORDER, fill_value=0)
max_count    = train_counts.max()

summary = pd.DataFrame({
    "clase":        ORDER,
    "nombre":       [CLASS_NAMES[c] for c in ORDER],
    "n_eventos":    train_counts.values,
    "ratio_vs_max": (train_counts / max_count).round(3).values,
    "target_sqrt":  (np.sqrt(train_counts) / np.sqrt(train_counts).sum() * train_counts.sum()).round(0).astype(int).values,
    "necesita_aug": (max_count / train_counts.replace(0, np.nan)).fillna(0).round(1).values,
})

print(summary.to_string(index=False))
print()
print("→ 'necesita_aug' = cuántas veces hay que replicar esa clase para igualar a la mayoritaria.")
print("→ 'target_sqrt'  = objetivo con strategy='proportional' (más conservador).")

train_wav  = (DATA_DIR / "train" / "wav").resolve()
train_json = (DATA_DIR / "train" / ANN_DIR).resolve()
test_wav   = (DATA_DIR / "test"  / "wav").resolve()
test_json  = (DATA_DIR / "test"  / ANN_DIR).resolve()


# ── Escribir la sección 'data:' directamente en config/pipeline.yaml ─────────
# Usamos ruamel.yaml en modo round-trip para no pisar comentarios ni el resto
# de las secciones (runner, stages, augmentation, ...) que ya viven en el yaml.
from ruamel.yaml import YAML

CONFIG_PATH = args.config

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=2, offset=0)

if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as fh:
        cfg = yaml.load(fh) or {}
else:
    cfg = {}
    print(f"⚠ {CONFIG_PATH} no existía, se crea desde cero.")

if "data" not in cfg:
    cfg["data"] = {}

cfg["data"]["dataset"]       = DATASET
if list(cfg["data"].get("allowed_classes") or []) != allowed:      # solo si cambió
    from ruamel.yaml.comments import CommentedSeq
    seq = CommentedSeq(allowed)
    seq.fa.set_flow_style()                                          # ["N", "W", ...]
    cfg["data"]["allowed_classes"] = seq
cfg["data"]["raw_wav_dir"]   = str(train_wav)
cfg["data"]["raw_json_dir"]  = str(train_json)
cfg["data"]["test_wav_dir"]  = str(test_wav)
cfg["data"]["test_json_dir"] = str(test_json)
cfg["data"]["output_dir"]    = str(OUTPUT_DIR)
cfg["data"]["task"]          = "1-2"

CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CONFIG_PATH, "w") as fh:
    yaml.dump(cfg, fh)

print(f"✓ Sección 'data:' actualizada en {CONFIG_PATH.resolve()}")