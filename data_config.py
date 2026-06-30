import argparse
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Escanea anotaciones BioCAS y actualiza config/pipeline.yaml"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/raw_data"),
        help="Carpeta con train/{wav,json} y test/{wav,json} (default: ./data/raw_data)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/processed"),
        help="Carpeta con train/{wav,json} y test/{wav,json} con los archivos procesados (default: ./data/processed)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("workflow/config/pipeline.yaml"),
        help="Ruta del yaml a actualizar (default: config/pipeline.yaml)",
    )
    return p.parse_args()


args     = parse_args()
DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir.resolve()


# ── Normalización de strings (igual que en EventExtractionStage) ─────────────
EVENT_TYPE_NORM = {
    "normal":         "N",
    "rhonchi":        "R",
    "wheeze":         "W",
    "stridor":        "S",
    "coarse crackle": "CC",
    "fine crackle":   "FC",
    "wheeze&crackle": "WC",
    "wheeze+crackle": "WC",
}

CLASS_NAMES = {
    "N":  "Normal",
    "R":  "Rhonchi",
    "W":  "Wheeze",
    "S":  "Stridor",
    "CC": "Coarse Crackle",
    "FC": "Fine Crackle",
    "WC": "Wheeze+Crackle",
}

def scan_events(json_dir: Path):
    """Lee todos los JSONs de un directorio y devuelve una lista de dicts por evento."""
    rows = []
    for jpath in tqdm(sorted(json_dir.glob("*.json"))):
        try:
            ann = json.loads(jpath.read_text())
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
train_rows = scan_events(DATA_DIR / "train" / "json")
test_rows  = scan_events(DATA_DIR / "test"  / "json")

df_train = pd.DataFrame(train_rows)
df_test  = pd.DataFrame(test_rows)

print(f"Train events : {len(df_train):,}")
print(f"Test  events : {len(df_test):,}")
print()
print(df_train["label"].value_counts().rename("train").to_frame()
      .join(df_test["label"].value_counts().rename("test"), how="outer")
      .fillna(0).astype(int))


# ── Resumen: cuánto augmentation necesita cada clase ─────────────────────────
# Esto es lo que hay que mirar ANTES de fijar los parámetros de la pipeline.

ORDER  = ["N", "R", "W", "S", "CC", "FC", "WC"]

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
train_json = (DATA_DIR / "train" / "json").resolve()
test_wav   = (DATA_DIR / "test"  / "wav").resolve()
test_json  = (DATA_DIR / "test"  / "json").resolve()


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