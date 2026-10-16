"""
paths.py
────────
Única fuente de rutas de `experiments/`. Los scripts de cada experimento hacen:

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    import paths

Convención dentro de cada experimento eNN_*/:
    results/  informes (report.md = conclusiones, report_auto.md = tablas generadas), figuras y CSV pequeños
    cache/    derivados grandes y reproducibles (ignorados por git: ver experiments/.gitignore)
"""
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
REPO = EXPERIMENTS.parent
RAW = REPO / "data" / "raw_data"                       # get_data.sh → train/{wav,json}, test/{wav,json}

COMMON = EXPERIMENTS / "common"
FEATURES = COMMON / "features"                         # tablas de features por evento (compartidas)
SEGMENTS_CSV = FEATURES / "segments_spectral.csv"      # extract_spectral_features.py (todas las clases)
PHYSICS_CSV = FEATURES / "physics_features.csv"        # physics_features.py (eventos W y FC)

GATE1 = EXPERIMENTS / "gate1_normal_vs_abnormal"
GATE2 = EXPERIMENTS / "gate2_event_type"

G1_E01 = GATE1 / "e01_spectral_baseline"
G2_E01 = GATE2 / "e01_flat_N_W_FC_augmentation"
G2_E02 = GATE2 / "e02_W_vs_FC_features"
G2_E03 = GATE2 / "e03_W_vs_FC_cnn_physics"
G2_E04 = GATE2 / "e04_synthetic_augmentation"

REFERENCE_AUDIO = REPO / "Auscultaion_medzcool_audio"  # MEDZCOOL teaching clips (44.1 kHz MP3): morphology references only
SYNTH_DIR = REPO / "data" / "synthetic"                 # synthetic recordings in SPRSound format (git-ignored)
