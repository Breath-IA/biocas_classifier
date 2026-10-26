"""
analysis_utils.py
─────────────────
Utilidades compartidas por los análisis de separabilidad (antes importadas de gate2_analysis.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

YEARS = ["2022", "2023", "2024", "2025"]
YCOL = {"2022": "#4C78A8", "2023": "#54A24B", "2024": "#F58518", "2025": "#E45756"}

# columnas que no son features
META = {"stem", "source", "split", "patient_id", "age", "gender", "location", "event_idx", "class",
        "start_ms", "end_ms", "rec_duration_s", "n_events_rec", "start_frac", "poor_quality", "n_frames",
        "is_aug", "aug_variant", "aug_copy", "orig_start_ms", "orig_end_ms", "keep_orig", "n_aug"}


def auc(x, y):
    """AUC univariada (x alto → positivo). NaN si hay menos de 15 muestras por lado."""
    x = np.asarray(x, float)
    m = np.isfinite(x)
    x, y = x[m], np.asarray(y)[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 15 or n0 < 15:
        return np.nan
    r = rankdata(x)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def groups(cols):
    """Agrupa las features por familia (duración, nivel, relativas, forma espectral, tonales, temporales, MFCC)."""
    return {
        "duration": [c for c in cols if c == "duration_s"],
        "level": [c for c in cols if c in ("rms_db",)],
        "rel": [c for c in cols if c in ("snr_vs_floor_db", "level_vs_rec_db") or c.startswith("rel_")],
        "shape": [c for c in cols if c.startswith(("spec_", "band_")) or c in ("dom_freq", "lf_hf_db")],
        "tonal": [c for c in cols if c in ("n_spec_peaks", "max_peak_prom_db", "dom_freq_std", "dom_persist",
                                           "frame_peakiness_db", "if_mean", "if_std", "acf_peak", "f0_acf",
                                           "tonal_run_s", "tonal_frac", "harm_ratio_db")],
        "temporal": [c for c in cols if c in ("crest_factor", "kurtosis", "skewness", "zcr", "n_transients",
                                              "transient_rate", "itv_cv", "env_kurtosis", "env_cv",
                                              "env_peak_to_median", "mf_peak_ratio", "mf_f0", "mf_dur_ms",
                                              "mf_rate", "energy_cv", "temporal_flatness")],
        "mfcc": [c for c in cols if c.startswith("mfcc_")],
    }


def md(df: pd.DataFrame, nd: int = 3) -> str:
    return df.round(nd).to_markdown()
