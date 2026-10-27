# gate 2 · e02 — W contra FC con features

Igual que e01 pero **sin la clase N**: clasificación binaria sibilancia (W) contra crepitante fino (FC), con RandomForest
y regresión logística (3 semillas por configuración).

- **Código:** `analyze.py`
- **Entradas:** `../../common/features/segments_spectral.csv` y las copias aumentadas de W y FC de
  `../e01_flat_N_W_FC_augmentation/cache/aug_<variante>.csv` (**requiere haber corrido antes e01/augment.py**)
- **Salidas:** `results/report.md` (conclusiones y sugerencias), `results/report_auto.md`, `results/g3_*.png`, `results/*.csv`
- **Comando:** `.venv/Scripts/python experiments/gate2_event_type/e02_W_vs_FC_features/analyze.py` (varios minutos: unos 400 ajustes pequeños)
- **Nombre anterior:** "gate 3" (`eda/gate3_analysis.py`, `eda/out/gate3/`). Las figuras conservan el prefijo `g3_`.

## Resultado clave

- AUC medio ≈ 0,90 (2022 0,94 · 2023 0,91 · 2024 0,93 · **2025 0,81**). Quitar N no mejora el par: la dificultad de e01 venía de N.
- Separan `env_kurtosis`, `crest_factor`, `kurtosis`, `env_cv` (AUC 0,71–0,88 en los 4 años); la diferencia espectral FC − W desaparece en 2025.
- La regresión logística iguala al RF en AUC y lo supera en balanced accuracy (0,81 contra 0,77); el RF se descalibra al cambiar de año.
- La aumentación aporta ≈ +0,01 de AUC, solo en 2025. `duration_s` es un atajo no confiable.
