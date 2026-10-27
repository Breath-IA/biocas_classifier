# gate 1 · e01 — Línea base con features espectrales

Análisis inicial (antes de separar por gates): separabilidad de las features por evento y clasificación con
RandomForest dejando un año fuera.

- **Cubre:** gate 1 (N contra anormal) y una primera mirada al gate 2 (W contra crepitantes, 6 clases). Los análisis
  dedicados al gate 2 están en [../../gate2_event_type/](../../gate2_event_type/).
- **Código:** `analyze_separability.py`
- **Entradas:** `../../common/features/segments_spectral.csv` (`common/extract_spectral_features.py`)
- **Salidas:** `results/report.md` (informe), `results/fig_*.png`, `results/*.csv`
- **Comando:** `.venv/Scripts/python experiments/gate1_normal_vs_abnormal/e01_spectral_baseline/analyze_separability.py` (~6 min)
- **Nota:** `report.md` y las figuras nombran las tareas "gate1 N/abn", "gate2 W/crk" y "gate2 6cls" (los dos gates de plan.md).

## Resultado clave

- Gate 1 (RF, features de forma + tonales + temporales): ROC AUC 0,85 / 0,84 / 0,80 al retener 2022 / 2023 / 2024 y **0,63 al retener 2025**.
- Las features MFCC y de nivel de señal generalizan peor aún; el desplazamiento de dominio train → 2025 es detectable (AUC 0,79).
- El espectro de las clases anormales respecto de N se invierte en 2025 en las bandas altas.
