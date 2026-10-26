# gate 2 · e03 — CNN y features físicas, W contra FC

CNN de 4 bloques convolucionales sobre log-mel con la misma validación por año que e02, primero sola (baseline del plan,
paso 1) y luego con una rama de las **features físicas de plan.md §1** concatenada (paso 2).

- **Código:** `train_cnn.py` (entrena y guarda resultados; reanuda lo ya hecho) → `report.py` (tablas y figuras)
  y `../../common/physics_features.py` (features físicas, a correr antes)
- **Entradas:** `../../common/features/segments_spectral.csv`, `../../common/features/physics_features.csv`;
  `report.py` también lee `../e02_W_vs_FC_features/results/classification_agg.csv` como referencia
- **Salidas:** `results/report.md` (conclusiones), `results/report_auto.md` (tablas), `results/g4_*.png`, `results/cnn_*.csv`,
  `results/logs/`; `cache/mel_cache.npz` (ignorado por git)
- **Comandos** (desde la raíz del repo):
  ```bash
  .venv/Scripts/python experiments/common/physics_features.py                                   # ~1 min
  .venv/Scripts/python experiments/gate2_event_type/e03_W_vs_FC_cnn_physics/train_cnn.py        # ~40 min en CPU (24 entrenamientos)
  .venv/Scripts/python experiments/gate2_event_type/e03_W_vs_FC_cnn_physics/report.py           # ~2 min
  ```
- **Nombre anterior:** "gate 4" (`eda/gate4_cnn.py`, `eda/gate4_report.py`, `eda/out/gate4/`). Las figuras conservan el prefijo `g4_`.
- **`results/v1/`:** primera corrida, con una versión defectuosa del matched filter de sibilancia (ver `report.md` §3). Las tablas
  actuales usan las features corregidas; el baseline `cnn` de la corrida actual proviene de la primera corrida porque no usa esas features.

## Resultado clave

- La CNN (solo log-mel) supera a las features hechas a mano: **AUC medio 0,923 contra 0,899** (RF de e02), y **0,863 contra 0,809 en 2025**.
  No depende de la duración del evento.
- La fusión con features físicas aporta **+0,005 de AUC** (+0,009 en 2025), dentro del ruido. Las features de decaimiento τ separan W de FC
  con sentido físico (τ más corto en FC), pero no añaden información a las features hechas a mano.
- El umbral 0,5 es inestable entre semillas (balanced accuracy con desviación de hasta 0,19): conviene calibrar.
