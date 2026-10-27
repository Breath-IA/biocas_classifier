# gate 2 · e01 — N / W / FC plano (sin gate 1) con aumentación

Clasificación directa de 3 clases, sin pasar por el gate 1, con y sin aumentación a nivel de grabación.

- **Código:** `augment.py` (genera copias aumentadas y extrae sus features) → `analyze.py` (análisis y clasificación)
- **Entradas:** `../../common/features/segments_spectral.csv`
- **Salidas:** `results/report.md` (conclusiones y sugerencias), `results/report_auto.md` (tablas), `results/g2_*.png`,
  `results/*.csv`, `results/plan.csv` (plan de aumentación); `cache/aug_<variante>.csv`, `cache/bg_pool.npz` (ignorados por git)
- **Comandos:** `augment.py` (~3 min por variante: full, eq, noise, jitter) y luego `analyze.py` (más de 10 min: más de 100 ajustes de RF)
- **Aumentación:** simétrica en N, W y FC (≈ 45–60 % de muestras aumentadas por clase): fondo real mezclado
  (`noise`), realce espectral tipo otro dispositivo y ganancia (`eq`), desplazamiento de fronteras (`jitter`) y las tres (`full`).
  Sin pitch ni time-stretch porque mueven f0 y τ.
- **Nombre anterior:** "gate 2" (`eda/gate2_*.py`, `eda/out/gate2/`)

## Resultado clave

- Balanced accuracy N/W/FC (azar 0,33): 0,65–0,69 (2022) → **0,46–0,50 (2025)**. N contra FC es el cuello de botella (AUC 0,60 en 2025).
- Las features que sobreviven al cambio de año son de impulsividad temporal; las de forma espectral por bandas no.
- La mejora real (0,37 → 0,49 en 2025) viene de submuestrear N, no de aumentar (+0,01 de media con aumentación).
- macro-F1 y balanced accuracy dan rankings opuestos por la prevalencia de N.
