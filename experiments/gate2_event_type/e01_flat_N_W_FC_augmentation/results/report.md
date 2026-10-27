# Gate 2 · e01 — N / W / FC plano (sin gate 1): features, aumentación y sugerencias

Clasificación directa de 3 clases sin pasar por el gate 1. Validación dejando un año fuera (2022, 2023, 2024, 2025), con
pacientes disjuntos entre años (≤1 paciente compartido). Solo se aumenta el entrenamiento; el año retenido se evalúa con
eventos originales y su prevalencia real. Tablas completas: [report_auto.md](report_auto.md). Código:
`../../../common/extract_spectral_features.py`, `../augment.py`, `../analyze.py`.

## 1. Resumen

1. **El problema plano N/W/FC es difícil y empeora con los años.** Balanced accuracy (azar = 0,33): 0,65–0,69 en 2022,
   0,64–0,67 en 2023, 0,51–0,56 en 2024 y **0,46–0,50 en 2025** (split oficial).
2. **El cuello de botella es N contra FC.** AUC por pares en 2025: N–W ≈ 0,82, W–FC ≈ 0,80, **N–FC ≈ 0,60**. Con el mejor
   modelo, el 72 % de los FC de 2025 se predicen como N (recall FC 0,20; recall W 0,44; recall N 0,84).
3. **Las features que sobreviven al cambio de año son temporales (impulsividad), no espectrales.** `env_kurtosis`,
   `crest_factor`, `kurtosis`, `env_cv`, `n_transients`. Las de forma espectral por bandas casi no aportan fuera de
   dominio porque la diferencia espectral entre clases cambia de signo según el año; la excepción es `spec_flux`
   (variación del espectro en el tiempo).
4. **La aumentación tal como la construí no ayuda en 2025.** La mejora grande (0,37 → 0,49 de balanced accuracy en 2025)
   viene de submuestrear N, no de aumentar. Las copias aumentadas dan +0,01 de media (+0,03 en 2022–23, −0,01 en 2025),
   dentro del ruido de una sola semilla.
5. **macro-F1 y balanced accuracy dan rankings opuestos.** Es un efecto de prevalencia, no de calidad del modelo (sección 5).

## 2. Datos

| clase | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| N | 6887 | 2458 | 7252 | 2175 |
| W | 865 | 194 | 141 | 305 |
| FC | 1167 | 309 | 169 | 1885 |

Duración mediana del evento anotado (s): W pasa de 0,57 (2022) a 1,84 (2025); FC de 0,91 a 1,92; N de 1,36 a 1,93. La
anotación pasó de eventos ajustados a ventanas de ~2 s. Prevalencia de N entre las tres clases: 85 % en 2022–24 (77 % en 2022, 83 % en 2023, 96 % en 2024) contra 50 % en 2025.

Entrenamiento con aumentación (2022–24), simétrica en las tres clases para que "suena aumentado" no sea pista de la
etiqueta: N 2989 (51 % aumentadas), W 3009 (60 %), FC 2997 (45 %). Transformaciones: fondo real de otro evento N del mismo
año a SNR 8–25 dB (`noise`), realce/atenuación espectral tipo otro dispositivo ±8 dB más ganancia ±10 dB (`eq`),
desplazamiento de fronteras ±15 % (`jitter`), y `full` = las tres. No se usó pitch ni time-stretch porque mueven f0 y τ.

## 3. Reporte de features

AUC por año (>0,5 = mayor en la segunda clase del par). `margen` = peor |AUC−0,5| entre los 4 años con el mismo signo.

### W contra FC (el par que las features resuelven mejor)

| feature | 2022 | 2023 | 2024 | 2025 | margen | lectura |
|---|---:|---:|---:|---:|---:|---|
| env_kurtosis | 0,88 | 0,80 | 0,83 | 0,75 | 0,25 | FC es impulsivo, W es sostenido |
| crest_factor | 0,82 | 0,79 | 0,77 | 0,73 | 0,23 | idem |
| kurtosis | 0,77 | 0,74 | 0,74 | 0,71 | 0,21 | idem |
| env_peak_to_median | 0,79 | 0,76 | 0,71 | 0,67 | 0,17 | idem |
| env_cv | 0,76 | 0,71 | 0,66 | 0,67 | 0,16 | idem |
| spec_flux | 0,30 | 0,24 | 0,18 | 0,31 | −0,19 | mayor en W que en FC (pero shift 0,32) |
| tonal_run_s (nueva) | 0,38 | 0,35 | 0,29 | 0,36 | −0,12 | W mantiene el tono más tiempo |
| harm_ratio_db (nueva) | – | – | – | – | – | AUC media 0,66, aporte modesto |

Con duración emparejada (1,5–2,5 s) el patrón se mantiene en 2023–2025 (`env_kurtosis` 0,73 / 0,86 / 0,77); en 2022 no hay
suficientes W en ese rango para evaluarlo.

### N contra W

| feature | 2022 | 2023 | 2024 | 2025 | margen | shift |
|---|---:|---:|---:|---:|---:|---:|
| snr_vs_floor_db | 0,73 | 0,78 | 0,77 | 0,68 | 0,18 | 0,10 |
| spec_slope_db_khz | 0,23 | 0,25 | 0,20 | 0,32 | −0,18 | 0,12 |
| crest_factor | 0,31 | 0,29 | 0,35 | 0,31 | −0,15 | 0,16 |
| f0_acf | 0,28 | 0,31 | 0,29 | 0,36 | −0,15 | 0,20 |
| max_peak_prom_db | 0,84 | 0,82 | 0,82 | **0,64** | 0,14 | 0,21 |

`max_peak_prom_db` pierde poder en 2025 (0,84 → 0,64). `snr_vs_floor_db` es la más estable y la de menor shift.

### N contra FC (par débil)

| feature | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| env_cv | 0,74 | 0,66 | 0,66 | 0,61 |
| env_peak_to_median | 0,70 | 0,66 | 0,65 | 0,59 |
| mf_peak_ratio | 0,71 | 0,67 | 0,66 | 0,58 |
| spec_slope_db_khz | 0,33 | 0,40 | 0,40 | 0,30 |

Ninguna feature supera 0,61 en 2025. Con duración emparejada (1,5–2,5 s) `env_kurtosis` pasa de 0,66–0,73 en 2022–24 a **0,55
en 2025**: **la duración sola no explica** la caída de 2025; los FC de 2025 son intrínsecamente más difíciles de separar de N
con estas features.

### Espectros por clase y año (`g2_spectra_by_year.png`)

La diferencia espectral entre clases **no es consistente entre años**: FC − N tiene un pico de +13,7 dB cerca de 500 Hz en
2023 y de −4 dB en 2025; FC − W vale +4 a +9 dB por encima de 1 kHz en 2022–24 y ≈ 0 en 2025. Por eso `shape` y
`sp32` (curva de 32 bins) no ayudan fuera de dominio. N de 2022 además tiene ~10 dB más ruido de alta frecuencia que los demás años.

### Calidad de las features nuevas

| feature | veredicto |
|---|---|
| tonal_run_s | modesta pero estable en W–FC (margen 0,12). Conservar. |
| harm_ratio_db | modesta (W–FC 0,66). Conservar con cautela. |
| tonal_frac | **saturada**: ≥ 0,99 en el 100 % de los eventos. Descartar. |
| itv_cv | shift 0,40 y AUC ≈ 0,5 en los pares útiles. Descartar. |

La saturación de `tonal_frac` (y la de `frame_peakiness_db`, mediana 69 dB) viene de que el espectro cae unos 60 dB entre
150 Hz y 1 kHz, así que "pico ≥ 10 dB sobre la mediana" se cumple siempre. La joroba de 100–200 Hz domina `dom_freq` y las
métricas de tonalidad, y esconde las sibilancias de 300–800 Hz.

### Redundancia (|ρ| > 0,85)

`crest_factor` ≈ `kurtosis`; `env_cv` ≈ `env_peak_to_median` ≈ `mf_peak_ratio`; `sp_00` ≈ `max_peak_prom_db`;
`mfcc_1_mean` ≈ `rms_db`. Un subconjunto compacto basta.

## 4. Aumentación: resultados

Balanced accuracy, features `robust`, RandomForest 200 árboles (una semilla):

| entrenamiento | 2022 | 2023 | 2024 | 2025 | media |
|---|---:|---:|---:|---:|---:|
| B0 (todo, `class_weight`) | 0,60 | 0,56 | 0,55 | **0,37** | 0,52 |
| B1 (N submuestreada, control sin aumentar) | 0,65 | 0,64 | 0,52 | 0,49 | 0,58 |
| A_eq | 0,68 | 0,67 | 0,53 | 0,48 | 0,59 |
| A_jitter | 0,67 | 0,65 | 0,54 | 0,48 | 0,59 |
| A_full | 0,67 | 0,67 | 0,51 | 0,48 | 0,58 |
| A_noise | 0,64 | 0,66 | 0,51 | 0,49 | 0,58 |

- B0 → B1 es la mejora real. En 2025, B0 predice N casi siempre (recall N 0,995, W 0,12, FC 0,00) porque entrena con N al
  85 % y 2025 tiene N al 50 %. `class_weight` no corrige ese desajuste de prevalencia.
- B1 → A: +0,01 de media, positivo en 2022–23, nulo o negativo en 2025. No hay evidencia de beneficio bajo cambio de dominio.
- `eq` no redujo el desplazamiento de dominio (|2·AUC−1| train vs 2025: 0,235 → 0,236). `noise` lo empeoró (0,260) y
  bajó el recall de FC. El cambio entre años no es un simple filtro de dispositivo: depende de la clase.

## 5. macro-F1 contra balanced accuracy

Por prevalencia, macro-F1 favorece a B0 en 2022–24 (0,64 / 0,61 / 0,45 contra 0,53 / 0,50 / 0,18 de A) y a A en 2025 (0,44 contra 0,29).
Un modelo entrenado con clases equilibradas dispara falsos positivos de W/FC cuando N es la gran mayoría del test (85–96 % en 2022–24). Hay que fijar
la prevalencia de despliegue antes de comparar modelos con macro-F1; balanced accuracy y recall por clase no dependen de ella.

## 6. Importancia fuera de dominio (permutación en el año retenido, A_eq + robust)

`env_kurtosis` 0,034 · `max_peak_prom_db` 0,016 · `spec_flux` 0,016 · `crest_factor` 0,011 · `n_transients` 0,010 · `env_cv` 0,008 ·
`f0_acf` 0,008. Por grupo (suma con signo; `spec_flux` es de forma espectral): temporal 0,092, tonal 0,034, forma espectral 0,007.

## 7. Sugerencias (por prioridad)

1. **Fijar el punto de operación.** Decidir qué prevalencia y qué métrica importan (recall por clase / balanced accuracy).
   Ante el desajuste de prevalencia, calibrar umbrales o priors (ajuste EM de priors sobre datos sin etiquetar, o umbrales
   por clase elegidos con validación por año) antes de seguir cambiando modelos.
2. **Re-ventanear los eventos.** Calcular features en sub-ventanas de longitud fija (p. ej. 0,5 s con salto de 0,25 s) y
   agregar por evento (máximo y media). Elimina la dependencia de la longitud anotada, que cambió de 0,6–0,9 s a ~1,9 s, y
   evita que unos pocos crepitantes se diluyan en una ventana de 2 s.
3. **Rehacer las features tonales con el espectro blanqueado** (dividir por el espectro mediano de la grabación o por una
   envolvente suavizada) y limitadas a 250–1200 Hz. Descartar `tonal_frac`, `frame_peakiness_db` e `itv_cv`.
4. **Reforzar las features de crepitante para N–FC**: estadísticos de los k picos transitorios más fuertes, curtosis
   multiescala y matched filter sobre señal blanqueada. Es el par con AUC 0,60 en 2025.
5. **Conjunto de features recomendado:** `env_kurtosis`, `crest_factor`, `env_cv`, `n_transients`, `spec_flux`,
   `max_peak_prom_db`, `f0_acf`, `tonal_run_s`, `harm_ratio_db`, `snr_vs_floor_db`. Excluir `duration_s`, `rms_db`, MFCC y las
   bandas absolutas por su desplazamiento de dominio.
6. **Aumentación:** no adoptar la actual. Mantener el submuestreo de N. Si se sigue con aumentación, usar solo `eq` + ganancia
   suaves y confirmar con varias semillas y comparación pareada. La síntesis de crepitantes sobre fondo N real propuesta en
   plan.md apunta justo al par débil (N–FC) y sería el siguiente experimento razonable.
7. **Inspeccionar los FC de 2025**: escuchar o graficar una muestra con SNR bajo y comparar con FC de 2023. Si la anotación
   de 2025 es más permisiva, ningún cambio de modelo cerrará esa brecha.
8. **Comparar contra la jerarquía** (gate 1 + gate 2). El gate 1 hereda el mismo problema N–FC, así que conviene medir si la
   jerarquía mejora de verdad frente a este plano.

## 8. Limitaciones

- Una sola semilla de RandomForest y un solo sorteo de aumentación: diferencias ≤ 0,03 de balanced accuracy no son concluyentes.
- W tiene 141–305 eventos por año retenido; su recall tiene un error del orden de ±0,03.
- La elección del conjunto `stable` usa solo años de entrenamiento, pero la comparación entre conjuntos de features y el
  ranking de configuraciones se miró sobre los mismos 4 años retenidos.
- CC, WC, R y S quedan fuera de este análisis.
- Las figuras están en esta carpeta: `g2_spectra_by_year`, `g2_auc_by_year`, `g2_dist_*`, `g2_scatter_best2`, `g2_aug_vs_orig`,
  `g2_models`, `g2_confusion_2025`, `g2_importance`.
