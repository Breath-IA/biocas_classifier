# Gate 2 · e02 — W (sibilancia) contra FC (crepitante fino), sin la clase N

Misma validación que e01 (N / W / FC plano) (dejar un año fuera; pacientes disjuntos entre años), pero solo con eventos W y FC.
Cada configuración de RandomForest se ajustó con 3 semillas; la regresión logística (LR) usa cuantiles normalizados.
Tablas completas: [report_auto.md](report_auto.md). Código: `../analyze.py` (reutiliza las copias
aumentadas de W y FC de `../../e01_flat_N_W_FC_augmentation/cache/`).

## 1. Resumen

1. **W contra FC se separa bien: AUC medio ≈ 0,90** (2022 0,94 · 2023 0,91 · 2024 0,93 · **2025 0,81**). Quitar N no mejora
   el par: en el problema de 3 clases el AUC W–FC de 2025 ya era ≈ 0,80. La dificultad del plano N / W / FC (e01) venía de N.
2. **El año 2025 sigue siendo el más difícil** (AUC 0,78–0,82), y la causa es la misma que antes: la diferencia espectral
   entre FC y W desaparece en 2025.
3. **Las features que separan son de impulsividad temporal**: `env_kurtosis`, `crest_factor`, `kurtosis`,
   `env_peak_to_median`, `env_cv` (AUC 0,71–0,88 en los 4 años), más `spec_flux`, `spec_crest`, `n_spec_peaks`.
4. **La regresión logística iguala al RandomForest en AUC y lo supera en balanced accuracy** (0,81 contra 0,77 de media). El RF
   conserva el ranking pero se descalibra al cambiar de año (sección 4).
5. **La aumentación aporta como máximo +0,01 de AUC, y solo en 2025.** No cambia nada en los otros años.

## 2. Datos

| | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| W | 865 | 194 | 141 | 305 |
| FC | 1167 | 309 | 169 | 1885 |
| % FC | 57 | 61 | 55 | **86** |

Duración mediana del evento (s): W 0,57 → 0,83 → 1,93 → 1,84; FC 0,91 → 1,22 → 1,89 → 1,92. Con W y FC ya casi
equilibradas en entrenamiento (1200 contra 1645 en 2022–24), no hizo falta submuestrear. Entrenamiento aumentado:
W 3009 (60 % aumentadas), FC 2997 (45 %).

## 3. Reporte de features (W contra FC)

AUC por año (>0,5 = mayor en FC). `margen` = peor |AUC−0,5| con el mismo signo en los 4 años; `shift` = desplazamiento de
dominio train → 2025 dentro de la misma clase.

| feature | 2022 | 2023 | 2024 | 2025 | margen | shift | lectura |
|---|---:|---:|---:|---:|---:|---:|---|
| env_kurtosis | 0,88 | 0,80 | 0,83 | 0,75 | 0,25 | 0,15 | FC impulsivo, W sostenido |
| crest_factor | 0,82 | 0,79 | 0,77 | 0,73 | 0,23 | 0,16 | idem |
| kurtosis | 0,77 | 0,74 | 0,74 | 0,71 | 0,21 | 0,11 | idem (≈ crest_factor) |
| env_peak_to_median | 0,79 | 0,76 | 0,71 | 0,67 | 0,17 | 0,14 | idem (≈ env_cv) |
| env_cv | 0,76 | 0,71 | 0,66 | 0,67 | 0,16 | 0,10 | idem |
| spec_flux | 0,30 | 0,24 | 0,18 | 0,31 | −0,19 | **0,36** | mayor en W; orden estable pero distribución desplazada |
| spec_crest | 0,30 | 0,21 | 0,23 | 0,36 | −0,14 | 0,15 | mayor en W |
| mf_peak_ratio | 0,78 | 0,75 | 0,68 | 0,64 | 0,14 | 0,15 | matched filter de crepitante |
| tonal_run_s (nueva) | 0,38 | 0,35 | 0,29 | 0,36 | −0,12 | 0,12 | W mantiene el tono más tiempo |
| n_spec_peaks | 0,23 | 0,29 | 0,39 | 0,38 | −0,11 | 0,13 | FC casi siempre 1 pico; W entre 1 y 3 |
| max_peak_prom_db | 0,19 | 0,16 | 0,20 | 0,39 | −0,11 | 0,23 | pierde poder en 2025 |
| harm_ratio_db (nueva) | 0,73 | 0,70 | 0,62 | 0,58 | 0,08 | 0,16 | modesta y decreciente |

**Tendencia con los años.** Las features de impulsividad pierden 0,07–0,12 de AUC de 2022 a 2025 (`env_kurtosis`
0,88 → 0,75). Con estratos de duración la caída persiste: en eventos de 1–2 s `env_kurtosis` va de 0,81 (2022) a 0,73
(2025), y en eventos de ≥2 s de 0,87 (2024) a 0,78 (2025). **No se debe solo a que los eventos se hicieron más largos.**

**Duración como atajo.** El AUC de `duration_s` por sí sola (W→FC) es 0,79 en 2022, 0,70 en 2023, 0,47 en 2024 y 0,59 en 2025:
W era más corta que FC en los primeros años y ya no. Un modelo que la use aprende una regla que no se sostiene.

**Espectro (`g3_spectra_by_year.png`).** FC − W vale +4 a +9 dB por encima de 1 kHz en 2022–24 y ≈ 0 en 2025 (con un hueco de
−6 dB a 500–700 Hz). Por eso las features de forma espectral por bandas y la curva de 32 bins tienen el mayor desplazamiento
(0,39–0,41) y no mejoran al agregarlas (`robust+sp32` ≈ `robust`).

**Features nuevas y descartes.** `tonal_run_s` y `harm_ratio_db` aportan poco pero con signo estable. `tonal_frac` es
constante (AUC exactamente 0,50 en los 4 años) e `itv_cv` tiene shift 0,56: descartar ambas.

**Redundancia.** `crest_factor` ≈ `kurtosis`; `env_peak_to_median` ≈ `env_cv` ≈ `mf_peak_ratio`; `sp_00` ≈ `max_peak_prom_db`.

**Dispersión (`g3_scatter_best2.png`).** En el plano `env_kurtosis` × `crest_factor`, W se agrupa abajo-izquierda y FC se
extiende hacia arriba-derecha. El solape está en valores bajos: muchos FC de 2025 tienen poca impulsividad y se parecen a W.

## 4. Clasificación

ROC AUC por año retenido (RF = media de 3 semillas):

| modelo · entrenamiento · features | 2022 | 2023 | 2024 | 2025 | media |
|---|---:|---:|---:|---:|---:|
| RF · A_eq · all | 0,945 | 0,911 | 0,934 | 0,813 | **0,901** |
| RF · B0 · all | 0,941 | 0,912 | 0,932 | 0,809 | 0,899 |
| LR · A_eq · robust | 0,932 | 0,899 | 0,911 | 0,819 | 0,890 |
| LR · B0 · robust | 0,932 | 0,903 | 0,913 | 0,804 | 0,888 |
| RF · B0 · robust | 0,933 | 0,903 | 0,913 | 0,787 | 0,884 |
| RF · B0 · stable (15 mejores por consistencia) | 0,922 | 0,906 | 0,896 | 0,768 | 0,873 |

- `all` (incluye duración, MFCC y nivel) supera a `robust` en ≈ +0,01 de AUC medio y +0,01–0,02 en 2025. Es una diferencia
  pequeña: no se confirma que esas features dañen aquí, pero tampoco hay evidencia para recomendarlas.
- La selección `stable` rinde menos que `robust`: elegir por consistencia entre años descarta señal útil.
- **Ruido de semilla:** desviación típica del AUC entre semillas de 0,002 (mediana) a 0,006 (máximo). Es más pequeño que el
  error de muestreo de un año con 141–305 W, así que diferencias de ~0,01 no son concluyentes.

**Calibración entre años (hallazgo).** Con las mismas features `robust` y entrenamiento B0, en 2024 el RF y la LR tienen el mismo AUC (0,913),
pero el RF acierta 0,71 de balanced accuracy (recall W 0,44 · FC 0,98) y la LR 0,81 (recall W 0,70 · FC 0,93). La diferencia AUC − balanced
accuracy es 0,08–0,20 para el RF (0,20 en 2024) y estable en 0,07–0,09 para la LR. El RF ordena bien los eventos pero su umbral
se desplaza al cambiar de año.

**2025 (mejor RF, `A_eq · all`, semilla 42).** Recall W 0,69 y FC 0,77. Como el 86 % de los eventos de 2025 son FC, 440 FC clasificados como W
dejan la precisión de W en 0,32 (211 aciertos de 651 predichos W). Con esa prevalencia, la métrica operativa depende de qué error
cuesta más.

## 5. Aumentación

ΔAUC contra B0, media sobre conjuntos de features:

| fold | A_full | A_eq | A_noise | A_jitter |
|---|---:|---:|---:|---:|
| 2022 | +0,005 | +0,003 | +0,004 | −0,002 |
| 2023 | −0,006 | 0,000 | −0,003 | +0,002 |
| 2024 | 0,000 | 0,000 | −0,001 | −0,002 |
| 2025 | **+0,012** | **+0,012** | +0,006 | +0,011 |

- La mejora de 2025 es pequeña pero positiva en las 4 variantes; mayor con conjuntos débiles (`stable`: 0,768 → 0,78–0,79) y casi
  nula con `all` (0,809 → 0,813–0,820). En los otros años no hay efecto.
- Ninguna variante reduce de forma apreciable el desplazamiento de dominio (|2·AUC−1| train vs 2025: 0,294 → 0,291 con `eq`);
  `noise` lo aumenta (0,311). El efecto, si existe, no pasa por acercar la distribución a 2025.

## 6. Importancia fuera de dominio (RF `A_eq · all`, caída de AUC al permutar, año retenido)

`spec_flux` 0,030 · `env_kurtosis` 0,025 · `n_spec_peaks` 0,012 · `max_peak_prom_db` 0,007 · `mfcc_11_mean` 0,004 · `zcr` 0,004 ·
`crest_factor` 0,003. Suma por grupo (solo positivas): temporal 0,041, forma espectral 0,039 (casi todo `spec_flux`), tonal 0,025, MFCC 0,015.
`n_spec_peaks` es la única de las tres primeras que aporta lo mismo en los 4 años (0,010–0,015, incluido 2025).
`max_peak_prom_db` baja a ≈ 0 en 2025.

## 7. Sugerencias

1. **Usar un modelo simple y calibrado.** La LR sobre features normalizadas por cuantiles iguala el AUC del RF, da +0,04 de
   balanced accuracy de media y se descalibra mucho menos. Antes de pasar a modelos más grandes, calibrar el umbral (Platt o
   isotónica con un año de validación) y probar normalización de cuantiles por año sin etiquetas.
2. **Núcleo de features para W–FC:** `env_kurtosis`, `crest_factor` (o `kurtosis`), `env_cv` (o `env_peak_to_median`),
   `spec_flux`, `spec_crest`, `n_spec_peaks`, `tonal_run_s`, `mf_peak_ratio`, `harm_ratio_db`. Excluir `duration_s`
   (el signo del efecto cambia entre años), `tonal_frac` e `itv_cv`. Mantener `max_peak_prom_db` solo si se valida en un año nuevo.
3. **Features por sub-ventana.** Calcularlas en ventanas de ~0,5 s y agregarlas (máximo y media) para que la impulsividad de pocos crepitantes no
   se diluya en eventos de ~1,9 s. Es la prueba más directa de la caída persistente de `env_kurtosis` dentro de cada estrato de duración.
4. **Cuidar el solape en impulsividad baja.** Los FC con poca impulsividad se confunden con W. Probar features de crepitante sobre
   señal blanqueada (conteo y energía de los k picos más fuertes) y de sibilancia con tonos rastreados entre 250 y 1200 Hz.
5. **Aumentación:** solo `eq` o `jitter` y solo si se mantiene con más años o con más semillas. La ganancia es de +0,01 y concentrada en 2025.
6. **Investigar 2025.** La desaparición de la ventaja de FC en alta frecuencia (+4 a +9 dB en 2022–24, ≈ 0 en 2025) apunta a un cambio
   de equipo, filtrado o protocolo de anotación; conviene revisarlo con la metadata de grabación antes de ajustar más el modelo.

## 8. Limitaciones

- W tiene 141–305 eventos por año retenido: el AUC de un solo año tiene un error de muestreo de ≈ ±0,02.
- La elección de configuraciones se mira sobre los mismos 4 años retenidos; no hay un conjunto final intacto.
- En 2022 hay muchos eventos de < 1 s y en 2024–2025 casi ninguno (y viceversa para ≥ 2 s), así que los estratos de duración
  dejan celdas vacías (NaN en el informe automático).
- Solo W y FC: CC, WC, R, S y N quedan fuera.
- Figuras: `g3_spectra_by_year`, `g3_auc_by_year`, `g3_dist_top12`, `g3_scatter_best2`, `g3_models`, `g3_auc_by_fold`,
  `g3_confusion`, `g3_importance`.
