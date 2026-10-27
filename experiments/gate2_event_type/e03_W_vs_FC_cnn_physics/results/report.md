# Gate 2 · e03 — CNN (4 bloques) y features físicas de plan.md, W contra FC

Misma validación que e02 (dejar un año fuera; pacientes disjuntos entre años; solo eventos W y FC), pero con una CNN
sobre log-mel y, después, con la rama de features físicas de plan.md §1. Tablas completas: [report_auto.md](report_auto.md).
Código: `../train_cnn.py`, `../report.py`, `../../../common/physics_features.py`.

## 1. Resumen

1. **La CNN supera a los modelos con features hechas a mano**: AUC medio 0,923 contra 0,899 del mejor RF de e02, y
   **0,863 contra 0,809 en 2025**. Su AUC no depende de la duración del evento (sección 4).
2. **Añadir las features físicas por concatenación aporta +0,005 de AUC medio**: +0,019 en 2022, −0,007 en 2023, −0,002 en 2024
   y +0,009 en 2025. Es del orden del ruido; con la primera versión (defectuosa) de las features el efecto en 2025 era −0,004.
3. **Las features físicas separan por sí solas** (RF: AUC 0,86; LR: 0,81), sobre todo las de decaimiento τ, pero **no añaden
   información a las features hechas a mano** en un modelo lineal (0,889 contra 0,888) y poco en un RF (+0,01).
4. **El umbral 0,5 es el punto débil de la CNN.** La balanced accuracy varía hasta 0,19 entre semillas del mismo fold
   (recall FC de 0,04 a 0,95 en 2022). El ranking es estable (AUC); la calibración no.

## 2. Diseño

- **Entrada:** log-mel de 64 × 95 (n_fft 512, hop 128, fmin 50, fmax 2000, 4 kHz) sobre ventanas de 3 s (eventos más cortos
  repetidos en *tile*, como en `pipeline.yaml`; más largos recortados al centro) con normalización por instancia.
- **CNN:** 4 bloques Conv3×3-BN-ReLU-MaxPool (16-32-64-128), pooling global, MLP de 64 unidades; ≈ 97 k parámetros. SpecAugment
  (una máscara de frecuencia y una de tiempo) y desplazamiento circular. BCE con `pos_weight` balanceado, AdamW, 20 épocas máximo.
- **Early stopping** con el 15 % de los **pacientes** de los años de entrenamiento; el año retenido nunca se mira.
- **Fusión (`cnn+phys`):** MLP de 32 unidades sobre las 28 features físicas (imputación + cuantiles normales ajustados solo con
  entrenamiento) concatenada con el embedding de la CNN. Corresponde a la "fusión por concatenación simple" de plan.md.
- **Semillas:** 3 por celda; 24 entrenamientos (≈ 40 min en CPU).

## 3. Features físicas (plan.md §1)

Familias: (A) decaimiento τ de la envolvente tras cada transitorio, (B) frecuencia instantánea 100–1900 Hz y su varianza,
(C) banco de matched filters de crepitante `A·exp(-t/τ)·sin(2πf0t)` con barrido (f0, τ), (D) matched filter de sibilancia
(ventana 100 ms, f0 100–1000 Hz), (E) persistencia espectral entre frames con espectro blanqueado.

AUC por año, W contra FC (>0,5 = mayor en FC); `margen` = peor |AUC−0,5| con el mismo signo en los 4 años:

| feature | 2022 | 2023 | 2024 | 2025 | margen | shift | lectura |
|---|---:|---:|---:|---:|---:|---:|---|
| ph_tau_p25 | 0,22 | 0,27 | 0,15 | 0,30 | −0,20 | 0,24 | τ más corto en FC (≈ 4–7 ms contra ≈ 10 ms en W), como predice el modelo físico |
| ph_env_slope_neg_frac | 0,67 | 0,66 | 0,77 | 0,74 | 0,16 | 0,07 | más ventanas con caída de envolvente en FC (shift mínimo) |
| ph_tau_med | 0,26 | 0,32 | 0,18 | 0,30 | −0,18 | 0,26 | idem |
| ph_decay_frac | 0,70 | 0,67 | 0,78 | 0,70 | 0,17 | 0,26 | fracción de transitorios con τ < 20 ms |
| ph_mf_cr_tau | 0,37 | 0,33 | 0,35 | 0,34 | −0,13 | 0,20 | τ óptimo del banco más corto en FC |
| ph_mf_cr_max | 0,77 | 0,74 | 0,66 | 0,62 | 0,12 | 0,15 | decae con los años |
| ph_mf_wh_frac70 | 0,42 | 0,38 | 0,34 | 0,38 | −0,09 | 0,28 | persistencia tonal de W (signo estable, efecto pequeño) |
| ph_mf_wh_f0_track_std | 0,40 | 0,40 | 0,34 | 0,34 | −0,10 | 0,20 | idem |
| ph_if_std, ph_if_cv, ph_if_stable_frac | – | – | – | – | 0 | 0,31–0,43 | **el signo cambia entre años**: no fiables |
| ph_persist_cos_mean, ph_persist_run_s | – | – | – | – | 0 | 0,20–0,45 | **el signo cambia en 2025**: no fiables |

Referencia de e02: `env_kurtosis` tiene margen 0,25 y `crest_factor` 0,23, es decir las features físicas de decaimiento son
más débiles que las de impulsividad, pero tienen sentido físico y signo estable.

**Defecto corregido durante el experimento.** La primera versión del matched filter de sibilancia normalizaba mal: el máximo teórico
de la correlación para un tono con ventana Hann es 0,577, por lo que los umbrales 0,5 y 0,7 eran inalcanzables y
`ph_mf_wh_frac70` valía exactamente 0,5 de AUC. Se reescaló, se centró la persistencia espectral (el coseno de espectros
positivos era casi constante en 0,95) y se añadió τ = 1 ms al banco de crepitantes. La fusión se repitió entera con las
features corregidas; la corrida anterior está en `v1/`. El baseline `cnn` no usa esas features y no se repitió.

**Clasificadores simples con las features (ROC AUC, RF / LR):**

| features | 2022 | 2023 | 2024 | 2025 | media |
|---|---:|---:|---:|---:|---:|
| solo físicas (RF) | 0,921 | 0,858 | 0,885 | 0,776 | 0,860 |
| solo físicas (LR) | 0,844 | 0,817 | 0,832 | 0,730 | 0,806 |
| `robust` de e02 + físicas (RF) | 0,954 | 0,905 | 0,926 | 0,797 | 0,895 |
| `robust` de e02 (RF) | 0,934 | 0,903 | 0,915 | 0,788 | 0,885 |
| `robust` de e02 + físicas (LR) | 0,934 | 0,905 | 0,911 | 0,806 | 0,889 |
| `robust` de e02 (LR) | 0,932 | 0,903 | 0,913 | 0,804 | 0,888 |

Primer chequeo del protocolo (¿las features solas separan?): sí, con RF alcanzan 0,86 de AUC medio, claramente sobre el azar y no
lejos de las features hechas a mano (0,885).

## 4. CNN

ROC AUC por año retenido (media de 3 semillas):

| modelo | 2022 | 2023 | 2024 | 2025 | media |
|---|---:|---:|---:|---:|---:|
| **CNN + físicas** | 0,951 | 0,926 | 0,959 | **0,873** | 0,927 |
| **CNN (solo log-mel)** | 0,932 | 0,933 | 0,961 | 0,863 | 0,923 |
| RF `all` (e02) | 0,941 | 0,912 | 0,932 | 0,809 | 0,899 |
| LR `robust` (e02) | 0,932 | 0,903 | 0,913 | 0,804 | 0,888 |

- Conjunto de semillas (probabilidades promediadas): CNN 0,946 / 0,937 / 0,964 / 0,871; CNN + físicas 0,956 / 0,937 / 0,962 / 0,877.
- ΔAUC pareada (CNN + físicas − CNN): +0,019 (σ 0,018) · −0,007 (σ 0,012) · −0,002 (σ 0,003) · +0,009 (σ 0,004); media +0,005.
- 2025 sigue siendo el año más difícil para todos los modelos; la CNN reduce la brecha pero no la cierra (0,86–0,88 contra 0,93–0,96 en 2022–24).

**Control 1: la CNN no usa la duración.** El *tile* repite los eventos cortos y podría delatar su longitud. AUC del conjunto de semillas
por estrato de duración (CNN): 1–2 s: 0,957 / 0,928 / 0,985 / 0,872; ≥ 2 s: 0,941 (2024) y 0,856 (2025); < 1 s: 0,938 / 0,930 / – / 0,899
(2022 / 2023 / 2024 / 2025). Es parecido al AUC global en cada año, así que la duración no explica el rendimiento.
Para comparar, una sola feature de e02 (`env_kurtosis`) llega a 0,73 en 2025 con eventos de 1–2 s.

**Control 2: lo que se pierde por el umbral.** Balanced accuracy a 0,5 contra el techo con el mejor umbral posible (no alcanzable
en la práctica, porque usa las etiquetas del año retenido), conjunto de semillas:

| fold | CNN @0,5 | CNN techo | CNN + físicas @0,5 | CNN + físicas techo |
|---|---:|---:|---:|---:|
| 2022 | 0,871 | 0,885 | 0,679 | 0,899 |
| 2023 | 0,860 | 0,872 | 0,851 | 0,865 |
| 2024 | 0,909 | 0,918 | 0,895 | 0,925 |
| 2025 | 0,724 | 0,800 | 0,802 | 0,814 |

El techo en 2025 es casi igual para las dos variantes (0,80 contra 0,81): **la "mejora" de la fusión en balanced accuracy de
2025 (0,72 → 0,80) es de calibración, no de poder discriminativo**. En 2022 la fusión necesitaría un umbral de 0,09 en lugar de 0,5.
Por semilla, la balanced accuracy a 0,5 tiene una desviación de hasta 0,19 (CNN, 2022) y el recall de FC va de 0,04 a 0,95.

## 5. Importancia de las features físicas en la fusión

Permutación sobre el año retenido (caída de AUC): `ph_mf_cr_tau` 0,0057 (0,0125 en 2025; positiva en los 4 años), `ph_if_std_best100` 0,0030,
`ph_decay_frac` 0,0021, `ph_tau_p75` 0,0018, `ph_dur_est_s` 0,0017 (0,0063 en 2025). Ninguna supera 0,013, coherente con la ganancia
de AUC casi nula. La más útil es el τ óptimo del banco de matched filters, es decir la constante de decaimiento del crepitante.

## 6. Sugerencias

1. **Usar la CNN como columna vertebral discriminativa y calibrar el umbral**: ganancia de AUC de ≈ +0,02 de media y +0,05 en 2025
   sobre las features. Calibrar con las probabilidades de la partición de validación por pacientes (Platt o temperatura), promediar
   semillas (+0,008 de AUC en 2025 al promediar) y reportar siempre AUC junto a balanced accuracy.
2. **Probar la fusión con más rigor antes de adoptarla**: ≥ 5 semillas y más folds; y una fusión tardía (regresión logística sobre el
   logit de la CNN + features físicas) en vez de concatenar al embedding. La ganancia observada (+0,005) no la justifica aún.
3. **Mejorar solo las features físicas que funcionaron**: las de decaimiento τ (`mf_cr_tau`, `tau_p25`, `decay_frac`). Ampliar la grilla
   (τ 0,5–20 ms, f0 hasta 1200 Hz), ajustar sobre señal blanqueada y estimar τ con ajustes ponderados por amplitud. Descartar las de
   frecuencia instantánea y la persistencia espectral (el signo cambia entre años).
4. **Resolución temporal de la entrada**: con hop de 128 muestras (32 ms) un crepitante de 5 ms ocupa una sola columna. Probar hop 32 /
   n_fft 128 o escalogramas wavelet, y ventanas de ~0,5 s agregadas por evento en lugar del *tile* de 3 s.
5. **Paso 3 del plan sin hacer**: síntesis paramétrica de crepitantes sobre fondo real, dirigida al par débil N–FC (AUC 0,60 en 2025 en
   e01), y aumentación de forma de onda en línea para la CNN (aquí solo se usó SpecAugment).
6. **Siguiente experimento natural**: repetir esta CNN en el gate 1 (N contra anormal), donde las features hechas a mano generalizan peor
   (AUC 0,63 en 2025 en gate 1 · e01), para ver si la ventaja de la CNN se mantiene allí.

## 7. Limitaciones

- 3 semillas por celda; la desviación entre semillas del AUC es 0,001–0,019, así que diferencias de ~0,01 por fold no son concluyentes.
- El early stopping usa una partición de validación del mismo dominio que el entrenamiento, por lo que no protege contra el cambio de año.
- Una sola arquitectura y entrenamiento solo en CPU; no se exploraron hiperparámetros.
- Las configuraciones se comparan sobre los mismos 4 años retenidos; no hay un conjunto final intacto.
- Las features físicas se calculan sobre el evento sin *tile* mientras la CNN ve el evento repetido a 3 s.
- Los estratos de duración dejan celdas vacías (2022 casi no tiene eventos de ≥ 2 s y 2024–2025 casi ninguno de < 1 s).
- Solo W y FC: CC, WC, R, S y N quedan fuera.
- Figuras: `g4_comparison`, `g4_ph_auc_by_year`, `g4_ph_dist_top12`, `g4_training_curves`, `g4_confusion`, `g4_ph_importance`.
