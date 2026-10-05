# Separabilidad espectral — informe automático

24,578 eventos extraídos; 0 descartados por 'poor quality' (solo anotado en parte del dataset) → 24,578. 65 features.

## 1. Eventos por clase y año

| class   |   2022 |   2023 |   2024 |   2025 |   total |
|:--------|-------:|-------:|-------:|-------:|--------:|
| N       |   6887 |   2458 |   7252 |   2175 |   18772 |
| W       |    865 |    194 |    141 |    305 |    1505 |
| FC      |   1167 |    309 |    169 |   1885 |    3530 |
| CC      |     66 |     34 |     12 |     65 |     177 |
| WC      |     34 |      2 |      6 |    261 |     303 |
| R       |     53 |     95 |     62 |      7 |     217 |
| S       |     17 |     32 |     17 |      8 |      74 |


## 2. Duración por clase (s)

| class   |   count |   25% |   50% |   75% |   max |
|:--------|--------:|------:|------:|------:|------:|
| N       |   18772 |  1.3  |  1.74 |  2.25 |  9.27 |
| W       |    1505 |  0.52 |  0.78 |  1.5  |  6.12 |
| FC      |    3530 |  1.02 |  1.58 |  2.03 |  7.17 |
| CC      |     177 |  0.92 |  1.4  |  1.67 |  3.02 |
| WC      |     303 |  1.57 |  1.99 |  2.22 |  4.62 |
| R       |     217 |  0.94 |  1.67 |  2.3  |  4.38 |
| S       |      74 |  0.96 |  1.55 |  2.38 |  5.71 |


Pacientes en train y test a la vez: 0

## 3. Features candidatas por par de clases

`auc_*`: AUC univariada (>0.5 = más alto en la 1ª clase del par). `min_year_margin`: peor |AUC−0.5| entre los 4 años con el mismo signo (0 si cambia de signo). `domain_shift`: 0 = train/test indistinguibles dentro de las mismas clases, 1 = separables. `score = margen × (1 − shift)`.

### abn_vs_N

|                   |   auc_train |   auc_test |   min_year_margin |   domain_shift |   score |
|:------------------|------------:|-----------:|------------------:|---------------:|--------:|
| spec_slope_db_khz |       0.326 |      0.293 |            -0.174 |          0.118 |   0.153 |
| snr_vs_floor_db   |       0.677 |      0.631 |             0.131 |          0.103 |   0.118 |
| mfcc_4_mean       |       0.351 |      0.326 |            -0.143 |          0.201 |   0.114 |
| mfcc_1_mean       |       0.611 |      0.719 |             0.121 |          0.312 |   0.083 |
| env_cv            |       0.577 |      0.579 |             0.079 |          0.117 |   0.069 |
| temporal_flatness |       0.435 |      0.407 |            -0.078 |          0.197 |   0.063 |
| band_1500_2000    |       0.454 |      0.3   |            -0.088 |          0.3   |   0.062 |
| max_peak_prom_db  |       0.64  |      0.578 |             0.078 |          0.208 |   0.061 |
| mfcc_8_mean       |       0.373 |      0.395 |            -0.086 |          0.312 |   0.059 |
| spec_centroid     |       0.67  |      0.61  |             0.083 |          0.284 |   0.059 |
| rel_high_db       |       0.602 |      0.581 |             0.071 |          0.17  |   0.059 |
| band_1250_1500    |       0.46  |      0.298 |            -0.085 |          0.306 |   0.059 |


### W_vs_N

|                   |   auc_train |   auc_test |   min_year_margin |   domain_shift |   score |
|:------------------|------------:|-----------:|------------------:|---------------:|--------:|
| snr_vs_floor_db   |       0.737 |      0.683 |             0.183 |          0.103 |   0.164 |
| spec_slope_db_khz |       0.256 |      0.324 |            -0.176 |          0.118 |   0.155 |
| mfcc_1_mean       |       0.72  |      0.709 |             0.209 |          0.312 |   0.144 |
| crest_factor      |       0.282 |      0.312 |            -0.148 |          0.156 |   0.125 |
| f0_acf            |       0.25  |      0.355 |            -0.145 |          0.204 |   0.116 |
| max_peak_prom_db  |       0.791 |      0.643 |             0.143 |          0.208 |   0.113 |
| env_kurtosis      |       0.315 |      0.307 |            -0.132 |          0.15  |   0.112 |
| rms_db            |       0.646 |      0.753 |             0.211 |          0.532 |   0.098 |
| mfcc_8_mean       |       0.309 |      0.342 |            -0.132 |          0.312 |   0.091 |
| zcr               |       0.796 |      0.616 |             0.116 |          0.286 |   0.083 |
| n_spec_peaks      |       0.762 |      0.586 |             0.086 |          0.117 |   0.076 |
| band_1250_1500    |       0.371 |      0.392 |            -0.108 |          0.306 |   0.075 |


### FC_vs_N

|                    |   auc_train |   auc_test |   min_year_margin |   domain_shift |   score |
|:-------------------|------------:|-----------:|------------------:|---------------:|--------:|
| env_cv             |       0.674 |      0.608 |             0.108 |          0.117 |   0.095 |
| spec_slope_db_khz  |       0.394 |      0.302 |            -0.103 |          0.118 |   0.091 |
| env_peak_to_median |       0.65  |      0.589 |             0.089 |          0.148 |   0.076 |
| spec_centroid      |       0.674 |      0.602 |             0.102 |          0.284 |   0.073 |
| spec_flux          |       0.281 |      0.395 |            -0.105 |          0.316 |   0.072 |
| snr_vs_floor_db    |       0.58  |      0.606 |             0.08  |          0.103 |   0.072 |
| band_200_300       |       0.559 |      0.629 |             0.08  |          0.108 |   0.072 |
| mf_peak_ratio      |       0.674 |      0.583 |             0.083 |          0.149 |   0.071 |
| kurtosis           |       0.604 |      0.565 |             0.065 |          0.121 |   0.058 |
| spec_crest         |       0.373 |      0.415 |            -0.068 |          0.163 |   0.057 |
| mfcc_4_mean        |       0.404 |      0.324 |            -0.069 |          0.201 |   0.055 |
| rel_high_db        |       0.649 |      0.562 |             0.062 |          0.17  |   0.051 |


### W_vs_crk

|                    |   auc_train |   auc_test |   min_year_margin |   domain_shift |   score |
|:-------------------|------------:|-----------:|------------------:|---------------:|--------:|
| env_kurtosis       |       0.145 |      0.256 |            -0.244 |          0.15  |   0.207 |
| crest_factor       |       0.198 |      0.277 |            -0.223 |          0.156 |   0.188 |
| kurtosis           |       0.24  |      0.294 |            -0.206 |          0.121 |   0.181 |
| env_peak_to_median |       0.23  |      0.332 |            -0.168 |          0.148 |   0.143 |
| env_cv             |       0.263 |      0.335 |            -0.16  |          0.117 |   0.141 |
| spec_flux          |       0.713 |      0.69  |             0.19  |          0.316 |   0.13  |
| mf_peak_ratio      |       0.236 |      0.364 |            -0.136 |          0.149 |   0.116 |
| spec_crest         |       0.71  |      0.635 |             0.135 |          0.163 |   0.113 |
| n_spec_peaks       |       0.74  |      0.619 |             0.093 |          0.117 |   0.082 |
| max_peak_prom_db   |       0.793 |      0.6   |             0.1   |          0.208 |   0.079 |
| snr_vs_floor_db    |       0.662 |      0.577 |             0.077 |          0.103 |   0.069 |
| mfcc_8_mean        |       0.379 |      0.404 |            -0.096 |          0.312 |   0.066 |


### CC_vs_FC

|                   |   auc_train |   auc_test |   min_year_margin |   domain_shift |   score |
|:------------------|------------:|-----------:|------------------:|---------------:|--------:|
| duration_s        |       0.537 |      0.262 |                -0 |          0.604 |       0 |
| spec_centroid     |       0.37  |      0.399 |                -0 |          0.284 |       0 |
| spec_bandwidth    |       0.339 |      0.427 |                -0 |          0.387 |       0 |
| spec_rolloff85    |       0.353 |      0.376 |                -0 |          0.317 |       0 |
| spec_rolloff95    |       0.364 |      0.372 |                -0 |          0.419 |       0 |
| spec_flatness     |       0.224 |      0.432 |                -0 |          0.388 |       0 |
| spec_entropy      |       0.3   |      0.278 |                -0 |          0.343 |       0 |
| spec_crest        |       0.72  |      0.797 |                 0 |          0.163 |       0 |
| dom_freq          |       0.43  |      0.391 |                -0 |          0.183 |       0 |
| spec_slope_db_khz |       0.265 |      0.394 |                -0 |          0.118 |       0 |
| band_50_100       |       0.442 |      0.431 |                -0 |          0.168 |       0 |
| band_100_200      |       0.652 |      0.669 |                 0 |          0.188 |       0 |


### WC_vs_FC

|                   |   auc_train |   auc_test |   min_year_margin |   domain_shift |   score |
|:------------------|------------:|-----------:|------------------:|---------------:|--------:|
| duration_s        |       0.598 |      0.545 |                 0 |          0.604 |       0 |
| spec_centroid     |       0.662 |      0.67  |                 0 |          0.284 |       0 |
| spec_bandwidth    |       0.463 |      0.502 |                -0 |          0.387 |       0 |
| spec_rolloff85    |       0.554 |      0.616 |                 0 |          0.317 |       0 |
| spec_rolloff95    |       0.526 |      0.591 |                 0 |          0.419 |       0 |
| spec_flatness     |       0.295 |      0.404 |                -0 |          0.388 |       0 |
| spec_entropy      |       0.474 |      0.488 |                -0 |          0.343 |       0 |
| spec_crest        |       0.57  |      0.542 |                 0 |          0.163 |       0 |
| dom_freq          |       0.789 |      0.665 |                 0 |          0.183 |       0 |
| spec_slope_db_khz |       0.249 |      0.353 |                -0 |          0.118 |       0 |
| band_50_100       |       0.271 |      0.336 |                -0 |          0.168 |       0 |
| band_100_200      |       0.355 |      0.374 |                -0 |          0.188 |       0 |


### WC_vs_W

|                   |   auc_train |   auc_test |   min_year_margin |   domain_shift |   score |
|:------------------|------------:|-----------:|------------------:|---------------:|--------:|
| duration_s        |       0.759 |      0.628 |                 0 |          0.604 |       0 |
| spec_centroid     |       0.525 |      0.647 |                 0 |          0.284 |       0 |
| spec_bandwidth    |       0.557 |      0.431 |                -0 |          0.387 |       0 |
| spec_rolloff85    |       0.524 |      0.617 |                 0 |          0.317 |       0 |
| spec_rolloff95    |       0.542 |      0.547 |                 0 |          0.419 |       0 |
| spec_flatness     |       0.533 |      0.378 |                -0 |          0.388 |       0 |
| spec_entropy      |       0.657 |      0.571 |                 0 |          0.343 |       0 |
| spec_crest        |       0.337 |      0.403 |                -0 |          0.163 |       0 |
| dom_freq          |       0.559 |      0.66  |                 0 |          0.183 |       0 |
| spec_slope_db_khz |       0.422 |      0.403 |                -0 |          0.118 |       0 |
| band_50_100       |       0.519 |      0.336 |                -0 |          0.168 |       0 |
| band_100_200      |       0.487 |      0.509 |                 0 |          0.188 |       0 |


### R_vs_N

|                   |   auc_train |   auc_test |   min_year_margin |   domain_shift |   score |
|:------------------|------------:|-----------:|------------------:|---------------:|--------:|
| duration_s        |       0.464 |        nan |                -0 |          0.604 |       0 |
| spec_centroid     |       0.273 |        nan |                -0 |          0.284 |       0 |
| spec_bandwidth    |       0.137 |        nan |                -0 |          0.387 |       0 |
| spec_rolloff85    |       0.197 |        nan |                -0 |          0.317 |       0 |
| spec_rolloff95    |       0.16  |        nan |                -0 |          0.419 |       0 |
| spec_flatness     |       0.149 |        nan |                -0 |          0.388 |       0 |
| spec_entropy      |       0.072 |        nan |                -0 |          0.343 |       0 |
| spec_crest        |       0.945 |        nan |                 0 |          0.163 |       0 |
| dom_freq          |       0.369 |        nan |                -0 |          0.183 |       0 |
| spec_slope_db_khz |       0.326 |        nan |                -0 |          0.118 |       0 |
| band_50_100       |       0.575 |        nan |                 0 |          0.168 |       0 |
| band_100_200      |       0.494 |        nan |                 0 |          0.188 |       0 |


### S_vs_N

|                   |   auc_train |   auc_test |   min_year_margin |   domain_shift |   score |
|:------------------|------------:|-----------:|------------------:|---------------:|--------:|
| duration_s        |       0.498 |        nan |                -0 |          0.604 |       0 |
| spec_centroid     |       0.596 |        nan |                 0 |          0.284 |       0 |
| spec_bandwidth    |       0.449 |        nan |                 0 |          0.387 |       0 |
| spec_rolloff85    |       0.523 |        nan |                 0 |          0.317 |       0 |
| spec_rolloff95    |       0.515 |        nan |                 0 |          0.419 |       0 |
| spec_flatness     |       0.187 |        nan |                -0 |          0.388 |       0 |
| spec_entropy      |       0.247 |        nan |                -0 |          0.343 |       0 |
| spec_crest        |       0.805 |        nan |                 0 |          0.163 |       0 |
| dom_freq          |       0.635 |        nan |                 0 |          0.183 |       0 |
| spec_slope_db_khz |       0.172 |        nan |                -0 |          0.118 |       0 |
| band_50_100       |       0.216 |        nan |                -0 |          0.168 |       0 |
| band_100_200      |       0.504 |        nan |                -0 |          0.188 |       0 |


## 4. Clasificación dejando un año fuera (RandomForest, 250 árboles)

Entrena con los otros 3 años y evalúa en el año retenido (2025 = split oficial). `year-stable top15` elige las 15 features con AUC consistente entre los años de **entrenamiento** (sin ver el año retenido). Umbral fijado con la prevalencia de train.

### gate1 N/abn — ROC AUC

| features             |   2022 |   2023 |   2024 |   2025 |   media |
|:---------------------|-------:|-------:|-------:|-------:|--------:|
| shape+tonal+temporal |  0.851 |  0.84  |  0.803 |  0.634 |   0.782 |
| all                  |  0.836 |  0.846 |  0.802 |  0.629 |   0.778 |
| temporal             |  0.814 |  0.853 |  0.804 |  0.628 |   0.775 |
| year-stable top15    |  0.744 |  0.766 |  0.78  |  0.582 |   0.718 |
| tonal                |  0.705 |  0.756 |  0.778 |  0.554 |   0.698 |
| shape                |  0.694 |  0.764 |  0.765 |  0.57  |   0.698 |
| level                |  0.674 |  0.732 |  0.746 |  0.599 |   0.688 |
| mfcc                 |  0.652 |  0.702 |  0.699 |  0.528 |   0.645 |


gate1 N/abn — macro-F1

| features             |   2022 |   2023 |   2024 |   2025 |   media |
|:---------------------|-------:|-------:|-------:|-------:|--------:|
| temporal             |  0.708 |  0.741 |  0.625 |  0.374 |   0.612 |
| shape+tonal+temporal |  0.742 |  0.741 |  0.561 |  0.397 |   0.61  |
| all                  |  0.724 |  0.754 |  0.563 |  0.391 |   0.608 |
| tonal                |  0.623 |  0.695 |  0.502 |  0.408 |   0.557 |
| year-stable top15    |  0.635 |  0.67  |  0.526 |  0.397 |   0.557 |
| shape                |  0.618 |  0.68  |  0.512 |  0.403 |   0.553 |
| level                |  0.567 |  0.673 |  0.5   |  0.438 |   0.545 |
| mfcc                 |  0.582 |  0.61  |  0.498 |  0.374 |   0.516 |


### gate2 W/crk — ROC AUC

| features             |   2022 |   2023 |   2024 |   2025 |   media |
|:---------------------|-------:|-------:|-------:|-------:|--------:|
| all                  |  0.937 |  0.902 |  0.907 |  0.812 |   0.889 |
| shape+tonal+temporal |  0.931 |  0.896 |  0.887 |  0.786 |   0.875 |
| year-stable top15    |  0.916 |  0.896 |  0.872 |  0.768 |   0.863 |
| temporal             |  0.913 |  0.846 |  0.835 |  0.76  |   0.838 |
| shape                |  0.843 |  0.835 |  0.837 |  0.761 |   0.819 |
| tonal                |  0.857 |  0.846 |  0.772 |  0.685 |   0.79  |
| mfcc                 |  0.84  |  0.779 |  0.762 |  0.757 |   0.785 |
| level                |  0.73  |  0.694 |  0.661 |  0.573 |   0.664 |


gate2 W/crk — macro-F1

| features             |   2022 |   2023 |   2024 |   2025 |   media |
|:---------------------|-------:|-------:|-------:|-------:|--------:|
| shape+tonal+temporal |  0.825 |  0.806 |  0.629 |  0.626 |   0.721 |
| all                  |  0.812 |  0.801 |  0.614 |  0.654 |   0.72  |
| year-stable top15    |  0.798 |  0.81  |  0.707 |  0.557 |   0.718 |
| tonal                |  0.786 |  0.771 |  0.68  |  0.569 |   0.701 |
| temporal             |  0.836 |  0.75  |  0.604 |  0.605 |   0.699 |
| shape                |  0.719 |  0.736 |  0.686 |  0.6   |   0.685 |
| mfcc                 |  0.734 |  0.73  |  0.543 |  0.59  |   0.649 |
| level                |  0.534 |  0.666 |  0.512 |  0.467 |   0.545 |


### gate2 6cls — macro-F1

| features             |   2022 |   2023 |   2024 |   2025 |   media |
|:---------------------|-------:|-------:|-------:|-------:|--------:|
| shape+tonal+temporal |  0.402 |  0.406 |  0.274 |  0.175 |   0.314 |
| year-stable top15    |  0.355 |  0.383 |  0.309 |  0.18  |   0.307 |
| all                  |  0.382 |  0.416 |  0.248 |  0.18  |   0.306 |
| shape                |  0.336 |  0.372 |  0.278 |  0.189 |   0.294 |
| tonal                |  0.368 |  0.326 |  0.257 |  0.175 |   0.281 |
| temporal             |  0.347 |  0.343 |  0.236 |  0.183 |   0.277 |
| level                |  0.235 |  0.33  |  0.234 |  0.183 |   0.246 |
| mfcc                 |  0.272 |  0.22  |  0.178 |  0.172 |   0.211 |


## 5. Features con separabilidad estable en los 4 años (|margen| > 0.05)

**abn_vs_N**: spec_slope_db_khz (-0.17), mfcc_4_mean (-0.14), snr_vs_floor_db (+0.13), mfcc_1_mean (+0.12), rms_db (+0.10), band_1500_2000 (-0.09), mfcc_8_mean (-0.09), band_1250_1500 (-0.08), spec_centroid (+0.08), zcr (+0.08), frame_peakiness_db (+0.08), env_cv (+0.08), temporal_flatness (-0.08), max_peak_prom_db (+0.08), rel_high_db (+0.07)

**W_vs_crk**: env_kurtosis (-0.24), crest_factor (-0.22), kurtosis (-0.21), spec_flux (+0.19), env_peak_to_median (-0.17), env_cv (-0.16), mf_peak_ratio (-0.14), spec_crest (+0.13), max_peak_prom_db (+0.10), mfcc_8_mean (-0.10), n_spec_peaks (+0.09), rms_db (+0.09), spec_entropy (-0.08), n_transients (-0.08), snr_vs_floor_db (+0.08)

**CC_vs_FC**: ninguna

**WC_vs_FC**: ninguna
