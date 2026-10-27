# Calibration of the synthesis

90 synthetic recordings per candidate; same hosts, profiles and seeds for every candidate (paired comparison). Distance D = mean |median(synthetic) − median(real)| / IQR(real).

## Result

- chain: high-pass 70 Hz (order 2), **low-pass 250 Hz (order 6)**
- wheeze effective chain: low-pass 500 Hz (order 6)
- crackle rate scale 0.99, level offset -1.70 dB (clipped to ±3 dB)
- wheeze harmonic amplitudes (1f0, 2f0, 3f0): [1.0, 0.2, 0.04]
- wheeze (gliding-burst model): parallel-tone probability 0.4, burst median 0.25 s, glide sd 0.25
- final distance: **D_FC = 0.20**, D_W = 0.30 (0 = identical medians; values below ≈ 0.3 mean the medians fall within a third of the real IQR)

## Independent check (not used as a target)

Fitted crackle decay on synthetic fine-crackle events: tau 5.8 ms, 2CD 7.9 ms (419 fits) vs. real SPRSound fine crackles: tau 6.8 ms, 2CD 8.1 ms. The source tau (≈ 4 ms) and 2CD (≈ 3 ms) of the reference clip were stretched by the calibrated low-pass; if the chain is right they land near the real values.

## Search tables

### Stage 1: low-pass (hp = 50 Hz)

|   lp_hz |   2.0 |   4.0 |   6.0 |
|--------:|------:|------:|------:|
|     120 | 0.548 | 0.288 | 0.35  |
|     150 | 0.663 | 0.289 | 0.298 |
|     175 | 0.747 | 0.3   | 0.296 |
|     200 | 0.806 | 0.306 | 0.288 |
|     225 | 0.857 | 0.38  | 0.257 |
|     250 | 0.903 | 0.492 | 0.257 |
|     300 | 0.984 | 0.694 | 0.497 |
|     400 | 1.071 | 0.948 | 0.826 |

### Stage 2: high-pass

|   hp_hz |   D_FC |
|--------:|-------:|
|      20 |  0.269 |
|      35 |  0.26  |
|      50 |  0.257 |
|      70 |  0.252 |
|     100 |  0.278 |
|     140 |  0.293 |
|      70 |  0.252 |
|      70 |  0.252 |
|      70 |  0.271 |
|      70 |  0.253 |

### Stage 4a: wheeze low-pass (D_W)

|   lp_hz |     2 |     4 |     6 |
|--------:|------:|------:|------:|
|     250 | 0.412 | 0.462 | 0.488 |
|     300 | 0.406 | 0.409 | 0.458 |
|     400 | 0.391 | 0.376 | 0.375 |
|     500 | 0.413 | 0.378 | 0.365 |
|     650 | 0.418 | 0.41  | 0.388 |
|     800 | 0.421 | 0.418 | 0.418 |

### Medians after calibration (synthetic / real)

|                       |   FC synth |   FC real |   FC |Δ|/IQR |
|:----------------------|-----------:|----------:|-------------:|
| spec_centroid         |    169.541 |   176.642 |        0.182 |
| spec_rolloff85        |    234.375 |   242.188 |        0.143 |
| spec_rolloff95        |    281.25  |   289.062 |        0.091 |
| spec_slope_db_khz     |    -32.123 |   -32.507 |        0.079 |
| band_50_100           |     -9.605 |   -11.735 |        0.414 |
| band_100_200          |     -2.169 |    -2.218 |        0.033 |
| band_200_300          |     -5.975 |    -6.381 |        0.125 |
| band_300_400          |    -15.745 |   -14.578 |        0.167 |
| band_400_600          |    -25.178 |   -27.077 |        0.119 |
| band_600_800          |    -42.021 |   -51.908 |        0.425 |
| spec_crest            |     18.845 |    20.091 |        0.15  |
| env_kurtosis          |     23.675 |    24.556 |        0.023 |
| crest_factor          |     10.175 |     9.839 |        0.062 |
| kurtosis              |     17.802 |    17.369 |        0.015 |
| env_cv                |      1.408 |     1.423 |        0.025 |
| env_peak_to_median    |     27.16  |    28.07  |        0.028 |
| n_transients          |     16     |    17     |        0.062 |
| transient_rate        |     12.376 |    11.854 |        0.083 |
| spec_flux             |      0.105 |     0.111 |        0.517 |
| n_spec_peaks          |      1     |     1     |        0     |
| mf_peak_ratio         |     54.104 |    55.758 |        0.026 |
| mf_rate               |     18.981 |    19.483 |        0.063 |
| snr_vs_floor_db       |     12.458 |    11.924 |        0.089 |
| rel_tilt_db           |      3.151 |     1.456 |        0.25  |
| ph_tau_p25            |      4.22  |     6.046 |        0.398 |
| ph_tau_med            |      6.662 |    11.432 |        0.453 |
| ph_decay_frac         |      0.81  |     0.667 |        0.491 |
| ph_env_slope_neg_frac |      0.189 |     0.155 |        0.665 |
| ph_mf_cr_max          |     54.1   |    56.023 |        0.031 |
| ph_mf_cr_tau          |      2     |     4     |        0.5   |
| ph_mf_cr_top5         |     29.081 |    33.24  |        0.129 |
| rel_low_db            |      5.93  |     4.409 |        0.343 |
| rel_mid_db            |      9.164 |     8.534 |        0.097 |
| rel_high_db           |      8.775 |     5.993 |        0.377 |
| level_vs_rec_db       |      5.994 |     4.562 |        0.317 |

### Wheeze (searched only on W-specific parameters)

|                       |   W synth |   W real |   W |Δ|/IQR |
|:----------------------|----------:|---------:|------------:|
| spec_centroid         |   171.54  |  191.854 |       0.293 |
| spec_rolloff85        |   226.562 |  257.812 |       0.286 |
| spec_slope_db_khz     |   -31.903 |  -34.788 |       0.402 |
| band_100_200          |    -1.739 |   -2.887 |       0.328 |
| band_200_300          |    -7.519 |   -6.52  |       0.203 |
| band_300_400          |   -15.511 |  -12.89  |       0.295 |
| band_400_600          |   -23.388 |  -19.904 |       0.243 |
| spec_crest            |    30.474 |   24.942 |       0.318 |
| env_kurtosis          |     1.886 |    4.547 |       0.262 |
| crest_factor          |     4.815 |    6.12  |       0.366 |
| env_cv                |     0.9   |    1.107 |       0.413 |
| n_spec_peaks          |     2     |    2     |       0     |
| max_peak_prom_db      |    18.156 |   18.706 |       0.05  |
| tonal_run_s           |     0.176 |    0.096 |       1     |
| harm_ratio_db         |   -14.254 |  -12.983 |       0.087 |
| snr_vs_floor_db       |    15.115 |   14.931 |       0.02  |
| ph_mf_wh_max          |     0.829 |    0.753 |       0.392 |
| ph_mf_wh_frac50       |     0.235 |    0.19  |       0.325 |
| ph_mf_wh_f0_track_std |    23.38  |   27.427 |       0.121 |
| ph_if_std_best100     |    13.815 |   20.188 |       0.455 |
| rel_low_db            |     7.77  |    5.722 |       0.274 |
| rel_mid_db            |    12.55  |    8.61  |       0.431 |
| rel_high_db           |    10.602 |    5.871 |       0.518 |
| rel_tilt_db           |     1.753 |    0.205 |       0.188 |
| level_vs_rec_db       |     7.951 |    6.031 |       0.249 |