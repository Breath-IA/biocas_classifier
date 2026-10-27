# Synthetic set QC — automatic report

## 1. Composition

| kind   |   2022 |   2023 |   2024 |   2025 |
|:-------|-------:|-------:|-------:|-------:|
| FC     |    126 |     30 |     97 |     47 |
| N      |    115 |     30 |    106 |     49 |
| W      |    134 |     28 |     97 |     41 |


Host recordings used: 812 (max reuse 2). All files are train-only (`train_only` = True); synthetic recordings whose host is a 2025 test recording: 137 (drop them for the official split).

## 2. Integrity checks

|                                                        |   value |
|:-------------------------------------------------------|--------:|
| recordings                                             |     900 |
| format_errors                                          |       0 |
| schema_errors                                          |       0 |
| events_outside                                         |       0 |
| recordings_with_overlapping_events                     |      17 |
| of_which_already_in_the_real_host                      |      17 |
| clipping_introduced_by_synthesis(sampled)              |       0 |
| host_samples_identical_outside_event(min over sampled) |       1 |
| host_samples_identical_outside_event(mean)             |       1 |


## 3. Realism: |2·AUC−1| between real and synthetic events (0 = indistinguishable, 1 = fully separable)

| class   |   n_features |   median_gap |   share_gap_below_0_2 |   share_gap_above_0_5 |
|:--------|-------------:|-------------:|----------------------:|----------------------:|
| FC      |           97 |        0.152 |                 0.608 |                  0.01 |
| N       |           69 |        0.04  |                 0.942 |                  0    |
| W       |           97 |        0.208 |                 0.495 |                  0.01 |


### Largest gaps — N

| feature           |   gap |   median_real |   median_synth |    ks |
|:------------------|------:|--------------:|---------------:|------:|
| band_1250_1500    | 0.331 |       -66.751 |        -61.353 | 0.232 |
| band_1500_2000    | 0.331 |       -64.927 |        -59.667 | 0.23  |
| band_1000_1250    | 0.318 |       -65.492 |        -60.419 | 0.23  |
| band_800_1000     | 0.238 |       -63.121 |        -57.506 | 0.18  |
| spec_slope_db_khz | 0.131 |       -30.307 |        -29.43  | 0.1   |
| mf_rate           | 0.116 |        18.185 |         16.902 | 0.101 |
| spec_flatness     | 0.102 |         0     |          0     | 0.094 |
| lf_hf_db          | 0.093 |        47.976 |         45.614 | 0.108 |
| snr_vs_floor_db   | 0.087 |        10.313 |          9.583 | 0.084 |
| duration_s        | 0.075 |         1.743 |          1.838 | 0.072 |


Key features — N:

| feature           |   gap |   median_real |   median_synth |
|:------------------|------:|--------------:|---------------:|
| spec_centroid     | 0.068 |       165.779 |        162.585 |
| spec_slope_db_khz | 0.131 |       -30.307 |        -29.43  |
| band_100_200      | 0.003 |        -1.913 |         -1.916 |
| band_300_400      | 0.046 |       -15.62  |        -15.966 |
| spec_flux         | 0.045 |         0.114 |          0.114 |
| crest_factor      | 0.004 |         8.394 |          8.397 |
| env_kurtosis      | 0.033 |         9.482 |         10.108 |
| snr_vs_floor_db   | 0.087 |        10.313 |          9.583 |


### Largest gaps — FC

| feature               |   gap |   median_real |   median_synth |    ks |
|:----------------------|------:|--------------:|---------------:|------:|
| mfcc_10_mean          | 0.574 |        -1.89  |         -0.476 | 0.472 |
| ph_tau_p75            | 0.492 |        30.89  |         10.142 | 0.383 |
| max_peak_prom_db      | 0.483 |        12.979 |          9.14  | 0.376 |
| ph_env_slope_neg_frac | 0.466 |         0.155 |          0.194 | 0.35  |
| ph_tau_med            | 0.465 |        11.432 |          5.757 | 0.421 |
| ph_persist_cos_mean   | 0.459 |         0.868 |          0.89  | 0.386 |
| mfcc_11_mean          | 0.458 |        -1.732 |         -0.795 | 0.382 |
| ph_decay_frac         | 0.451 |         0.667 |          0.875 | 0.366 |
| mfcc_7_mean           | 0.415 |        -0.869 |         -2.615 | 0.307 |
| ph_if_std             | 0.414 |        34.665 |         28.482 | 0.312 |


Key features — FC:

| feature         |   gap |   median_real |   median_synth |
|:----------------|------:|--------------:|---------------:|
| spec_centroid   | 0.196 |       176.642 |        169.546 |
| band_300_400    | 0.167 |       -14.578 |        -15.792 |
| crest_factor    | 0.003 |         9.839 |          9.742 |
| n_transients    | 0.093 |        17     |         16     |
| env_kurtosis    | 0.045 |        24.556 |         21.668 |
| snr_vs_floor_db | 0.011 |        11.924 |         11.869 |
| ph_tau_p25      | 0.343 |         6.046 |          4.086 |
| ph_mf_cr_max    | 0.119 |        56.023 |         49.745 |


### Largest gaps — W

| feature             |   gap |   median_real |   median_synth |    ks |
|:--------------------|------:|--------------:|---------------:|------:|
| ph_persist_cos_mean | 0.566 |         0.876 |          0.905 | 0.472 |
| tonal_run_s         | 0.478 |         0.096 |          0.208 | 0.399 |
| env_cv              | 0.422 |         1.107 |          0.871 | 0.331 |
| ph_if_stable_frac   | 0.413 |         0.835 |          0.931 | 0.344 |
| ph_if_std           | 0.41  |        38.064 |         27.161 | 0.329 |
| env_peak_to_median  | 0.398 |        12.626 |          6.564 | 0.32  |
| kurtosis            | 0.395 |         5.609 |          2.107 | 0.328 |
| transient_rate      | 0.394 |         9.148 |          3.026 | 0.333 |
| ph_wh_peak_db       | 0.39  |        17.314 |         23.275 | 0.298 |
| ph_persist_cos_p10  | 0.383 |         0.781 |          0.814 | 0.296 |


Key features — W:

| feature          |   gap |   median_real |   median_synth |
|:-----------------|------:|--------------:|---------------:|
| band_200_300     | 0.102 |        -6.52  |         -6.949 |
| n_spec_peaks     | 0.155 |         2     |          2     |
| max_peak_prom_db | 0.036 |        18.706 |         19.056 |
| tonal_run_s      | 0.478 |         0.096 |          0.208 |
| crest_factor     | 0.275 |         6.12  |          4.895 |
| env_kurtosis     | 0.27  |         4.547 |          1.413 |
| snr_vs_floor_db  | 0.071 |        14.931 |         14.082 |
| ph_mf_wh_max     | 0.265 |         0.753 |          0.825 |


### Gap of the key features by year

| class   |   year |   n_real |   n_synth |   median_gap_key_features |
|:--------|-------:|---------:|----------:|--------------------------:|
| FC      |   2022 |     1167 |       126 |                     0.201 |
| FC      |   2023 |      309 |        30 |                     0.222 |
| FC      |   2024 |      169 |        97 |                     0.058 |
| FC      |   2025 |     1885 |        47 |                     0.05  |
| W       |   2022 |      865 |       134 |                     0.174 |
| W       |   2023 |      194 |        28 |                     0.343 |
| W       |   2024 |      141 |        97 |                     0.25  |
| W       |   2025 |      305 |        41 |                     0.184 |


## 4. Placement and level

- achieved vs target level (snr_vs_floor_db), median absolute error: 0.00 dB
- events placed inside a detected breath burst: 94.0%; relative position in the burst (median): FC 0.13, W 0.33
- event duration (s), median synthetic vs real: FC 1.36 vs 1.58, W 1.03 vs 0.78