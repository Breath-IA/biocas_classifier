# Real-vs-synthetic probe

400 synthetic events per class and set (same seed); real events: up to 2000 per class, random. Out-of-fold AUC, 5 folds grouped by patient (real) / host patient (synthetic). 0.5 = indistinguishable.

## Probe AUC and univariate gap

| set                | class   |   probe_auc_RF |   probe_auc_LR |   median_univariate_gap |   n_synth |   n_real |   n_features |
|:-------------------|:--------|---------------:|---------------:|------------------------:|----------:|---------:|-------------:|
| v1 (e04 generator) | FC      |          0.935 |          0.929 |                   0.146 |       400 |     2000 |           97 |
| v1 (e04 generator) | W       |          0.904 |          0.936 |                   0.204 |       400 |     1505 |           97 |
| v2 (improved)      | FC      |          0.916 |          0.921 |                   0.131 |       400 |     2000 |           97 |
| v2 (improved)      | W       |          0.835 |          0.858 |                   0.181 |       400 |     1505 |           97 |

## Features that betray the synthetic events (RF importance)

- **v1 (e04 generator) · FC**: mfcc_10_mean (0.05), band_600_800 (0.04), lf_hf_db (0.04), band_800_1000 (0.03), itv_cv (0.03), ph_if_std (0.03), if_std (0.03), mfcc_2_mean (0.03)
- **v1 (e04 generator) · W**: ph_persist_cos_mean (0.07), ph_wh_peak_db (0.04), tonal_run_s (0.03), mfcc_3_mean (0.02), rel_mid_db (0.02), ph_if_std (0.02), ph_if_stable_frac (0.02), temporal_flatness (0.02)
- **v2 (improved) · FC**: mfcc_10_mean (0.08), ph_persist_cos_mean (0.05), mfcc_9_mean (0.04), ph_tau_med (0.03), band_600_800 (0.03), mfcc_5_mean (0.03), ph_tau_p25 (0.03), itv_cv (0.03)
- **v2 (improved) · W**: ph_wh_peak_db (0.04), ph_persist_cos_mean (0.03), spec_crest (0.03), acf_peak (0.03), rel_mid_db (0.02), spec_entropy (0.02), ph_if_stable_frac (0.02), mfcc_3_mean (0.02)