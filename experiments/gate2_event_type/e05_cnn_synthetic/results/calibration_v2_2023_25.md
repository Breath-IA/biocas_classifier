# Calibration v2 — 2023_25 (years 2023, 2024, 2025)

90 synthetic events per candidate; paired seeds; hosts matched to the real event's location/age.

Pooled e04 calibration applied to this group: D_FC = 0.265, D_W = 0.303

## Result

- crackle chain: high-pass 70 Hz, low-pass 250 Hz (order 6)
- crackle source tau 4.1 ms, rate scale 0.92, level offset -1.55 dB
- wheeze chain low-pass 400 Hz; burst gap 0.3 s, clicks 0.0/s, slow envelope depth 0.6
- distances: D_FC 0.265 → **0.273**, D_W 0.303 → **0.217** (pooled e04 calibration → refined)
- independent check (fitted, not a target): synthetic tau 5.9 ms / 2CD 8.2 ms vs real 6.8 / 8.1 ms (428 fits)

## Search log

| stage           |   value |     D |    gap |   clicks |   slow_env |
|:----------------|--------:|------:|-------:|---------:|-----------:|
| S1_lp           |   180   | 0.271 | nan    |      nan |      nan   |
| S1_lp           |   225   | 0.279 | nan    |      nan |      nan   |
| S1_lp           |   250   | 0.265 | nan    |      nan |      nan   |
| S1_lp           |   300   | 0.512 | nan    |      nan |      nan   |
| S1_lp           |   350   | 0.761 | nan    |      nan |      nan   |
| S1_hp           |    50   | 0.272 | nan    |      nan |      nan   |
| S1_hp           |    70   | 0.265 | nan    |      nan |      nan   |
| S1_hp           |   100   | 0.272 | nan    |      nan |      nan   |
| S2_tau          |     4.1 | 0.265 | nan    |      nan |      nan   |
| S2_tau          |     5   | 0.276 | nan    |      nan |      nan   |
| S2_tau          |     5.7 | 0.284 | nan    |      nan |      nan   |
| S2_tau          |     7   | 0.296 | nan    |      nan |      nan   |
| S4_wheeze_lp    |   400   | 0.272 | nan    |      nan |      nan   |
| S4_wheeze_lp    |   500   | 0.313 | nan    |      nan |      nan   |
| S4_wheeze_lp    |   650   | 0.415 | nan    |      nan |      nan   |
| S4_wheeze_lp    |   800   | 0.472 | nan    |      nan |      nan   |
| S5_wheeze_irreg |   nan   | 0.272 |   0.15 |        0 |        0   |
| S5_wheeze_irreg |   nan   | 0.271 |   0.15 |        0 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.249 |   0.15 |        0 |        0.6 |
| S5_wheeze_irreg |   nan   | 0.272 |   0.15 |        3 |        0   |
| S5_wheeze_irreg |   nan   | 0.259 |   0.15 |        3 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.243 |   0.15 |        3 |        0.6 |
| S5_wheeze_irreg |   nan   | 0.221 |   0.15 |        8 |        0   |
| S5_wheeze_irreg |   nan   | 0.267 |   0.15 |        8 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.244 |   0.15 |        8 |        0.6 |
| S5_wheeze_irreg |   nan   | 0.267 |   0.3  |        0 |        0   |
| S5_wheeze_irreg |   nan   | 0.249 |   0.3  |        0 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.217 |   0.3  |        0 |        0.6 |
| S5_wheeze_irreg |   nan   | 0.241 |   0.3  |        3 |        0   |
| S5_wheeze_irreg |   nan   | 0.249 |   0.3  |        3 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.225 |   0.3  |        3 |        0.6 |
| S5_wheeze_irreg |   nan   | 0.263 |   0.3  |        8 |        0   |
| S5_wheeze_irreg |   nan   | 0.249 |   0.3  |        8 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.219 |   0.3  |        8 |        0.6 |