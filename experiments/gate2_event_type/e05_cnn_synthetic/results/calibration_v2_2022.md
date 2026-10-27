# Calibration v2 — 2022 (years 2022)

90 synthetic events per candidate; paired seeds; hosts matched to the real event's location/age.

Pooled e04 calibration applied to this group: D_FC = 0.278, D_W = 0.389

## Result

- crackle chain: high-pass 50 Hz, low-pass 300 Hz (order 6)
- crackle source tau 4.1 ms, rate scale 0.78, level offset -1.99 dB
- wheeze chain low-pass 500 Hz; burst gap 0.3 s, clicks 3.0/s, slow envelope depth 0.6
- distances: D_FC 0.278 → **0.208**, D_W 0.389 → **0.241** (pooled e04 calibration → refined)
- independent check (fitted, not a target): synthetic tau 4.6 ms / 2CD 6.8 ms vs real 6.8 / 8.1 ms (414 fits)

## Search log

| stage           |   value |     D |    gap |   clicks |   slow_env |
|:----------------|--------:|------:|-------:|---------:|-----------:|
| S1_lp           |   180   | 0.492 | nan    |      nan |      nan   |
| S1_lp           |   225   | 0.352 | nan    |      nan |      nan   |
| S1_lp           |   250   | 0.278 | nan    |      nan |      nan   |
| S1_lp           |   300   | 0.213 | nan    |      nan |      nan   |
| S1_lp           |   350   | 0.409 | nan    |      nan |      nan   |
| S1_hp           |    50   | 0.211 | nan    |      nan |      nan   |
| S1_hp           |    70   | 0.213 | nan    |      nan |      nan   |
| S1_hp           |   100   | 0.228 | nan    |      nan |      nan   |
| S2_tau          |     4.1 | 0.211 | nan    |      nan |      nan   |
| S2_tau          |     5   | 0.216 | nan    |      nan |      nan   |
| S2_tau          |     5.7 | 0.222 | nan    |      nan |      nan   |
| S2_tau          |     7   | 0.229 | nan    |      nan |      nan   |
| S4_wheeze_lp    |   400   | 0.384 | nan    |      nan |      nan   |
| S4_wheeze_lp    |   500   | 0.375 | nan    |      nan |      nan   |
| S4_wheeze_lp    |   650   | 0.415 | nan    |      nan |      nan   |
| S4_wheeze_lp    |   800   | 0.428 | nan    |      nan |      nan   |
| S5_wheeze_irreg |   nan   | 0.375 |   0.15 |        0 |        0   |
| S5_wheeze_irreg |   nan   | 0.37  |   0.15 |        0 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.331 |   0.15 |        0 |        0.6 |
| S5_wheeze_irreg |   nan   | 0.366 |   0.15 |        3 |        0   |
| S5_wheeze_irreg |   nan   | 0.326 |   0.15 |        3 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.316 |   0.15 |        3 |        0.6 |
| S5_wheeze_irreg |   nan   | 0.361 |   0.15 |        8 |        0   |
| S5_wheeze_irreg |   nan   | 0.333 |   0.15 |        8 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.308 |   0.15 |        8 |        0.6 |
| S5_wheeze_irreg |   nan   | 0.284 |   0.3  |        0 |        0   |
| S5_wheeze_irreg |   nan   | 0.286 |   0.3  |        0 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.277 |   0.3  |        0 |        0.6 |
| S5_wheeze_irreg |   nan   | 0.277 |   0.3  |        3 |        0   |
| S5_wheeze_irreg |   nan   | 0.266 |   0.3  |        3 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.241 |   0.3  |        3 |        0.6 |
| S5_wheeze_irreg |   nan   | 0.306 |   0.3  |        8 |        0   |
| S5_wheeze_irreg |   nan   | 0.281 |   0.3  |        8 |        0.3 |
| S5_wheeze_irreg |   nan   | 0.254 |   0.3  |        8 |        0.6 |