# Splits report

K=250 per class · max 8/patient/class · seed 42 (+0 retries for years) · val fraction 0.15 · 5 folds

Patients: test 77 · pool 866 (val 130)

## Test set (events, patients) per class and year

| class | 2022 | 2023 | 2024 | 2025 | patients |
|---|---|---|---|---|---|
| N | 106 | 57 | 50 | 37 | 64 |
| W | 100 | 56 | 50 | 44 | 59 |
| FC | 129 | 32 | 11 | 78 | 50 |

## Pool events per class and role

| class | train | val |
|---|---|---|
| N | 14377 | 2248 |
| W | 900 | 125 |
| FC | 2265 | 394 |
| other | 648 | 68 |

Dropped (events of test patients not in the test set): 2803
Assertions passed: test = exactly K per class, no patient shared between test / pool, train / val, or across the folds of a CV split.
