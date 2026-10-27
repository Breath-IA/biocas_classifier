# Confusion matrix of the best e05 model (`v2_pre_b2000`, seed ensemble, threshold 0.5)

Rows = true class, columns = predicted class (W = wheeze, FC = fine crackle). Each year was held out from training.

## Official split (2022–2024 train, **2025 test**)

| true \ predicted | W | FC |
|---|---:|---:|
| W | 233 | 72 |
| FC | 278 | 1607 |

## Precision, recall and F1 — official split (2025 test)

|              |   precision |   recall |    f1 |   support |
|:-------------|------------:|---------:|------:|----------:|
| W            |       0.456 |    0.764 | 0.571 |       305 |
| FC           |       0.957 |    0.853 | 0.902 |      1885 |
| accuracy     |     nan     |  nan     | 0.84  |      2190 |
| macro avg    |       0.707 |    0.808 | 0.736 |      2190 |
| weighted avg |       0.887 |    0.84  | 0.856 |      2190 |

`macro avg` = unweighted mean of the two classes; `weighted avg` = mean weighted by class support (2025 is 86 % FC); `accuracy` is the share of correct predictions.

## Metrics per held-out year

| held_out_year      |    n |   auc |   balanced_accuracy |   recall_W |   recall_FC |   precision_W |   precision_FC |
|:-------------------|-----:|------:|--------------------:|-----------:|------------:|--------------:|---------------:|
| 2022               | 2032 | 0.948 |               0.873 |      0.901 |       0.845 |         0.811 |          0.92  |
| 2023               |  503 | 0.945 |               0.872 |      0.866 |       0.877 |         0.816 |          0.912 |
| 2024               |  310 | 0.965 |               0.911 |      0.887 |       0.935 |         0.919 |          0.908 |
| 2025               | 2190 | 0.889 |               0.808 |      0.764 |       0.853 |         0.456 |          0.957 |
| all years (pooled) | 5035 | 0.932 |               0.862 |      0.867 |       0.856 |         0.72  |          0.938 |

## Precision, recall and F1 for every held-out year

### 2022

|              |   precision |   recall |    f1 |   support |
|:-------------|------------:|---------:|------:|----------:|
| W            |       0.811 |    0.901 | 0.854 |       865 |
| FC           |       0.92  |    0.845 | 0.881 |      1167 |
| accuracy     |     nan     |  nan     | 0.869 |      2032 |
| macro avg    |       0.866 |    0.873 | 0.867 |      2032 |
| weighted avg |       0.874 |    0.869 | 0.869 |      2032 |

### 2023

|              |   precision |   recall |    f1 |   support |
|:-------------|------------:|---------:|------:|----------:|
| W            |       0.816 |    0.866 | 0.84  |       194 |
| FC           |       0.912 |    0.877 | 0.894 |       309 |
| accuracy     |     nan     |  nan     | 0.873 |       503 |
| macro avg    |       0.864 |    0.872 | 0.867 |       503 |
| weighted avg |       0.875 |    0.873 | 0.873 |       503 |

### 2024

|              |   precision |   recall |    f1 |   support |
|:-------------|------------:|---------:|------:|----------:|
| W            |       0.919 |    0.887 | 0.903 |       141 |
| FC           |       0.908 |    0.935 | 0.921 |       169 |
| accuracy     |     nan     |  nan     | 0.913 |       310 |
| macro avg    |       0.914 |    0.911 | 0.912 |       310 |
| weighted avg |       0.913 |    0.913 | 0.913 |       310 |

### all years (pooled)

|              |   precision |   recall |    f1 |   support |
|:-------------|------------:|---------:|------:|----------:|
| W            |       0.72  |    0.867 | 0.787 |      1505 |
| FC           |       0.938 |    0.856 | 0.895 |      3530 |
| accuracy     |     nan     |  nan     | 0.859 |      5035 |
| macro avg    |       0.829 |    0.862 | 0.841 |      5035 |
| weighted avg |       0.873 |    0.859 | 0.863 |      5035 |

## 2025 per seed (single networks, before ensembling)

|   seed |   TN(W→W) |   FP(W→FC) |   FN(FC→W) |   TP(FC→FC) |   auc |   balanced_accuracy |
|-------:|----------:|-----------:|-----------:|------------:|------:|--------------------:|
|      0 |       216 |         89 |        235 |        1650 | 0.872 |               0.792 |
|      1 |       266 |         39 |        525 |        1360 | 0.894 |               0.797 |
|      2 |       196 |        109 |        160 |        1725 | 0.869 |               0.779 |