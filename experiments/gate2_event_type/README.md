# Gate 2 — Tipo de sonido anormal

Segunda compuerta de [plan.md](../../plan.md): dado un evento anormal, ¿es crepitante o sibilancia? Aquí se evalúa
**aislada** (gate 1 "perfecto"): solo eventos W (sibilancia) y FC (crepitante fino), salvo e01 que incluye N para
medir qué pasa si se prescinde del gate 1.

| Orden | Experimento | Pregunta | Clases | Modelo |
|---|---|---|---|---|
| e01 | [e01_flat_N_W_FC_augmentation/](e01_flat_N_W_FC_augmentation/) | ¿Se puede clasificar N/W/FC directamente y ayuda aumentar los datos? | N, W, FC | RandomForest sobre features |
| e02 | [e02_W_vs_FC_features/](e02_W_vs_FC_features/) | ¿Cuánto separan W de FC las features y qué modelo generaliza mejor entre años? | W, FC | RF y regresión logística |
| e03 | [e03_W_vs_FC_cnn_physics/](e03_W_vs_FC_cnn_physics/) | ¿Una CNN sobre log-mel mejora a las features, y ayudan las features físicas de plan.md? | W, FC | CNN de 4 bloques, con y sin rama física |
| e04 | [e04_synthetic_augmentation/](e04_synthetic_augmentation/) | Can plan.md's parametric synthesis produce realistic N / FC / W recordings in the SPRSound format? | N, FC, W | none (no training) |
| e05 | [e05_cnn_synthetic/](e05_cnn_synthetic/) | Does the synthetic data help the CNN, and how should it be used? | W, FC | e03 CNN trained with synthetic data (mixed / pretrained) |

Cada experimento reutiliza resultados del anterior (e02 usa las copias aumentadas de e01; e03 compara contra e02).

## Resultados clave (detalle en cada `results/report.md`)

- **e01:** el plano N/W/FC tiene balanced accuracy 0,65–0,69 en 2022 y 0,46–0,50 en 2025; el par débil es N–FC (AUC 0,60 en 2025).
  La aumentación no mejora 2025; el mayor aporte viene de submuestrear N.
- **e02:** W contra FC tiene AUC medio ≈ 0,90 (0,81 en 2025). Separan las features de impulsividad temporal
  (`env_kurtosis`, `crest_factor`, `kurtosis`, `env_cv`); la regresión logística iguala al RF y calibra mejor entre años.
- **e04:** synthetic augmentation (no training): crackles and wheezes inserted into real SPRSound normal recordings through the dataset's own channel (effective low-pass ≈ 250 Hz for crackles, ≈ 500 Hz for wheezes); median real-vs-synthetic feature gap N 0.04, FC 0.15, W 0.21; host audio bit-identical outside the inserted event. See its `results/report.md` for steps and suggestions.
- **e05:** synthetic events mixed into the training batches hurt the CNN (1:1: AUC 0.923 → 0.905 / 0.911); pretraining on them and fine-tuning on real data helps (AUC 0.931, 2025: 0.878 vs 0.863; balanced accuracy 0.854 vs 0.795 with a 7× smaller seed spread). See its `results/report.md`.
- **e03:** una CNN sobre log-mel supera a las features (AUC medio 0,92 contra 0,90; 0,86 contra 0,81 en 2025). Las features físicas
  de plan.md separan W de FC con sentido físico (τ de decaimiento) pero la fusión aporta solo ≈ +0,005 de AUC. El umbral 0,5 de la CNN es inestable.
