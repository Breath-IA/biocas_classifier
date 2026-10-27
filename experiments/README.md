# experiments/

Experimentos de clasificación de sonidos respiratorios organizados según los **dos gates de [plan.md](../plan.md)**:

| Gate (plan.md) | Pregunta | Carpeta |
|---|---|---|
| **Gate 1** | Normal contra anormal | [gate1_normal_vs_abnormal/](gate1_normal_vs_abnormal/) |
| **Gate 2** | Tipo de sonido anormal (crepitante / sibilancia), condicionado a gate 1 = anormal | [gate2_event_type/](gate2_event_type/) |

Los experimentos de gate 2 se evalúan **de forma aislada** (gate 1 "perfecto"): solo entran eventos W y FC, o N/W/FC
en el plano sin gate 1 (e01). La compuerta jerárquica completa (MoE, pasos 4–6 del plan) aún no está implementada.

## Estructura

```
experiments/
├── README.md
├── .gitignore                       # ignora cache/ y *.npz
├── common/                          # código y tablas compartidos
│   ├── paths.py                     # ÚNICA fuente de rutas
│   ├── analysis_utils.py            # auc, groups, md, YEARS, META
│   ├── extract_spectral_features.py # features por evento (todas las clases)
│   ├── physics_features.py          # features físicas de plan.md §1 (eventos W y FC)
│   ├── synthesis/                   # synthetic auscultation toolkit (generators, SPRSound channel, insertion, I/O)
│   ├── features/                    # segments_spectral.csv, physics_features.csv
│   └── cache/                       # versiones anteriores (ignorado por git)
├── gate1_normal_vs_abnormal/
│   └── e01_spectral_baseline/
└── gate2_event_type/
    ├── e01_flat_N_W_FC_augmentation/
    ├── e02_W_vs_FC_features/
    ├── e03_W_vs_FC_cnn_physics/
    ├── e04_synthetic_augmentation/     # synthetic N / FC / W in SPRSound format (no training)
    └── e05_cnn_synthetic/              # e03 CNN trained with the synthetic data (mixing vs pretraining)
```

Cada experimento `eNN_*/` tiene la misma forma:

```
eNN_*/
├── README.md     objetivo, comando, entradas/salidas, resultado clave
├── *.py          código del experimento
├── results/      report.md (conclusiones) · report_auto.md (tablas generadas) · figuras · CSV pequeños
└── cache/        derivados grandes y reproducibles (ignorado por git)
```

La numeración `eNN` indica el **orden en que se hicieron** y cada experimento puede depender de los anteriores
(se indica en su README). Los scripts se pueden ejecutar desde cualquier directorio.

## Orden de ejecución

Desde la raíz del repo (`.venv` creado con `requirements.txt`; `numba`/`librosa` no se usan):

```bash
bash get_data.sh                                                              # 0. datos → data/raw_data/{train,test}
.venv/Scripts/python experiments/common/extract_spectral_features.py          # 1. features por evento (~2,5 min)

# gate 1
.venv/Scripts/python experiments/gate1_normal_vs_abnormal/e01_spectral_baseline/analyze_separability.py

# gate 2
G2=experiments/gate2_event_type
.venv/Scripts/python $G2/e01_flat_N_W_FC_augmentation/augment.py              # copias aumentadas (~3 min por variante)
.venv/Scripts/python $G2/e01_flat_N_W_FC_augmentation/analyze.py
.venv/Scripts/python $G2/e02_W_vs_FC_features/analyze.py                       # usa las copias de e01
.venv/Scripts/python experiments/common/physics_features.py                   # features físicas (~1 min)
.venv/Scripts/python $G2/e03_W_vs_FC_cnn_physics/train_cnn.py                 # CNN, CPU (~40 min)
.venv/Scripts/python $G2/e03_W_vs_FC_cnn_physics/report.py

# synthetic augmentation (e04; see its README for details)
E4=$G2/e04_synthetic_augmentation
.venv/Scripts/python $E4/analyze_references.py && .venv/Scripts/python $E4/analyze_sprsound_events.py
.venv/Scripts/python $E4/calibrate_synthesis.py                                # ~25 min
.venv/Scripts/python $E4/build_synthetic.py --n-per-class 300 --overwrite    # writes data/synthetic/
.venv/Scripts/python $E4/qc_synthetic.py

# CNN with synthetic data (e05; see its README, one training condition per train_cnn_synth.py call)
E5=$G2/e05_cnn_synthetic
.venv/Scripts/python $E5/build_synth_mels.py --name v2 --calibration $E5/results/calibration_map_v2.json --n-per-class 2500 --seed 22 --match-hosts
.venv/Scripts/python $E5/train_cnn_synth.py --name v2_pre_b2000 --synth v2 --ratio 1.0 --mode pretrain
.venv/Scripts/python $E5/report.py
```

## Protocolo común

- Validación **dejando un año fuera** (2022, 2023, 2024, 2025); el año retenido nunca se aumenta ni se usa para elegir
  umbrales o hacer early stopping. Pacientes disjuntos entre años (≤ 1 compartido).
- Métricas: ROC AUC, balanced accuracy, macro-F1 y recall por clase, siempre por año (como pide plan.md).
- Varias semillas por configuración cuando el entrenamiento es estocástico.

## Correspondencia con el plan (protocolo de plan.md)

| Paso del protocolo | Dónde |
|---|---|
| 1. Baseline documentado | gate1/e01 (RF sobre features espectrales) · gate2/e02 (W vs FC) · gate2/e03 (CNN `cnn`) |
| 2. Solo features físicas | gate2/e03 (rama de features físicas, variante `cnn+phys`) |
| 3. Solo augmentación sintética | gate2/e04: parametric synthesis built and quality-checked; gate2/e05: used to train the CNN (helps only as pretraining, hurts when mixed in); gate2/e01 y e02 usan aumentación genérica |
| 4–6. Combinación, pérdida auxiliar, MoE completo | pendiente |

## Nombres anteriores → ubicación actual

Antes se llamaban "gate 2", "gate 3" y "gate 4" según el orden de trabajo, no según el plan. Todos eran experimentos
del **gate 2 del plan** (o del análisis inicial de ambos gates):

| Nombre anterior | Ahora |
|---|---|
| `eda/analyze_separability.py` y `eda/out/*` (análisis inicial) | `gate1_normal_vs_abnormal/e01_spectral_baseline/` |
| "gate 2" (`eda/gate2_*.py`, `eda/out/gate2/`) | `gate2_event_type/e01_flat_N_W_FC_augmentation/` |
| "gate 3" (`eda/gate3_analysis.py`, `eda/out/gate3/`) | `gate2_event_type/e02_W_vs_FC_features/` |
| "gate 4" (`eda/gate4_*.py`, `eda/out/gate4/`) | `gate2_event_type/e03_W_vs_FC_cnn_physics/` |
| `eda/extract_spectral_features.py`, `eda/physics_features.py` | `common/` |
| `eda/out/segments_spectral.csv` | `common/features/segments_spectral.csv` |
| `eda/out/segments_spectral_v1.csv` | `common/cache/segments_spectral_v1.csv` |

`eda/eda.ipynb` y `workflow/` no se tocaron.
