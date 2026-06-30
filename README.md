# BioCAS Classifier 

## Buenas practicas
* Actualizar el requirements.txt cuando se agreguen nuevas librerías
* Colocar nombres de archivos y variables descriptivos
* Realizar commits antes de cualquier cambio que pueda impactar la lógica del codigo
* Poner comentarios descriptivos para cada commit (Español o Ingles)
  
## TODO
- [ ] Loop de entrenamiento
- [ ] EDA
- [ ] Hiperparametros del paper de la WEEF
- [ ] Integración con WandB
- [ ] Entrenamiento con Optuna 
- [ ] Preprocesamientos faltantes en el paper de la WEEF
  

# Documentación Técnica

Pipeline de preprocesamiento y clasificación de sonidos respiratorios sobre el
dataset **SPRSound / BioCAS** (2022–2025). Convierte grabaciones WAV anotadas
en JSON a espectrogramas log-mel balanceados por clase, listos para entrenar
una CNN en PyTorch.

> Generado a partir de una revisión completa del código fuente. Donde el
> comportamiento real difiere de lo documentado en docstrings, o hay código
> con bugs conocidos, se marca explícitamente con ⚠.

---

## Índice

- [BioCAS Classifier](#biocas-classifier)
  - [Buenas practicas](#buenas-practicas)
  - [TODO](#todo)
- [Documentación Técnica](#documentación-técnica)
  - [Índice](#índice)
  - [1. Arquitectura general](#1-arquitectura-general)
  - [2. Estructura del repositorio](#2-estructura-del-repositorio)
  - [3. Obtención de datos](#3-obtención-de-datos)
    - [3.1 `get_data.sh`](#31-get_datash)
    - [3.2 `data_config.py`](#32-data_configpy)
  - [4. Configuración](#4-configuración)
  - [5. La Pipeline](#5-la-pipeline)
    - [Fase 1 — metadata liviana (`_phase1`)](#fase-1--metadata-liviana-_phase1)
    - [Fase 2 — procesamiento en chunks con GPU (`_phase2`)](#fase-2--procesamiento-en-chunks-con-gpu-_phase2)
    - [Selección de device](#selección-de-device)
    - [`pipeline.summary()`](#pipelinesummary)
  - [6. Stages — referencia detallada](#6-stages--referencia-detallada)
    - [`AudioSample` (dataclass)](#audiosample-dataclass)
    - [Tabla de stages](#tabla-de-stages)
    - [Mapeo de labels](#mapeo-de-labels)
  - [7. `dataset.py` — `BioCASDataset`](#7-datasetpy--biocasdataset)
  - [8. `pipeline_api.py` — API para notebooks](#8-pipeline_apipy--api-para-notebooks)
  - [9. `run.py` — entry point CLI (Hydra)](#9-runpy--entry-point-cli-hydra)
  - [10. `train.py` y `models/`](#10-trainpy-y-models)
    - [`train.py`](#trainpy)
    - [`models/`](#models)
  - [11. Notebooks](#11-notebooks)
    - [`eda.ipynb`](#edaipynb)
    - [`research.ipynb`](#researchipynb)
  - [12. Issues conocidos / TODO](#12-issues-conocidos--todo)
  - [`pipeline.py` actual (posible alias legado de una versión anterior).](#pipelinepy-actual-posible-alias-legado-de-una-versión-anterior)
  - [13. Instalación rápida](#13-instalación-rápida)

---

## 1. Arquitectura general

```
WAV + JSON (SPRSound)
        │
        ▼
┌───────────────────────┐
│ get_data.sh            │  clona SPRSound, reorganiza en train/test
└──────────┬─────────────┘
           ▼
┌───────────────────────┐
│ data_config.py          │  escanea anotaciones, calcula balanceo,
└──────────┬─────────────┘  escribe rutas reales en pipeline.yaml
           ▼
┌───────────────────────┐
│ PreprocessingPipeline   │  Fase 1 (metadata) + Fase 2 (chunks, GPU)
│ (pipeline.py)           │
└──────────┬─────────────┘
           ▼
   List[AudioSample]  (.feature = log-mel, .label_int)
           │
           ▼
┌───────────────────────┐
│ BioCASDataset           │  torch.utils.data.Dataset
│ (dataset.py)            │
└──────────┬─────────────┘
           ▼
     DataLoader → train.py (EfficientNet-B0 1-canal)
```

**Decisión de diseño clave:** la pipeline está partida en dos fases para no
agotar la RAM con datasets grandes (problema original que motivó este
refactor):

- **Fase 1** — solo metadata (rutas, anotaciones JSON, timestamps). Ningún
  waveform en memoria. ~1 KB por evento.
- **Fase 2** — procesa en chunks de `cfg.runner.chunk_size` eventos. Carga el
  audio del chunk, lo procesa de punta a punta (resampling → ... →
  normalization) y descarta el waveform antes de pasar al siguiente chunk.
  Las etapas pesadas (resampling, augmentation, log-mel, spec-augment,
  normalization) corren en GPU si hay una disponible.

---

## 2. Estructura del repositorio

```
biocas_classifier/
├── data_config.py          # escanea JSONs, calcula balanceo, actualiza pipeline.yaml
├── get_data.sh              # descarga + reorganiza SPRSound en train/test
├── requirements.txt
├── README.md
├── LICENSE
└── workflow/
    ├── config/
    │   └── pipeline.yaml     # config central (Hydra/OmegaConf)
    ├── pipeline.py            # orquestador de 2 fases
    ├── pipeline_api.py        # API para notebooks (load_config, run_pipeline, ...)
    ├── dataset.py             # BioCASDataset (torch Dataset)
    ├── run.py                 # entry point CLI con @hydra.main
    ├── train.py                # loop de entrenamiento 
    ├── eda.ipynb       # notebook: exploración de datos
    ├── research.ipynb         # notebook: correr pipeline interactivamente
    ├── stages/
    │   ├── base.py            # AudioSample, Batch, BaseStage, Stage Protocol
    │   └── stages.py          # los 10(+2) stages concretos
    └── models/
        ├── cnns.py            # TODO
        ├── losses.py          # multiclass_focal_loss
        └── __init__.py        # TODO
```

---

## 3. Obtención de datos

### 3.1 `get_data.sh`

Descarga el repo [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git)
y reorganiza los WAV/JSON de las distintas ediciones (BioCAS2022–2025) en una
estructura plana `train/{wav,json}` y `test/{wav,json}`, prefijando cada
archivo con el año de origen para evitar colisiones de nombre.

**Mapeo de fuentes → split** (definido en el arreglo `SOURCES`):

| Origen | Split actual |
|---|---|
| BioCAS2022 `train2022` | `train` |
| BioCAS2022 `test2022` | `train` |
| BioCAS2023 `test2023` | `train` |
| BioCAS2024 `test2024` | `train` |
| BioCAS2025 `test2025` | `test` |

> ⚠ Todas las fuentes excepto BioCAS2025 van a `train`, incluyendo los
> `test*` de 2022–2024.

Al final del script, copia/symlinkea los archivos (`link_or_copy`, controlado
por `USE_SYMLINKS`) y llama automáticamente a `data_config.py`:

```bash
python3 data_config.py --data-dir "$DATA_DIR_ARG" \
                        --output-dir "$DATA_OUT_DIR_ARG" \
                        --config "$CONFIG_ARG"
```

> ⚠ Las variables `DATA_DIR_ARG`, `DATA_OUT_DIR_ARG`, `CONFIG_ARG` están
> **hardcodeadas** al inicio del script (`./data/raw_data`,
> `./data/processed`, `config/pipeline.yaml`) — no se parsean como argumentos
> de línea de comandos pese al comentario que dice "se reenvía". Si necesitás
> cambiarlas, editá esas tres líneas directamente o agregá un `getopts`/loop
> de parsing (ver versión anterior de este script si la tenés en tu
> historial de git).

Uso:

```bash
bash get_data.sh
```

### 3.2 `data_config.py`

Script de una sola pasada que:

1. Escanea recursivamente los JSON de `--data-dir/{train,test}/json`
2. Normaliza los tipos de evento (`EVENT_TYPE_NORM`) a las 7 clases canónicas
   (`N`, `R`, `W`, `S`, `CC`, `FC`, `WC`)
3. Imprime conteos por clase y una tabla de cuánto *augmentation*/balanceo
   necesita cada una (`ratio_vs_max`, `target_sqrt`, `necesita_aug`)
4. Actualiza **in-place** la sección `data:` de `workflow/config/pipeline.yaml`
   usando `ruamel.yaml` en modo *round-trip*, preservando comentarios y el
   resto de las secciones del config

CLI:

```bash
python3 data_config.py \
    --data-dir ./data/raw_data \
    --output-dir ./data/processed \
    --config workflow/config/pipeline.yaml
```

| Argumento | Default | Descripción |
|---|---|---|
| `--data-dir` | `./data/raw_data` | Carpeta con `train/{wav,json}` y `test/{wav,json}` |
| `--output-dir` | `./data/processed` | Se escribe en `cfg.data.output_dir` |
| `--config` | `workflow/config/pipeline.yaml` | Yaml a actualizar |

---

## 4. Configuración

`workflow/config/pipeline.yaml` es la única fuente de verdad de todos los
hiperparámetros. Se carga con Hydra (`run.py`) o con `pipeline_api.load_config()`
desde notebooks.

```yaml
data:
  raw_wav_dir / raw_json_dir / test_wav_dir / test_json_dir / output_dir
  task: "1-1" (binario N vs resto) | "1-2" (7 clases)

stages:                    # toggles enabled/disabled por stage
  loading, resampling, event_extraction, balanced_sampling,
  concatenation, padding, augmentation, log_mel, spec_augment, normalization

loading:
  target_sr: 4000           # Hz — resamplea todo a esta frecuencia
  keep_poor_quality: false  # descarta grabaciones "Poor Quality"

event_extraction:
  min_duration_s: 0.2       # descarta eventos más cortos
  keep_unlabelled: false    # descarta tipos de evento no reconocidos

balanced_sampling:
  strategy: "proportional"  # "proportional"/"sqrt" | "equal"
  target_samples_per_class: null   # override manual
  random_seed: 42

concatenation:
  min_duration_s / max_duration_s   # pega clips cortos del mismo label

padding:
  mode: "zero" | "reflect" | "tile"
  target_duration_s

augmentation:
  spec_augment: {enabled, freq_mask_param, time_mask_param, num_freq_masks, num_time_masks, mask_value}
  time_shift:   {enabled, max_shift_s}
  add_noise:    {enabled, snr_db_range}
  pitch_shift:  {enabled, semitone_range}   # ⚠ definido en yaml pero NO implementado en stages.py
  time_stretch: {enabled, rate_range}

log_mel:
  n_fft, hop_length, n_mels, fmin, fmax

normalization:
  strategy: "instance" | "global" | "none"
  global_stats_path   # requerido solo si strategy="global"

runner:
  chunk_size: 256      # eventos por chunk en la Fase 2
  batch_size: 64        # ⚠ no usado por pipeline.py actual (alias legado)
  device: "auto" | "cpu" | "cuda" | "cuda:0"
  log_level: "INFO"
  fail_fast: true
```

> ⚠ `augmentation.pitch_shift` está definido en el YAML (`enabled: false`)
> pero no existe ninguna implementación correspondiente en
> `WaveformAugmentationStage` (`stages.py`). Si lo activás, no tiene efecto.

> ⚠ El YAML de ejemplo incluido trae rutas absolutas de Windows
> (`C:\\Users\\camsp\\...`) hardcodeadas en `data.raw_wav_dir` etc. Estas se
> sobreescriben automáticamente al correr `get_data.sh` / `data_config.py`,
> pero si corrés `run.py` sin haber corrido antes ese paso, va a fallar
> buscando esas rutas en tu máquina.

---

## 5. La Pipeline

`workflow/pipeline.py` → clase `PreprocessingPipeline`.

```python
from pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline.from_config(cfg)
print(pipeline.summary())
batch = pipeline.run(augment=True)   # augment=False para test set
```

### Fase 1 — metadata liviana (`_phase1`)

| Paso | Stage | Qué hace |
|---|---|---|
| 1 | `LoadingStage(load_audio=False)` | Lee WAV+JSON del disco pero **no** carga el audio (`waveform=[]`, `sr=-1`). Solo guarda paths y anotación cruda. |
| 2 | `EventExtractionStage` | Detecta `sr==-1` (modo lazy) y, en vez de cortar el waveform, guarda `start_ms`/`end_ms` en `meta` para cortarlo después. |
| 3 | `BalancedSamplingStage` | Opera solo sobre metadata (label_int) — instantáneo, sin I/O de audio. |

Resultado: lista de `AudioSample` con `waveform=[]` pero `label_int`, `meta`
y conteo de clases ya balanceado. Footprint: ~1 KB/muestra.

### Fase 2 — procesamiento en chunks con GPU (`_phase2`)

Para cada chunk de `cfg.runner.chunk_size` eventos:

```
WaveformLoadingStage → SliceEventsStage → ResamplingStage → ConcatenationStage
→ PaddingStage → WaveformAugmentationStage → LogMelStage → SpecAugmentStage
→ NormalizationStage
```

Al terminar cada chunk, el waveform se reemplaza por un array vacío
(`s.waveform = np.array([])`) para liberar RAM, y si hay GPU se llama
`torch.cuda.empty_cache()`. Solo queda en memoria `s.feature` (el espectrograma).

`augment=False` (usado para el set de test) salta `WaveformAugmentationStage`
y `SpecAugmentStage` dentro del loop de chunks.

### Selección de device

```python
self.device = torch.device("cuda") if cfg.runner.device == "auto" and torch.cuda.is_available() else ...
```

Las stages de Fase 2 reciben `device` en el constructor y mueven sus tensores
ahí (`ResamplingStage`, `WaveformAugmentationStage`, `LogMelStage`,
`SpecAugmentStage`, `NormalizationStage`). `PaddingStage` y
`ConcatenationStage` trabajan en CPU/numpy.

> ⚠ Cada corrida de `pipeline.run()` reprocesa todo desde cero.

### `pipeline.summary()`

Imprime device, chunk size, task, sample rate, duración objetivo, mel bins y
qué augmentations están activas — útil para verificar la config antes de una
corrida larga.

---

## 6. Stages — referencia detallada

Todos los stages heredan de `BaseStage` (`stages/base.py`) y son *callables*
con la firma `batch_out = stage(batch_in, cfg)`. `BaseStage.__call__` chequea
automáticamente `cfg.stages.<name>.enabled` y saltea el stage si está en
`false`.

### `AudioSample` (dataclass)

```python
sample_id:  str
waveform:   np.ndarray       # [] si no está cargado (modo lazy)
sr:         int              # -1 si no está cargado
label_str:  str
label_int:  int = -1
meta:       Dict[str, Any]
feature:    Optional[np.ndarray] = None   # log-mel una vez calculado
```

### Tabla de stages

| # | Stage | Device | Entrada → Salida | Notas |
|---|---|---|---|---|
| 1 | `LoadingStage` | CPU | disco → 1 `AudioSample`/recording | `load_audio=False` para modo lazy (Fase 1) |
| 1b | `WaveformLoadingStage` | CPU | metadata → con audio cargado | Solo Fase 2; cachea WAVs repetidos dentro del chunk (`wav_cache`) |
| 2 | `ResamplingStage` | GPU | — | Resamplea a `cfg.loading.target_sr` con `torchaudio.functional.resample` |
| 3 | `EventExtractionStage` | CPU | 1 recording → N eventos | Usa `event_annotation` del JSON; detecta modo lazy automáticamente |
| 3b | `SliceEventsStage` | CPU | — | Complemento de (3) en modo lazy: corta el waveform ya cargado usando `start_ms`/`end_ms` de `meta` |
| 4 | `BalancedSamplingStage` | CPU | — | `proportional`/`sqrt` (∝√count) o `equal`; upsampling con repetición + draw aleatorio, downsampling sin reemplazo |
| 5 | `ConcatenationStage` | CPU | — | Pega hasta 10 donantes aleatorios del mismo `label_int` para alcanzar `min_duration_s` |
| 6 | `PaddingStage` | CPU | — | `zero` / `reflect` / `tile` hasta `target_duration_s` |
| 7 | `WaveformAugmentationStage` | GPU | — | time-shift (`torch.roll`), ruido blanco a SNR aleatorio, time-stretch (resample) |
| 8 | `LogMelStage` | GPU | waveform → feature | `torchaudio.transforms.MelSpectrogram` + `AmplitudeToDB` |
| 9 | `SpecAugmentStage` | GPU | — | Frequency + time masking (Park et al. 2019) sobre `s.feature` |
| 10 | `NormalizationStage` | GPU | — | `instance` (por muestra) o `global` (stats precalculadas en JSON) |

### Mapeo de labels

```python
LABEL_MAP_1_1 = {"N": 0, "R": 1, "W": 1, "S": 1, "CC": 1, "FC": 1, "WC": 1}  # binario
LABEL_MAP_1_2 = {"N": 0, "R": 1, "W": 2, "S": 3, "CC": 4, "FC": 5, "WC": 6}  # 7 clases
```

Seleccionado por `cfg.data.task` (`"1-1"` o `"1-2"`).

---

## 7. `dataset.py` — `BioCASDataset`

`torch.utils.data.Dataset` con dos modos de construcción:

```python
# Modo en memoria — features ya calculados por pipeline.run()
ds = BioCASDataset(batch)

# Modo lazy desde disco — cada __getitem__ lee su propio .npy
ds = BioCASDataset.from_index(index_df, base_dir=Path("data/processed/train"))
```

El modo `from_index` está diseñado para ser seguro con
`DataLoader(num_workers>0)` en Windows, ya que cada worker lee su propio
archivo sin compartir memoria ni file descriptors.

Propiedades útiles:

- `ds.num_classes` — derivado de `max(label_int) + 1`
- `ds.class_weights` — pesos inverso-frecuencia, listos para
  `nn.CrossEntropyLoss(weight=...)`
- `ds.feature_shape` — shape de un item individual, `(1, n_mels, T)`

---

## 8. `pipeline_api.py` — API para notebooks

Expone la pipeline sin depender del decorador `@hydra.main` (que no funciona
bien en Jupyter), usando `hydra.compose()` + `hydra.initialize_config_dir()`.

| Función | Descripción |
|---|---|
| `load_config(config_dir, config_name, overrides)` | Carga el YAML con overrides estilo Hydra (`["data.task=1-1", ...]`) |
| `show_config(cfg)` | Pretty-print del config activo |
| `patch_config(cfg, updates)` | Aplica overrides como `dict` con keys en notación de punto, sin tocar el YAML en disco |
| `build_pipeline(cfg)` | `PreprocessingPipeline.from_config(cfg)` |
| `run_pipeline(cfg, save=True)` | Corre la pipeline completa y devuelve un `BioCASDataset`; guarda en `cfg.data.output_dir` si `save=True` |
| `save_dataset(dataset, output_dir)` | Serializa a `index.csv` + `features/*.npy` (un .npy por muestra) |
| `load_dataset(output_dir)` | Carga el índice CSV; los `.npy` se leen bajo demanda |
| `processed_exists(output_dir)` | `True` si ya existe `index.csv` + carpeta `features/` |
| `load_or_run(cfg)` | Si hay datos procesados los carga; si no, corre la pipeline completa y guarda — punto de entrada recomendado para notebooks |

Ejemplo típico en notebook:

```python
from pipeline_api import load_config, load_or_run

cfg = load_config()
dataset = load_or_run(cfg)   # primera vez: corre la pipeline; después: instantáneo
```
---

## 9. `run.py` — entry point CLI (Hydra)

Punto de entrada para correr la pipeline desde terminal, fuera de notebooks.

```bash
python run.py                                    # config por defecto
python run.py data.task=1-1                      # tarea binaria
python run.py stages.augmentation.enabled=false   # saltear augmentation
python run.py runner.chunk_size=128
```

Al finalizar, llama a `save_processed_dataset(batch, cfg.data.output_dir)`,
que guarda:

```
output_dir/
  features.npy    # (N, n_mels, T) float32 — TODO el batch en un solo array
  labels.npy      # (N,) int32
  metadata.json   # lista de {sample_id, label_str, meta}
```

> [!CAUTION]
> ⚠ Este formato de guardado es **distinto** al usado por
> `pipeline_api.save_dataset()` (que guarda un `.npy` por muestra +
> `index.csv`). Son dos formatos de persistencia incompatibles entre sí:
> `run.py` produce `features.npy`/`labels.npy`/`metadata.json`, mientras que
> `pipeline_api.py`/`train.py` esperan `index.csv` + `features/*.npy`. Si
> corrés `run.py` y después intentás `load_dataset()` desde `pipeline_api`,
> va a fallar con `FileNotFoundError` porque no existe `index.csv`. Hay que
> unificar a uno de los dos formatos. 
> **Por arreglar** -> Usar desde el notebook provicionalmente

---

## 10. `train.py` y `models/`

### `train.py`

Loop de entrenamiento standalone (no integrado con Hydra ni con
`pipeline_api.run_pipeline`):

```python
test_dataset  = load_dataset("data/processed_v3/test")
train_dataset = load_dataset("data/processed_v3/train")
train(test_dataset, train_dataset)
```

Modelo: `torchvision.models.efficientnet_b0`, con la primera capa
convolucional reemplazada para aceptar 1 canal (espectrogramas en vez de
RGB) y la cabeza de clasificación ajustada a 7 clases (hardcodeado).

> ⚠ **Issues detectados en `train.py`:**
> - La ruta `"data/processed_v3/..."` no coincide con `cfg.data.output_dir`
>   por defecto (`data/processed` o `data\processed` según el YAML) — es un
>   path manual, probablemente de una corrida específica.
> - El número de clases está hardcodeado en `7` (`nn.Linear(num_ftrs, 7)`),
>   ignorando `cfg.data.task`. Si corrés con `task="1-1"` (2 clases), el
>   modelo va a tener una cabeza con 7 salidas pero solo 2 labels reales.
> - `train_loader` y `val_loader` se definen como variables globales dentro
>   de `if __name__ == "__main__":` pero se usan dentro de la función
>   `train()` sin pasarlos como argumento — funciona solo porque Python
>   resuelve nombres globales en tiempo de ejecución, pero `train.py` no es
>   importable como módulo sin romper (`train()` fallaría si se llama desde
>   otro script sin antes definir esas globales).
> - No hay validación/early stopping — el loop solo entrena, nunca evalúa
>   sobre `val_loader` dentro del loop de épocas.
> - Requiere CUDA explícitamente (`model.to("cuda")`, sin chequeo de
>   disponibilidad) — falla en máquinas sin GPU.

> [!WARNING]
> Por terminar para generalizar train para distintas arquitecturas / OPTUNA

### `models/`


> [!WARNING]
> Por hacer
> 
| Archivo | Estado |
|---|---|
| `cnns.py` | ⚠ Vacío — solo tiene los imports (`torch`, `nn`, `F`). Ninguna arquitectura CNN custom implementada todavía pese al nombre del archivo. |
| `losses.py` | `multiclass_focal_loss(inputs, targets, alpha, gamma=2.0)` — Focal Loss multiclase, pensada para usar junto con `dataset.class_weights` como `alpha`. |
| `__init__.py` | ⚠ Vacío — no re-exporta nada; hay que importar con paths completos (`from models.losses import multiclass_focal_loss`). |

---

## 11. Notebooks

### `eda.ipynb`
> [!WARNING]
> Por hacer


### `research.ipynb`

Notebook de exploración interactiva de la pipeline usando `pipeline_api`:

```python
cfg = load_config()
show_config(cfg)
cfg_test = load_config(overrides=[
    "data.raw_wav_dir=data/test/wav",
    "data.raw_json_dir=data/test/json",
    "stages.balanced_sampling.enabled=false",
    ...
])
```

Útil como referencia de overrides típicos al armar el set de test (sin
balanceo de clases, sin augmentation).

---

## 12. Issues conocidos / TODO

Resumen consolidado de todo lo marcado con ⚠ arriba, más lo declarado en el
`README.md` original:

- [ ] **Formato de persistencia inconsistente** entre `run.py`
  (`features.npy`/`labels.npy`/`metadata.json`) y `pipeline_api.py`/
  `dataset.py` (`index.csv` + `features/*.npy`). Unificar a uno solo.
- [ ] **`augmentation.pitch_shift`** está en el YAML pero no implementado en
  `WaveformAugmentationStage`.
- [ ] **`train.py`**: número de clases hardcodeado (7), requiere CUDA sin
  fallback, sin loop de validación, variables globales implícitas.
- [ ] **`models/cnns.py`** y **`models/__init__.py`** están vacíos —
  arquitectura CNN custom pendiente.
- [ ] **`get_data.sh`**: variables de path hardcodeadas pese al comentario
  que sugiere que son argumentos reenviables; mapeo de splits envía casi
  todo a `train` salvo BioCAS2025.
- [ ] **`cfg.runner.batch_size`** declarado en el YAML pero no consumido por
  `pipeline.py` actual (posible alias legado de una versión anterior).
---

## 13. Instalación rápida

```bash
git clone <este-repo>
cd biocas_classifier
pip install -r requirements.txt

bash get_data.sh                       # descarga + organiza datos + actualiza yaml
python workflow/run.py                  # corre la pipeline completa (CLI/Hydra)
# o, desde notebook/research.ipynb:
#   from pipeline_api import load_config, load_or_run
#   dataset = load_or_run(load_config())
```

**Requisitos de hardware:** GPU recomendada (CUDA) para que la Fase 2 de la
pipeline (resampling, augmentation, log-mel, spec-augment, normalization)
corra a velocidad razonable — todas esas stages mueven tensores a
`cfg.runner.device`. Funciona en CPU (`device: "cpu"`) pero más lento.