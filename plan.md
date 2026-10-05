# Proyecto: Clasificación physics-informed de sonidos respiratorios

## Contexto del proyecto

Clasificación de sonidos adventicios pulmonares (crepitantes/crackles, sibilancias/wheezes)
usando deep learning. El baseline (CNN sobre mel-spectrogramas + SpecAugment) hace overfitting
rápido, principalmente por desbalance de clases y poca capacidad de generalización con dataset
pequeño. La estrategia actual combina:

1. Un clasificador jerárquico tipo **mixture of experts**: gate 1 (normal vs. anormal) → gate 2
   (tipo de sonido anormal: crepitante / sibilancia / otro), condicionado a que gate 1 diga "anormal".
2. Técnicas **physics-informed** para inyectar conocimiento del dominio y reducir la dependencia
   de datos: features derivadas de modelos físicos, augmentación sintética generada por esos
   mismos modelos, y pérdidas auxiliares multitask.

<!-- AJUSTAR: dataset(s) que se está usando ahora mismo -->
**Dataset:** SPRSound (https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) y/o ICBHI 2017.

- **ICBHI**: ciclos respiratorios anotados como normal / crackle / wheeze / both. Fuerte
  desbalance hacia "normal". 4 dispositivos de grabación distintos (posible fuente de leakage
  si el split no es cuidadoso). Métrica estándar de comparación en literatura: ICBHI score =
  (Sensitivity + Specificity) / 2.
- **SPRSound**: anotación a nivel de evento (timestamps) además de nivel de registro, incluye
  clases más finas (fine/coarse crackle, wheeze, stridor, rhonchi) y etiquetas de calidad de
  grabación. Población pediátrica.

**Regla no negociable:** todos los splits deben ser **por paciente**, nunca por evento/ciclo
suelto. Verificar esto antes de confiar en cualquier métrica.

## Modelos físicos de referencia (para features, augmentación y pseudo-etiquetas)

- **Crepitante**: oscilación sinusoidal amortiguada exponencialmente.
  `y(t) = A · exp(-t/τ) · sin(2π f0 t + φ)`
  - Fino: f0 ≈ 650 Hz, τ corto (duración ~5 ms)
  - Grueso: f0 ≈ 350 Hz, τ mayor (duración ~15 ms)
- **Sibilancia**: oscilación auto-sostenida inducida por flujo (análoga a un oscilador
  autosostenido / flutter aeroelástico). Cuasi-sinusoidal, armónicos estables,
  f0 típicamente > 100–400 Hz, duración > 80 ms, baja varianza de frecuencia instantánea.

Estos modelos son la base de las tres piezas physics-informed de abajo. Si algún componente no
está funcionando, el primer sospechoso es que el modelo paramétrico no está calibrado a los
datos reales (ruido de fondo, dispositivo, población pediátrica vs. adulta).

## Componentes a implementar

### 1. Features físicas (extracción, no aprendida)
Ubicación sugerida: `<!-- AJUSTAR: ej. src/features/physics_features.py -->`

- Envolvente + tasa de decaimiento τ: Hilbert transform → envolvente → regresión log-lineal
  en ventanas cortas.
- Frecuencia instantánea y su varianza: filtro pasa-banda (100–2000 Hz) → Hilbert → derivada
  de fase instantánea.
- Banco de matched filters (Gabor/Morlet amortiguadas) barriendo `(f0, τ)` en rango fisiológico;
  guardar máxima correlación y el `(f0*, τ*)` óptimo.
- Persistencia espectral entre frames (spectral flux) para estimar duración de evento.

**Antes de integrar cualquiera de estas al modelo grande**: validar con un clasificador simple
(regresión logística / MLP pequeño) que las features solas separan las clases en un subset de
validación. Si no separan nada, no avanzar — recalibrar el modelo físico primero.

### 2. Augmentación sintética
Ubicación sugerida: `<!-- AJUSTAR: ej. src/synth/generators.py -->`

- `synth_crackle(fs, tau_range, f0_range)`: genera crepitante paramétrico (ver fórmula arriba).
- `synth_wheeze(fs, f0_range, jitter)`: oscilador auto-sostenido (van der Pol o FM con jitter
  controlado de frecuencia).
- `insert_event(background_real, event, snr_db, phase_aware=True)`: inserta el evento sintético
  sobre **fondo real** (nunca fondo simulado), respetando fase respiratoria (crepitantes más
  probables en inspiración, sibilancias en espiración).

**Reglas no negociables:**
- Fondo siempre real, evento sintético.
- Barrer un rango amplio de SNR, incluyendo casos difíciles.
- Usar la síntesis para balancear las clases minoritarias del **gate 2** (tipo de sonido), no
  para inflar artificialmente la clase "anormal" del gate 1.
- **Nunca meter datos sintéticos en validación o test.** Solo en train.
- Correr periódicamente un probe sintético-vs-real (clasificador simple sobre embeddings
  intermedios) para detectar domain gap. Si separa con alta precisión, aumentar variabilidad
  del generador (ruido de canal, distorsión no lineal leve).

### 3. Pérdidas auxiliares / multitask
Ubicación sugerida: `<!-- AJUSTAR: ej. src/losses/physics_losses.py -->`

```
loss = ce_loss(class_logits, y_class)
     + lambda1 * mse_loss(tau_pred, tau_pseudolabel)
     + lambda2 * mse_loss(f0_pred, f0_pseudolabel)
```

- Las pseudo-etiquetas `(τ, f0)` vienen del matched filter del punto 1 — son señal de
  regularización débil, no ground truth clínico.
- Barrer `lambda` en {0, 0.05, 0.1, 0.3, 0.5}, medir impacto en macro-F1 de validación.
- Gate 1 (normal/anormal): preferir un término de **consistencia** (¿la energía atribuida a
  "anormal" cae dentro de la banda de frecuencia/duración esperada?) en vez de regresión directa.
- Gate 2 (tipo de sonido): regresión directa de `(τ, f0)` tiene sentido porque es lo que
  distingue crepitante de sibilancia.

### 4. Arquitectura MoE
Ubicación sugerida: `<!-- AJUSTAR: ej. src/models/moe.py -->`

- Tronco CNN compartido sobre mel-spectrograma + rama de features físicas → fusión (empezar
  por concatenación simple antes de probar FiLM u otras fusiones más caras).
- Gate 1: normal vs. anormal (+ pérdida de consistencia física).
- Gate 2 (condicional, solo si gate 1 = anormal): crepitante / sibilancia / otro
  (+ regresión multitask τ, f0).

## Protocolo de experimentación — ORDEN ESTRICTO

No saltar pasos. Cada paso debe compararse contra el anterior de forma aislada para poder
atribuir la mejora al componente correcto.

1. **Baseline documentado**: CNN actual, split por paciente verificado, matriz de confusión
   completa, macro-F1 y sensibilidad/especificidad por clase.
2. **Solo features físicas** (fusión temprana simple), sin augmentación ni pérdidas nuevas.
3. **Solo augmentación sintética**, arquitectura sin cambios.
4. **2 + 3 combinados.**
5. **Añadir pérdida auxiliar** sobre el mejor resultado de 4, con barrido de `lambda`.
6. **MoE completo** (features + augmentación + pérdidas + arquitectura jerárquica) vs. el mismo
   MoE sin ningún componente físico, como comparación final.

Si un paso no mejora sobre el anterior, no avanzar al siguiente arrastrándolo — volver a
diagnosticar (ver sección de features físicas arriba) antes de sumar más complejidad.

## Reglas de evaluación (no negociables en ningún experimento)

- Split por paciente, nunca por evento/ciclo.
- Nunca sintéticos en validación/test.
- Reportar siempre: macro-F1, sensibilidad y especificidad por clase, matriz de confusión.
- Si se usa ICBHI, reportar también el score oficial (Sens+Spec)/2 para comparar con literatura.
- Cada experimento debe loguearse con su configuración completa (semilla, hiperparámetros,
  qué componentes physics-informed están activos) para poder reproducirlo.

## Stack y convenciones de código

<!-- AJUSTAR: confirmar/framework real del repo -->
- Lenguaje: Python. Framework de deep learning: PyTorch (asumido — corregir si es TensorFlow/Keras).
- Experimentos config-driven (ej. YAML + argparse o Hydra) — un config por experimento,
  no hardcodear hiperparámetros en el script de entrenamiento.
- Semillas fijas en todos los experimentos (`numpy`, `torch`, `random`).
- Logs y checkpoints en `experiments/<fecha>_<nombre-experimento>/`.

<!-- AJUSTAR: comandos reales del repo -->
### Comandos
- Entrenar: `python src/train.py --config configs/<experimento>.yaml`
- Evaluar: `python src/eval.py --checkpoint <path> --split test`
- Tests: `pytest tests/`

## Estructura de repo esperada

<!-- AJUSTAR a la estructura real -->
```
data/                  # raw + processed (ICBHI/SPRSound)
src/
  data/                # loaders, splits por paciente, preprocesamiento
  features/            # features físicas (matched filter, envolvente, IF)
  synth/               # generadores sintéticos de crepitante/sibilancia
  models/              # CNN trunk, gates MoE, capas de fusión
  losses/              # pérdidas auxiliares / multitask
  train.py
  eval.py
experiments/           # configs y resultados por corrida
notebooks/             # EDA, validación de features físicas (paso 1 del protocolo)
tests/
```

## Cosas a vigilar (pitfalls conocidos de este dominio)

- Overfitting "rápido" casi siempre es leakage de paciente entre train/test, no falta de prior
  físico — descartar esto antes de atribuir mejoras a los componentes physics-informed.
- Variabilidad de dispositivo en ICBHI puede filtrarse como "firma del micrófono" en vez de
  patología real.
- `lambda` de pérdida auxiliar demasiado alto puede degradar la tarea principal si las
  pseudo-etiquetas físicas son ruidosas — es señal para recalibrar el modelo físico, no para
  bajar el lambda y seguir de largo.
- El gate 1 (normal/anormal) y el gate 2 (tipo de sonido) tienen necesidades physics-informed
  distintas: el gate 1 se beneficia más de consistencia y augmentación con "normales difíciles";
  el gate 2 se beneficia más de features físicas discriminativas y regresión multitask.

## Referencias de dominio (consultar/verificar, no citar como definitivas sin revisar)

- Fredberg & Holford — modelo de crepitante como oscilación amortiguada.
- Gavriely et al. / Sovijärvi et al. (ERS Task Force) — modelos y taxonomía de sibilancias.
- Engel et al. 2020 (DDSP) — inspiración para capas de síntesis diferenciable si se llega a
  la fase de arquitectura "analysis-by-synthesis".

## Cómo trabajar en este repo (para Claude Code)

- Antes de escribir código nuevo, correr los tests existentes y confirmar que el baseline del
  paso 1 del protocolo sigue reproduciéndose.
- No implementar los pasos 4-6 del protocolo de experimentación si 2 y 3 no están cada uno
  validados por separado primero.
- Cualquier cambio de arquitectura del MoE (gates, fusión) debe proponerse antes de implementar,
  no aplicarse directamente sobre el pipeline de entrenamiento actual.
- Al reportar resultados de un experimento, incluir siempre macro-F1 y métricas por clase, no
  solo accuracy — el dataset está desbalanceado y accuracy es engañosa.
