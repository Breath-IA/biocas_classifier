# Gate 1 — Normal contra anormal

Primera compuerta de [plan.md](../../plan.md): decide si un evento es normal o anormal. Si es anormal pasa al gate 2.

| Orden | Experimento | Pregunta | Estado |
|---|---|---|---|
| e01 | [e01_spectral_baseline/](e01_spectral_baseline/) | ¿Qué features espectrales/temporales separan N de anormal y sobreviven al cambio de año? | hecho |

Pendiente para este gate (plan.md): features físicas (paso 2), consistencia física en la pérdida del gate 1 (paso 5) y
augmentación con "normales difíciles" (paso 3).

**Hallazgo que condiciona el resto:** el gate 1 es el que peor generaliza a 2025 (ver e01), por desplazamiento de
dominio y de prevalencia (17 % de anormales en 2022–24, 53 % en 2025).
