"""
models/base.py
──────────────
Clase abstracta que deben implementar todos los modelos del proyecto.

Contrato
────────
Cada arquitectura:
  1. Hereda de BaseAudioClassifier (que hereda de nn.Module).
  2. Implementa `build(trial, num_classes, in_channels)` como classmethod —
     recibe un `optuna.Trial` y devuelve una instancia ya configurada.
     Si `trial` es None el modelo se construye con sus hiperparámetros por
     defecto (útil para runs sin búsqueda).
  3. Expone `name` como propiedad de clase.

Registro
────────
Los modelos se registran con @register_model("nombre").
train.py usa el registro para que Optuna pueda sugerir qué arquitectura probar.

    from models.base import MODEL_REGISTRY
    cls = MODEL_REGISTRY["efficientnet_b0"]
    model = cls.build(trial, num_classes=7, in_channels=1)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Type

import optuna
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Registro global
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Type["BaseAudioClassifier"]] = {}


def register_model(name: str):
    """
    Decorador de clase.  Uso:

        @register_model("mi_modelo")
        class MiModelo(BaseAudioClassifier):
            ...
    """
    def decorator(cls: Type["BaseAudioClassifier"]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Modelo '{name}' ya registrado.")
        MODEL_REGISTRY[name] = cls
        cls.name: ClassVar[str] = name
        return cls
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Clase base
# ─────────────────────────────────────────────────────────────────────────────

class BaseAudioClassifier(nn.Module, ABC):
    """
    Base para todos los clasificadores de audio del proyecto.

    Los subclases deben implementar:
        - `build(trial, num_classes, in_channels)` → instancia del modelo
        - `name` → str (normalmente inyectado por @register_model)
        - `forward(x)` → Tensor  (herencia normal de nn.Module)
    """

    name: ClassVar[str] = "base"

    @classmethod
    @abstractmethod
    def build(
        cls,
        trial: Optional[optuna.Trial],
        num_classes: int,
        in_channels: int = 1,
    ) -> "BaseAudioClassifier":
        """
        Construye el modelo, opcionalmente guiado por Optuna.

        Parameters
        ----------
        trial       : optuna.Trial activo, o None para defaults.
        num_classes : número de clases de salida.
        in_channels : canales de entrada (1 para espectrogramas mono).
        """
        ...

    def n_parameters(self) -> int:
        """Número de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"params={self.n_parameters():,})"
        )
