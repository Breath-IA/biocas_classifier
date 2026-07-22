"""
models/cnns.py
──────────────
Arquitecturas concretas registradas en MODEL_REGISTRY.

Cada `build(trial, ...)` usa `trial.suggest_*` para explorar variantes
cuando Optuna está activo, y caen a defaults razonables si trial=None.

Modelos disponibles
───────────────────
  "efficientnet_b0"  –  EfficientNet-B0 preentrenado, primer conv adaptado a 1 canal
  "resnet18"         –  ResNet-18 preentrenado, igual adaptación

Para agregar un modelo nuevo:
    @register_model("mi_modelo")
    class MiModelo(BaseAudioClassifier):
        @classmethod
        def build(cls, trial, num_classes, in_channels=1):
            ...
"""
from __future__ import annotations

from typing import Optional

import optuna
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet18, ResNet18_Weights,
)

from .base import BaseAudioClassifier, register_model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_or_default(trial: Optional[optuna.Trial], name: str, default, **kwargs):
    """
    Si trial es None devuelve default.
    Si trial está activo, llama al método de sugerencia adecuado según el tipo
    de default (float → suggest_float, int → suggest_int, str → suggest_categorical).
    """
    if trial is None:
        return default
    if isinstance(default, float):
        return trial.suggest_float(name, **kwargs)
    if isinstance(default, int):
        return trial.suggest_int(name, **kwargs)
    if isinstance(default, str):
        choices = kwargs.get("choices", [default])
        return trial.suggest_categorical(name, choices)
    return default


def _replace_first_conv(model: nn.Module, in_channels: int, conv_path) -> None:
    """
    Reemplaza la primera capa convolucional de un modelo torchvision para
    aceptar `in_channels` canales (por defecto 3 en ImageNet).

    Parameters
    ----------
    conv_path : referencia a la conv a reemplazar (ej. model.features[0][0])
    """
    old = conv_path
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    return new


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet-B0
# ─────────────────────────────────────────────────────────────────────────────

@register_model("efficientnet_b0")
class EfficientNetB0Classifier(BaseAudioClassifier):
    """
    EfficientNet-B0 adaptado para espectrogramas de 1 canal.

    Hiperparámetros que Optuna puede explorar:
      dropout  (0.0 – 0.5)  : dropout antes de la capa de clasificación
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    @classmethod
    def build(
        cls,
        trial: Optional[optuna.Trial],
        num_classes: int,
        in_channels: int = 1,
    ) -> "EfficientNetB0Classifier":
        dropout = _suggest_or_default(
            trial, "dropout", default=0.2,
            low=0.0, high=0.5
        )

        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Adaptar primer conv a in_channels
        model.features[0][0] = _replace_first_conv(
            model, in_channels, model.features[0][0]
        )

        # Reemplazar cabeza de clasificación
        in_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_ftrs, num_classes),
        )

        return cls(backbone=model)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-18
# ─────────────────────────────────────────────────────────────────────────────

@register_model("resnet18")
class ResNet18Classifier(BaseAudioClassifier):
    """
    ResNet-18 adaptado para espectrogramas de 1 canal.

    Hiperparámetros que Optuna puede explorar:
      dropout  (0.0 – 0.5)  : dropout extra antes de la cabeza final
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    @classmethod
    def build(
        cls,
        trial: Optional[optuna.Trial],
        num_classes: int,
        in_channels: int = 1,
    ) -> "ResNet18Classifier":
        dropout = _suggest_or_default(
            trial, "dropout", default=0.2,
            low=0.0, high=0.5
        )

        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adaptar primer conv a in_channels
        model.conv1 = _replace_first_conv(model, in_channels, model.conv1)

        # Reemplazar cabeza de clasificación
        in_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_ftrs, num_classes),
        )

        return cls(backbone=model)