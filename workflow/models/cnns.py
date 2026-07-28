"""
──────────────
CNN custom para clasificación de espectrogramas respiratorios.
Diseñada para NAS con Optuna — toda la arquitectura emerge de los params
sugeridos por el trial, sin pesos preentrenados.

Espacio de búsqueda
───────────────────
  n_blocks        : número de bloques convolucionales (2–6)
  base_channels   : canales del primer bloque (16 / 32 / 64)
  channel_growth  : cómo crecen los canales entre bloques
                    "double"   → [C, 2C, 4C, ...]
                    "constant" → [C, C,  C,  ...]
                    "linear"   → [C, C+16, C+32, ...]
  kernel_size     : kernel temporal (3 / 5 / 7) — global para todos los bloques
  norm            : "batch" / "group" / "instance" / "none"
  activation      : "relu" / "gelu" / "leaky_relu" / "elu"
  pool_type       : "max" / "avg" aplicado tras cada bloque
  use_se          : Squeeze-and-Excitation en cada bloque (True / False)
  use_residual    : skip connections (True / False)
  global_pool     : cómo se colapsa el mapa espacial final
                    "avg" / "max" / "concat" (avg || max, dobla el vector)
  dropout         : dropout antes de la capa lineal final (0.0 – 0.5)

Uso desde Optuna
────────────────
    from models.cnns import CustomAudioCNN
    model = CustomAudioCNN.build(trial, num_classes=7, in_channels=1)

Uso con defaults (sin trial)
────────────────────────────
    model = CustomAudioCNN.build(None, num_classes=7, in_channels=1)
"""
from __future__ import annotations

import math
from typing import List, Optional

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAudioClassifier, register_model


# ─────────────────────────────────────────────────────────────────────────────
# Bloques reutilizables
# ─────────────────────────────────────────────────────────────────────────────

def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        # num_groups debe dividir channels; usamos min(8, channels)
        n_groups = min(8, channels)
        while channels % n_groups != 0:
            n_groups -= 1
        return nn.GroupNorm(n_groups, channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    return nn.Identity()   # "none"


def _make_activation(act: str) -> nn.Module:
    return {
        "relu":       nn.ReLU(inplace=True),
        "gelu":       nn.GELU(),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "elu":        nn.ELU(inplace=True),
    }[act]


def _make_pool(pool_type: str) -> nn.Module:
    if pool_type == "max":
        return nn.MaxPool2d(kernel_size=2, stride=2)
    return nn.AvgPool2d(kernel_size=2, stride=2)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al. 2018).
    Recalibra los canales aprendiendo qué features son más importantes.
    reduction=8 en vez de 16 porque los canales son más chicos que en ResNet.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class ConvBlock(nn.Module):
    """
    Un bloque convolucional completo:
        Conv2d → Norm → Activation → (SE opcional) → Pool

    Si use_residual=True y in_channels != out_channels, se agrega una
    proyección 1×1 para que las dimensiones coincidan en el skip.
    El pool se aplica FUERA del residual para no romper la suma.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        norm: str,
        activation: str,
        pool_type: str,
        use_se: bool,
        use_residual: bool,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=(norm == "none"))
        self.norm = _make_norm(norm, out_ch)
        self.act  = _make_activation(activation)
        self.se   = SEBlock(out_ch) if use_se else nn.Identity()
        self.pool = _make_pool(pool_type)

        self.use_residual = use_residual
        self.proj: Optional[nn.Module] = None
        if use_residual and in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                _make_norm(norm, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.se(out)

        if self.use_residual:
            if self.proj is not None:
                identity = self.proj(identity)
            out = out + identity

        out = self.pool(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CNN custom completa
# ─────────────────────────────────────────────────────────────────────────────

def _channel_schedule(base: int, growth: str, n_blocks: int) -> List[int]:
    """Devuelve la lista de canales para cada bloque."""
    if growth == "double":
        return [min(base * (2 ** i), 512) for i in range(n_blocks)]
    if growth == "linear":
        return [base + 16 * i for i in range(n_blocks)]
    return [base] * n_blocks   # "constant"


@register_model("custom_cnn")
class CustomAudioCNN(BaseAudioClassifier):
    """
    CNN configurable completamente por Optuna.
    Sin pesos preentrenados — aprende desde cero sobre espectrogramas.
    """

    def __init__(self, blocks: nn.Sequential, head: nn.Sequential) -> None:
        super().__init__()
        self.blocks = blocks
        self.head   = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        feat = self.blocks(x)           # (B, C, H', W')
        feat = self._global_pool(feat)  # (B, C') — C' depende del global_pool
        return self.head(feat)

    # global_pool se inyecta como atributo en build()
    def _global_pool(self, x: torch.Tensor) -> torch.Tensor:
        return self._gpool(x)

    @classmethod
    def build(
        cls,
        trial: Optional[optuna.Trial],
        num_classes: int,
        in_channels: int = 1,
    ) -> "CustomAudioCNN":

        def s_int(name, low, high, **kw):
            return trial.suggest_int(name, low, high, **kw) if trial else (low + high) // 2

        def s_cat(name, choices):
            return trial.suggest_categorical(name, choices) if trial else choices[0]

        def s_float(name, low, high, **kw):
            return trial.suggest_float(name, low, high, **kw) if trial else (low + high) / 2

        # ── Hiperparámetros de arquitectura ───────────────────────────────
        n_blocks      = s_int("n_blocks",     2, 6)
        base_channels = s_cat("base_channels", [16, 32, 64])
        channel_growth= s_cat("channel_growth",["double", "constant", "linear"])
        kernel_size   = s_cat("kernel_size",   [3, 5, 7])
        norm          = s_cat("norm",          ["batch", "group", "instance", "none"])
        activation    = s_cat("activation",    ["relu", "gelu", "leaky_relu", "elu"])
        pool_type     = s_cat("pool_type",     ["max", "avg"])
        use_se        = s_cat("use_se",        [True, False])
        use_residual  = s_cat("use_residual",  [True, False])
        global_pool   = s_cat("global_pool",   ["avg", "max", "concat"])
        dropout       = s_float("dropout",     0.0, 0.5)

        # ── Construir bloques ─────────────────────────────────────────────
        channels = _channel_schedule(base_channels, channel_growth, n_blocks)
        block_list = []
        ch_in = in_channels

        for ch_out in channels:
            block_list.append(ConvBlock(
                in_ch       = ch_in,
                out_ch      = ch_out,
                kernel_size = kernel_size,
                norm        = norm,
                activation  = activation,
                pool_type   = pool_type,
                use_se      = use_se,
                use_residual= use_residual,
            ))
            ch_in = ch_out

        blocks = nn.Sequential(*block_list)
        last_ch = channels[-1]

        # ── Global pooling ────────────────────────────────────────────────
        # "concat" combina avg y max → dobla el nº de features
        if global_pool == "avg":
            gpool_fn = nn.AdaptiveAvgPool2d(1)
            head_in  = last_ch
        elif global_pool == "max":
            gpool_fn = nn.AdaptiveMaxPool2d(1)
            head_in  = last_ch
        else:   # "concat"
            gpool_fn = None          # se maneja en _ConcatPool
            head_in  = last_ch * 2

        # ── Cabeza clasificadora ──────────────────────────────────────────
        head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(head_in, max(64, head_in // 2)),
            _make_activation(activation),
            nn.Linear(max(64, head_in // 2), num_classes),
        )

        # ── Ensamblar modelo ──────────────────────────────────────────────
        model = cls(blocks=blocks, head=head)

        # Inyectar global pool como método interno
        if global_pool == "concat":
            avg_pool = nn.AdaptiveAvgPool2d(1)
            max_pool = nn.AdaptiveMaxPool2d(1)

            def _gpool(x):
                avg = avg_pool(x).flatten(1)
                mx  = max_pool(x).flatten(1)
                return torch.cat([avg, mx], dim=1)
        else:
            _pool = gpool_fn

            def _gpool(x):
                return _pool(x).flatten(1)

        import types
        model._gpool = types.MethodType(lambda self, x: _gpool(x), model)

        # Inicialización de pesos
        model._init_weights(norm)

        return model

    def _init_weights(self, norm: str) -> None:
        """
        Inicialización de pesos apropiada según el tipo de normalización.
        Con BatchNorm/GroupNorm usamos He (kaiming), con none usamos Xavier.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if norm in ("batch", "group", "instance"):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)