"""
models_small.py — one builder for every FPGA-sized CNN variant, driven by a config dict (see search_space.py), plus parameter / MAC counting.

    model = build_model(cfg, n_mels)          # input (B, 1, n_mels, 95) -> logit (B,)
    profile(model, (n_mels, 95))              # params, BN-folded params, MACs, peak activation KB (int8)

Only Conv / BatchNorm / ReLU-family / pooling / Linear (+ Sigmoid, Mul for the optional squeeze-excite) are used, so the exported ONNX is FPGA-friendly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

ACTS = {"relu": lambda: nn.ReLU(inplace=True), "relu6": lambda: nn.ReLU6(inplace=True), "leaky": lambda: nn.LeakyReLU(0.1, inplace=True)}


def stage_widths(cfg: dict) -> list[int]:
    w, out = float(cfg["base_width"]), []
    for _ in range(int(cfg["n_stages"])):
        out.append(int(min(256, max(4, 4 * round(w / 4)))))
        w *= float(cfg["growth"])
    return out


class SE(nn.Module):
    def __init__(self, c: int, r: int = 4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(4, c // r), 1), nn.ReLU(inplace=True), nn.Conv2d(max(4, c // r), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


def conv_bn(c_in, c_out, k, groups=1):
    return [nn.Conv2d(c_in, c_out, k, padding=k // 2, groups=groups, bias=False), nn.BatchNorm2d(c_out)]


class Block(nn.Module):
    def __init__(self, kind, c_in, c_out, k, act, se):
        super().__init__()
        self.kind = kind
        if kind == "dws" and c_in > 1:
            self.body = nn.Sequential(*conv_bn(c_in, c_in, k, groups=c_in), ACTS[act](), *conv_bn(c_in, c_out, 1))
        elif kind == "res":
            self.body = nn.Sequential(*conv_bn(c_in, c_out, k), ACTS[act](), *conv_bn(c_out, c_out, 3))
            self.short = nn.Sequential(*conv_bn(c_in, c_out, 1)) if c_in != c_out else nn.Identity()
        else:
            self.body = nn.Sequential(*conv_bn(c_in, c_out, k))
        self.se = SE(c_out) if se else nn.Identity()
        self.act = ACTS[act]()

    def forward(self, x):
        y = self.se(self.body(x))
        if self.kind == "res":
            y = y + self.short(x)
        return self.act(y)


class SmallNet(nn.Module):
    def __init__(self, cfg: dict, n_mels: int):
        super().__init__()
        widths, c_in, layers = stage_widths(cfg), 1, []
        for i, w in enumerate(widths):
            k = int(cfg["kernel_first"]) if i == 0 else 3
            layers += [Block(cfg["block"], c_in, w, k, cfg["act"], bool(cfg["se"]) and i >= 1)]
            layers += [nn.MaxPool2d(2, ceil_mode=True) if cfg["pool"] == "max" else nn.AvgPool2d(2, ceil_mode=True)]
            c_in = w
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        h = int(cfg["head_hidden"])
        drop = float(cfg["dropout"])
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(c_in, h), nn.ReLU(inplace=True), nn.Linear(h, 1)) if h else nn.Sequential(nn.Dropout(drop), nn.Linear(c_in, 1))

    def forward(self, x):
        return self.head(self.gap(self.features(x)).flatten(1)).squeeze(1)


def build_model(cfg: dict, n_mels: int) -> SmallNet:
    return SmallNet(cfg, n_mels)


def count_params(model: nn.Module) -> tuple[int, int]:
    """(trainable parameters, trainable parameters with every BatchNorm folded into its convolution: -1 per BN channel)."""
    raw = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bn = sum(m.num_features for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    return raw, raw - bn


@torch.no_grad()
def profile(model: nn.Module, in_hw: tuple[int, int]) -> dict:
    """Parameters, MACs per inference and peak activation memory at int8 (largest conv/linear input + output, in KB)."""
    macs, peak, hooks = [0], [0], []

    def conv_hook(m, i, o):
        macs[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        peak[0] = max(peak[0], (i[0].numel() + o.numel()) // o.shape[0])

    def lin_hook(m, i, o):
        macs[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was = model.training
    model.eval()
    model(torch.zeros(1, 1, *in_hw))
    model.train(was)
    for h in hooks:
        h.remove()
    raw, folded = count_params(model)
    return dict(params=raw, params_folded=folded, macs=int(macs[0]), peak_act_kb=round(peak[0] / 1024, 1), weights_int8_kb=round(folded / 1024, 1))
