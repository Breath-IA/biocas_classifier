"""
train.py
─────────
Loop de entrenamiento + búsqueda de hiperparámetros con Optuna + logging en WandB.

Cada trial de Optuna:
  • elige una arquitectura del MODEL_REGISTRY
  • sugiere lr, weight_decay, batch_size, dropout (dentro de cada modelo)
  • crea un run de WandB asociado al trial
  • loguea loss y accuracy de train y validación por época
  • loguea una imagen aleatoria de validación con label verdadera vs predicha

Uso
───
    # Búsqueda con Optuna (N trials, maximizando val_acc)
    python train.py --n-trials 20 --n-epochs 30 \
                    --train-dir data/processed/train \
                    --test-dir  data/processed/test  \
                    --wandb-project biocas

    # Un solo run sin búsqueda (trial=None, usa defaults)
    python train.py --no-optuna --n-epochs 100 \
                    --model efficientnet_b0

    # Ver todos los argumentos
    python train.py --help
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importa el registro y todos los modelos (el import registra las clases)
from models.base import MODEL_REGISTRY
import models.networks  # noqa: F401 — side-effect: registra EfficientNet, ResNet, ...

from pipeline_api import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Un epoch de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    n_epochs: int,
) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Train {epoch}/{n_epochs}", leave=False)
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
  
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%",
        })

    return {
        "train/loss": total_loss / total,
        "train/acc": 100.0 * correct / total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Un epoch de validación
# ─────────────────────────────────────────────────────────────────────────────

def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    n_epochs: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Val   {epoch}/{n_epochs}", leave=False)
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return {
        "val/loss": total_loss / total,
        "val/acc": 100.0 * correct / total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Log de imagen de validación
# ─────────────────────────────────────────────────────────────────────────────

# Nombres de las 7 clases según LABEL_MAP_1_2
CLASS_NAMES = {0: "Normal", 1: "Rhonchi", 2: "Wheeze",
               3: "Stridor", 4: "Coarse Crackle",
               5: "Fine Crackle", 6: "Wheeze+Crackle"}


def log_val_image(
    model: nn.Module,
    val_dataset,
    device: torch.device,
    epoch: int,
    n_samples: int = 4,
) -> None:
    """
    Elige `n_samples` muestras aleatorias del set de validación,
    predice sus etiquetas y las loguea en WandB como tabla de imágenes.

    El espectrograma (n_mels × T) se visualiza con matplotlib y se pasa
    como `wandb.Image` con caption "true: X | pred: Y".
    """
    import matplotlib.pyplot as plt

    model.eval()
    indices = random.sample(range(len(val_dataset)), min(n_samples, len(val_dataset)))

    wandb_images = []
    with torch.no_grad():
        for idx in indices:
            feature, true_label = val_dataset[idx]          # (1, n_mels, T), scalar
            inp = feature.unsqueeze(0).to(device)            # (1, 1, n_mels, T)
            logits = model(inp)
            pred_label = int(torch.argmax(logits, dim=1).item())

            # Visualizar el espectrograma
            spec = feature.squeeze(0).cpu().numpy()          # (n_mels, T)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
            ax.set_xlabel("Time frames")
            ax.set_ylabel("Mel bins")
            true_name = CLASS_NAMES.get(int(true_label), str(int(true_label)))
            pred_name = CLASS_NAMES.get(pred_label, str(pred_label))
            ax.set_title(f"true: {true_name}  |  pred: {pred_name}")
            fig.tight_layout()

            wandb_images.append(
                wandb.Image(fig, caption=f"true={true_name} | pred={pred_name}")
            )
            plt.close(fig)

    wandb.log({"val/sample_predictions": wandb_images, "epoch": epoch})


# ─────────────────────────────────────────────────────────────────────────────
# Objetivo Optuna (un trial = un run de WandB)
# ─────────────────────────────────────────────────────────────────────────────

def objective(
    trial: Optional[optuna.Trial],
    train_dataset,
    val_dataset,
    args: argparse.Namespace,
) -> float:
    """
    Entrena un modelo durante `args.n_epochs` épocas y devuelve la val_acc
    de la mejor época (métrica que Optuna maximiza).

    Si trial=None se usan los defaults y solo se hace un run simple.
    """
    device = resolve_device(args.device)
    seed_everything(args.seed)

    # ── Hiperparámetros sugeridos por Optuna (o defaults) ─────────────────
    if trial is not None:
        model_name = trial.suggest_categorical(
            "model", list(MODEL_REGISTRY.keys())
        )
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    else:
        model_name = args.model
        lr = args.lr
        weight_decay = args.weight_decay
        batch_size = args.batch_size

    # ── Construir modelo ──────────────────────────────────────────────────
    num_classes = train_dataset.num_classes
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls.build(trial, num_classes=num_classes, in_channels=1)
    model = model.to(device)

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Loss, optimizer, scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=train_dataset.class_weights.to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs
    )

    # ── WandB run ─────────────────────────────────────────────────────────
    run_config = {
        "model": model_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "n_epochs": args.n_epochs,
        "num_classes": num_classes,
        "n_params": model.n_parameters(),
        "trial": trial.number if trial else None,
    }
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        config=run_config,
        reinit=True,
    )
    wandb.watch(model, log="gradients", log_freq=100)

    # ── Loop de entrenamiento ─────────────────────────────────────────────
    best_val_acc = 0.0

    for epoch in range(1, args.n_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.n_epochs
        )
        val_metrics = val_epoch(
            model, val_loader, criterion, device, epoch, args.n_epochs
        )
        scheduler.step()

        # Loguear métricas numéricas
        wandb.log({
            **train_metrics,
            **val_metrics,
            "lr": scheduler.get_last_lr()[0],
            "epoch": epoch,
        })

        # Loguear imágenes de validación cada N épocas
        if epoch % args.log_image_every == 0 or epoch == args.n_epochs:
            log_val_image(model, val_dataset, device, epoch, n_samples=4)

        val_acc = val_metrics["val/acc"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Pruning de Optuna: si el trial va mal, cortarlo temprano
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                run.finish(exit_code=1)
                raise optuna.exceptions.TrialPruned()

        print(
            f"Epoch {epoch:3d}/{args.n_epochs} | "
            f"train_loss={train_metrics['train/loss']:.4f} "
            f"train_acc={train_metrics['train/acc']:.2f}% | "
            f"val_loss={val_metrics['val/loss']:.4f} "
            f"val_acc={val_acc:.2f}%"
        )

    run.finish()
    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena un clasificador de sonidos respiratorios (BioCAS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Datos
    p.add_argument("--train-dir", default="data/processed/train",
                   help="Directorio del dataset de entrenamiento (index.csv + features/)")
    p.add_argument("--test-dir",  default="data/processed/test",
                   help="Directorio del dataset de test/validación")

    # Entrenamiento
    p.add_argument("--n-epochs",     type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=256,
                   help="Ignorado si Optuna está activo (lo sugiere por trial)")
    p.add_argument("--lr",           type=float, default=1e-3,
                   help="Ignorado si Optuna está activo")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Ignorado si Optuna está activo")
    p.add_argument("--model",        default="efficientnet_b0",
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Ignorado si Optuna está activo")
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--device",       default="auto",
                   help="'auto' | 'cpu' | 'cuda' | 'cuda:0'")
    p.add_argument("--seed",         type=int,   default=42)

    # Optuna
    p.add_argument("--no-optuna",   action="store_true",
                   help="Correr un solo run sin búsqueda de hiperparámetros")
    p.add_argument("--n-trials",    type=int, default=20,
                   help="Número de trials de Optuna")
    p.add_argument("--study-name",  default="biocas_study",
                   help="Nombre del estudio de Optuna")

    # WandB
    p.add_argument("--wandb-project", default="biocas",
                   help="Nombre del proyecto en WandB")
    p.add_argument("--wandb-group",   default=None,
                   help="Grupo de runs en WandB (por defecto None)")
    p.add_argument("--log-image-every", type=int, default=10,
                   help="Loguear imágenes de validación cada N épocas")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Cargar datasets
    print(f"Cargando train desde: {args.train_dir}")
    train_dataset = load_dataset(args.train_dir)
    print(f"Cargando test desde:  {args.test_dir}")
    val_dataset   = load_dataset(args.test_dir)

    print(f"  Train: {train_dataset}")
    print(f"  Val:   {val_dataset}")
    print(f"  Device: {resolve_device(args.device)}")
    print(f"  Modelos disponibles: {list(MODEL_REGISTRY.keys())}")

    if args.no_optuna:
        # ── Run único sin búsqueda ─────────────────────────────────────────
        print(f"\nRun único | model={args.model} | epochs={args.n_epochs}")
        best_acc = objective(None, train_dataset, val_dataset, args)
        print(f"\nBest val acc: {best_acc:.2f}%")

    else:
        # ── Búsqueda con Optuna ────────────────────────────────────────────
        # MedianPruner: poda trials que vayan por debajo de la mediana
        # de los trials anteriores en esa época.
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",    # maximizar val/acc
            pruner=pruner,
        )

        study.optimize(
            lambda trial: objective(trial, train_dataset, val_dataset, args),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        # Resumen final
        best = study.best_trial
        print("\n" + "=" * 50)
        print(f"Mejor trial: #{best.number}")
        print(f"  val/acc  : {best.value:.2f}%")
        print(f"  Params   : {best.params}")
        print("=" * 50)