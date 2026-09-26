from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset
from src.data.transform import (
    get_train_transform,
    get_eval_transform,
)
from src.models.EfficientNetB0.model import (
    create_efficientnet_b0,
)
from src.training.trainer import Trainer
from src.utils.device import get_device



# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "train.csv"
)

VAL_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "val.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "EfficientNetB0"
    / "checkpoint"
    / "best.pth"
)


# Configuration
BATCH_SIZE = 16
NUM_EPOCHS = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 5

NUM_WORKERS = 0


# Class weights
def get_class_weights(manifest_path):

    df = pd.read_csv(manifest_path)

    labels = df["label"].values

    classes = df["label"].unique()
    classes.sort()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    return weights



# Main
def main():

    # Device
    device = get_device()

    print("Device:", device)

    # Datasets
    print("\nLoading datasets...")

    train_dataset = DRDataset(
        TRAIN_MANIFEST,
        transform=get_train_transform(),
    )

    val_dataset = DRDataset(
        VAL_MANIFEST,
        transform=get_eval_transform(),
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )


    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )


    # Model
    print("\nCreating EfficientNet-B0...")

    model = create_efficientnet_b0(
        num_classes=NUM_CLASSES,
        pretrained=True,
    )

    model = model.to(device)


    # Class weights
    class_weights = get_class_weights(
        TRAIN_MANIFEST
    )

    print("\nClass weights:")
    for i, weight in enumerate(class_weights):
        print(
            f"Class {i}: {weight:.4f}"
        )

    class_weights = class_weights.to(device)


    # Loss
    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Learning-rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )


    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
    )


    # Train
    history = trainer.fit()

    print("\nTraining complete.")

    print(
        "\nBest validation QWK:",
        trainer.best_qwk,
    )


if __name__ == "__main__":
    main()