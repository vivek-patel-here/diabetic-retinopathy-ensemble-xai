from pathlib import Path
from tqdm import tqdm

import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)


class Trainer:
    """
    Generic training engine for multi-class image classification.

    This class is model-independent and can be reused for:
        - EfficientNet-B0
        - DenseNet121
        - ResNet18
        - MobileNetV2
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        scheduler=None,
        num_epochs=20,
        checkpoint_path=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.num_epochs = num_epochs

        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )

        if self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        # Best validation QWK seen so far.
        self.best_qwk = float("-inf")

        # Store training history.
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_macro_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "val_qwk": [],
        }


    # Training
    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        all_predictions = []
        all_labels = []

        progress_bar = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        )

        for batch in progress_bar:

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = (
            running_loss
            / len(self.train_loader.dataset)
        )

        accuracy = accuracy_score(
            all_labels,
            all_predictions,
        )

        macro_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )

        return {
            "loss": epoch_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

    # Validation
    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        all_predictions = []
        all_labels = []

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
        )

        for batch in progress_bar:

            images = batch["image"].to(
                self.device
            )

            labels = batch["label"].to(
                self.device
            )

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu()
                .numpy()
            )

            all_labels.extend(
                labels.cpu()
                .numpy()
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = (
            running_loss
            / len(self.val_loader.dataset)
        )

        accuracy = accuracy_score(
            all_labels,
            all_predictions,
        )

        macro_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )

        # Quadratic Weighted Kappa.
        qwk = cohen_kappa_score(
            all_labels,
            all_predictions,
            weights="quadratic",
        )

        return {
            "loss": epoch_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "qwk": qwk,
        }


    # Save checkpoint
    def save_checkpoint(
        self,
        epoch,
        val_metrics,
        filename,
    ):

        if self.checkpoint_path is None:
            return

        checkpoint = {
            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_macro_f1":
                val_metrics["macro_f1"],

            "val_qwk":
                val_metrics["qwk"],
        }

        if self.scheduler is not None:

            checkpoint[
                "scheduler_state_dict"
            ] = self.scheduler.state_dict()

        checkpoint_path = (
            self.checkpoint_path.parent
            / filename
        )

        torch.save(
            checkpoint,
            checkpoint_path,
        )

    # Full training
    def fit(self):

        print("\nStarting training...")

        print(
            f"Device: {self.device}"
        )

        print(
            f"Epochs: {self.num_epochs}"
        )

        for epoch in range(
            1,
            self.num_epochs + 1,
        ):

            print(
                f"\nEpoch "
                f"{epoch}/{self.num_epochs}"
            )

            # -------------------------------------------------
            # Train
            # -------------------------------------------------

            train_metrics = (
                self.train_one_epoch()
            )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            val_metrics = (
                self.validate()
            )

            # -------------------------------------------------
            # Scheduler
            # -------------------------------------------------

            if self.scheduler is not None:

                self.scheduler.step()

            # -------------------------------------------------
            # Store history
            # -------------------------------------------------

            self.history[
                "train_loss"
            ].append(
                train_metrics["loss"]
            )

            self.history[
                "train_accuracy"
            ].append(
                train_metrics["accuracy"]
            )

            self.history[
                "train_macro_f1"
            ].append(
                train_metrics["macro_f1"]
            )

            self.history[
                "val_loss"
            ].append(
                val_metrics["loss"]
            )

            self.history[
                "val_accuracy"
            ].append(
                val_metrics["accuracy"]
            )

            self.history[
                "val_macro_f1"
            ].append(
                val_metrics["macro_f1"]
            )

            self.history[
                "val_qwk"
            ].append(
                val_metrics["qwk"]
            )

            # Print results
            print(
                f"Train Loss: "
                f"{train_metrics['loss']:.4f}"
            )

            print(
                f"Train Accuracy: "
                f"{train_metrics['accuracy']:.4f}"
            )

            print(
                f"Train Macro F1: "
                f"{train_metrics['macro_f1']:.4f}"
            )

            print(
                f"Val Loss: "
                f"{val_metrics['loss']:.4f}"
            )

            print(
                f"Val Accuracy: "
                f"{val_metrics['accuracy']:.4f}"
            )

            print(
                f"Val Macro F1: "
                f"{val_metrics['macro_f1']:.4f}"
            )

            print(
                f"Val QWK: "
                f"{val_metrics['qwk']:.4f}"
            )

            # Save checkpoint after EVERY epoch
            self.save_checkpoint(
                epoch,
                val_metrics,
                f"epoch_{epoch}.pth",
            )

            print(
                f"✓ Epoch {epoch} checkpoint saved"
            )

            # Save BEST model
            if (
                val_metrics["qwk"]
                > self.best_qwk
            ):

                self.best_qwk = (
                    val_metrics["qwk"]
                )

                self.save_checkpoint(
                    epoch,
                    val_metrics,
                    "best.pth",
                )

                print(
                    "✓ New best model saved"
                )

        print("\nTraining finished.")

        print(
            f"Best validation QWK: "
            f"{self.best_qwk:.4f}"
        )

        return self.history