from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import DRDataset
from src.data.transform import (
    get_train_transform,
    get_eval_transform,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def main():

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = DRDataset(
        TRAIN_MANIFEST,
        transform=get_train_transform(),
    )

    val_dataset = DRDataset(
        VAL_MANIFEST,
        transform=get_eval_transform(),
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Inspect one batch
    # --------------------------------------------------------

    batch = next(iter(train_loader))

    print("TRAIN DATASET")
    print("----------------------------")

    print("Number of samples:", len(train_dataset))

    print("Image shape:", batch["image"].shape)

    print("Label shape:", batch["label"].shape)

    print("Labels:", batch["label"])

    print("Image IDs:")
    print(batch["image_id"][:5])

    print("Datasets:")
    print(batch["dataset"][:5])

    print("\nVAL DATASET")
    print("----------------------------")

    print("Number of samples:", len(val_dataset))


if __name__ == "__main__":
    main()