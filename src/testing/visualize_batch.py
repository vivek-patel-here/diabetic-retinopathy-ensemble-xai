import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset
from src.data.transform import get_eval_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "val.csv"
)


LABEL_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


def main():

    dataset = DRDataset(
        MANIFEST,
        transform=get_eval_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    images = batch["image"]
    labels = batch["label"]
    image_ids = batch["image_id"]
    datasets = batch["dataset"]

    # Undo ImageNet normalization for visualization
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    images = images * std + mean
    images = images.clamp(0, 1)

    n = min(16, len(images))

    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 12),
    )

    axes = axes.flatten()

    for i in range(n):

        image = images[i].permute(
            1, 2, 0
        ).numpy()

        label = labels[i].item()

        axes[i].imshow(image)

        axes[i].set_title(
            f"{datasets[i]}\n"
            f"{LABEL_NAMES[label]}\n"
            f"{image_ids[i]}",
            fontsize=8,
        )

        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()