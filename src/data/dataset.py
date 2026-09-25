from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DRDataset(Dataset):
    """
    Dataset for diabetic retinopathy classification.

    Expected manifest columns:
        image_id
        image_path
        dataset
        label
        split
    """

    def __init__(self, manifest_path, transform=None):
        self.manifest_path = Path(manifest_path)
        self.transform = transform

        self.df = pd.read_csv(self.manifest_path)

        required_columns = {
            "image_id",
            "image_path",
            "dataset",
            "label",
            "split",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in manifest: {missing}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_id = row["image_id"]
        image_path = PROJECT_ROOT / row["image_path"]
        label = int(row["label"])
        dataset = row["dataset"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": image_id,
            "dataset": dataset,
        }