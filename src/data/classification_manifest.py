from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "classification"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
VAL_RATIO = 0.15


# ============================================================
# APTOS
# ============================================================

def load_aptos():

    aptos_dir = RAW_DIR / "aptos"

    rows = []

    split_files = {
        "train": "train_1.csv",
        "val": "valid.csv",
        "test": "test.csv",
    }

    image_dirs = {
        "train": aptos_dir / "train_images" / "train_images",
        "val": aptos_dir / "val_images" / "val_images",
        "test": aptos_dir / "test_images" / "test_images",
    }

    for split, csv_name in split_files.items():

        df = pd.read_csv(aptos_dir / csv_name)

        for _, row in df.iterrows():

            image_id = row["id_code"]
            label = int(row["diagnosis"])

            image_path = image_dirs[split] / f"{image_id}.png"

            rows.append({
                "image_id": f"aptos_{image_id}",
                "image_path": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "dataset": "aptos",
                "label": label,
                "split": split,
            })

    return pd.DataFrame(rows)


# ============================================================
# EyePACS
# ============================================================

def load_eyepacs():

    eyepacs_dir = RAW_DIR / "eyepac"

    df = pd.read_csv(
        eyepacs_dir / "trainLabels.csv"
    )

    label_folders = {
        0: "No_DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferate_DR",
    }

    rows = []

    for _, row in df.iterrows():

        image_id = row["image"]
        label = int(row["level"])

        image_path = (
            eyepacs_dir
            / "colored_images"
            / "colored_images"
            / label_folders[label]
            / f"{image_id}.png"
        )

        rows.append({
            "image_id": f"eyepacs_{image_id}",
            "image_path": str(
                image_path.relative_to(PROJECT_ROOT)
            ),
            "dataset": "eyepacs",
            "label": label,
        })

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Create our own stratified train/validation split
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_RATIO,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"

    return pd.concat(
        [train_df, val_df],
        ignore_index=True
    )


# ============================================================
# IDRiD
# ============================================================

def load_idrid_test():

    idrid_dir = RAW_DIR / "idrid"

    csv_path = (
        idrid_dir
        / "B. Disease Grading"
        / "2. Groundtruths"
        / "b. IDRiD_Disease Grading_Testing Labels.csv"
    )

    image_dir = (
        idrid_dir
        / "B. Disease Grading"
        / "1. Original Images"
        / "b. Testing Set"
    )

    df = pd.read_csv(csv_path)

    rows = []

    for _, row in df.iterrows():

        image_id = row["Image name"]
        label = int(row["Retinopathy grade"])

        image_path = image_dir / f"{image_id}.jpg"

        rows.append({
            "image_id": f"idrid_{image_id}",
            "image_path": str(
                image_path.relative_to(PROJECT_ROOT)
            ),
            "dataset": "idrid",
            "label": label,
            "split": "external_test",
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    print("Loading APTOS...")
    aptos = load_aptos()

    print("Loading EyePACS...")
    eyepacs = load_eyepacs()

    print("Loading IDRiD...")
    idrid = load_idrid_test()

    # --------------------------------------------------------
    # APTOS
    # --------------------------------------------------------

    aptos_train = aptos[
        aptos["split"] == "train"
    ]

    aptos_val = aptos[
        aptos["split"] == "val"
    ]

    # APTOS test is not used initially because
    # IDRiD will serve as our external test set.
    # We retain it separately for possible later experiments.

    aptos_test = aptos[
        aptos["split"] == "test"
    ]

    # --------------------------------------------------------
    # EyePACS
    # --------------------------------------------------------

    eyepacs_train = eyepacs[
        eyepacs["split"] == "train"
    ]

    eyepacs_val = eyepacs[
        eyepacs["split"] == "val"
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_df = pd.concat(
        [
            aptos_train,
            eyepacs_train,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_df = pd.concat(
        [
            aptos_val,
            eyepacs_val,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # External test
    # --------------------------------------------------------

    external_test_df = idrid

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    val_df.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
    )

    external_test_df.to_csv(
        OUTPUT_DIR / "external_test.csv",
        index=False,
    )

    # Keep APTOS test separately for later comparison.
    aptos_test.to_csv(
        OUTPUT_DIR / "aptos_test.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n========================================")
    print("MANIFEST SUMMARY")
    print("========================================")

    print("\nTrain:")
    print(len(train_df))
    print(train_df.groupby(["dataset", "label"]).size())

    print("\nValidation:")
    print(len(val_df))
    print(val_df.groupby(["dataset", "label"]).size())

    print("\nExternal Test (IDRiD):")
    print(len(external_test_df))
    print(
        external_test_df
        .groupby(["dataset", "label"])
        .size()
    )

    print("\nFiles created:")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "val.csv")
    print(OUTPUT_DIR / "external_test.csv")
    print(OUTPUT_DIR / "aptos_test.csv")


if __name__ == "__main__":
    main()