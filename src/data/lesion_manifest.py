from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "idrid"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lesion"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Segmentation directories
# ============================================================

SEGMENTATION_ROOT = (
    RAW_DIR
    / "A. Segmentation"
)

IMAGE_ROOT = (
    SEGMENTATION_ROOT
    / "1. Original Images"
)

MASK_ROOT = (
    SEGMENTATION_ROOT
    / "2. All Segmentation Groundtruths"
)


# ============================================================
# Build mask lookup
# ============================================================

def build_mask_lookup():

    mask_types = {
        "ma_mask": (
            "1. Microaneurysms",
            "_MA.tif",
        ),
        "he_mask": (
            "2. Haemorrhages",
            "_HE.tif",
        ),
        "ex_mask": (
            "3. Hard Exudates",
            "_EX.tif",
        ),
        "se_mask": (
            "4. Soft Exudates",
            "_SE.tif",
        ),
    }

    lookup = {}

    for split_folder in [
        "a. Training Set",
        "b. Testing Set",
    ]:

        for column, (folder, suffix) in mask_types.items():

            directory = (
                MASK_ROOT
                / split_folder
                / folder
            )

            if not directory.exists():
                continue

            for mask_path in directory.glob("*.tif"):

                image_id = mask_path.stem.replace(
                    suffix.replace(".tif", ""),
                    ""
                )

                if image_id not in lookup:
                    lookup[image_id] = {}

                lookup[image_id][column] = str(
                    mask_path.relative_to(PROJECT_ROOT)
                )

    return lookup


# ============================================================
# Build image lookup
# ============================================================

def build_image_lookup():

    lookup = {}

    for split_folder in [
        "a. Training Set",
        "b. Testing Set",
    ]:

        directory = (
            IMAGE_ROOT
            / split_folder
        )

        for image_path in directory.glob("*.jpg"):

            image_id = image_path.stem

            lookup[image_id] = {
                "image_path": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "split": (
                    "train"
                    if split_folder == "a. Training Set"
                    else "test"
                ),
            }

    return lookup


# ============================================================
# Main
# ============================================================

def main():

    image_lookup = build_image_lookup()
    mask_lookup = build_mask_lookup()

    rows = []

    for image_id, image_info in image_lookup.items():

        masks = mask_lookup.get(
            image_id,
            {}
        )

        rows.append({
            "image_id": image_id,
            "image_path": image_info["image_path"],
            "split": image_info["split"],
            "ma_mask": masks.get("ma_mask"),
            "he_mask": masks.get("he_mask"),
            "ex_mask": masks.get("ex_mask"),
            "se_mask": masks.get("se_mask"),
        })

    manifest = pd.DataFrame(rows)

    output_path = (
        OUTPUT_DIR
        / "idrid_lesion.csv"
    )

    manifest.to_csv(
        output_path,
        index=False,
    )

    print("\nLesion manifest created:")
    print(output_path)

    print("\nTotal IDRiD segmentation images:")
    print(len(manifest))

    print("\nMask availability:")

    for column in [
        "ma_mask",
        "he_mask",
        "ex_mask",
        "se_mask",
    ]:
        print(
            f"{column}: "
            f"{manifest[column].notna().sum()}"
        )


if __name__ == "__main__":
    main()