
import os
import re
import shutil
import random
from pathlib import Path

import cv2


# ============================================================
# 1. PATHS
# ============================================================

# Change this only if your dataset is somewhere else.
SOURCE_DIR = Path("Dataset_BUSI_with_GT")
OUTPUT_DIR = Path("BUSI_processed")


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = (256, 256)

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# 3. FIND IMAGE/MASK PAIRS
# ============================================================

def get_image_id(filename):
    """
    Convert a BUSI filename into its base image ID.

    Examples:
        benign (100).png
            -> benign (100)

        benign (100)_mask.png
            -> benign (100)

        benign (100)_mask_1.png
            -> benign (100)
    """

    name = Path(filename).stem

    # Remove _mask, _mask_1, _mask_2, etc.
    name = re.sub(r"_mask(?:_\d+)?$", "", name)

    return name


def collect_samples():
    """
    Find all BUSI lesion images and their masks.

    We use only:
        benign/
        malignant/

    The normal/ directory is intentionally ignored.
    """

    samples = []

    for class_name in ["benign", "malignant"]:

        class_dir = SOURCE_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Could not find directory: {class_dir}"
            )

        # Find every PNG file that is NOT a mask.
        image_files = [
            f for f in class_dir.glob("*.png")
            if "_mask" not in f.stem
        ]

        for image_path in image_files:

            image_id = get_image_id(image_path.name)

            # Find all masks belonging to this image.
            mask_files = sorted(
                class_dir.glob(f"{image_id}_mask*.png")
            )

            if len(mask_files) == 0:
                print(
                    f"WARNING: No mask found for {image_path.name}"
                )
                continue

            samples.append({
                "class": class_name,
                "image": image_path,
                "masks": mask_files
            })

    return samples


# ============================================================
# 4. COMBINE MULTIPLE MASKS
# ============================================================

def load_and_combine_masks(mask_paths):
    """
    Load all masks for one image and combine them.

    If an image has:

        image.png
        image_mask.png
        image_mask_1.png

    the masks are combined using a logical OR.

    The final result is one binary mask:
        background = 0
        lesion     = 255
    """

    combined_mask = None

    for mask_path in mask_paths:

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise ValueError(
                f"Could not read mask: {mask_path}"
            )

        # Convert mask to binary.
        mask = (mask > 0).astype("uint8") * 255

        if combined_mask is None:
            combined_mask = mask
        else:
            # Logical OR.
            combined_mask = cv2.bitwise_or(
                combined_mask,
                mask
            )

    return combined_mask


# ============================================================
# 5. CREATE OUTPUT DIRECTORIES
# ============================================================

def create_output_directories():

    splits = ["train", "val", "test"]

    for split in splits:

        image_dir = OUTPUT_DIR / split / "images"
        mask_dir = OUTPUT_DIR / split / "masks"

        image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        mask_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 6. SPLIT DATA
# ============================================================

def split_samples(samples):

    random.seed(RANDOM_SEED)

    # Shuffle a copy so the original list isn't modified.
    samples = samples.copy()

    random.shuffle(samples)

    total = len(samples)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_samples = samples[:train_count]

    val_samples = samples[
        train_count:
        train_count + val_count
    ]

    test_samples = samples[
        train_count + val_count:
    ]

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }


# ============================================================
# 7. PROCESS ONE IMAGE
# ============================================================

def process_sample(sample, split, index):

    image_path = sample["image"]
    mask_paths = sample["masks"]

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # --------------------------------------------------------
    # Resize image
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Load and combine masks
    # --------------------------------------------------------

    mask = load_and_combine_masks(mask_paths)

    # --------------------------------------------------------
    # Resize mask
    #
    # IMPORTANT:
    # Use nearest-neighbor interpolation for masks.
    # --------------------------------------------------------

    mask = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    # Make absolutely sure the mask is binary.
    mask = (mask > 0).astype("uint8") * 255

    # --------------------------------------------------------
    # Give the processed file a clean name.
    #
    # Example:
    # benign_00001.png
    # benign_00001_mask.png
    # --------------------------------------------------------

    class_name = sample["class"]

    output_name = f"{class_name}_{index:05d}.png"

    output_image_path = (
        OUTPUT_DIR
        / split
        / "images"
        / output_name
    )

    output_mask_path = (
        OUTPUT_DIR
        / split
        / "masks"
        / f"{class_name}_{index:05d}_mask.png"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        str(output_image_path),
        image
    )

    cv2.imwrite(
        str(output_mask_path),
        mask
    )


# ============================================================
# 8. PROCESS ENTIRE DATASET
# ============================================================

def process_dataset(splits):

    for split_name, samples in splits.items():

        print()
        print(f"Processing {split_name}: {len(samples)} images")

        for index, sample in enumerate(samples, start=1):

            process_sample(
                sample,
                split_name,
                index
            )

        print(
            f"Finished {split_name}"
        )


# ============================================================
# 9. VERIFY DATASET
# ============================================================

def verify_dataset(splits):

    print()
    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    total = 0

    for split_name, samples in splits.items():

        image_dir = (
            OUTPUT_DIR
            / split_name
            / "images"
        )

        mask_dir = (
            OUTPUT_DIR
            / split_name
            / "masks"
        )

        image_count = len(
            list(image_dir.glob("*.png"))
        )

        mask_count = len(
            list(mask_dir.glob("*.png"))
        )

        print(
            f"{split_name:5s}: "
            f"{image_count} images, "
            f"{mask_count} masks"
        )

        if image_count != mask_count:
            raise ValueError(
                f"Mismatch in {split_name}: "
                f"{image_count} images vs "
                f"{mask_count} masks"
            )

        total += image_count

    print("-" * 60)
    print(f"Total: {total} images")

    if total != len(
        [s for samples in splits.values() for s in samples]
    ):
        raise ValueError(
            "Processed image count does not match "
            "the number of samples."
        )

    print("Verification passed!")


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BUSI PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Check source directory
    # --------------------------------------------------------

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{SOURCE_DIR.resolve()}\n\n"
            "Make sure you run this script from the directory "
            "containing Dataset_BUSI_with_GT."
        )

    print()
    print(f"Source: {SOURCE_DIR.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")

    # --------------------------------------------------------
    # Find samples
    # --------------------------------------------------------

    print()
    print("Finding BUSI images and masks...")

    samples = collect_samples()

    print(
        f"Found {len(samples)} lesion images."
    )

    # --------------------------------------------------------
    # Check expected number
    # --------------------------------------------------------

    if len(samples) != 647:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected 647 lesion images, "
            f"but found {len(samples)}."
        )

        print(
            "Please check the dataset before continuing."
        )

        return

    # --------------------------------------------------------
    # Count classes
    # --------------------------------------------------------

    benign_count = sum(
        1
        for sample in samples
        if sample["class"] == "benign"
    )

    malignant_count = sum(
        1
        for sample in samples
        if sample["class"] == "malignant"
    )

    print()
    print("Class counts:")
    print(f"  Benign:    {benign_count}")
    print(f"  Malignant: {malignant_count}")
    print(f"  Total:     {len(samples)}")

    # --------------------------------------------------------
    # Check multiple masks
    # --------------------------------------------------------

    multiple_mask_samples = [
        sample
        for sample in samples
        if len(sample["masks"]) > 1
    ]

    print()
    print(
        f"Images with multiple masks: "
        f"{len(multiple_mask_samples)}"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_output_directories()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    print()
    print("Creating 80/10/10 split...")

    splits = split_samples(samples)

    print(
        f"Train: {len(splits['train'])}"
    )

    print(
        f"Validation: {len(splits['val'])}"
    )

    print(
        f"Test: {len(splits['test'])}"
    )


    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    process_dataset(splits)

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_dataset(splits)

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print()
    print("Processed dataset saved at:")

    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()