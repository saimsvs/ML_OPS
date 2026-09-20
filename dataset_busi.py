from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BUSIDataset(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = Path(root_dir)
        self.split = split

        self.image_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"

        self.images = sorted(self.image_dir.glob("*.png"))

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        print(
            f"{split}: {len(self.images)} images found"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        # Find corresponding mask
        mask_path = self.mask_dir / f"{image_path.stem}_mask.png"

        # Read grayscale image
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        # Read grayscale mask
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        # Convert image from:
        # [H, W]
        # to:
        # [1, H, W]
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # Convert mask to binary
        mask = (mask > 127).astype(np.float32)

        # Convert to PyTorch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask


if __name__ == "__main__":

    dataset_root = "/Users/saimasad/Desktop/ML_PROJ/BUSI_processed"

    train_dataset = BUSIDataset(
        dataset_root,
        "train"
    )

    val_dataset = BUSIDataset(
        dataset_root,
        "val"
    )

    test_dataset = BUSIDataset(
        dataset_root,
        "test"
    )

    print()
    print("Dataset test")
    print("-------------------------")

    image, mask = train_dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)

    print("Image min:", image.min().item())
    print("Image max:", image.max().item())

    print("Mask unique values:", torch.unique(mask))

    print()
    print("Dataset test completed successfully.")