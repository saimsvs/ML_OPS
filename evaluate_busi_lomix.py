
import torch
from torch.utils.data import DataLoader

from unet_lomix import UNetLoMix
from dataset_busi import BUSIDataset


# ============================================================
# 1. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# 2. DATASET
# ============================================================

DATASET_PATH = "/Users/saimasad/Desktop/ML_PROJ/BUSI_processed"

test_dataset = BUSIDataset(
    DATASET_PATH,
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)

print("test:", len(test_dataset), "images found")


# ============================================================
# 3. CREATE MODEL
# ============================================================

model = UNetLoMix(
    n_channels=1,
    n_classes=1
).to(device)


# ============================================================
# 4. LOAD BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    "busi_lomix_best.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Loaded best model from epoch:",
    checkpoint["epoch"]
)

print(
    "Training loss:",
    checkpoint["train_loss"]
)

print(
    "Validation loss:",
    checkpoint["val_loss"]
)


# ============================================================
# 5. DICE FUNCTION
# ============================================================

def calculate_dice(prediction, target, smooth=1e-5):

    prediction = prediction.reshape(
        prediction.shape[0], -1
    )

    target = target.reshape(
        target.shape[0], -1
    )

    intersection = (
        prediction * target
    ).sum(dim=1)

    dice = (
        2 * intersection + smooth
    ) / (
        prediction.sum(dim=1)
        + target.sum(dim=1)
        + smooth
    )

    return dice


# ============================================================
# 6. IoU FUNCTION
# ============================================================

def calculate_iou(prediction, target, smooth=1e-5):

    prediction = prediction.reshape(
        prediction.shape[0], -1
    )

    target = target.reshape(
        target.shape[0], -1
    )

    intersection = (
        prediction * target
    ).sum(dim=1)

    union = (
        prediction.sum(dim=1)
        + target.sum(dim=1)
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou


# ============================================================
# 7. TEST
# ============================================================

total_dice = 0.0
total_iou = 0.0

total_images = 0


with torch.no_grad():

    for batch_idx, (images, masks) in enumerate(test_loader):

        images = images.to(device)
        masks = masks.to(device)

        # ----------------------------------------------------
        # Dataset mask:
        # [B, 256, 256]
        #
        # Model output:
        # [B, 1, 256, 256]
        #
        # Add channel dimension.
        # ----------------------------------------------------

        masks = masks.unsqueeze(1)

        # ----------------------------------------------------
        # U-Net produces 4 predictions
        # ----------------------------------------------------

        predictions = model(images)

        # ----------------------------------------------------
        # For inference we use the FINAL decoder prediction.
        #
        # LoMix is a training/supervision method and is not
        # required during inference.
        # ----------------------------------------------------

        final_prediction = predictions[-1]

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            final_prediction
        )

        # ----------------------------------------------------
        # Convert probabilities to binary masks
        # ----------------------------------------------------

        predicted_masks = (
            probabilities >= 0.5
        ).float()

        # ----------------------------------------------------
        # Calculate Dice
        # ----------------------------------------------------

        dice_scores = calculate_dice(
            predicted_masks,
            masks
        )

        # ----------------------------------------------------
        # Calculate IoU
        # ----------------------------------------------------

        iou_scores = calculate_iou(
            predicted_masks,
            masks
        )

        # ----------------------------------------------------
        # Add batch results
        # ----------------------------------------------------

        batch_size = images.shape[0]

        total_dice += dice_scores.sum().item()
        total_iou += iou_scores.sum().item()

        total_images += batch_size

        print(
            f"Batch {batch_idx + 1}/{len(test_loader)} "
            f"Dice: {dice_scores.mean().item():.4f} "
            f"IoU: {iou_scores.mean().item():.4f}"
        )


# ============================================================
# 8. FINAL RESULTS
# ============================================================

average_dice = total_dice / total_images
average_iou = total_iou / total_images


print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test images: {total_images}"
)

print(
    f"Dice: {average_dice:.4f}"
)

print(
    f"IoU:  {average_iou:.4f}"
)

print("=" * 60)

