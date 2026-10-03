python
import torch
from torch.utils.data import DataLoader

from dataset_busi import BUSIDataset
from unet_lomix import UNetLoMix


# --------------------------------------------------
# Settings
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4

TEST_DIR = "BUSI_processed/test"

MODEL_PATH = "busi_baseline_epoch_55.pth"


# --------------------------------------------------
# Dice score
# --------------------------------------------------

def dice_score(pred, target):

    intersection = (pred * target).sum()

    dice = (
        (2 * intersection + 1e-6)
        /
        (pred.sum() + target.sum() + 1e-6)
    )

    return dice


# --------------------------------------------------
# IoU score
# --------------------------------------------------

def iou_score(pred, target):

    intersection = (pred * target).sum()

    union = (
        pred.sum()
        + target.sum()
        - intersection
    )

    iou = (
        (intersection + 1e-6)
        /
        (union + 1e-6)
    )

    return iou


# --------------------------------------------------
# Dataset
# --------------------------------------------------

test_dataset = BUSIDataset(TEST_DIR)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNetLoMix(
    n_channels=1,
    n_classes=1
).to(DEVICE)


# --------------------------------------------------
# Load trained checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded model from epoch:", checkpoint["epoch"])
print("Device:", DEVICE)
print("Test images:", len(test_dataset))


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

total_dice = 0.0
total_iou = 0.0
total_images = 0


with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # U-Net produces p1, p2, p3, p4
        outputs = model(images)

        # Standard U-Net uses ONLY the final output
        logits = outputs[3]

        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Convert probabilities to binary masks
        predictions = (
            probabilities > 0.5
        ).float()

        # Calculate metrics for every image
        for i in range(images.size(0)):

            dice = dice_score(
                predictions[i],
                masks[i]
            )

            iou = iou_score(
                predictions[i],
                masks[i]
            )

            total_dice += dice.item()
            total_iou += iou.item()

            total_images += 1


# --------------------------------------------------
# Final results
# --------------------------------------------------

average_dice = (
    total_dice / total_images
)

average_iou = (
    total_iou / total_images
)


print()
print("===================================")
print("Standard U-Net Evaluation")
print("===================================")
print(f"Dice: {average_dice:.4f}")
print(f"IoU : {average_iou:.4f}")
print("===================================")
