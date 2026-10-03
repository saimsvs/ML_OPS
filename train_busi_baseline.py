import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_busi import BUSIDataset
from unet_lomix import UNetLoMix


# --------------------------------------------------
# Settings
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4

TRAIN_DIR = "BUSI_processed/train"
VAL_DIR = "BUSI_processed/val"

SAVE_PATH = "busi_baseline_best.pth"


# --------------------------------------------------
# Dice loss
# --------------------------------------------------

def dice_loss(logits, targets):

    probs = torch.sigmoid(logits)

    smooth = 1e-6

    intersection = (probs * targets).sum(dim=(1, 2, 3))

    union = (
        probs.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (
        (2 * intersection + smooth)
        / (union + smooth)
    )

    return 1 - dice.mean()


# --------------------------------------------------
# Combined loss
# --------------------------------------------------

def combined_loss(logits, targets):

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets.float()
    )

    dice = dice_loss(
        logits,
        targets
    )

    return 0.3 * bce + 0.7 * dice


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = BUSIDataset(TRAIN_DIR)
val_dataset = BUSIDataset(VAL_DIR)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
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
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training settings
# --------------------------------------------------

best_val_loss = float("inf")


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Dataset masks: [B, H, W]
        # Model output:   [B, 1, H, W]
        masks = masks.unsqueeze(1)

        optimizer.zero_grad()

        # U-Net produces p1, p2, p3, p4
        outputs = model(images)

        # Standard U-Net uses ONLY final prediction
        p4 = outputs[3]

        loss = combined_loss(
            p4,
            masks
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            masks = masks.unsqueeze(1)

            outputs = model(images)

            p4 = outputs[3]

            loss = combined_loss(
                p4,
                masks
            )

            total_val_loss += loss.item()

    average_val_loss = (
        total_val_loss / len(val_loader)
    )


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Train Loss: {average_train_loss:.4f} "
        f"| Val Loss: {average_val_loss:.4f}"
    )


    # --------------------------------------------------
    # Save ONLY the best model
    # --------------------------------------------------

    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": average_train_loss,
                "val_loss": average_val_loss
            },
            SAVE_PATH
        )

        print(
            f"New best model saved: {SAVE_PATH}"
        )


# --------------------------------------------------
# Training complete
# --------------------------------------------------

print()
print("Training completed.")
print("Best validation loss:", best_val_loss)
print("Best model saved to:", SAVE_PATH)
