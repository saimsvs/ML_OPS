
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet_lomix import UNetLoMix
from dataset_busi import BUSIDataset
from lomix import CombinatorialMutationsLossModule


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
# 2. DATASET PATH
# ============================================================

DATASET_PATH = "/Users/saimasad/Desktop/ML_PROJ/BUSI_processed"


# ============================================================
# 3. DATASETS
# ============================================================

train_dataset = BUSIDataset(
    DATASET_PATH,
    split="train"
)

val_dataset = BUSIDataset(
    DATASET_PATH,
    split="val"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

print("train:", len(train_dataset), "images found")
print("val:", len(val_dataset), "images found")


# ============================================================
# 4. U-NET MODEL
# ============================================================

model = UNetLoMix(
    n_channels=1,
    n_classes=1
).to(device)


# ============================================================
# 5. LOMIX
# ============================================================

lomix = CombinatorialMutationsLossModule(
    original_num_maps=4,
    num_classes=1,
    selecetd_num_maps=4,
    operations=["add", "mul", "wf", "concat"],
    use_learnable_weights=True
).to(device)


# ============================================================
# 6. BCE LOSS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


# ============================================================
# 7. DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction, target, softmax=False):

        # ----------------------------------------------------
        # For binary segmentation:
        # sigmoid converts logits into probabilities.
        #
        # softmax=True is supported because the LoMix
        # implementation expects this argument.
        # ----------------------------------------------------

        if softmax:
            prediction = torch.softmax(prediction, dim=1)
        else:
            prediction = torch.sigmoid(prediction)

        # ----------------------------------------------------
        # Flatten each image
        # ----------------------------------------------------

        prediction = prediction.reshape(
            prediction.shape[0], -1
        )

        target = target.reshape(
            target.shape[0], -1
        )

        # ----------------------------------------------------
        # Intersection
        # ----------------------------------------------------

        intersection = (
            prediction * target
        ).sum(dim=1)

        # ----------------------------------------------------
        # Dice coefficient
        # ----------------------------------------------------

        dice = (
            2 * intersection + self.smooth
        ) / (
            prediction.sum(dim=1)
            + target.sum(dim=1)
            + self.smooth
        )

        # ----------------------------------------------------
        # Dice loss
        # ----------------------------------------------------

        return 1 - dice.mean()


dice_loss = DiceLoss()


# ============================================================
# 8. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    list(model.parameters()) +
    list(lomix.parameters()),
    lr=1e-4
)


# ============================================================
# 9. TRAINING SETTINGS
# ============================================================

num_epochs = 3

best_val_loss = float("inf")


# ============================================================
# 10. TRAINING
# ============================================================

for epoch in range(num_epochs):

    model.train()
    lomix.train()

    total_train_loss = 0.0

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAINING BATCHES
    # --------------------------------------------------------

    for batch_idx, (images, masks) in enumerate(train_loader):

        # ----------------------------------------------------
        # Move images and masks to MPS/CPU
        # ----------------------------------------------------

        images = images.to(device)
        masks = masks.to(device)

        # ----------------------------------------------------
        # Dataset mask shape:
        #
        # [B, 256, 256]
        #
        # Model prediction shape:
        #
        # [B, 1, 256, 256]
        #
        # Add channel dimension to mask.
        # ----------------------------------------------------

        masks = masks.unsqueeze(1)

        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass through U-Net
        #
        # Returns 4 decoder predictions.
        # ----------------------------------------------------

        predictions = model(images)

        # ----------------------------------------------------
        # LoMix
        #
        # Combines the 4 predictions using:
        #
        # Addition
        # Multiplication
        # Weighted fusion
        # Concatenation
        #
        # and calculates the loss.
        # ----------------------------------------------------

        loss, ds_loss, mutation_loss = lomix(
            predictions,
            masks,
            bce_loss,
            dice_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update model + LoMix parameters
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Record loss
        # ----------------------------------------------------

        total_train_loss += loss.item()

        print(
            f"Batch {batch_idx + 1}/{len(train_loader)} "
            f"Loss: {loss.item():.4f}"
        )

    # ========================================================
    # 11. AVERAGE TRAINING LOSS
    # ========================================================

    avg_train_loss = (
        total_train_loss / len(train_loader)
    )

    print()
    print(
        f"Epoch {epoch + 1} "
        f"average training loss: "
        f"{avg_train_loss:.4f}"
    )


    # ========================================================
    # 12. VALIDATION
    # ========================================================

    model.eval()
    lomix.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            # ------------------------------------------------
            # Move data to device
            # ------------------------------------------------

            images = images.to(device)
            masks = masks.to(device)

            # ------------------------------------------------
            # Same mask shape fix used during training
            # ------------------------------------------------

            masks = masks.unsqueeze(1)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            predictions = model(images)

            # ------------------------------------------------
            # Calculate validation loss
            # ------------------------------------------------

            loss, ds_loss, mutation_loss = lomix(
                predictions,
                masks,
                bce_loss,
                dice_loss
            )

            total_val_loss += loss.item()


    # ========================================================
    # 13. AVERAGE VALIDATION LOSS
    # ========================================================

    avg_val_loss = (
        total_val_loss / len(val_loader)
    )

    print(
        f"Epoch {epoch + 1} "
        f"validation loss: "
        f"{avg_val_loss:.4f}"
    )


    # ========================================================
    # 14. SAVE CHECKPOINT
    # ========================================================

    checkpoint = {
        "epoch": epoch + 1,

        "model_state_dict":
            model.state_dict(),

        "lomix_state_dict":
            lomix.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            avg_train_loss,

        "val_loss":
            avg_val_loss
    }

    torch.save(
        checkpoint,
        f"busi_lomix_epoch_{epoch + 1}.pth"
    )

    print(
        f"Saved: "
        f"busi_lomix_epoch_{epoch + 1}.pth"
    )


    # ========================================================
    # 15. SAVE BEST MODEL
    # ========================================================

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            checkpoint,
            "busi_lomix_best.pth"
        )

        print("New best model saved!")


# ============================================================
# 16. TRAINING COMPLETE
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    "Best model: "
    "busi_lomix_best.pth"
)
