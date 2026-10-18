import hashlib
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from unet_lomix import UNetLoMix


# ============================================================
# 1. CHECKPOINT SETTINGS
# ============================================================

CHECKPOINT_NAME = "busi_lomix_epoch_55.pth"

# The checkpoint is stored as a GitHub Release asset.
# It is downloaded once on first use and cached on disk.
CHECKPOINT_URL = (
    "https://github.com/saimsvs/ML_OPS/releases/download/"
    "lomix-epoch55/busi_lomix_epoch_55.pth"
)

# SHA256 of the real checkpoint file
# (same hash as the Git LFS object ID).
CHECKPOINT_SHA256 = (
    "5e85470e18071c3509350587bef23bf39e0295381226c8e38121ca2f59173909"
)

IMAGE_SIZE = (256, 256)

CACHE_DIR = Path.home() / ".cache" / "lomix_busi"


# ============================================================
# 2. CHECKPOINT DOWNLOAD
# ============================================================

def is_git_lfs_pointer(path):
    """
    A checkout without Git LFS leaves a small text file
    that starts with this line instead of the real weights.
    """

    try:
        with open(path, "rb") as file:
            header = file.read(100)
    except OSError:
        return False

    return header.startswith(
        b"version https://git-lfs.github.com/spec/v1"
    )


def sha256_of(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_checkpoint():

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    destination = CACHE_DIR / CHECKPOINT_NAME

    # Reuse the cached file only if its hash matches.
    if destination.exists():
        if sha256_of(destination) == CHECKPOINT_SHA256:
            return destination
        destination.unlink()

    temporary = destination.with_suffix(".part")

    try:
        with urllib.request.urlopen(
            CHECKPOINT_URL,
            timeout=120
        ) as response, open(temporary, "wb") as file:

            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                file.write(chunk)

    except Exception as error:

        if temporary.exists():
            temporary.unlink()

        raise RuntimeError(
            f"Could not download the model checkpoint from "
            f"{CHECKPOINT_URL}. "
            f"Original error: {error}"
        )

    if sha256_of(temporary) != CHECKPOINT_SHA256:

        temporary.unlink()

        raise RuntimeError(
            "The downloaded checkpoint failed the SHA256 "
            "integrity check. Download aborted."
        )

    temporary.replace(destination)

    return destination


def find_checkpoint():
    """
    Search order:

    1. Real checkpoint next to inference.py (local checkout
       with Git LFS pulled).
    2. Previously downloaded copy in the cache.
    3. Download from the GitHub Release.
    """

    local_path = Path(__file__).parent / CHECKPOINT_NAME

    if local_path.exists() and not is_git_lfs_pointer(local_path):
        return local_path

    cached_path = CACHE_DIR / CHECKPOINT_NAME

    if cached_path.exists() and not is_git_lfs_pointer(cached_path):
        if sha256_of(cached_path) == CHECKPOINT_SHA256:
            return cached_path

    return download_checkpoint()


# ============================================================
# 3. MODEL LOADING
# ============================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model():

    device = get_device()

    checkpoint_path = find_checkpoint()

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        # Older torch versions do not have weights_only.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        epoch = None
    else:
        raise RuntimeError(
            "Unrecognized checkpoint format."
        )

    model = UNetLoMix(
        n_channels=1,
        n_classes=1
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    return model, device, epoch, str(checkpoint_path)


# ============================================================
# 4. PREPROCESSING AND INFERENCE
# ============================================================
#
# Matches dataset_busi.py and preprocess_busi.py:
#
#   grayscale -> resize 256 x 256 (INTER_AREA)
#   -> divide by 255 -> shape [1, 1, 256, 256]
#
# ============================================================

def preprocess(gray_image):

    resized = cv2.resize(
        gray_image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    normalized = resized.astype(np.float32) / 255.0
    normalized = np.expand_dims(normalized, axis=0)

    return torch.from_numpy(normalized).unsqueeze(0)


@torch.inference_mode()
def predict(model, device, image_tensor):

    predictions = model(image_tensor.to(device))

    # Same combination as evaluate_busi_lomix.py:
    # sum the four logits -> sigmoid -> threshold at 0.5
    final_prediction = torch.stack(
        predictions, dim=0
    ).sum(dim=0)

    probabilities = torch.sigmoid(final_prediction)

    predicted_mask = (
        probabilities >= 0.5
    ).float()

    return predicted_mask.squeeze().cpu().numpy()


def build_overlay(display_image, mask_256, alpha=0.45):

    height, width = display_image.shape[:2]

    mask_resized = cv2.resize(
        (mask_256 * 255).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    if display_image.ndim == 2:
        base = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
    else:
        base = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

    overlay = base.astype(np.float32)

    lesion_color = np.array(
        [255.0, 60.0, 60.0], dtype=np.float32
    )

    lesion = mask_resized > 0

    overlay[lesion] = (
        overlay[lesion] * (1.0 - alpha)
        + lesion_color * alpha
    )

    return base, mask_resized, overlay.astype(np.uint8)
