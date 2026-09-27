In this reproduction, we implemented the core LoMix method for medical image segmentation and adapted it to the Breast Ultrasound Images (BUSI) dataset.
We prepared the dataset by selecting the 647 benign and malignant lesion images, generating binary segmentation masks, resizing the images and masks to 256×256, and dividing the data into training, validation, and testing sets. Instead of the PVT-EMCAD-B2 backbone used in the original BUSI experiment, we adapted LoMix to a compact U-Net architecture to suit our available computational resources. We implemented the LoMix mechanism to combine multi-scale decoder predictions using different fusion operations and learnable weights during training. The implemented model will subsequently be evaluated and compared with standard single-output supervision and deep supervision using Dice and IoU metrics.

Section 1 - Preprocessing

The paper evaluates BUSI as one of its segmentation benchmarks, using the dataset of Al-Dhabyani et al. (780 images: 437 benign, 210 malignant, 133 normal). Only the 647 benign and malignant lesion images contain an actual lesion to segment, so the normal class is excluded; we follow this same exclusion.

Each raw image is matched with its mask file or files. Some images have more than one mask because a single scan can contain several distinct lesion regions; the paper does not state how it handles this. We combine all masks for an image into one binary mask with a pixel-wise logical OR, so a pixel marked as lesion in any mask becomes lesion in the final mask. This gives one training sample per image and keeps the total at 647.

The paper resizes BUSI images to 256x256 pixels, and we match this. Images use area interpolation (smooths grayscale ultrasound data), while masks use nearest-neighbor interpolation to preserve hard binary edges instead of introducing gray values along boundaries. The paper does not specify its interpolation choices, so this follows standard practice.

We replicate the paper's 80/10/10 train/validation/test split on our 647-image subset with a fixed seed (42) for reproducibility, and apply no augmentation, matching the paper; its BUSI batch size is 16.

Section 2 - Training

The training pipeline comprises unet_lomix.py (network), lomix.py (LoMix loss module), dataset_busi.py (data loader), and train_busi.py (training loop).

unet_lomix.py is a compact U-Net (classic DoubleConv, max-pool, and skip-connection design) with one 1x1 prediction head after each of its four decoder stages. It therefore emits four logit maps at different resolutions, from a 32x32 coarse map up to the final 256x256 prediction; the coarse maps are bilinearly upsampled so all four align pixel-by-pixel, and forward() returns them coarse to fine. This matches the contract of the paper's EMCADNet, which likewise hands four full-resolution logit maps (upsampled by x4/x8/x16/x32) to the LoMix loss.

The substantive difference is architectural: the paper uses a pretrained PVT-V2-B2 transformer encoder with the EMCAD multi-scale attention decoder, while our U-Net is fully convolutional and trained from scratch. Since LoMix is a training-time supervision method that works on any U-shaped network emitting several decoder-stage predictions, our adaptation is valid. Our absolute scores should not be compared with the paper's 80.25-81.32% DICE range for PVT-EMCAD-B2 + LoMix, but the relative gain of LoMix over single-output and deep-supervision baselines remains a meaningful comparison.

Section 3 - Evaluation
