# PROVENANCE.md

This document records where every major piece of this codebase came from.

Reference material
------------------
- Paper (the "reference method"): Md Mostafijur Rahman and Radu Marculescu,
  "LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image
  Segmentation", NeurIPS 2025. https://arxiv.org/abs/2510.22995
- Official implementation repo: https://github.com/SLDGroup/LoMix
  (commit 464a945ced3ecd5e9a0faa7ab2612a7c8a82240a, Feb 2026)
- This repo was at commit 72b27decdb62d931733491d6c860b1d26fb52f74 when this
  file was written.

Per-file provenance
-------------------

### preprocess_busi.py                    -> Written by us
Prepares the Breast Ultrasound Images (BUSI) dataset for training.
The official LoMix repo ships NO BUSI preprocessing: its data pipeline
targets the 3D Synapse CT dataset (utils/preprocess_synapse_data.py) and
points to downloads for ACDC and polyp data. BUSI appears only in the paper's
appendix, so everything here is our own work, including the BUSI-specific
handling of multiple masks per image merged with a logical OR, resizing to
256x256, and the 80/10/10 train/val/test split.

### dataset_busi.py                       -> Written by us
A simple PyTorch Dataset that loads the processed 2D BUSI images and masks.
The official repo only ships volumetric loaders (utils/dataset_synapse.py,
utils/dataset_ACDC.py); there is no BUSI loader to reuse. Standard,
generic boilerplate with nothing copied.

### unet_lomix.py                         -> Adapted
- The U-Net building blocks (DoubleConv, Down, Up, OutConv) follow the
  canonical public-domain PyTorch U-Net implementation
  (https://github.com/milesial/Pytorch-UNet, MIT license, released 2018).
  The behavior is the standard U-Net; we rewrote it in our own style.
  This code does NOT come from the LoMix authors (they use a PVT-V2-B2
  transformer backbone with an EMCAD decoder, not a plain U-Net).
- The modification that matters for this project is ours: four 1x1
  prediction heads (out1..out4), one after each decoder stage, emitting
  four logit maps (coarse -> fine) that are upsampled to the same size and
  returned as a list for the LoMix loss. This mirrors the paper's
  "multi-scale decoder logits" design idea, expressed as a small U-Net.

### lomix.py                              -> Adapted from SLDGroup/LoMix
Based on trainer.py in the official repo (classes WeightedFusion and
CombinatorialMutationsLossModule). We kept the core LoMix machinery:
softplus-transformed learnable weights per prediction, the same fusion
operators (add, mul, concat, weighted_fusion), and the same loss
aggregation (lc1*CE + lc2*Dice). Our changes are substantive, not cosmetic:
1. Changed the loss to BCE + Dice for BUSI binary segmentation, including
   casting targets to float for BCEWithLogitsLoss (the paper uses
   softmax/CE for multi-class data).
2. Fixed a real bug in the official code: there the concat 1x1 conv is
   created inside the forward loop, so it is not a registered parameter and
   is never updated by the optimizer. We pre-register these conv layers in
   __init__ (self.concat_modules) so they actually train.
3. Simplified the weight logging (raw params removed).

### train_busi.py                         -> Written by us
Training loop written from scratch for 2D BUSI data: plain epoch loop,
AdamW/Adam, BCE + Dice losses, checkpointing, best-model saving.
The official training loop lives in trainer.py / train_synapse_lomix.py and
is built around 3D volumetric dataloaders, TensorBoard, a poly learning-rate
schedule, and an iteration counter; none of that code is reused here.
The DiceLoss class is the standard textbook Dice formula (no external
source). What follows the paper is the *recipe* we call: the LoMix loss
module from lomix.py with operations = add/mul/wf/concat, learnable
softplus weights, and supervision='lomix'.

### evaluate_busi_lomix.py                -> Written by us
Evaluation script for 2D BUSI. Dice and IoU are standard definitions
written by us. The official evaluation code (utils/utils.py
val_single_volume plus inference() in trainer.py) is tailored to volumetric
sliding-window evaluation and is not reused. Inference uses only the final
decoder prediction, matching the paper's claim of zero inference overhead.

### README.md, PROVENANCE.md              -> Written by us
Documentation only.
