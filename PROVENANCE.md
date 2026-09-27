# PROVENANCE.md

This document records where every major code component came from.

Reference material:
- Paper: Rahman and Marculescu, "LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation", NeurIPS 2025. https://arxiv.org/abs/2510.22995
- Official code: https://github.com/SLDGroup/LoMix @ commit 464a945 (Feb 2026).
- This repo was at commit 72b27de when this file was written.

### preprocess_busi.py -> Written by us
BUSI preprocessing. The official repo ships no BUSI pipeline (its data prep targets Synapse 3D CT), so the multi-mask OR merging, 256x256 resizing, and 80/10/10 split are entirely our own work.

### dataset_busi.py -> Written by us
Plain 2D PyTorch Dataset for the processed images and masks. The official repo only has volumetric loaders; nothing reused.

### unet_lomix.py -> Adapted
U-Net blocks follow the canonical MIT PyTorch U-Net (milesial/Pytorch-UNet), rewritten in our style; the four per-stage prediction heads producing the multi-scale logit list are our addition. Not author code.

### lomix.py -> Adapted from SLDGroup/LoMix
Based on CombinatorialMutationsLossModule/WeightedFusion in trainer.py (@464a945). Kept the softplus weights, four operators, and loss aggregation; switched to BCE+Dice for binary BUSI, and fixed the official bug where the concat conv was created in forward() and never trained.

### train_busi.py -> Written by us
Minimal Adam training loop with checkpointing. Only the LoMix recipe (four operators, learnable weights, supervision='lomix') follows the paper; the official volumetric training script is not reused.

### evaluate_busi_lomix.py -> Written by us
Standard Dice/IoU on the final prediction. Official volumetric sliding-window evaluator not reused.

### README.md, PROVENANCE.md -> Written by us
Documentation.
