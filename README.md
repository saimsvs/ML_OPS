In this reproduction, we implemented the core LoMix method for medical image segmentation and adapted it to the Breast Ultrasound Images (BUSI) dataset.
We prepared the dataset by selecting the 647 benign and malignant lesion images, generating binary segmentation masks, resizing the images and masks to 256×256, and dividing the data into training, validation, and testing sets. Instead of the PVT-EMCAD-B2 backbone used in the original BUSI experiment, we adapted LoMix to a compact U-Net architecture to suit our available computational resources. We implemented the LoMix mechanism to combine multi-scale decoder predictions using different fusion operations and learnable weights during training. The implemented model will subsequently be evaluated and compared with standard single-output supervision and deep supervision using Dice and IoU metrics.

Section 1 - Preprocessing

Section 2 - Training

Section 3 - Evaluation
