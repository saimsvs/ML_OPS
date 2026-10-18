import numpy as np
import cv2
import streamlit as st

from inference import (
    CHECKPOINT_NAME,
    build_overlay,
    load_model,
    predict,
    preprocess,
)


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Breast Ultrasound Segmentation",
    page_icon="🩻",
    layout="wide"
)


# ============================================================
# 2. CACHED MODEL LOADING
# ============================================================

@st.cache_resource(
    show_spinner="Loading model checkpoint (373 MB) ..."
)
def load_model_cached():
    return load_model()


try:
    model, device, epoch, checkpoint_path = load_model_cached()
except Exception as error:

    st.error(f"Could not load the model checkpoint: {error}")

    st.info(
        "For local development, run `git lfs pull` inside "
        "the repository so that `busi_lomix_epoch_55.pth` "
        "contains the real weights instead of an LFS pointer."
    )

    st.stop()


# ============================================================
# 3. SIDEBAR
# ============================================================

st.sidebar.title("LoMix U-Net")

st.sidebar.markdown(
    """
    Breast ultrasound lesion segmentation
    adapted from the LoMix paper on a compact U-Net
    trained on the BUSI dataset.
    """
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "**Checkpoint:** `busi_lomix_epoch_55.pth`  \n"
    "**Input:** 256 x 256 grayscale  \n"
    "**Inference:** sum of four logits -> sigmoid -> 0.5"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Research prototype. This app does not provide a "
    "medical diagnosis and is not a substitute for "
    "professional medical care."
)


# ============================================================
# 4. MAIN PAGE
# ============================================================

st.title("Breast Ultrasound Lesion Segmentation")

st.caption(
    "Upload a breast ultrasound image to get the "
    "predicted lesion mask and an overlay."
)


# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an ultrasound image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
)

if uploaded_file is None:

    with st.expander("What happens to my image?"):
        st.markdown(
            "1. The image is converted to grayscale.  \n"
            "2. It is resized to 256 x 256.  \n"
            "3. Pixel values are scaled to the 0 to 1 range.  \n"
            "4. The U-Net predicts four segmentation maps that "
            "are summed, passed through a sigmoid, and "
            "thresholded at 0.5.  \n\n"
            "The image is only processed in memory and is "
            "not stored anywhere."
        )

    st.stop()


# ------------------------------------------------------------
# Decode the uploaded file
# ------------------------------------------------------------

file_bytes = np.frombuffer(
    uploaded_file.getvalue(),
    dtype=np.uint8
)

gray_image = cv2.imdecode(
    file_bytes,
    cv2.IMREAD_GRAYSCALE
)

if gray_image is None:

    st.error(
        "Could not read the uploaded file as an image. "
        "Please upload a valid PNG, JPEG, BMP, TIFF, or "
        "WebP image."
    )

    st.stop()


# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------

image_tensor = preprocess(gray_image)

with st.spinner("Running the model ..."):

    try:
        mask_256 = predict(
            model, device, image_tensor
        )
    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )

        st.stop()


# ------------------------------------------------------------
# Build display images
# ------------------------------------------------------------

display_image = cv2.imdecode(
    file_bytes,
    cv2.IMREAD_UNCHANGED
)

base_rgb, mask_display, overlay = build_overlay(
    display_image, mask_256
)

coverage = float(mask_256.mean()) * 100.0


# ------------------------------------------------------------
# Show results
# ------------------------------------------------------------

st.markdown("### Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Original image**")
    st.image(base_rgb, use_container_width=True)

with col2:
    st.markdown("**Predicted mask**")
    st.image(mask_display, use_container_width=True)

with col3:
    st.markdown("**Overlay**")
    st.image(overlay, use_container_width=True)

if coverage == 0.0:
    st.warning(
        "No lesion region was predicted: every pixel is "
        "below the 0.5 probability threshold."
    )
else:
    st.success(
        f"Predicted lesion region: {coverage:.1f}% of "
        f"the image area."
    )

st.caption(
    "The mask is the model's prediction. Dice and IoU "
    "cannot be computed for uploaded images because they "
    "have no ground-truth mask. Test-set metrics were "
    "measured on the BUSI test split only."
)


# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------

with st.expander("Model information"):

    st.markdown(
        f"""
        - **Checkpoint file:** `{CHECKPOINT_NAME}`
        - **Checkpoint epoch:** {epoch}
        - **Loaded from:** `{checkpoint_path}`
        - **Device:** `{device}`
        - **Architecture:** `UNetLoMix` with four
          decoder prediction heads
        - **Input size:** 256 x 256 grayscale,
          normalized to 0 to 1
        - **Post-processing:** sum of the four logits,
          sigmoid, threshold 0.5
        """
    )
