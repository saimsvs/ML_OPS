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
# 1. PAGE CONFIG AND THEME
# ============================================================

st.set_page_config(
    page_title="Breast Ultrasound Segmentation",
    layout="wide"
)

st.markdown(
    """
    <style>
    @keyframes lomix-fade-down {
        from { opacity: 0; transform: translateY(-14px); }
        to   { opacity: 1; transform: none; }
    }
    @keyframes lomix-fade-up {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: none; }
    }
    @keyframes lomix-pop {
        0%   { opacity: 0; transform: scale(0.97); }
        100% { opacity: 1; transform: scale(1); }
    }
    @keyframes lomix-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(0, 119, 182, 0.35); }
        70%  { box-shadow: 0 0 0 10px rgba(0, 119, 182, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 119, 182, 0); }
    }
    .lomix-header {
        background: linear-gradient(90deg, #0077B6 0%, #00B4D8 100%);
        border-radius: 12px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.0rem;
        box-shadow: 0 2px 8px rgba(0, 119, 182, 0.25);
        animation: lomix-fade-down 0.5s ease-out;
    }
    .lomix-header h1 {
        color: #FFFFFF;
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
    }
    .lomix-header p {
        color: #DFF3FB;
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }
    .lomix-section {
        color: #0B2A3A;
        font-size: 1.15rem;
        font-weight: 650;
        border-bottom: 2px solid #B6D9EA;
        padding-bottom: 0.35rem;
        margin: 1.2rem 0 0.8rem 0;
        animation: lomix-fade-up 0.45s ease-out both;
    }
    .lomix-panel-title {
        animation: lomix-fade-up 0.5s ease-out both;
    }
    .lomix-panel-title {
        color: #0077B6;
        font-weight: 650;
        font-size: 0.95rem;
        margin: 0 0 0.4rem 0;
    }
    .lomix-note {
        border: 1px solid #B6D9EA;
        background: #F4FAFD;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #0B2A3A;
        font-size: 0.9rem;
        animation: lomix-fade-up 0.5s ease-out both;
    }
    .lomix-footer {
        margin-top: 2rem;
        padding: 0.9rem 1.1rem;
        border-left: 4px solid #0077B6;
        background: #F4FAFD;
        color: #34505E;
        font-size: 0.85rem;
        border-radius: 0 10px 10px 0;
    }
    div[data-testid="stBorder"] {
        background: #FFFFFF;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
        animation: lomix-fade-up 0.55s ease-out both;
    }
    div[data-testid="stBorder"]:hover {
        box-shadow: 0 4px 14px rgba(0, 119, 182, 0.18);
        transform: translateY(-2px);
    }
    div[data-testid="stSuccess"] {
        animation: lomix-pop 0.4s ease-out;
    }
    div[data-testid="stWarning"] {
        animation: lomix-pop 0.4s ease-out;
    }
    section[data-testid="stSidebar"] {
        background-color: #EAF4F8;
    }
    </style>

    <div class="lomix-header">
        <h1>Breast Ultrasound Lesion Segmentation</h1>
        <p>AI-assisted lesion masking &middot; LoMix U-Net &middot; BUSI dataset &middot; Research use only</p>
    </div>
    """,
    unsafe_allow_html=True
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

st.sidebar.markdown(
    """
    <div style="margin-bottom: 0.6rem;">
        <div style="color:#0077B6; font-size:1.25rem; font-weight:700;">
            Clinical Viewer
        </div>
        <div style="color:#34505E; font-size:0.85rem;">
            LoMix U-Net segmentation
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.markdown("**Model**")
st.sidebar.markdown(
    f"""
    - Checkpoint: `busi_lomix_epoch_55.pth`
    - Epoch: {epoch}
    - Architecture: `UNetLoMix` (4 heads)
    - Input: 256 &times; 256 grayscale
    - Inference: sum &rarr; sigmoid &rarr; 0.5
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown("**Pipeline**")
st.sidebar.markdown(
    """
    1. Grayscale conversion
    2. Resize to 256 &times; 256
    3. Normalize to 0&ndash;1
    4. U-Net segmentation
    5. Threshold at 0.5
    """
)


# ============================================================
# 4. UPLOAD
# ============================================================

st.markdown(
    '<div class="lomix-section">Image upload</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a breast ultrasound image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
    label_visibility="collapsed"
)

if uploaded_file is None:

    st.markdown(
        """
        <div class="lomix-note">
        Upload a PNG or JPEG breast ultrasound image to compute the
        lesion mask. The image is processed in memory only and is
        not stored anywhere.<br><br>
        <b>Pipeline:</b> grayscale &rarr; 256&times;256 resize &rarr; 0&ndash;1
        normalization &rarr; U-Net &rarr; sum of four logits &rarr; sigmoid
        &rarr; threshold 0.5.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# 5. DECODE AND PREDICT
# ============================================================

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


image_tensor = preprocess(gray_image)

with st.spinner("Running segmentation ..."):

    try:
        mask_256 = predict(
            model, device, image_tensor
        )
    except Exception as error:

        st.error(f"Prediction failed: {error}")

        st.stop()


# ============================================================
# 6. RESULTS
# ============================================================

display_image = cv2.imdecode(
    file_bytes,
    cv2.IMREAD_UNCHANGED
)

base_rgb, mask_display, overlay = build_overlay(
    display_image, mask_256
)

coverage = float(mask_256.mean()) * 100.0

st.markdown(
    '<div class="lomix-section">Segmentation results</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        '<div class="lomix-panel-title">Original image</div>',
        unsafe_allow_html=True
    )
    with st.container(border=True):
        st.image(base_rgb, use_container_width=True)

with col2:
    st.markdown(
        '<div class="lomix-panel-title">Predicted mask</div>',
        unsafe_allow_html=True
    )
    with st.container(border=True):
        st.image(mask_display, use_container_width=True)

with col3:
    st.markdown(
        '<div class="lomix-panel-title">Lesion overlay</div>',
        unsafe_allow_html=True
    )
    with st.container(border=True):
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

st.markdown(
    """
    <div class="lomix-note" style="margin-top: 0.8rem;">
    The mask is the model's raw prediction. Dice and IoU cannot be
    computed for uploaded images because they have no ground-truth
    mask. Test-set metrics were measured on the BUSI test split only.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 7. MODEL INFORMATION AND DISCLAIMER
# ============================================================

with st.expander("Model information"):

    st.markdown(
        f"""
        - **Checkpoint file:** `{CHECKPOINT_NAME}`
        - **Checkpoint epoch:** {epoch}
        - **Loaded from:** `{checkpoint_path}`
        - **Device:** `{device}`
        - **Architecture:** `UNetLoMix` with four decoder heads
        - **Input size:** 256 &times; 256 grayscale, normalized to 0&ndash;1
        - **Post-processing:** sum of four logits, sigmoid, threshold 0.5
        - **Trained on:** BUSI (benign + malignant), 80/10/10 split
        """
    )

st.markdown(
    """
    <div class="lomix-footer">
    <b>Medical disclaimer:</b> this application is an academic research
    prototype adapted from the LoMix paper. It is not a certified
    medical device, provides no diagnosis, and must not be used for
    clinical decision-making. Always consult a qualified clinician.
    </div>
    """,
    unsafe_allow_html=True
)
