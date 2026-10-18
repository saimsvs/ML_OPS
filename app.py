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
    layout="wide"
)


# ============================================================
# 2. THEME CSS
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- animations ---------- */
    @keyframes lomix-fade-down {
        from { opacity: 0; transform: translateY(-16px); }
        to   { opacity: 1; transform: none; }
    }
    @keyframes lomix-fade-up {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: none; }
    }
    @keyframes lomix-pop {
        0%   { opacity: 0; transform: scale(0.96); }
        100% { opacity: 1; transform: scale(1); }
    }
    @keyframes lomix-gradient-shift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes lomix-dot-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.35; transform: scale(0.7); }
    }

    /* ---------- header ---------- */
    .lomix-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        background: linear-gradient(120deg, #005F8F, #0077B6, #00B4D8, #0077B6);
        background-size: 300% 300%;
        animation: lomix-gradient-shift 12s ease infinite,
                   lomix-fade-down 0.55s ease-out;
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 4px 16px rgba(0, 95, 143, 0.28);
        margin-bottom: 0.4rem;
    }
    .lomix-eyebrow {
        color: #BFE7F7;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .lomix-header h1 {
        color: #FFFFFF;
        margin: 0;
        font-size: 1.85rem;
        font-weight: 750;
        letter-spacing: -0.01em;
    }
    .lomix-header p {
        color: #E3F4FC;
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
    }
    .lomix-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .lomix-badge {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        backdrop-filter: blur(4px);
        white-space: nowrap;
    }

    /* ---------- sections ---------- */
    .lomix-section {
        color: #4A6B7E;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.7rem 0;
        animation: lomix-fade-up 0.45s ease-out both;
    }
    .lomix-section::before {
        content: "";
        display: inline-block;
        width: 9px;
        height: 9px;
        background: #0077B6;
        border-radius: 2px;
        margin-right: 0.55rem;
        transform: rotate(45deg);
    }
    .lomix-panel-title {
        color: #0B2A3A;
        font-weight: 650;
        font-size: 0.95rem;
        margin: 0 0 0.45rem 0;
        animation: lomix-fade-up 0.5s ease-out both;
    }
    .lomix-panel-sub {
        color: #7A95A5;
        font-size: 0.75rem;
        margin: -0.3rem 0 0.4rem 0;
    }

    /* ---------- result cards (staggered entrance) ---------- */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1)
        [data-testid="stBorder"] { animation-delay: 0.05s; }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2)
        [data-testid="stBorder"] { animation-delay: 0.18s; }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3)
        [data-testid="stBorder"] { animation-delay: 0.31s; }

    div[data-testid="stBorder"] {
        background: #FFFFFF;
        transition: box-shadow 0.3s ease, transform 0.3s ease;
        animation: lomix-fade-up 0.6s ease-out both;
    }
    div[data-testid="stBorder"]:hover {
        box-shadow: 0 8px 22px rgba(0, 119, 182, 0.18);
        transform: translateY(-3px);
    }

    /* ---------- metric readout ---------- */
    .lomix-metric {
        display: flex;
        align-items: baseline;
        gap: 0.9rem;
        background: #F4FAFD;
        border: 1px solid #B6D9EA;
        border-left: 4px solid #0077B6;
        border-radius: 10px;
        padding: 0.9rem 1.3rem;
        animation: lomix-pop 0.5s ease-out both;
        margin-top: 0.4rem;
    }
    .lomix-metric.warn { border-left-color: #E8A100; }
    .lomix-metric-value {
        font-size: 1.9rem;
        font-weight: 750;
        color: #0077B6;
        line-height: 1;
    }
    .lomix-metric.warn .lomix-metric-value { color: #C77E00; }
    .lomix-metric-label {
        color: #34505E;
        font-size: 0.92rem;
    }

    /* ---------- notes, citation, footer ---------- */
    .lomix-note {
        border: 1px solid #B6D9EA;
        background: #F4FAFD;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        color: #0B2A3A;
        font-size: 0.9rem;
        animation: lomix-fade-up 0.5s ease-out both;
    }
    .lomix-cite {
        border: 1px dashed #9FC9DF;
        background: #FAFCFE;
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        color: #34505E;
        font-size: 0.82rem;
        font-style: italic;
    }
    .lomix-footer {
        margin-top: 2rem;
        padding: 0.95rem 1.2rem;
        border-left: 4px solid #0077B6;
        background: #F4FAFD;
        color: #34505E;
        font-size: 0.85rem;
        border-radius: 0 10px 10px 0;
        animation: lomix-fade-up 0.5s ease-out both;
    }

    /* ---------- spinner and feedback ---------- */
    div[data-testid="stSpinner"] p {
        color: #0077B6;
        font-weight: 650;
        letter-spacing: 0.03em;
    }
    div[data-testid="stSuccess"] { animation: lomix-pop 0.4s ease-out; }
    div[data-testid="stWarning"] { animation: lomix-pop 0.4s ease-out; }
    div[data-testid="stError"]   { animation: lomix-pop 0.4s ease-out; }

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #EAF4F8;
    }
    </style>

    <div class="lomix-header">
        <div>
            <div class="lomix-eyebrow">Clinical Image Analysis &middot; Research Prototype</div>
            <h1>Breast Ultrasound Lesion Segmentation</h1>
            <p>LoMix-adapted four-head U-Net &middot; Trained on the BUSI benchmark</p>
        </div>
        <div class="lomix-badges">
            <span class="lomix-badge">EPOCH 55</span>
            <span class="lomix-badge">256 &times; 256 INPUT</span>
            <span class="lomix-badge">CPU READY</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. CACHED MODEL LOADING
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
# 4. SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="margin-bottom: 0.4rem;">
        <div style="color:#0077B6; font-size:1.2rem; font-weight:750;">
            Clinical Viewer
        </div>
        <div style="color:#34505E; font-size:0.82rem; letter-spacing:0.06em;">
            LoMix U-Net &middot; segmentation
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
    - Epoch: **{epoch}**
    - Architecture: `UNetLoMix`, 4 decoder heads
    - Input: 256 &times; 256 grayscale, [0, 1]
    - Fusion: sum &rarr; sigmoid &rarr; threshold 0.5
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown("**Inference pipeline**")
st.sidebar.markdown(
    """
    1. Grayscale conversion
    2. Resize to 256 &times; 256 (`INTER_AREA`)
    3. Normalize to [0, 1]
    4. Forward pass &rarr; 4 logit maps
    5. Sum logits &rarr; sigmoid &rarr; 0.5
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="color:#34505E; font-size:0.78rem; line-height:1.5;">
    No image leaves your session. Inference runs entirely
    in the app container's memory.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. UPLOAD
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
        normalization &rarr; U-Net forward pass &rarr; sum of four
        logits &rarr; sigmoid &rarr; threshold 0.5.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# 6. DECODE AND PREDICT
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
# 7. RESULTS
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
        '<div class="lomix-panel-title">Original image</div>'
        '<div class="lomix-panel-sub">As uploaded</div>',
        unsafe_allow_html=True
    )
    with st.container(border=True):
        st.image(base_rgb, use_container_width=True)

with col2:
    st.markdown(
        '<div class="lomix-panel-title">Predicted mask</div>'
        '<div class="lomix-panel-sub">Binary, threshold 0.5</div>',
        unsafe_allow_html=True
    )
    with st.container(border=True):
        st.image(mask_display, use_container_width=True)

with col3:
    st.markdown(
        '<div class="lomix-panel-title">Lesion overlay</div>'
        '<div class="lomix-panel-sub">Mask fused with source</div>',
        unsafe_allow_html=True
    )
    with st.container(border=True):
        st.image(overlay, use_container_width=True)

if coverage == 0.0:
    st.markdown(
        """
        <div class="lomix-metric warn">
            <span class="lomix-metric-value">0.0%</span>
            <span class="lomix-metric-label">
                predicted lesion area &mdash; no pixels exceeded
                the 0.5 probability threshold
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div class="lomix-metric">
            <span class="lomix-metric-value">{coverage:.1f}%</span>
            <span class="lomix-metric-label">
                of the image area classified as lesion
            </span>
        </div>
        """,
        unsafe_allow_html=True
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
# 8. TECHNICAL DETAILS, CITATION, DISCLAIMER
# ============================================================

with st.expander("Technical details & model card"):

    st.markdown(
        f"""
        - **Checkpoint file:** `{CHECKPOINT_NAME}`
        - **Checkpoint epoch:** {epoch}
        - **Loaded from:** `{checkpoint_path}`
        - **Device:** `{device}`
        - **Architecture:** `UNetLoMix` with four decoder prediction heads
        - **Input size:** 256 &times; 256 grayscale, normalized to [0, 1]
        - **Post-processing:** sum of four logits, sigmoid, threshold 0.5
        - **Dataset:** BUSI (benign + malignant only), 80/10/10 split,
          seed 42, 647 images, no augmentation
        - **Training:** Adam, lr 1e-4, BCE + Dice inside the LoMix
          mutation loss
        """
    )

st.markdown(
    """
    <div class="lomix-cite" style="margin-top: 1rem;">
    Reference: Rahman and Marculescu, &ldquo;LoMix: Learnable Weighted
    Multi-Scale Logits Mixing for Medical Image Segmentation,&rdquo;
    NeurIPS 2025. arXiv:2510.22995 &mdash; this deployment is an
    independent academic adaptation using a compact U-Net.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="lomix-footer">
    <b>Medical disclaimer:</b> this application is an academic research
    prototype. It is not a certified medical device, provides no
    diagnosis, and must not be used for clinical decision-making.
    Always consult a qualified clinician.
    </div>
    """,
    unsafe_allow_html=True
)
