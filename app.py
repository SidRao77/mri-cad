"""
app.py

Local Streamlit demo: upload a brain MRI image, run it through the trained
ResNet18 checkpoint, and display the prediction, confidence, and a
Grad-CAM heatmap showing what region of the image the model focused on.

Run:
    streamlit run app.py

This is a learning/portfolio project, not a diagnostic tool. See the
disclaimer below and in README.md.
"""

import base64
import io
import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# Make src/ importable so the app reuses the exact same model-building,
# preprocessing, and Grad-CAM code that train.py / evaluate.py use —
# no duplicated logic that could drift out of sync.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from calibration import load_temperature  # noqa: E402
from dataset import eval_transform  # noqa: E402
from model import build_model, get_device  # noqa: E402
from gradcam import build_gradcam, overlay_heatmap  # noqa: E402

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"

CLASS_LABELS = {0: "No Tumor Detected", 1: "Tumor Detected"}

# Confidence display bounds. A model trained on ~250 images can't justify
# claims more precise than this, so we cap the readout instead of showing
# "100.0%", and flag readings close to a coin flip for extra caution.
MAX_DISPLAY_CONFIDENCE = 0.99
LOW_CONFIDENCE_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Model loading — cached so it only happens once per session, not on every
# widget interaction (Streamlit reruns the whole script on each interaction).
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    device = get_device()
    model = build_model().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    gradcam = build_gradcam(model)
    return model, gradcam, device, load_temperature()


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string, for embedding in HTML."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def run_prediction(image: Image.Image, model, gradcam, device, temperature):
    """Preprocess the image, run inference, and generate a Grad-CAM overlay."""
    input_tensor = eval_transform(image.convert("RGB")).unsqueeze(0).to(device)

    # Divide logits by the fitted temperature so the reported confidence is
    # calibrated (see src/calibration.py). This never changes the prediction.
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits / temperature, dim=1).squeeze(0).cpu()
    predicted_class = probs.argmax().item()

    # Grad-CAM needs its own forward+backward pass (it needs gradients),
    # so it's run separately from the no_grad inference above.
    heatmap, _, _ = gradcam.generate(input_tensor, target_class=predicted_class)
    overlay = overlay_heatmap(image, heatmap)

    return {
        "predicted_class": predicted_class,
        "confidence": probs[predicted_class].item(),
        "probs": probs,
        "overlay": overlay,
    }


# ---------------------------------------------------------------------------
# Page setup + styling
#
# Design direction: modeled on real radiology AI software chrome (e.g. the
# dark, compact, single-accent-color scan viewers used by tools like Aidoc)
# rather than an editorial "designed" web page. True near-black flat
# background, dense functional type, a status-dot + pill vocabulary, and
# exactly one accent (amber) reserved only for a flagged finding.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MRI-CAD", page_icon="🩻", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #0b0c0f;
    --panel: #131519;
    --border: #22252c;
    --text: #e7e8ea;
    --text-dim: #7d818a;
    --accent: #ffab4a;
    --online: #5fd992;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 860px; }

/* ---- Top bar ---- */
.topbar { display: flex; align-items: center; justify-content: space-between; }
.topbar-brand { display: flex; align-items: center; gap: 0.55rem; }
.brand-mark {
    width: 18px; height: 18px;
    border: 1.5px solid var(--text);
    border-radius: 4px;
    position: relative;
    flex-shrink: 0;
}
.brand-mark::before {
    content: "";
    position: absolute; top: 50%; left: 50%;
    width: 4px; height: 4px;
    background: var(--accent);
    border-radius: 50%;
    transform: translate(-50%, -50%);
}
.brand-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.92rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.status-pill {
    display: flex; align-items: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.03em;
    text-transform: uppercase;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.3rem 0.7rem;
}
.pill-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--online);
    margin-right: 0.45rem;
    flex-shrink: 0;
}
.topbar-rule { border-top: 1px solid var(--border); margin: 1.1rem 0 1.6rem 0; }

.section-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 1.6rem 0 0.7rem 0;
}

/* ---- Upload control ---- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg { opacity: 0.35; }

.viewport-empty {
    text-align: center;
    padding: 2.6rem 1rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--panel);
    color: var(--text-dim);
    margin-top: 0.8rem;
}
.empty-glyph { font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; color: var(--text-dim); margin-bottom: 0.5rem; }
.empty-title { font-size: 0.88rem; color: var(--text); margin-bottom: 0.25rem; }
.empty-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.02em; }

.viewer-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.02em;
    margin-bottom: 0.6rem;
}

/* ---- Scan panels ---- */
.scan-panel {
    position: relative;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: #000;
    animation: fadeIn 0.4s ease-out both;
}
.scan-panel img { width: 100%; display: block; }
.scan-tag {
    position: absolute; top: 7px; left: 7px;
    background: rgba(0, 0, 0, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.45rem;
    border-radius: 3px;
}

/* ---- Findings ---- */
.findings-bar {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin-top: 1rem;
    animation: fadeIn 0.5s ease-out both;
    animation-delay: 0.08s;
}
.findings-row {
    display: flex; align-items: center; gap: 0.55rem;
    margin-bottom: 0.85rem;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.status-label { font-weight: 600; font-size: 0.98rem; letter-spacing: 0.01em; flex: 1; }
.conf-readout { font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; }
.conf-track { height: 4px; background: #1c1f26; border-radius: 2px; overflow: hidden; }
.conf-fill {
    height: 100%; border-radius: 2px;
    transform-origin: left;
    animation: fillBar 0.8s cubic-bezier(0.22, 1, 0.36, 1) both;
    animation-delay: 0.15s;
}
.findings-meta {
    margin-top: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.02em;
}
.findings-note {
    margin-top: 0.55rem;
    font-size: 0.76rem;
    line-height: 1.5;
    color: var(--text-dim);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fillBar {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
}

.footnote {
    margin-top: 2.2rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
device_label = str(get_device()).upper()
st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">
        <span class="brand-mark"></span>
        <span class="brand-name">MRI-CAD</span>
    </div>
    <div class="status-pill"><span class="pill-dot"></span>Model Ready &middot; {device_label}</div>
</div>
<div class="topbar-rule"></div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Upload + inference
# ---------------------------------------------------------------------------
st.markdown('<div class="section-tag">Scan Input</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a brain MRI image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div class="viewport-empty">
        <div class="empty-glyph">⌁</div>
        <div class="empty-title">No scan loaded</div>
        <div class="empty-sub">Upload a JPG or PNG MRI slice to run inference</div>
    </div>
    """, unsafe_allow_html=True)
else:
    if not CHECKPOINT_PATH.exists():
        st.error(
            f"No checkpoint found at {CHECKPOINT_PATH}. Run `python src/train.py` "
            "first to produce a trained model."
        )
    else:
        image = Image.open(uploaded_file)
        width, height = image.size

        with st.spinner("Running inference..."):
            model, gradcam, device, temperature = load_model()
            result = run_prediction(image, model, gradcam, device, temperature)

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        label = CLASS_LABELS[predicted_class]
        is_positive = predicted_class == 1

        original_b64 = image_to_base64(image.convert("RGB"))
        overlay_b64 = image_to_base64(result["overlay"])

        st.markdown(
            f'<div class="viewer-meta">VIEWING &middot; {uploaded_file.name} &middot; {width}&times;{height}px</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="scan-panel">
                <img src="data:image/png;base64,{original_b64}" />
                <div class="scan-tag">Original</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="scan-panel">
                <img src="data:image/png;base64,{overlay_b64}" />
                <div class="scan-tag">Grad-CAM</div>
            </div>
            """, unsafe_allow_html=True)

        # ---- Findings ----
        if is_positive:
            dot_color = label_color = fill_color = "var(--accent)"
            status_text = "Tumor Detected"
        else:
            dot_color = label_color = fill_color = "var(--text)"
            status_text = "No Tumor Detected"

        note = (
            "A negative reading does not rule out disease — this model is trained "
            "on a small research dataset and has not been clinically validated."
            if not is_positive else
            "This is a research model's output, not a diagnosis — any real finding "
            "requires review by a qualified radiologist."
        )
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            note = "Low-confidence reading — the model is uncertain about this scan. " + note

        shown_confidence = min(confidence, MAX_DISPLAY_CONFIDENCE)
        conf_text = (
            f"&gt;{MAX_DISPLAY_CONFIDENCE:.0%}" if confidence > MAX_DISPLAY_CONFIDENCE
            else f"{confidence:.0%}"
        )

        st.markdown(f"""
        <div class="findings-bar">
            <div class="findings-row">
                <span class="status-dot" style="background: {dot_color};"></span>
                <span class="status-label" style="color: {label_color};">{status_text}</span>
                <span class="conf-readout">{conf_text}</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width: {shown_confidence * 100:.1f}%; background: {fill_color};"></div>
            </div>
            <div class="findings-meta">ResNet18 &middot; fine-tuned &middot; attention basis: Grad-CAM, final residual block</div>
            <div class="findings-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    '<div class="footnote">MRI-CAD &middot; Research Prototype &middot; Not For Clinical Use</div>',
    unsafe_allow_html=True,
)
