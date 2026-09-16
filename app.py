from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from analyzer import analyze_plt_image, annotate_result, make_result_table
from model import load_model

st.set_page_config(
    page_title="PLT Histogram Analyzer",
    page_icon="📈",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "plt_roi_peak50_unet_v8_best.pt"

st.title("PLT Histogram Analyzer")
st.caption("V8 ROI-based peak and 50% intersection measurement")

@st.cache_resource(show_spinner="Loading V8 model...")
def get_model():
    return load_model(MODEL_PATH if MODEL_PATH.exists() else None)

try:
    model, checkpoint_path, device = get_model()
except Exception as exc:
    st.error(f"Model loading failed: {type(exc).__name__}: {exc}")
    st.info("Place the downloaded V8 best checkpoint at models/plt_roi_peak50_unet_v8_best.pt")
    st.stop()

with st.sidebar:
    st.subheader("Model")
    st.write(f"Checkpoint: `{checkpoint_path.name}`")
    st.write(f"Device: `{device}`")
    threshold = st.slider("Curve segmentation threshold", 0.15, 0.65, 0.28, 0.01)
    st.markdown("---")
    st.subheader("X-axis")
    x_min = st.number_input("X-axis minimum (fL)", value=0.0, step=1.0)
    x_max = st.number_input("X-axis maximum (fL)", value=40.0, step=1.0)

uploaded = st.file_uploader(
    "Upload a PLT graph image",
    type=["png", "jpg", "jpeg", "webp"],
    help="Upload the graph image containing the two vertical dashed reference boundaries.",
)

if uploaded is not None:
    try:
        image_bytes = uploaded.getvalue()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(pil_image, caption="Uploaded image", use_container_width=True)

        run = st.button("Analyze graph", type="primary", use_container_width=True)
        if run:
            with st.spinner("Detecting ROI, tracing curve and calculating 50% intersections..."):
                result = analyze_plt_image(
                    model,
                    device,
                    pil_image,
                    x_min,
                    x_max,
                    threshold,
                )

            annotated = annotate_result(result)
            st.subheader("Detected graph")
            st.image(annotated, caption="Green = ROI boundaries | Orange = 50% level | Magenta = peak | Red = genuine intersections", use_container_width=True)

            st.subheader("Measurement results")
            st.dataframe(make_result_table(result), use_container_width=True, hide_index=True)

            if result.get("status") == "measure_width":
                st.success(f"Width at 50%: {result['width_50_fl']:.3f} fL")
            elif result.get("status") == "single_intersection":
                st.warning("Only one genuine 50% intersection was detected, so width is not calculated.")
            elif result.get("status") == "no_intersection":
                st.info("The tracked curve does not genuinely cross the 50% level inside the ROI.")
            else:
                st.warning(result.get("warning") or "The measurement could not be completed.")

    except Exception as exc:
        st.error(f"ERROR: {type(exc).__name__}: {exc}")
else:
    st.info("Upload a PLT graph image to begin.")
