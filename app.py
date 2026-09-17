from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from analyzer import (
    analyze_plt_image,
    annotate_result,
    make_result_table,
    SENSITIVITY_PRESETS,
)
from model import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PLT Histogram Analyzer",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# IMPORTANT:
# This is the renamed checkpoint filename you are now using.
MODEL_PATH = PROJECT_DIR / "models" / "plt_roi_peak50_unet_v8_best.pt"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("PLT Histogram Analyzer")
st.caption(
    "V8 ROI-based peak and 50% intersection measurement"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading V8 model...")
def get_model():
    return load_model(MODEL_PATH)


try:
    model, checkpoint_path, device = get_model()

except Exception as exc:
    st.error(
        f"Model loading failed: {type(exc).__name__}: {exc}"
    )

    st.info(
        "Make sure the checkpoint exists at:\n\n"
        "`models/plt_roi_peak50_unet_v8_best.pt`"
    )

    st.stop()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.subheader("Analysis Settings")

    # --------------------------------------------------------
    # Sensitivity
    # --------------------------------------------------------

    sensitivity_name = st.selectbox(
        "Curve detection sensitivity",
        options=list(SENSITIVITY_PRESETS.keys()),
        index=1,
        help=(
            "Higher sensitivity keeps weaker curve responses. "
            "Lower sensitivity rejects more weak/noisy responses."
        ),
    )

    selected_threshold = SENSITIVITY_PRESETS[sensitivity_name]

    st.caption(
        f"Segmentation threshold: `{selected_threshold:.2f}`"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # X Axis
    # --------------------------------------------------------

    st.subheader("X-axis")

    x_min = st.number_input(
        "X-axis minimum (fL)",
        min_value=-1000.0,
        max_value=10000.0,
        value=0.0,
        step=1.0,
    )

    x_max = st.number_input(
        "X-axis maximum (fL)",
        min_value=-1000.0,
        max_value=10000.0,
        value=40.0,
        step=1.0,
    )

    # --------------------------------------------------------
    # Y Axis
    # --------------------------------------------------------

    st.subheader("Y-axis")

    y_max = st.number_input(
        "Y-axis maximum (fL)",
        min_value=0.001,
        max_value=10000.0,
        value=10.0,
        step=0.5,
        help=(
            "Maximum numerical value represented by the top "
            "of the plotting area. This is used to convert "
            "the detected peak and 50% level into Y-axis values."
        ),
    )

    st.markdown("---")

    st.subheader("Model")

    st.write(
        f"Checkpoint: `{checkpoint_path.name}`"
    )

    st.write(
        f"Device: `{device}`"
    )


# ============================================================
# MAIN UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload a PLT graph image",
    type=["png", "jpg", "jpeg", "webp"],
    help=(
        "Upload the PLT graph image containing the "
        "two vertical dashed reference boundaries."
    ),
)


# ============================================================
# IMAGE + ANALYSIS
# ============================================================

if uploaded is not None:

    try:

        image_bytes = uploaded.getvalue()

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        st.image(
            pil_image,
            caption="Uploaded PLT graph",
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Parameter summary
        # ----------------------------------------------------

        st.markdown("### Selected analysis settings")

        setting_col1, setting_col2, setting_col3 = st.columns(3)

        with setting_col1:
            st.metric(
                "Sensitivity",
                sensitivity_name,
            )

        with setting_col2:
            st.metric(
                "X-axis",
                f"{x_min:g} – {x_max:g} fL",
            )

        with setting_col3:
            st.metric(
                "Y-axis maximum",
                f"{y_max:g} fL",
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        valid_inputs = True

        if x_max <= x_min:
            st.error(
                "X-axis maximum must be greater than X-axis minimum."
            )
            valid_inputs = False

        if y_max <= 0:
            st.error(
                "Y-axis maximum must be greater than 0."
            )
            valid_inputs = False

        # ----------------------------------------------------
        # Analyze button
        # ----------------------------------------------------

        analyze_button = st.button(
            "Analyze graph",
            type="primary",
            use_container_width=True,
            disabled=not valid_inputs,
        )

        # ----------------------------------------------------
        # Analysis
        # ----------------------------------------------------

        if analyze_button:

            with st.spinner(
                "Detecting ROI, tracing curve and calculating 50% intersections..."
            ):

                result = analyze_plt_image(
                    model=model,
                    device=device,
                    image_input=pil_image,
                    x_min_fl=x_min,
                    x_max_fl=x_max,
                    y_max_fl=y_max,
                    threshold=selected_threshold,
                )

            # ------------------------------------------------
            # Detected image
            # ------------------------------------------------

            st.subheader("Detected graph")

            annotated = annotate_result(result)

            st.image(
                annotated,
                caption=(
                    "Green = ROI boundaries | "
                    "Orange = 50% level | "
                    "Magenta = detected peak | "
                    "Red = genuine 50% intersections"
                ),
                use_container_width=True,
            )

            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            st.subheader("Measurement results")

            result_table = make_result_table(result)

            st.dataframe(
                result_table,
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------
            # Status messages
            # ------------------------------------------------

            status = result.get("status")

            if status == "measure_width":

                st.success(
                    f"Width at 50%: "
                    f"{result['width_50_fl']:.3f} fL"
                )

            elif status == "single_intersection":

                st.warning(
                    "Only one genuine 50% intersection was detected, "
                    "so width is not calculated."
                )

            elif status == "no_intersection":

                st.info(
                    "The tracked curve does not genuinely cross "
                    "the 50% level inside the ROI."
                )

            elif status == "roi_not_found":

                st.error(
                    "The two vertical dashed ROI boundaries "
                    "could not be detected reliably."
                )

            elif status == "curve_not_found":

                st.error(
                    "A reliable curve trace could not be found "
                    "inside the ROI."
                )

            elif status == "peak_not_found":

                st.error(
                    "A sufficiently strong curve peak could not "
                    "be detected."
                )

            else:

                warning = result.get("warning")

                if warning:
                    st.warning(warning)

            # ------------------------------------------------
            # Additional information
            # ------------------------------------------------

            with st.expander("Detailed analysis information"):

                st.json(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in {
                            "image_rgb",
                            "curve_probability",
                            "curve_mask",
                            "centerline_x",
                            "centerline_y",
                            "centerline_confidence",
                        }
                    }
                )

    except Exception as exc:

        st.error(
            f"ERROR: {type(exc).__name__}: {exc}"
        )

else:

    st.info(
        "Upload a PLT graph image to begin."
    )