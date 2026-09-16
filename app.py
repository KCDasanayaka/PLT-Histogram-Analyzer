# ============================================================
# app.py
# PLT Histogram Analyzer - Streamlit Application
# ============================================================

from __future__ import annotations

import io

from pathlib import Path

import streamlit as st

from PIL import Image

from model import (
    load_model
)

from analyzer import (
    analyze_plt_image,
    annotate_result,
    make_result_table,
    sensitivity_to_threshold
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(

    page_title="PLT Histogram Analyzer",

    page_icon="📈",

    layout="wide"

)


# ============================================================
# Model path
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

MODEL_PATH = (
    BASE_DIR
    /
    "models"
    /
    "plt_roi_peak50_v4.pt"
)


# ============================================================
# Page header
# ============================================================

st.title(
    "PLT Histogram Analyzer"
)

st.caption(
    "ROI-based peak and 50% intersection measurement - V4"
)

st.write(
    "Upload a PLT histogram image, configure the "
    "calibration values and sensitivity, and analyze "
    "the distribution inside the detected ROI."
)


# ============================================================
# Load model
# ============================================================

@st.cache_resource(
    show_spinner="Loading V4 model..."
)
def get_model():

    return load_model(
        MODEL_PATH
    )


try:

    (
        model,
        checkpoint_path,
        device,
        image_size,
        saved_threshold
    ) = get_model()

except Exception as exc:

    st.error(
        "Model loading failed."
    )

    st.exception(
        exc
    )

    st.info(
        "Make sure your renamed checkpoint exists at:\n\n"
        "models/plt_roi_peak50_v4.pt"
    )

    st.stop()


# ============================================================
# Sidebar controls
# ============================================================

with st.sidebar:

    st.header(
        "Analysis Settings"
    )

    st.subheader(
        "Model"
    )

    st.write(
        f"Checkpoint: `{checkpoint_path.name}`"
    )

    st.write(
        f"Device: `{device}`"
    )

    st.write(
        f"Image size: `{image_size} × {image_size}`"
    )

    st.markdown(
        "---"
    )

    # --------------------------------------------------------
    # Sensitivity
    # --------------------------------------------------------

    st.subheader(
        "Curve Detection"
    )

    sensitivity = st.slider(

        "Sensitivity",

        min_value=0,

        max_value=100,

        value=60,

        step=1,

        help=(
            "Higher sensitivity lowers the segmentation "
            "threshold and allows weaker curve probabilities."
        )

    )

    threshold = sensitivity_to_threshold(
        sensitivity
    )

    st.caption(
        f"Calculated segmentation threshold: "
        f"`{threshold:.3f}`"
    )

    st.markdown(
        "---"
    )

    # --------------------------------------------------------
    # X-axis calibration
    # --------------------------------------------------------

    st.subheader(
        "X-axis Calibration"
    )

    x_min = st.number_input(

        "X-axis minimum (fL)",

        min_value=-100000.0,

        max_value=100000.0,

        value=0.0,

        step=1.0,

        format="%.3f"

    )

    x_max = st.number_input(

        "X-axis maximum (fL)",

        min_value=-100000.0,

        max_value=100000.0,

        value=40.0,

        step=1.0,

        format="%.3f"

    )

    st.markdown(
        "---"
    )

    # --------------------------------------------------------
    # Y-axis calibration
    # --------------------------------------------------------

    st.subheader(
        "Y-axis Calibration"
    )

    y_max = st.number_input(

        "Y-axis maximum (fL)",

        min_value=0.001,

        max_value=100000.0,

        value=40.0,

        step=1.0,

        format="%.3f",

        help=(
            "Maximum Y-axis value used to estimate "
            "the vertical peak and 50% level."
        )

    )


# ============================================================
# File uploader
# ============================================================

uploaded = st.file_uploader(

    "Upload a PLT graph image",

    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],

    help=(
        "Upload a graph image containing the two "
        "vertical dashed reference boundaries."
    )

)


# ============================================================
# Application
# ============================================================

if uploaded is None:

    st.info(
        "Upload a PLT graph image to begin."
    )

else:

    try:

        image_bytes = uploaded.getvalue()

        pil_image = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert(
            "RGB"
        )

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            pil_image,
            caption="Input graph",
            use_container_width=True
        )

        st.markdown(
            "---"
        )

        # ----------------------------------------------------
        # Configuration summary
        # ----------------------------------------------------

        summary_col1, summary_col2, summary_col3 = (
            st.columns(3)
        )

        with summary_col1:

            st.metric(
                "Sensitivity",
                f"{sensitivity}%"
            )

        with summary_col2:

            st.metric(
                "Segmentation threshold",
                f"{threshold:.3f}"
            )

        with summary_col3:

            st.metric(
                "Y-axis maximum",
                f"{y_max:.3f} fL"
            )

        # ----------------------------------------------------
        # Analyze button
        # ----------------------------------------------------

        run = st.button(

            "Analyze Graph",

            type="primary",

            use_container_width=True

        )

        if run:

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if x_max <= x_min:

                st.error(
                    "X-axis maximum must be greater "
                    "than X-axis minimum."
                )

                st.stop()

            if y_max <= 0:

                st.error(
                    "Y-axis maximum must be greater than zero."
                )

                st.stop()

            # ------------------------------------------------
            # Analysis
            # ------------------------------------------------

            with st.spinner(

                "Detecting ROI, tracing curve and "
                "calculating 50% intersections..."

            ):

                result = analyze_plt_image(

                    model=model,

                    device=device,

                    image=pil_image,

                    x_min=x_min,

                    x_max=x_max,

                    threshold=threshold,

                    y_max=y_max,

                    sensitivity=sensitivity,

                    image_size=image_size

                )

            # ------------------------------------------------
            # Annotated image
            # ------------------------------------------------

            st.subheader(
                "Detected Graph"
            )

            annotated = annotate_result(
                result
            )

            st.image(

                annotated,

                caption=(
                    "Green = ROI boundaries | "
                    "Orange = 50% level | "
                    "Magenta = peak | "
                    "Red = intersections"
                ),

                use_container_width=True

            )

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            status = result.get(
                "status"
            )

            if status == "measure_width":

                st.success(
                    "Two or more genuine intersections "
                    "were detected."
                )

            elif status == "single_intersection":

                st.warning(
                    "Only one genuine 50% intersection "
                    "was detected. Width is not calculated."
                )

            elif status == "no_intersection":

                st.info(
                    "No genuine 50% intersection was "
                    "detected inside the ROI."
                )

            else:

                st.warning(
                    result.get(
                        "warning"
                    )
                    or
                    "The measurement could not be completed."
                )

            # ------------------------------------------------
            # Measurement table
            # ------------------------------------------------

            st.subheader(
                "Measurement Results"
            )

            table = make_result_table(
                result
            )

            st.dataframe(

                table,

                use_container_width=True,

                hide_index=True

            )

            # ------------------------------------------------
            # Width result
            # ------------------------------------------------

            st.subheader(
                "Width at 50%"
            )

            width_50 = result.get(
                "width_50_fl"
            )

            if width_50 is not None:

                st.metric(

                    "Width",

                    f"{width_50:.4f} fL"

                )

            else:

                st.info(

                    "Width is not available. "
                    "At least two genuine intersections "
                    "are required."

                )

            # ------------------------------------------------
            # Intersection details
            # ------------------------------------------------

            st.subheader(
                "Intersection Values"
            )

            intersections_fl = result.get(
                "intersections_fl",
                []
            )

            if intersections_fl:

                for index, value in enumerate(
                    intersections_fl,
                    start=1
                ):

                    st.write(
                        f"Crossing {index}: "
                        f"`{value:.4f} fL`"
                    )

            else:

                st.write(
                    "No X-axis intersection values available."
                )

            # ------------------------------------------------
            # Technical information
            # ------------------------------------------------

            with st.expander(
                "Technical Details"
            ):

                st.write(
                    f"Reference detection method: "
                    f"{result.get('reference_method')}"
                )

                st.write(
                    f"Reference confidence: "
                    f"{result.get('reference_confidence'):.4f}"
                )

                st.write(
                    f"ROI left: "
                    f"{result.get('reference_left_px')} px"
                )

                st.write(
                    f"ROI right: "
                    f"{result.get('reference_right_px')} px"
                )

                st.write(
                    f"Baseline: "
                    f"{result.get('baseline_y_px')} px"
                )

                st.write(
                    f"Sensitivity: "
                    f"{result.get('sensitivity')}"
                )

                st.write(
                    f"Threshold: "
                    f"{result.get('threshold'):.4f}"
                )

                st.write(
                    f"Device: `{device}`"
                )

                st.write(
                    f"Model image size: "
                    f"{image_size} × {image_size}"
                )

            # ------------------------------------------------
            # Warning
            # ------------------------------------------------

            if result.get(
                "warning"
            ):

                st.warning(
                    result[
                        "warning"
                    ]
                )

    except Exception as exc:

        st.error(
            f"Analysis error: "
            f"{type(exc).__name__}: {exc}"
        )

        st.exception(
            exc
        )