from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from PIL import Image

from model import (
    load_plt_model,
)

from analyzer import (
    analyze_plt_graph,
    MIN_PEAK_HEIGHT_RATIO,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=(
        "PLT Histogram Analyzer"
    ),
    page_icon="📈",
    layout="wide",
    initial_sidebar_state=(
        "expanded"
    ),
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(
        __file__
    )
    .resolve()
    .parent
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "plt_curve_unet_fixed50_best.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL ONCE
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model checkpoint "
            "was not found:\n"
            f"{MODEL_PATH}"
        )

    (
        model,
        checkpoint,
        default_threshold,
    ) = load_plt_model(
        checkpoint_path=(
            MODEL_PATH
        ),
        device=DEVICE,
    )

    return (
        model,
        checkpoint,
        default_threshold,
    )


try:

    (
        model,
        checkpoint,
        DEFAULT_THRESHOLD,
    ) = load_model()

except Exception as error:

    st.error(
        "Unable to load "
        "the trained model."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 PLT Histogram Analyzer"
)

st.caption(
    "Detects the PLT distribution curve and "
    "measures intersections at 50% of the "
    "detected curve peak."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Analysis Settings"
    )

    x_axis_max = st.selectbox(
        "X-axis maximum (fL)",
        options=[
            40,
            30,
        ],
        index=0,
        help=(
            "Select the maximum X-axis "
            "value shown on the graph."
        ),
    )

    segmentation_threshold = (
        st.slider(
            "Curve Detection Sensitivity",
            min_value=0.15,
            max_value=0.75,
            value=float(
                DEFAULT_THRESHOLD
            ),
            step=0.01,
            help=(
                "Lower values detect "
                "fainter curves. Higher "
                "values require clearer "
                "curve pixels."
            ),
        )
    )

    st.divider()

    st.subheader(
        "Measurement Logic"
    )

    st.markdown(
        f"""
The system first detects the actual PLT curve.

**Step 1 — Distribution check**

The curve peak must reach at least:

**{MIN_PEAK_HEIGHT_RATIO * 100:.0f}% of the graph height**

If the distribution is lower than this, no 50% measurement is performed.

**Step 2 — Curve 50%**

For a valid distribution:

**Curve peak = 100%**

**Baseline = 0%**

The measurement line is placed at:

**50% of the detected curve peak**

It is not the fixed middle of the Y-axis.
"""
    )

    st.divider()

    st.caption(
        f"Running on: {DEVICE}"
    )


# ============================================================
# UPLOAD
# ============================================================

st.subheader(
    "1. Upload PLT Histogram"
)

uploaded_file = (
    st.file_uploader(
        "Upload a cropped PLT graph",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        help=(
            "Upload the complete cropped "
            "PLT graph with the X-axis "
            "and Y-axis visible."
        ),
    )
)


if uploaded_file is None:

    st.info(
        "Upload a PLT histogram "
        "graph to begin."
    )

    st.stop()


try:

    input_image = (
        Image.open(
            uploaded_file
        )
        .convert(
            "RGB"
        )
    )

except Exception as error:

    st.error(
        "Unable to read "
        "the uploaded image."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# SHOW INPUT
# ============================================================

input_column, settings_column = (
    st.columns(
        [
            2,
            1,
        ]
    )
)


with input_column:

    st.image(
        input_image,
        caption=(
            "Uploaded PLT Histogram"
        ),
        use_container_width=True,
    )


with settings_column:

    st.markdown(
        "### Current Settings"
    )

    st.metric(
        "X-axis Range",
        f"0–{x_axis_max} fL",
    )

    st.metric(
        "Curve Sensitivity",
        f"{segmentation_threshold:.2f}",
    )

    st.metric(
        "Minimum Peak Height",
        (
            f"{MIN_PEAK_HEIGHT_RATIO * 100:.0f}%"
        ),
    )


# ============================================================
# ANALYZE
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Graph",
    type="primary",
    use_container_width=True,
)


if analyze_button:

    try:

        with st.spinner(
            "Analyzing PLT histogram..."
        ):

            (
                annotated_image,
                result_table,
                result_json,
            ) = analyze_plt_graph(
                input_image=(
                    input_image
                ),
                model=model,
                device=DEVICE,
                x_axis_max_fl=float(
                    x_axis_max
                ),
                segmentation_threshold=float(
                    segmentation_threshold
                ),
            )

        st.divider()

        st.subheader(
            "2. Analysis Result"
        )

        status = (
            result_json.get(
                "status",
                "unknown",
            )
        )

        reason = (
            result_json.get(
                "reason",
                "None",
            )
        )

        # ====================================================
        # RESULT MESSAGE
        # ====================================================

        if reason == "peak_too_low":

            st.info(
                "The detected PLT distribution "
                "is too low for half-peak "
                "measurement. No 50% point "
                "was selected."
            )

        elif status == "measure_width":

            st.success(
                "Two or more valid "
                "half-peak intersections "
                "were detected."
            )

        elif status == (
            "single_intersection"
        ):

            st.warning(
                "Only one valid half-peak "
                "intersection was detected. "
                "Width cannot be calculated."
            )

        elif status == (
            "no_intersection"
        ):

            st.info(
                "No valid half-peak "
                "intersection was detected."
            )

        elif status == "reject":

            st.error(
                "The PLT curve could not "
                "be traced reliably."
            )

        else:

            st.warning(
                f"Analysis status: {status}"
            )

        # ====================================================
        # METRICS
        # ====================================================

        intersection_count = int(
            result_json.get(
                "intersection_count",
                0,
            )
        )

        minimum_x = (
            result_json.get(
                "minimum_intersection_fl"
            )
        )

        maximum_x = (
            result_json.get(
                "maximum_intersection_fl"
            )
        )

        width_50 = (
            result_json.get(
                "width_50_fl"
            )
        )

        peak_height_percent = (
            result_json.get(
                "peak_height_percent"
            )
        )

        metric_1, metric_2, metric_3, metric_4 = (
            st.columns(4)
        )


        with metric_1:

            st.metric(
                "Peak Height",
                (
                    f"{peak_height_percent:.1f}%"
                    if peak_height_percent
                    is not None
                    else "N/A"
                ),
            )


        with metric_2:

            st.metric(
                "Intersections",
                intersection_count,
            )


        with metric_3:

            st.metric(
                "X Range",
                (
                    f"{minimum_x:.2f} – "
                    f"{maximum_x:.2f} fL"
                    if (
                        minimum_x
                        is not None
                        and maximum_x
                        is not None
                    )
                    else "N/A"
                ),
            )


        with metric_4:

            st.metric(
                "Width at 50%",
                (
                    f"{width_50:.3f} fL"
                    if width_50
                    is not None
                    else "N/A"
                ),
            )

        # ====================================================
        # ANNOTATED GRAPH
        # ====================================================

        st.markdown(
            "### Detected Graph"
        )

        st.image(
            annotated_image,
            caption=(
                "Detected PLT curve and "
                "half-peak intersections"
            ),
            use_container_width=True,
        )

        # ====================================================
        # INTERSECTION TABLE
        # ====================================================

        intersections = (
            result_json.get(
                "intersections",
                [],
            )
        )

        if intersections:

            st.markdown(
                "### Detected 50% Points"
            )

            intersection_dataframe = (
                pd.DataFrame(
                    {
                        "Point": [
                            (
                                f"Point "
                                f"{index + 1}"
                            )
                            for index
                            in range(
                                len(
                                    intersections
                                )
                            )
                        ],

                        "X Position (fL)": [
                            round(
                                float(
                                    value
                                ),
                                3,
                            )
                            for value
                            in intersections
                        ],
                    }
                )
            )

            st.dataframe(
                intersection_dataframe,
                use_container_width=True,
                hide_index=True,
            )

        # ====================================================
        # FULL TABLE
        # ====================================================

        st.markdown(
            "### Measurement Details"
        )

        st.dataframe(
            result_table,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # WARNING
        # ====================================================

        warning = result_json.get(
            "warning"
        )

        if (
            warning
            and warning != "None"
        ):

            st.warning(
                warning
            )

        # ====================================================
        # JSON
        # ====================================================

        with st.expander(
            "View Detailed JSON Result"
        ):

            st.json(
                result_json
            )

    except Exception as error:

        st.error(
            "An error occurred while "
            "analyzing the graph."
        )

        st.exception(
            error
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research prototype for PLT histogram "
    "graph analysis. Not intended for "
    "medical diagnosis or clinical decisions."
)