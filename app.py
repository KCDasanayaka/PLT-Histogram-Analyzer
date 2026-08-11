from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from model import load_plt_model
from analyzer import analyze_plt_graph


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PLT Histogram Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

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
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    model, checkpoint, default_threshold = (
        load_plt_model(
            checkpoint_path=MODEL_PATH,
            device=DEVICE,
        )
    )

    return (
        model,
        checkpoint,
        default_threshold,
    )


try:

    model, checkpoint, DEFAULT_THRESHOLD = (
        load_model()
    )

except Exception as error:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(error)

    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0rem;
    }

    .sub-title {
        font-size: 1rem;
        color: #777;
        margin-bottom: 1.5rem;
    }

    .status-box {
        padding: 14px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
        border: 1px solid rgba(128,128,128,0.3);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        📈 PLT Histogram Analyzer
    </div>

    <div class="sub-title">
        Deep-learning-based PLT histogram curve analysis
        using a fixed 50% graph-height measurement.
    </div>
    """,
    unsafe_allow_html=True,
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
            "Choose the maximum value represented "
            "by the graph's X-axis."
        ),
    )

    segmentation_threshold = st.slider(
        "Curve Detection Sensitivity",
        min_value=0.15,
        max_value=0.75,
        value=float(
            DEFAULT_THRESHOLD
        ),
        step=0.01,
        help=(
            "Lower values detect fainter curve lines. "
            "Higher values require clearer curve lines."
        ),
    )

    st.divider()

    st.subheader(
        "Measurement Logic"
    )

    st.markdown(
        """
        The graph plotting height is interpreted as:

        **Top = 100%**

        **Middle = 50%**

        **X-axis = 0%**

        The 50% level is fixed and does not depend
        on the curve's highest point.
        """
    )

    st.divider()

    st.caption(
        f"Running on: {DEVICE}"
    )


# ============================================================
# INPUT
# ============================================================

st.subheader(
    "1. Upload PLT Histogram"
)

uploaded_file = st.file_uploader(
    "Upload a cropped PLT graph",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
    ],
    help=(
        "For best results, upload only the complete "
        "PLT histogram area with its X-axis and Y-axis visible."
    ),
)


if uploaded_file is None:

    st.info(
        "Upload a cropped PLT histogram graph to begin."
    )

    st.stop()


try:

    input_image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

except Exception as error:

    st.error(
        "Unable to read the uploaded image."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SHOW INPUT
# ============================================================

input_column, information_column = (
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
        caption="Uploaded PLT Graph",
        use_container_width=True,
    )


with information_column:

    st.markdown(
        "### Selected Settings"
    )

    st.metric(
        "X-axis Range",
        f"0–{x_axis_max} fL",
    )

    st.metric(
        "Curve Sensitivity",
        f"{segmentation_threshold:.2f}",
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Graph",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

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
                input_image=input_image,
                model=model,
                device=DEVICE,
                x_axis_max_fl=float(
                    x_axis_max
                ),
                segmentation_threshold=float(
                    segmentation_threshold
                ),
            )


        # ====================================================
        # STATUS
        # ====================================================

        st.divider()

        st.subheader(
            "2. Analysis Result"
        )

        status = result_json.get(
            "status",
            "unknown",
        )

        if status == "measure_width":

            st.success(
                "Two or more valid 50% intersections "
                "were detected."
            )

        elif status == "single_intersection":

            st.warning(
                "Only one valid 50% intersection "
                "was detected. Width cannot be calculated."
            )

        elif status == "no_intersection":

            st.info(
                "The detected curve does not intersect "
                "the fixed 50% graph level."
            )

        elif status == "reject":

            st.error(
                "The curve could not be traced reliably."
            )

        else:

            st.warning(
                f"Analysis status: {status}"
            )


        # ====================================================
        # MAIN METRICS
        # ====================================================

        intersection_count = int(
            result_json.get(
                "intersection_count",
                0,
            )
        )

        minimum_x = result_json.get(
            "minimum_intersection_fl"
        )

        maximum_x = result_json.get(
            "maximum_intersection_fl"
        )

        width_50 = result_json.get(
            "width_50_fl"
        )


        metric_1, metric_2, metric_3, metric_4 = (
            st.columns(4)
        )


        with metric_1:

            st.metric(
                "Intersections",
                intersection_count,
            )


        with metric_2:

            st.metric(
                "Minimum X",
                (
                    f"{minimum_x:.3f} fL"
                    if minimum_x is not None
                    else "N/A"
                ),
            )


        with metric_3:

            st.metric(
                "Maximum X",
                (
                    f"{maximum_x:.3f} fL"
                    if maximum_x is not None
                    else "N/A"
                ),
            )


        with metric_4:

            st.metric(
                "Width at 50%",
                (
                    f"{width_50:.3f} fL"
                    if width_50 is not None
                    else "N/A"
                ),
            )


        # ====================================================
        # RESULT IMAGE
        # ====================================================

        st.markdown(
            "### Detected Graph"
        )

        st.image(
            annotated_image,
            caption=(
                "Fixed 50% level and detected intersections"
            ),
            use_container_width=True,
        )


        # ====================================================
        # INTERSECTION VALUES
        # ====================================================

        intersections = result_json.get(
            "intersections",
            [],
        )


        if intersections:

            st.markdown(
                "### Detected 50% Intersections"
            )

            intersection_dataframe = (
                pd.DataFrame(
                    {
                        "Point": [
                            f"Point {index + 1}"
                            for index
                            in range(
                                len(
                                    intersections
                                )
                            )
                        ],
                        "X Position (fL)": [
                            round(
                                float(value),
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
        # COMPLETE RESULT TABLE
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
            "An error occurred while analyzing the graph."
        )

        st.exception(
            error
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research prototype for PLT histogram graph analysis. "
    "Not intended for medical diagnosis or clinical decisions."
)