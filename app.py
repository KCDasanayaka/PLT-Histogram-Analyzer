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
# PAGE
# ============================================================

st.set_page_config(
    page_title="PLT Histogram Analyzer",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# MODEL
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "plt_roi_peak50_unet_v8_best.pt"
)


st.title(
    "PLT Histogram Analyzer"
)

st.caption(
    "V8 ROI-based PLT 50% width analyzer"
)


@st.cache_resource(
    show_spinner="Loading model..."
)
def get_model():

    return load_model(
        MODEL_PATH
    )


try:

    model, checkpoint_path, device = (
        get_model()
    )

except Exception as exc:

    st.error(
        f"Model loading failed: "
        f"{type(exc).__name__}: {exc}"
    )

    st.info(
        "Expected model:\n\n"
        "`models/plt_roi_peak50_unet_v8_best.pt`"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader(
        "Analysis Settings"
    )

    # --------------------------------------------------------
    # Sensitivity preset
    # --------------------------------------------------------

    sensitivity = st.selectbox(
        "Curve detection sensitivity",
        options=list(
            SENSITIVITY_PRESETS.keys()
        ),
        index=1,
        help=(
            "Use High or Very High for faint/low-visibility "
            "curves. Higher sensitivity can also increase "
            "noise, so review the annotated result."
        ),
    )

    preset_threshold = SENSITIVITY_PRESETS[
        sensitivity
    ][
        "threshold"
    ]

    # --------------------------------------------------------
    # Fine tuning
    # --------------------------------------------------------

    threshold = st.slider(
        "Fine-tune curve sensitivity",
        min_value=0.08,
        max_value=0.45,
        value=float(
            preset_threshold
        ),
        step=0.01,
        help=(
            "Lower threshold = higher sensitivity. "
            "Increase sensitivity for faint curves."
        ),
    )

    st.caption(
        f"Current threshold: `{threshold:.2f}`"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    st.subheader(
        "X-axis range"
    )

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

    st.caption(
        "Enter the actual numerical range shown by the graph. "
        "The application no longer tries to recognize 10/20/30/40 labels."
    )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    st.subheader(
        "Y-axis range"
    )

    y_max = st.number_input(
        "Y-axis maximum (fL)",
        min_value=0.001,
        max_value=10000.0,
        value=10.0,
        step=0.5,
    )

    st.markdown("---")

    st.subheader(
        "Model"
    )

    st.write(
        f"Checkpoint: `{checkpoint_path.name}`"
    )

    st.write(
        f"Device: `{device}`"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload a PLT graph image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
    ],
)


# ============================================================
# MAIN
# ============================================================

if uploaded is None:

    st.info(
        "Upload a PLT graph image to begin."
    )

else:

    try:

        image = Image.open(
            io.BytesIO(
                uploaded.getvalue()
            )
        ).convert(
            "RGB"
        )

        st.image(
            image,
            caption="Uploaded PLT graph",
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Settings summary
        # ----------------------------------------------------

        st.markdown(
            "### Selected settings"
        )

        c1, c2, c3 = st.columns(
            3
        )

        with c1:

            st.metric(
                "Sensitivity",
                sensitivity,
            )

        with c2:

            st.metric(
                "X-axis",
                f"{x_min:g} – {x_max:g} fL",
            )

        with c3:

            st.metric(
                "Y-axis maximum",
                f"{y_max:g} fL",
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        valid = True

        if x_max <= x_min:

            st.error(
                "X-axis maximum must be greater "
                "than X-axis minimum."
            )

            valid = False

        if y_max <= 0:

            st.error(
                "Y-axis maximum must be greater than 0."
            )

            valid = False

        # ----------------------------------------------------
        # Analyze
        # ----------------------------------------------------

        analyze = st.button(
            "Analyze graph",
            type="primary",
            use_container_width=True,
            disabled=not valid,
        )

        if analyze:

            with st.spinner(
                "Detecting ROI, locating the X-axis, "
                "enhancing the curve and calculating 50% intersections..."
            ):

                result = analyze_plt_image(
                    model=model,
                    device=device,
                    image_input=image,
                    x_min_fl=x_min,
                    x_max_fl=x_max,
                    y_max_fl=y_max,
                    threshold=threshold,
                    sensitivity_name=sensitivity,
                )

            # ------------------------------------------------
            # Annotation
            # ------------------------------------------------

            st.subheader(
                "Detected graph"
            )

            annotated = annotate_result(
                result
            )

            st.image(
                annotated,
                caption=(
                    "Green = ROI boundaries | "
                    "Yellow/Cyan = actual X-axis endpoints | "
                    "Orange = 50% level | "
                    "Magenta = peak | "
                    "Red = genuine intersections"
                ),
                use_container_width=True,
            )

            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            st.subheader(
                "Measurement results"
            )

            st.dataframe(
                make_result_table(
                    result
                ),
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            status = result.get(
                "status"
            )

            if status == "measure_width":

                st.success(
                    (
                        f"Width at 50%: "
                        f"{result['width_50_fl']:.3f} fL"
                    )
                )

            elif status == "single_intersection":

                st.warning(
                    "Only one genuine 50% intersection "
                    "was detected, so width is not calculated."
                )

            elif status == "no_intersection":

                st.info(
                    "The traced curve does not genuinely "
                    "cross the 50% level inside the ROI."
                )

            elif status == "curve_not_found":

                st.error(
                    "The curve could not be reliably traced. "
                    "Try High or Very High sensitivity."
                )

            elif status == "roi_not_found":

                st.error(
                    "The two dashed ROI boundaries "
                    "could not be detected."
                )

            elif status == "peak_not_found":

                st.error(
                    "A reliable curve peak could not be detected."
                )

            else:

                warning = result.get(
                    "warning"
                )

                if warning:

                    st.warning(
                        warning
                    )

            # ------------------------------------------------
            # Axis debug information
            # ------------------------------------------------

            with st.expander(
                "X-axis detection details"
            ):

                st.write(
                    "Actual X-axis start pixel:",
                    result.get(
                        "x_axis_start_px"
                    ),
                )

                st.write(
                    "Actual X-axis end pixel:",
                    result.get(
                        "x_axis_end_px"
                    ),
                )

                st.write(
                    "X-axis Y pixel:",
                    result.get(
                        "x_axis_y_px"
                    ),
                )

                st.write(
                    "X-axis detection method:",
                    result.get(
                        "x_axis_method"
                    ),
                )

                st.write(
                    "X-axis confidence:",
                    result.get(
                        "x_axis_confidence"
                    ),
                )

            # ------------------------------------------------
            # Detailed output
            # ------------------------------------------------

            with st.expander(
                "Detailed analysis output"
            ):

                filtered = {
                    key: value
                    for key, value in result.items()
                    if key not in {
                        "image_rgb",
                        "curve_probability",
                        "visual_curve_score",
                        "curve_mask",
                        "centerline_x",
                        "centerline_y",
                        "centerline_confidence",
                    }
                }

                st.json(
                    filtered
                )

    except Exception as exc:

        st.error(
            f"ERROR: {type(exc).__name__}: {exc}"
        ) 