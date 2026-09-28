from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from analyzer import (
    analyze_plt_image,
    annotate_result,
    make_result_table,
    get_sensitivity_description,
)

from model import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PLT Histogram Analyzer",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# PROJECT / MODEL PATH
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

# User renamed the checkpoint using the second method.
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "plt_roi_peak50_unet_v8_best.pt"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "PLT Histogram Analyzer"
)

st.caption(
    "V9 — ROI-based 50% curve measurement with manual X-axis calibration"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(
    show_spinner="Loading PLT model..."
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
        "Model loading failed:\n\n"
        f"{type(exc).__name__}: {exc}"
    )

    st.info(
        "Expected model file:\n\n"
        "`models/plt_roi_peak50_unet_v8_best.pt`"
    )

    st.stop()


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
    help=(
        "Upload the original PLT graph image. "
        "The graph should contain the two vertical dashed "
        "ROI boundaries and its horizontal X-axis."
    ),
)


# ============================================================
# SIDEBAR — ANALYSIS SETTINGS
# ============================================================

with st.sidebar:

    st.subheader(
        "Analysis Settings"
    )

    # --------------------------------------------------------
    # MANUAL SENSITIVITY
    # --------------------------------------------------------

    st.markdown(
        "**Curve detection sensitivity**"
    )

    sensitivity_threshold = st.slider(
        "Sensitivity threshold",
        min_value=0.05,
        max_value=0.50,
        value=0.22,
        step=0.01,
        help=(
            "Lower value = higher sensitivity. "
            "Use a lower value for faint or low-visibility curves. "
            "Use a higher value when the graph contains noise."
        ),
    )

    st.caption(
        f"Current threshold: `{sensitivity_threshold:.2f}`"
    )

    st.caption(
        get_sensitivity_description(
            sensitivity_threshold
        )
    )

    # --------------------------------------------------------
    # Y AXIS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Y-axis"
    )

    y_max = st.number_input(
        "Y-axis maximum (fL)",
        min_value=0.001,
        max_value=10000.0,
        value=10.0,
        step=0.5,
        help=(
            "Maximum numerical value represented at "
            "the top of the plotting area."
        ),
    )

    # --------------------------------------------------------
    # MODEL INFO
    # --------------------------------------------------------

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
# MAIN APP
# ============================================================

if uploaded is None:

    st.info(
        "Upload a PLT graph image to begin."
    )

    st.markdown(
        """
### How to calibrate the X-axis

After uploading the graph, place the two calibration
positions on the **actual horizontal X-axis**:

- X-axis minimum position = where the numerical X-axis starts
- X-axis maximum position = where the numerical X-axis ends

These are separate from the green ROI boundaries.

For a 0–40 fL graph, the application then mathematically
reconstructs:

`0 → 10 → 20 → 30 → 40 fL`

between those manually selected endpoints.

No OCR or automatic recognition of the printed 10/20/30/40
numbers is used.
        """
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

        image_width = image.width
        image_height = image.height

        st.image(
            image,
            caption=(
                f"Uploaded graph — "
                f"{image_width} × {image_height}px"
            ),
            use_container_width=True,
        )

        # ====================================================
        # MANUAL X-AXIS CALIBRATION
        # ====================================================

        st.subheader(
            "Manual X-axis calibration"
        )

        st.info(
            "Set the two positions on the real horizontal "
            "X-axis. Do NOT use the green ROI boundaries unless "
            "they actually coincide with the numerical X-axis endpoints."
        )

        calibration_col1, calibration_col2 = st.columns(
            2
        )

        # ----------------------------------------------------
        # Defaults
        #
        # These are ONLY starting values.
        # They are not an automatic axis detector.
        # ----------------------------------------------------

        default_axis_start = max(
            0,
            int(
                round(
                    image_width
                    * 0.08
                )
            ),
        )

        default_axis_end = min(
            image_width - 1,
            int(
                round(
                    image_width
                    * 0.88
                )
            ),
        )

        with calibration_col1:

            x_axis_start_px = st.number_input(
                "X-axis minimum position (px)",
                min_value=0,
                max_value=image_width - 1,
                value=default_axis_start,
                step=1,
                help=(
                    "Pixel position where the numerical "
                    "X-axis starts."
                ),
            )

        with calibration_col2:

            x_axis_end_px = st.number_input(
                "X-axis maximum position (px)",
                min_value=0,
                max_value=image_width - 1,
                value=default_axis_end,
                step=1,
                help=(
                    "Pixel position where the numerical "
                    "X-axis ends."
                ),
            )

        # ----------------------------------------------------
        # Numerical X-axis
        # ----------------------------------------------------

        st.markdown(
            "**Numerical X-axis range**"
        )

        x_col1, x_col2 = st.columns(
            2
        )

        with x_col1:

            x_min = st.number_input(
                "X-axis minimum (fL)",
                min_value=-1000.0,
                max_value=10000.0,
                value=0.0,
                step=1.0,
            )

        with x_col2:

            x_max = st.number_input(
                "X-axis maximum (fL)",
                min_value=-1000.0,
                max_value=10000.0,
                value=40.0,
                step=1.0,
            )

        # ----------------------------------------------------
        # Calibration preview
        # ----------------------------------------------------

        if (
            x_axis_end_px
            <= x_axis_start_px
        ):

            st.error(
                "X-axis maximum position must be greater "
                "than X-axis minimum position."
            )

            axis_valid = False

        else:

            axis_valid = True

        if (
            x_max
            <= x_min
        ):

            st.error(
                "X-axis maximum value must be greater "
                "than X-axis minimum value."
            )

            axis_valid = False

        if (
            y_max
            <= 0
        ):

            st.error(
                "Y-axis maximum must be greater than 0."
            )

            axis_valid = False

        # ----------------------------------------------------
        # Show calibration values
        # ----------------------------------------------------

        if axis_valid:

            st.caption(
                (
                    f"Manual calibration: "
                    f"{x_axis_start_px}px → {x_min:g} fL    |    "
                    f"{x_axis_end_px}px → {x_max:g} fL"
                )
            )

            # Mathematical reconstruction of expected
            # major ticks. These are NOT detected from the image.
            major_tick_values = []

            start_10 = int(
                __import__(
                    "math"
                ).ceil(
                    x_min / 10.0
                )
                * 10
            )

            end_10 = int(
                __import__(
                    "math"
                ).floor(
                    x_max / 10.0
                )
                * 10
            )

            if end_10 >= start_10:

                major_tick_values = list(
                    range(
                        start_10,
                        end_10 + 1,
                        10,
                    )
                )

            st.caption(
                (
                    "Reconstructed major ticks: "
                    + (
                        ", ".join(
                            f"{v:g} fL"
                            for v in major_tick_values
                        )
                        if major_tick_values
                        else "None"
                    )
                )
            )

        # ====================================================
        # SELECTED SETTINGS SUMMARY
        # ====================================================

        st.markdown(
            "---"
        )

        st.subheader(
            "Selected analysis settings"
        )

        s1, s2, s3 = st.columns(
            3
        )

        with s1:

            st.metric(
                "Sensitivity",
                f"{sensitivity_threshold:.2f}",
            )

        with s2:

            st.metric(
                "X-axis",
                f"{x_min:g} – {x_max:g} fL",
            )

        with s3:

            st.metric(
                "Y max",
                f"{y_max:g} fL",
            )

        # ====================================================
        # ANALYZE
        # ====================================================

        analyze_button = st.button(
            "Analyze graph",
            type="primary",
            use_container_width=True,
            disabled=not axis_valid,
        )

        if analyze_button:

            with st.spinner(
                "Detecting ROI, enhancing the curve, "
                "tracing the graph and calculating 50% intersections..."
            ):

                result = analyze_plt_image(
                    model=model,
                    device=device,
                    image_input=image,
                    x_min_fl=x_min,
                    x_max_fl=x_max,
                    y_max_fl=y_max,
                    threshold=sensitivity_threshold,
                    manual_axis_start_px=x_axis_start_px,
                    manual_axis_end_px=x_axis_end_px,
                )

            # =================================================
            # ANNOTATION
            # =================================================

            st.subheader(
                "Analysis result"
            )

            annotated = annotate_result(
                result
            )

            st.image(
                annotated,
                caption=(
                    "Green = ROI boundaries | "
                    "Yellow = manually calibrated X-axis | "
                    "Yellow tick marks = reconstructed 10 fL spacing | "
                    "Orange = 50% level | "
                    "Magenta = peak | "
                    "Red = genuine intersections"
                ),
                use_container_width=True,
            )

            # =================================================
            # RESULT TABLE
            # =================================================

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

            # =================================================
            # STATUS
            # =================================================

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
                    (
                        "Only one genuine 50% intersection "
                        "was detected. Width is not calculated."
                    )
                )

            elif status == "no_intersection":

                st.info(
                    (
                        "The tracked curve does not genuinely "
                        "cross the 50% level inside the ROI."
                    )
                )

            elif status == "curve_not_found":

                st.error(
                    (
                        "A reliable curve trace could not be "
                        "established. Try lowering the sensitivity "
                        "threshold."
                    )
                )

            elif status == "roi_not_found":

                st.error(
                    (
                        "The two vertical dashed ROI boundaries "
                        "could not be detected."
                    )
                )

            elif status == "peak_not_found":

                st.error(
                    (
                        "A reliable curve peak could not be detected."
                    )
                )

            else:

                if result.get(
                    "warning"
                ):

                    st.warning(
                        result[
                            "warning"
                        ]
                    )

            # =================================================
            # CALIBRATION DETAILS
            # =================================================

            with st.expander(
                "X-axis calibration details"
            ):

                st.write(
                    "Manual X-axis start:",
                    result.get(
                        "x_axis_start_px"
                    ),
                    "px",
                    "→",
                    f"{x_min:g} fL",
                )

                st.write(
                    "Manual X-axis end:",
                    result.get(
                        "x_axis_end_px"
                    ),
                    "px",
                    "→",
                    f"{x_max:g} fL",
                )

                st.write(
                    "Pixels per fL:",
                    result.get(
                        "pixels_per_fl"
                    ),
                )

                st.write(
                    "Calibration method:",
                    result.get(
                        "x_axis_method"
                    ),
                )

            # =================================================
            # DETAILED OUTPUT
            # =================================================

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
            (
                f"ERROR: "
                f"{type(exc).__name__}: {exc}"
            )
        )