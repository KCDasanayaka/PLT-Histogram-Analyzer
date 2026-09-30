from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from analyzer import (
    analyze_plt_image,
    annotate_result,
    make_result_table,
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
# PROJECT / MODEL
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


# ============================================================
# HEADER
# ============================================================

st.title(
    "PLT Histogram Analyzer"
)

st.caption(
    "ROI-based PLT 50% width analyzer"
)


# ============================================================
# LOAD MODEL
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
        f"Model loading failed: "
        f"{type(exc).__name__}: {exc}"
    )

    st.info(
        "Expected checkpoint:\n\n"
        "`models/plt_roi_peak50_unet_v8_best.pt`"
    )

    st.stop()


# ============================================================
# UPLOAD
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


if uploaded is None:

    st.info(
        "Upload a PLT graph image to begin."
    )

    st.markdown(
        """
### Measurement workflow

The application will:

1. Detect the two vertical dashed ROI boundaries.
2. Detect the actual horizontal X-axis.
3. Run the trained curve segmentation model.
4. Enhance faint curve visibility.
5. Trace the actual curve inside the ROI.
6. Detect the highest point reached by the curve.
7. Calculate 50% of the detected peak height.
8. Find only genuine 50% intersections.
9. Convert intersection positions using the calibrated X-axis.
10. Calculate width only when at least two intersections exist.

The numerical X-axis values are entered manually. The application does **not**
try to recognize the printed `10`, `20`, `30`, `40` labels.
        """
    )

    st.stop()


# ============================================================
# IMAGE
# ============================================================

try:

    image = Image.open(
        io.BytesIO(
            uploaded.getvalue()
        )
    ).convert(
        "RGB"
    )

except Exception as exc:

    st.error(
        f"Image loading failed: "
        f"{type(exc).__name__}: {exc}"
    )

    st.stop()


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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader(
        "Analysis Settings"
    )

    # --------------------------------------------------------
    # CURVE SENSITIVITY
    # --------------------------------------------------------

    sensitivity_threshold = st.slider(
        "Curve sensitivity",
        min_value=0.05,
        max_value=0.50,
        value=0.22,
        step=0.01,
        help=(
            "Lower value = higher sensitivity. "
            "Use lower values for faint curves."
        ),
    )

    if sensitivity_threshold <= 0.12:

        st.caption(
            "Very high sensitivity"
        )

    elif sensitivity_threshold <= 0.18:

        st.caption(
            "High sensitivity — useful for faint curves"
        )

    elif sensitivity_threshold <= 0.28:

        st.caption(
            "Medium sensitivity"
        )

    elif sensitivity_threshold <= 0.36:

        st.caption(
            "Low sensitivity"
        )

    else:

        st.caption(
            "Very conservative sensitivity"
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
# AUTOMATIC X-AXIS VALUES
# ============================================================

auto_start = st.session_state.get(
    "auto_axis_start",
    int(
        round(
            image_width * 0.08
        )
    ),
)

auto_end = st.session_state.get(
    "auto_axis_end",
    int(
        round(
            image_width * 0.88
        )
    ),
)


# ============================================================
# X-AXIS CALIBRATION
# ============================================================

st.subheader(
    "X-axis calibration"
)

st.info(
    "The green ROI boundaries are NOT used as the numerical "
    "X-axis. The application detects the actual horizontal "
    "axis, then lets you make small manual corrections."
)

axis_col1, axis_col2 = st.columns(
    2
)

with axis_col1:

    x_min = st.number_input(
        "X-axis minimum (fL)",
        min_value=-1000.0,
        max_value=10000.0,
        value=0.0,
        step=1.0,
    )

with axis_col2:

    x_max = st.number_input(
        "X-axis maximum (fL)",
        min_value=-1000.0,
        max_value=10000.0,
        value=40.0,
        step=1.0,
    )


# ============================================================
# DETECT X AXIS
# ============================================================

detect_axis_button = st.button(
    "Detect actual X-axis",
    use_container_width=True,
)


if detect_axis_button:

    from analyzer import detect_horizontal_x_axis

    raw_rgb = __import__(
        "numpy"
    ).asarray(
        image
    )

    detected_axis = detect_horizontal_x_axis(
        raw_rgb
    )

    if detected_axis.get(
        "ok"
    ):

        st.session_state[
            "auto_axis_start"
        ] = int(
            detected_axis[
                "start_x"
            ]
        )

        st.session_state[
            "auto_axis_end"
        ] = int(
            detected_axis[
                "end_x"
            ]
        )

        st.session_state[
            "auto_axis_confidence"
        ] = float(
            detected_axis.get(
                "confidence",
                0.0,
            )
        )

        st.success(
            (
                "Actual horizontal X-axis detected: "
                f"{detected_axis['start_x']}–"
                f"{detected_axis['end_x']} px"
            )
        )

        auto_start = (
            detected_axis[
                "start_x"
            ]
        )

        auto_end = (
            detected_axis[
                "end_x"
            ]
        )

    else:

        st.warning(
            "Automatic X-axis detection was not reliable. "
            "Set the manual corrections below."
        )


# Re-read state after possible detection.
auto_start = st.session_state.get(
    "auto_axis_start",
    auto_start,
)

auto_end = st.session_state.get(
    "auto_axis_end",
    auto_end,
)

auto_confidence = st.session_state.get(
    "auto_axis_confidence",
    0.0,
)


# ============================================================
# MANUAL CORRECTION
# ============================================================

st.markdown(
    "### Manual correction"
)

st.caption(
    (
        "Offsets are applied to the automatically detected "
        "axis. Example: set end offset to -100 when Auto ends "
        "100 px too far to the right."
    )
)

offset_col1, offset_col2 = st.columns(
    2
)

with offset_col1:

    start_offset = st.slider(
        "X-axis start correction (px)",
        min_value=-200,
        max_value=200,
        value=0,
        step=1,
        key="axis_start_offset",
    )

with offset_col2:

    end_offset = st.slider(
        "X-axis end correction (px)",
        min_value=-300,
        max_value=300,
        value=0,
        step=1,
        key="axis_end_offset",
    )


x_axis_start_px = int(
    auto_start
    + start_offset
)

x_axis_end_px = int(
    auto_end
    + end_offset
)

# Keep inside image.
x_axis_start_px = max(
    0,
    min(
        image_width - 1,
        x_axis_start_px,
    ),
)

x_axis_end_px = max(
    0,
    min(
        image_width - 1,
        x_axis_end_px,
    ),
)


# ============================================================
# AXIS SUMMARY
# ============================================================

st.write(
    (
        f"Auto detected: "
        f"`{auto_start}px → {auto_end}px`"
    )
)

st.write(
    (
        f"Final X-axis: "
        f"`{x_axis_start_px}px → {x_axis_end_px}px`"
    )
)

if auto_confidence > 0:

    st.caption(
        (
            f"Automatic axis confidence: "
            f"{auto_confidence * 100:.1f}%"
        )
    )


# ============================================================
# VALIDATION
# ============================================================

axis_valid = True

if (
    x_axis_end_px
    <= x_axis_start_px
):

    st.error(
        "X-axis end position must be greater than "
        "the X-axis start position."
    )

    axis_valid = False


if x_max <= x_min:

    st.error(
        "X-axis maximum value must be greater "
        "than X-axis minimum value."
    )

    axis_valid = False


if y_max <= 0:

    st.error(
        "Y-axis maximum must be greater than 0."
    )

    axis_valid = False


# ============================================================
# SHOW RECONSTRUCTED TICKS
# ============================================================

if axis_valid:

    ticks = []

    start_tick = int(
        __import__(
            "math"
        ).ceil(
            x_min
            / 10.0
        )
        * 10
    )

    end_tick = int(
        __import__(
            "math"
        ).floor(
            x_max
            / 10.0
        )
        * 10
    )

    if end_tick >= start_tick:

        ticks = list(
            range(
                start_tick,
                end_tick + 1,
                10,
            )
        )

    st.caption(
        (
            "Reconstructed axis ticks: "
            + (
                ", ".join(
                    f"{v:g} fL"
                    for v in ticks
                )
                if ticks
                else "None"
            )
        )
    )


# ============================================================
# SHOW GUIDE SETTINGS
# ============================================================

show_guides = st.checkbox(
    "Show peak/intersection guide lines",
    value=True,
    help=(
        "Displays vertical/horizontal measurement guides "
        "instead of colored point markers."
    ),
)


# ============================================================
# ANALYZE
# ============================================================

analyze_button = st.button(
    "Analyze graph",
    type="primary",
    use_container_width=True,
    disabled=not axis_valid,
)


if analyze_button:

    with st.spinner(
        "Detecting ROI, enhancing the curve, tracing the graph "
        "and calculating genuine 50% intersections..."
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

    # --------------------------------------------------------
    # Result image
    # --------------------------------------------------------

    st.subheader(
        "Analysis result"
    )

    annotated = annotate_result(
        result,
        show_guides=show_guides,
    )

    st.image(
        annotated,
        caption=(
            "Green = ROI boundaries | "
            "Cyan = calibrated X-axis | "
            "Orange = 50% level | "
            "Magenta = peak guides | "
            "Red = genuine intersection guides"
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

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
                "The traced curve does not genuinely "
                "cross the 50% level inside the ROI."
            )
        )

    elif status == "curve_not_found":

        st.error(
            (
                "The curve could not be traced reliably. "
                "Try lowering the sensitivity threshold."
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
                "The peak could not be reliably detected."
            )
        )

    if result.get(
        "warning"
    ):

        st.warning(
            result[
                "warning"
            ]
        )

    # --------------------------------------------------------
    # Axis details
    # --------------------------------------------------------

    with st.expander(
        "X-axis calibration details"
    ):

        st.write(
            "Automatic axis start:",
            auto_start,
            "px",
        )

        st.write(
            "Automatic axis end:",
            auto_end,
            "px",
        )

        st.write(
            "Start correction:",
            start_offset,
            "px",
        )

        st.write(
            "End correction:",
            end_offset,
            "px",
        )

        st.write(
            "Final axis start:",
            x_axis_start_px,
            "px",
        )

        st.write(
            "Final axis end:",
            x_axis_end_px,
            "px",
        )

        st.write(
            "Pixels per fL:",
            result.get(
                "pixels_per_fl"
            ),
        )

    # --------------------------------------------------------
    # Detailed output
    # --------------------------------------------------------

    with st.expander(
        "Detailed analysis output"
    ):

        hidden = {
            "image_rgb",
            "curve_probability",
            "visual_curve_score",
            "curve_mask",
            "centerline_x",
            "centerline_y",
            "centerline_confidence",
        }

        filtered = {
            key: value
            for key, value in result.items()
            if key not in hidden
        }

        st.json(
            filtered
        )