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
# MODEL PATH
# ============================================================

PROJECT_DIR = Path(
    __file__
).resolve().parent

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
    "V8 ROI-based peak and 50% intersection measurement"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading V8 model..."
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
        "Make sure the renamed checkpoint exists at:\n\n"
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
    # Sensitivity
    # --------------------------------------------------------

    sensitivity = st.selectbox(
        "Curve detection sensitivity",
        options=list(
            SENSITIVITY_PRESETS.keys()
        ),
        index=1,
        help=(
            "Low = more conservative. "
            "Medium = balanced. "
            "High = more sensitive to weaker curve responses."
        ),
    )

    threshold = SENSITIVITY_PRESETS[
        sensitivity
    ]

    st.caption(
        f"Segmentation threshold: "
        f"{threshold:.2f}"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # X AXIS
    # --------------------------------------------------------

    st.subheader(
        "X-axis"
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

    # --------------------------------------------------------
    # Y AXIS
    # --------------------------------------------------------

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
            "Maximum numerical Y-axis value represented "
            "at the top of the plotting area."
        ),
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL INFO
    # --------------------------------------------------------

    st.subheader(
        "Model"
    )

    st.write(
        f"Checkpoint: "
        f"`{checkpoint_path.name}`"
    )

    st.write(
        f"Device: `{device}`"
    )


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
    help=(
        "Upload the graph containing the two "
        "vertical dashed reference boundaries."
    ),
)


# ============================================================
# MAIN
# ============================================================

if uploaded is not None:

    try:

        image_bytes = (
            uploaded.getvalue()
        )

        pil_image = (
            Image.open(
                io.BytesIO(
                    image_bytes
                )
            )
            .convert("RGB")
        )

        st.image(
            pil_image,
            caption="Uploaded PLT graph",
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Selected settings
        # ----------------------------------------------------

        st.markdown(
            "### Selected analysis settings"
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
        # Validate
        # ----------------------------------------------------

        valid_input = True

        if x_max <= x_min:

            st.error(
                "X-axis maximum must be greater "
                "than X-axis minimum."
            )

            valid_input = False

        if y_max <= 0:

            st.error(
                "Y-axis maximum must be greater than 0."
            )

            valid_input = False

        # ----------------------------------------------------
        # Analyze
        # ----------------------------------------------------

        analyze_button = st.button(
            "Analyze graph",
            type="primary",
            use_container_width=True,
            disabled=not valid_input,
        )

        if analyze_button:

            with st.spinner(
                "Detecting ROI, calibrating X-axis, "
                "tracing curve and calculating 50% intersections..."
            ):

                result = analyze_plt_image(
                    model=model,
                    device=device,
                    image_input=pil_image,
                    x_min_fl=x_min,
                    x_max_fl=x_max,
                    y_max_fl=y_max,
                    threshold=threshold,
                )

            # ------------------------------------------------
            # Annotated result
            # ------------------------------------------------

            st.subheader(
                "Detected graph"
            )

            annotated = (
                annotate_result(
                    result
                )
            )

            st.image(
                annotated,
                caption=(
                    "Green = ROI boundaries | "
                    "Cyan = detected X-axis ticks | "
                    "Orange = 50% level | "
                    "Magenta = peak | "
                    "Red = genuine intersections"
                ),
                use_container_width=True,
            )

            # ------------------------------------------------
            # Table
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
            # Status message
            # ------------------------------------------------

            status = result.get(
                "status"
            )

            if (
                status
                == "measure_width"
            ):

                st.success(
                    (
                        "Width at 50%: "
                        f"{result['width_50_fl']:.3f} fL"
                    )
                )

            elif (
                status
                == "single_intersection"
            ):

                st.warning(
                    "Only one genuine 50% intersection "
                    "was detected, so width is not calculated."
                )

            elif (
                status
                == "no_intersection"
            ):

                st.info(
                    "The traced curve does not genuinely "
                    "cross the 50% level inside the ROI."
                )

            elif (
                status
                == "roi_not_found"
            ):

                st.error(
                    "The two vertical dashed ROI boundaries "
                    "could not be detected reliably."
                )

            elif (
                status
                == "curve_not_found"
            ):

                st.error(
                    "A reliable curve trace could not "
                    "be found inside the ROI."
                )

            elif (
                status
                == "peak_not_found"
            ):

                st.error(
                    "A reliable curve peak could not "
                    "be detected."
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
            # Debug information
            # ------------------------------------------------

            with st.expander(
                "X-axis calibration details"
            ):

                calibration = (
                    result.get(
                        "x_axis_calibration"
                    )
                    or {}
                )

                st.write(
                    "Calibration method:",
                    calibration.get(
                        "method"
                    ),
                )

                st.write(
                    "Detected tick pixels:",
                    result.get(
                        "x_tick_positions_px"
                    ),
                )

                st.write(
                    "Detected tick values:",
                    result.get(
                        "x_tick_values_fl"
                    ),
                )

                st.write(
                    "Pixels per 10 fL:",
                    calibration.get(
                        "pixels_per_10fl"
                    ),
                )

                st.write(
                    "Calibration confidence:",
                    calibration.get(
                        "confidence"
                    ),
                )

                st.write(
                    "Candidate tick pixels:",
                    calibration.get(
                        "tick_candidates_px"
                    ),
                )

    except Exception as exc:

        st.error(
            f"ERROR: {type(exc).__name__}: {exc}"
        )

else:

    st.info(
        "Upload a PLT graph image to begin."
    )