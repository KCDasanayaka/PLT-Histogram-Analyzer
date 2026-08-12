import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image

from preprocessing import (
    prepare_inference_image,
    restore_probability_mask,
    detect_plot_bounds,
    trace_curve_centerline,
    find_true_runs,
)


# ============================================================
# MEASUREMENT SETTINGS
# ============================================================

# The curve peak must reach at least this percentage
# of the complete plotting height before half-peak
# measurement is allowed.
#
# 0.50 = peak must reach at least 50% of graph height.
MIN_PEAK_HEIGHT_RATIO = 0.50

# Used for detecting a curve that touches the
# half-peak line without clearly crossing through it.
TANGENT_TOLERANCE_PIXELS = 1.25

# Crossings closer than this are treated as
# the same intersection.
MERGE_DISTANCE_PIXELS = 2.5


# ============================================================
# BASIC UTILITIES
# ============================================================

def interpolate_crossing(
    x1,
    y1,
    x2,
    y2,
    target_y,
):

    denominator = y2 - y1

    if abs(denominator) < 1e-8:
        return float(
            (x1 + x2) / 2.0
        )

    fraction = (
        target_y - y1
    ) / denominator

    fraction = float(
        np.clip(
            fraction,
            0.0,
            1.0,
        )
    )

    return float(
        x1
        + fraction
        * (x2 - x1)
    )


def merge_close_values(
    values,
    minimum_distance=MERGE_DISTANCE_PIXELS,
):

    if not values:
        return []

    values = sorted(
        float(value)
        for value in values
    )

    groups = [
        [values[0]]
    ]

    for value in values[1:]:

        if (
            value
            - groups[-1][-1]
            <= minimum_distance
        ):

            groups[-1].append(
                value
            )

        else:

            groups.append(
                [value]
            )

    return [
        float(
            np.mean(group)
        )
        for group in groups
    ]


def pixel_to_fl(
    x_pixel,
    x_left,
    x_right,
    x_min_fl,
    x_max_fl,
):

    denominator = max(
        float(
            x_right - x_left
        ),
        1.0,
    )

    normalized_position = (
        float(x_pixel)
        - float(x_left)
    ) / denominator

    normalized_position = float(
        np.clip(
            normalized_position,
            0.0,
            1.0,
        )
    )

    return float(
        x_min_fl
        + normalized_position
        * (
            x_max_fl
            - x_min_fl
        )
    )


def format_fl(value):

    if value is None:
        return "Not available"

    return f"{float(value):.3f} fL"


# ============================================================
# CURVE PEAK + 50% OF CURVE
# ============================================================

def calculate_curve_peak(
    trace_result,
    plot_bounds,
    minimum_peak_height_ratio=MIN_PEAK_HEIGHT_RATIO,
):

    if not trace_result.get(
        "valid",
        False,
    ):

        return {
            "valid": False,
            "reason": "curve_trace_invalid",
            "peak_is_high_enough": False,
            "curve_half_y": None,
        }

    y_values = np.asarray(
        trace_result["y"],
        dtype=np.float32,
    )

    x_values = np.asarray(
        trace_result["x"],
        dtype=np.float32,
    )

    finite_mask = np.isfinite(
        y_values
    )

    if finite_mask.sum() < 5:

        return {
            "valid": False,
            "reason": "insufficient_curve_points",
            "peak_is_high_enough": False,
            "curve_half_y": None,
        }

    baseline_y = float(
        plot_bounds[
            "baseline_y"
        ]
    )

    y_top = float(
        plot_bounds[
            "y_top"
        ]
    )

    plot_height = (
        baseline_y - y_top
    )

    if plot_height <= 5:

        return {
            "valid": False,
            "reason": "invalid_plot_height",
            "peak_is_high_enough": False,
            "curve_half_y": None,
        }

    # --------------------------------------------------------
    # Find highest point of the detected curve.
    #
    # Image Y coordinates decrease as the graph gets higher,
    # therefore the smallest Y value is the curve peak.
    # --------------------------------------------------------

    finite_indexes = np.where(
        finite_mask
    )[0]

    finite_y_values = (
        y_values[
            finite_indexes
        ]
    )

    peak_local_index = int(
        np.argmin(
            finite_y_values
        )
    )

    peak_index = int(
        finite_indexes[
            peak_local_index
        ]
    )

    peak_y = float(
        y_values[
            peak_index
        ]
    )

    peak_x = float(
        x_values[
            peak_index
        ]
    )

    # Height of the actual curve above baseline.
    peak_height_pixels = (
        baseline_y - peak_y
    )

    peak_height_pixels = max(
        0.0,
        peak_height_pixels,
    )

    # How high is the curve compared with the
    # entire Y-axis plotting height?
    peak_height_ratio = (
        peak_height_pixels
        / plot_height
    )

    peak_height_ratio = float(
        np.clip(
            peak_height_ratio,
            0.0,
            1.5,
        )
    )

    peak_is_high_enough = (
        peak_height_ratio
        >= minimum_peak_height_ratio
    )

    # --------------------------------------------------------
    # Important:
    #
    # Do NOT calculate curve-half level when the
    # distribution is too low.
    # --------------------------------------------------------

    if not peak_is_high_enough:

        return {
            "valid": True,

            "reason":
                "peak_too_low",

            "peak_is_high_enough":
                False,

            "peak_x":
                peak_x,

            "peak_y":
                peak_y,

            "peak_height_pixels":
                peak_height_pixels,

            "peak_height_ratio":
                peak_height_ratio,

            "plot_height_pixels":
                plot_height,

            "curve_half_y":
                None,
        }

    # --------------------------------------------------------
    # 50% OF THE CURVE PEAK
    #
    # baseline = 0%
    # curve peak = 100% of this distribution
    #
    # Half-peak is halfway between baseline
    # and the detected curve peak.
    # --------------------------------------------------------

    curve_half_height = (
        peak_height_pixels
        * 0.50
    )

    curve_half_y = (
        baseline_y
        - curve_half_height
    )

    return {
        "valid":
            True,

        "reason":
            None,

        "peak_is_high_enough":
            True,

        "peak_x":
            peak_x,

        "peak_y":
            peak_y,

        "peak_height_pixels":
            peak_height_pixels,

        "peak_height_ratio":
            peak_height_ratio,

        "plot_height_pixels":
            plot_height,

        "curve_half_height_pixels":
            curve_half_height,

        "curve_half_y":
            float(
                curve_half_y
            ),
    }


# ============================================================
# FIND ALL GENUINE HALF-PEAK INTERSECTIONS
# ============================================================

def detect_curve_half_crossings(
    trace_result,
    curve_half_y,
    tangent_tolerance_pixels=TANGENT_TOLERANCE_PIXELS,
):

    if not trace_result.get(
        "valid",
        False,
    ):

        return {
            "valid": False,
            "reason": "invalid_curve_trace",
            "crossings_px": [],
        }

    if curve_half_y is None:

        return {
            "valid": False,
            "reason": "half_level_not_available",
            "crossings_px": [],
        }

    x_values = np.asarray(
        trace_result["x"],
        dtype=np.float32,
    )

    y_values = np.asarray(
        trace_result["y"],
        dtype=np.float32,
    )

    finite_mask = np.isfinite(
        y_values
    )

    crossings = []

    continuous_segments = (
        find_true_runs(
            finite_mask
        )
    )

    for segment_start, segment_end in (
        continuous_segments
    ):

        segment_length = (
            segment_end
            - segment_start
            + 1
        )

        if segment_length < 4:
            continue

        segment_x = x_values[
            segment_start:
            segment_end + 1
        ]

        segment_y = y_values[
            segment_start:
            segment_end + 1
        ]

        differences = (
            segment_y
            - curve_half_y
        )

        # ----------------------------------------------------
        # Normal crossings
        # ----------------------------------------------------

        for index in range(
            len(segment_x) - 1
        ):

            x1 = float(
                segment_x[
                    index
                ]
            )

            x2 = float(
                segment_x[
                    index + 1
                ]
            )

            y1 = float(
                segment_y[
                    index
                ]
            )

            y2 = float(
                segment_y[
                    index + 1
                ]
            )

            d1 = float(
                differences[
                    index
                ]
            )

            d2 = float(
                differences[
                    index + 1
                ]
            )

            # Curve genuinely moves from one
            # side of half-level to the other.
            if d1 * d2 < 0:

                crossing = (
                    interpolate_crossing(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        target_y=curve_half_y,
                    )
                )

                crossings.append(
                    crossing
                )

        # ----------------------------------------------------
        # Tangency / touching detection
        #
        # Example:
        #
        # curve approaches 50%
        # touches it
        # then moves back to same side.
        # ----------------------------------------------------

        absolute_difference = np.abs(
            differences
        )

        for index in range(
            1,
            len(segment_x) - 1,
        ):

            current_distance = float(
                absolute_difference[
                    index
                ]
            )

            if (
                current_distance
                > tangent_tolerance_pixels
            ):
                continue

            previous_distance = float(
                absolute_difference[
                    index - 1
                ]
            )

            next_distance = float(
                absolute_difference[
                    index + 1
                ]
            )

            previous_sign = float(
                differences[
                    index - 1
                ]
            )

            next_sign = float(
                differences[
                    index + 1
                ]
            )

            same_side = (
                previous_sign
                * next_sign
                > 0
            )

            local_minimum = (
                current_distance
                <= previous_distance
                and
                current_distance
                <= next_distance
            )

            if (
                same_side
                and local_minimum
            ):

                crossings.append(
                    float(
                        segment_x[
                            index
                        ]
                    )
                )

    crossings = merge_close_values(
        crossings,
        minimum_distance=(
            MERGE_DISTANCE_PIXELS
        ),
    )

    return {
        "valid":
            True,

        "reason":
            None,

        "crossings_px":
            sorted(
                crossings
            ),
    }


# ============================================================
# RESULT TABLE
# ============================================================

def make_result_table(result):

    peak_ratio = result.get(
        "peak_height_ratio"
    )

    if peak_ratio is None:

        peak_display = (
            "Not available"
        )

    else:

        peak_display = (
            f"{peak_ratio * 100:.1f}% "
            "of graph height"
        )

    rows = [
        [
            "Status",
            result[
                "status"
            ],
        ],

        [
            "Measurement",
            (
                "50% of detected "
                "curve peak"
            ),
        ],

        [
            "Minimum peak requirement",
            (
                f"{result['minimum_peak_height_ratio'] * 100:.0f}% "
                "of graph height"
            ),
        ],

        [
            "Detected peak height",
            peak_display,
        ],

        [
            "X-axis range",
            (
                f"{result['x_min_fl']:.0f}"
                f"–"
                f"{result['x_max_fl']:.0f}"
                " fL"
            ),
        ],

        [
            "50% intersections",
            result[
                "intersection_count"
            ],
        ],

        [
            "All intersections",
            (
                result[
                    "all_intersections_fl"
                ]
                or "None"
            ),
        ],

        [
            "Minimum X",
            format_fl(
                result[
                    "minimum_intersection_fl"
                ]
            ),
        ],

        [
            "Maximum X",
            format_fl(
                result[
                    "maximum_intersection_fl"
                ]
            ),
        ],

        [
            "Width at 50%",
            format_fl(
                result[
                    "width_50_fl"
                ]
            ),
        ],

        [
            "Curve support",
            (
                f"{result['curve_support']:.1f}%"
            ),
        ],

        [
            "Reason",
            result.get(
                "reason",
                "None",
            ),
        ],

        [
            "Warning",
            result[
                "warning"
            ],
        ],
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Measurement",
            "Result",
        ],
    )


# ============================================================
# ANNOTATION
# ============================================================

def annotate_result(
    image_rgb,
    probability_mask,
    plot_bounds,
    crossings_px,
    crossings_fl,
    curve_half_y,
    width_50_fl,
    status,
    reason,
    segmentation_threshold,
):

    annotated = cv2.cvtColor(
        image_rgb.copy(),
        cv2.COLOR_RGB2BGR,
    )

    height, width = (
        annotated.shape[:2]
    )

    # --------------------------------------------------------
    # Green overlay = model's detected curve
    # --------------------------------------------------------

    overlay = annotated.copy()

    curve_mask = (
        probability_mask
        >= segmentation_threshold
    )

    overlay[
        curve_mask
    ] = (
        0,
        180,
        0,
    )

    annotated = cv2.addWeighted(
        annotated,
        0.84,
        overlay,
        0.16,
        0,
    )

    x_left = int(
        plot_bounds[
            "x_left"
        ]
    )

    x_right = int(
        plot_bounds[
            "x_right"
        ]
    )

    baseline_y = int(
        plot_bounds[
            "baseline_y"
        ]
    )

    # --------------------------------------------------------
    # Draw half-peak line ONLY when measurement
    # is actually allowed.
    # --------------------------------------------------------

    if curve_half_y is not None:

        half_y = int(
            round(
                curve_half_y
            )
        )

        cv2.line(
            annotated,
            (
                x_left,
                half_y,
            ),
            (
                x_right,
                half_y,
            ),
            (
                0,
                165,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            "50% Curve",
            (
                min(
                    width - 80,
                    x_left + 4,
                ),
                max(
                    13,
                    half_y - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (
                0,
                120,
                220,
            ),
            1,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Draw every genuine intersection
        # ----------------------------------------------------

        for index, (
            x_pixel,
            x_value_fl,
        ) in enumerate(
            zip(
                crossings_px,
                crossings_fl,
            )
        ):

            x_point = int(
                round(
                    x_pixel
                )
            )

            cv2.circle(
                annotated,
                (
                    x_point,
                    half_y,
                ),
                4,
                (
                    0,
                    0,
                    255,
                ),
                -1,
                cv2.LINE_AA,
            )

            text_y = (
                half_y - 8
                if index % 2 == 0
                else half_y + 17
            )

            cv2.putText(
                annotated,
                (
                    f"{x_value_fl:.2f}"
                    " fL"
                ),
                (
                    max(
                        1,
                        min(
                            width - 68,
                            x_point - 20,
                        ),
                    ),
                    max(
                        12,
                        min(
                            height - 4,
                            text_y,
                        ),
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (
                    0,
                    0,
                    215,
                ),
                1,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    if reason == "peak_too_low":

        summary = (
            "Status: no_intersection "
            "| Distribution too low"
        )

    else:

        summary = (
            f"Status: {status}"
        )

        if width_50_fl is not None:

            summary += (
                f" | W50: "
                f"{width_50_fl:.2f} fL"
            )

    cv2.rectangle(
        annotated,
        (0, 0),
        (
            min(
                width - 1,
                max(
                    225,
                    len(summary) * 6,
                ),
            ),
            21,
        ),
        (
            255,
            255,
            255,
        ),
        -1,
    )

    cv2.putText(
        annotated,
        summary,
        (
            4,
            14,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (
            20,
            20,
            20,
        ),
        1,
        cv2.LINE_AA,
    )

    return Image.fromarray(
        cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB,
        )
    )


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

@torch.inference_mode()
def analyze_plt_graph(
    input_image,
    model,
    device,
    x_axis_max_fl=40.0,
    segmentation_threshold=0.35,
):

    # --------------------------------------------------------
    # 1. Prepare image
    # --------------------------------------------------------

    (
        original_image,
        image_tensor,
        transform_info,
    ) = prepare_inference_image(
        input_image,
        device,
    )

    # --------------------------------------------------------
    # 2. Segment the PLT curve
    # --------------------------------------------------------

    logits = model(
        image_tensor
    )

    probability_padded = (
        torch.sigmoid(
            logits
        )[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    probability_mask = (
        restore_probability_mask(
            probability_padded,
            transform_info,
        )
    )

    # --------------------------------------------------------
    # 3. Detect graph boundaries
    # --------------------------------------------------------

    plot_bounds = (
        detect_plot_bounds(
            original_image
        )
    )

    # --------------------------------------------------------
    # 4. Trace actual curve
    # --------------------------------------------------------

    trace_result = (
        trace_curve_centerline(
            probability_mask,
            plot_bounds,
            float(
                segmentation_threshold
            ),
        )
    )

    # --------------------------------------------------------
    # 5. Find actual curve peak
    # --------------------------------------------------------

    peak_result = (
        calculate_curve_peak(
            trace_result,
            plot_bounds,
            minimum_peak_height_ratio=(
                MIN_PEAK_HEIGHT_RATIO
            ),
        )
    )

    x_min_fl = 0.0
    x_max_fl = float(
        x_axis_max_fl
    )

    crossings_px = []
    crossings_fl = []

    reason = peak_result.get(
        "reason"
    )

    curve_half_y = peak_result.get(
        "curve_half_y"
    )

    # --------------------------------------------------------
    # 6. Decide whether half-peak analysis is allowed
    # --------------------------------------------------------

    if not peak_result.get(
        "valid",
        False,
    ):

        status = "reject"

    elif not peak_result.get(
        "peak_is_high_enough",
        False,
    ):

        # IMPORTANT:
        #
        # Low graph like your example:
        # no 50% line
        # no points
        # no width
        status = (
            "no_intersection"
        )

        crossings_px = []
        crossings_fl = []

    else:

        crossing_result = (
            detect_curve_half_crossings(
                trace_result,
                curve_half_y,
            )
        )

        crossings_px = (
            crossing_result.get(
                "crossings_px",
                [],
            )
        )

        crossings_fl = [
            pixel_to_fl(
                x_pixel,
                plot_bounds[
                    "x_left"
                ],
                plot_bounds[
                    "x_right"
                ],
                x_min_fl,
                x_max_fl,
            )
            for x_pixel
            in crossings_px
        ]

        # Sort from left to right.
        pairs = sorted(
            zip(
                crossings_px,
                crossings_fl,
            ),
            key=lambda item:
                item[0],
        )

        crossings_px = [
            float(
                item[0]
            )
            for item
            in pairs
        ]

        crossings_fl = [
            float(
                item[1]
            )
            for item
            in pairs
        ]

        number_of_crossings = len(
            crossings_fl
        )

        if number_of_crossings == 0:

            status = (
                "no_intersection"
            )

        elif number_of_crossings == 1:

            status = (
                "single_intersection"
            )

        else:

            status = (
                "measure_width"
            )

    # --------------------------------------------------------
    # 7. Calculate measurements
    # --------------------------------------------------------

    number_of_crossings = len(
        crossings_fl
    )

    if crossings_fl:

        minimum_intersection = float(
            min(
                crossings_fl
            )
        )

        maximum_intersection = float(
            max(
                crossings_fl
            )
        )

    else:

        minimum_intersection = None
        maximum_intersection = None

    if number_of_crossings >= 2:

        width_50 = float(
            maximum_intersection
            - minimum_intersection
        )

    else:

        width_50 = None

    # --------------------------------------------------------
    # 8. Generate warnings
    # --------------------------------------------------------

    warnings = []

    if not plot_bounds.get(
        "automatic",
        False,
    ):

        warnings.append(
            "Graph boundaries were "
            "estimated using fallback logic."
        )

    if status == "reject":

        warnings.append(
            "The curve could not be "
            "traced reliably."
        )

    elif reason == "peak_too_low":

        warnings.append(
            "The detected distribution peak "
            "is too low for half-peak measurement. "
            "No 50% point was calculated."
        )

    elif status == (
        "no_intersection"
    ):

        warnings.append(
            "No genuine intersection with "
            "the curve's 50% peak level "
            "was detected."
        )

    elif status == (
        "single_intersection"
    ):

        warnings.append(
            "Only one genuine half-peak "
            "intersection was detected. "
            "Width cannot be calculated."
        )

    elif number_of_crossings > 2:

        warnings.append(
            "More than two genuine "
            "intersections were detected. "
            "Width uses maximum X minus "
            "minimum X."
        )

    # --------------------------------------------------------
    # 9. Curve support
    # --------------------------------------------------------

    support_ratio = float(
        trace_result.get(
            "support_ratio",
            0.0,
        )
    )

    peak_height_ratio = (
        peak_result.get(
            "peak_height_ratio"
        )
    )

    # --------------------------------------------------------
    # 10. Result JSON
    # --------------------------------------------------------

    result = {
        "status":
            status,

        "reason":
            (
                reason
                if reason is not None
                else "None"
            ),

        "measurement_definition":
            "50_percent_of_detected_curve_peak",

        "minimum_peak_height_ratio":
            MIN_PEAK_HEIGHT_RATIO,

        "peak_height_ratio":
            (
                None
                if peak_height_ratio is None
                else round(
                    float(
                        peak_height_ratio
                    ),
                    4,
                )
            ),

        "peak_height_percent":
            (
                None
                if peak_height_ratio is None
                else round(
                    float(
                        peak_height_ratio
                    )
                    * 100.0,
                    2,
                )
            ),

        "x_min_fl":
            x_min_fl,

        "x_max_fl":
            x_max_fl,

        "intersection_count":
            number_of_crossings,

        "intersections":
            [
                round(
                    value,
                    3,
                )
                for value
                in crossings_fl
            ],

        "all_intersections_fl":
            (
                None
                if not crossings_fl
                else ", ".join(
                    f"{value:.3f}"
                    for value
                    in crossings_fl
                )
            ),

        "minimum_intersection_fl":
            (
                None
                if minimum_intersection
                is None
                else round(
                    minimum_intersection,
                    3,
                )
            ),

        "maximum_intersection_fl":
            (
                None
                if maximum_intersection
                is None
                else round(
                    maximum_intersection,
                    3,
                )
            ),

        "width_50_fl":
            (
                None
                if width_50 is None
                else round(
                    width_50,
                    3,
                )
            ),

        "curve_support":
            round(
                support_ratio
                * 100.0,
                1,
            ),

        "curve_half_y_pixel":
            (
                None
                if curve_half_y is None
                else round(
                    float(
                        curve_half_y
                    ),
                    2,
                )
            ),

        "plot_top_pixel":
            int(
                plot_bounds[
                    "y_top"
                ]
            ),

        "baseline_pixel":
            int(
                plot_bounds[
                    "baseline_y"
                ]
            ),

        "warning":
            (
                "None"
                if not warnings
                else " ".join(
                    warnings
                )
            ),
    }

    # --------------------------------------------------------
    # 11. Annotated result
    # --------------------------------------------------------

    annotated = (
        annotate_result(
            image_rgb=(
                original_image
            ),
            probability_mask=(
                probability_mask
            ),
            plot_bounds=(
                plot_bounds
            ),
            crossings_px=(
                crossings_px
            ),
            crossings_fl=(
                crossings_fl
            ),
            curve_half_y=(
                curve_half_y
            ),
            width_50_fl=(
                width_50
            ),
            status=(
                status
            ),
            reason=(
                reason
            ),
            segmentation_threshold=float(
                segmentation_threshold
            ),
        )
    )

    result_table = (
        make_result_table(
            result
        )
    )

    return (
        annotated,
        result_table,
        result,
    )