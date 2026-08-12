import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
from scipy.ndimage import median_filter

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

# A low distribution must first reach at least 50%
# of the available graph height.
#
# If it does NOT:
# - no curve 50% line
# - no intersection
# - no width
MIN_PEAK_HEIGHT_RATIO = 0.50


# Ignore very small region next to the Y-axis.
LEFT_EDGE_IGNORE_FRACTION = 0.025


# Robust peak detection neighborhood.
PEAK_NEIGHBORHOOD_FRACTION = 0.035


# Intersection settings.
TANGENT_TOLERANCE_PIXELS = 1.25

MERGE_DISTANCE_PIXELS = 2.5


# ============================================================
# UTILITIES
# ============================================================

def interpolate_crossing(
    x1,
    y1,
    x2,
    y2,
    target_y,
):

    denominator = (
        y2 - y1
    )

    if (
        abs(
            denominator
        ) < 1e-8
    ):

        return float(
            (
                x1 + x2
            )
            / 2.0
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
        * (
            x2 - x1
        )
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
        [
            values[0]
        ]
    ]

    for value in values[1:]:

        if (
            value
            - groups[-1][-1]
            <= minimum_distance
        ):

            groups[
                -1
            ].append(
                value
            )

        else:

            groups.append(
                [
                    value
                ]
            )

    return [
        float(
            np.mean(
                group
            )
        )
        for group
        in groups
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
            x_right
            - x_left
        ),
        1.0,
    )

    normalized_position = (
        float(
            x_pixel
        )
        - float(
            x_left
        )
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


def format_fl(
    value,
):

    if value is None:

        return (
            "Not available"
        )

    return (
        f"{float(value):.3f} fL"
    )


# ============================================================
# FIND THE REAL DISTRIBUTION PEAK
# ============================================================

def calculate_distribution_peak(
    trace_result,
    plot_bounds,
    minimum_peak_height_ratio=MIN_PEAK_HEIGHT_RATIO,
):

    if not trace_result.get(
        "valid",
        False,
    ):

        return {
            "valid":
                False,

            "reason":
                "curve_trace_invalid",

            "peak_is_high_enough":
                False,

            "curve_half_y":
                None,
        }

    x_values = np.asarray(
        trace_result[
            "x"
        ],
        dtype=np.float32,
    )

    y_values = np.asarray(
        trace_result[
            "y"
        ],
        dtype=np.float32,
    )

    finite_mask = np.isfinite(
        y_values
    )

    if (
        finite_mask.sum()
        < 5
    ):

        return {
            "valid":
                False,

            "reason":
                "insufficient_curve_points",

            "peak_is_high_enough":
                False,

            "curve_half_y":
                None,
        }

    x_left = float(
        plot_bounds[
            "x_left"
        ]
    )

    x_right = float(
        plot_bounds[
            "x_right"
        ]
    )

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

    plot_width = max(
        1.0,
        x_right
        - x_left,
    )

    plot_height = (
        baseline_y
        - y_top
    )

    if plot_height <= 5:

        return {
            "valid":
                False,

            "reason":
                "invalid_plot_height",

            "peak_is_high_enough":
                False,

            "curve_half_y":
                None,
        }

    # ========================================================
    # IGNORE Y-AXIS REGION
    # ========================================================

    minimum_peak_x = (
        x_left
        + plot_width
        * LEFT_EDGE_IGNORE_FRACTION
    )

    peak_candidate_mask = (
        finite_mask
        &
        (
            x_values
            >= minimum_peak_x
        )
        &
        (
            x_values
            <= x_right
        )
    )

    candidate_indexes = np.where(
        peak_candidate_mask
    )[0]

    if (
        len(
            candidate_indexes
        )
        < 5
    ):

        return {
            "valid":
                False,

            "reason":
                "insufficient_peak_candidates",

            "peak_is_high_enough":
                False,

            "curve_half_y":
                None,
        }

    candidate_y = (
        y_values[
            candidate_indexes
        ]
    )

    # ========================================================
    # ROBUST PEAK
    #
    # Do not allow one false high pixel to become the peak.
    # ========================================================

    smooth_size = max(
        3,
        int(
            round(
                len(
                    candidate_y
                )
                * PEAK_NEIGHBORHOOD_FRACTION
            )
        ),
    )

    if (
        smooth_size
        % 2 == 0
    ):

        smooth_size += 1

    smooth_size = min(
        smooth_size,
        11,
    )

    if (
        smooth_size >= 3
        and
        len(
            candidate_y
        )
        >= smooth_size
    ):

        robust_y = median_filter(
            candidate_y,
            size=smooth_size,
            mode="nearest",
        )

    else:

        robust_y = (
            candidate_y.copy()
        )

    # Image coordinates decrease upward.
    # Therefore lowest Y = highest curve point.
    local_peak_position = int(
        np.argmin(
            robust_y
        )
    )

    peak_index = int(
        candidate_indexes[
            local_peak_position
        ]
    )

    peak_y = float(
        robust_y[
            local_peak_position
        ]
    )

    peak_x = float(
        x_values[
            peak_index
        ]
    )

    # ========================================================
    # DISTRIBUTION HEIGHT
    # ========================================================

    peak_height_pixels = max(
        0.0,
        baseline_y
        - peak_y,
    )

    peak_height_ratio = float(
        peak_height_pixels
        / plot_height
    )

    peak_height_ratio = float(
        np.clip(
            peak_height_ratio,
            0.0,
            1.25,
        )
    )

    # ========================================================
    # LOW DISTRIBUTION REJECTION
    # ========================================================

    peak_is_high_enough = (
        peak_height_ratio
        >= minimum_peak_height_ratio
    )

    if not peak_is_high_enough:

        return {
            "valid":
                True,

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

    # ========================================================
    # 50% OF THE DISTRIBUTION
    #
    # baseline = 0
    # actual detected peak = 100%
    # halfway = 50%
    # ========================================================

    curve_half_height_pixels = (
        peak_height_pixels
        * 0.50
    )

    curve_half_y = (
        baseline_y
        - curve_half_height_pixels
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
            curve_half_height_pixels,

        "curve_half_y":
            float(
                curve_half_y
            ),
    }


# ============================================================
# ALL INTERSECTIONS WITH 50% OF DISTRIBUTION
# ============================================================

def detect_distribution_half_crossings(
    trace_result,
    plot_bounds,
    curve_half_y,
    tangent_tolerance_pixels=TANGENT_TOLERANCE_PIXELS,
):

    if not trace_result.get(
        "valid",
        False,
    ):

        return {
            "valid":
                False,

            "reason":
                "invalid_curve_trace",

            "crossings_px":
                [],
        }

    if curve_half_y is None:

        return {
            "valid":
                False,

            "reason":
                "half_level_not_available",

            "crossings_px":
                [],
        }

    x_values = np.asarray(
        trace_result[
            "x"
        ],
        dtype=np.float32,
    )

    y_values = np.asarray(
        trace_result[
            "y"
        ],
        dtype=np.float32,
    )

    finite_mask = np.isfinite(
        y_values
    )

    x_left = float(
        plot_bounds[
            "x_left"
        ]
    )

    x_right = float(
        plot_bounds[
            "x_right"
        ]
    )

    plot_width = max(
        1.0,
        x_right
        - x_left,
    )

    minimum_valid_x = (
        x_left
        + plot_width
        * LEFT_EDGE_IGNORE_FRACTION
    )

    crossings = []

    continuous_segments = (
        find_true_runs(
            finite_mask
        )
    )

    for (
        segment_start,
        segment_end,
    ) in continuous_segments:

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
            - float(
                curve_half_y
            )
        )

        # ====================================================
        # REAL CROSSINGS
        # ====================================================

        for index in range(
            len(
                segment_x
            )
            - 1
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

            if (
                max(
                    x1,
                    x2,
                )
                < minimum_valid_x
            ):

                continue

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

            if d1 * d2 < 0:

                crossing = (
                    interpolate_crossing(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        target_y=float(
                            curve_half_y
                        ),
                    )
                )

                if (
                    crossing
                    >= minimum_valid_x
                ):

                    crossings.append(
                        crossing
                    )

        # ====================================================
        # TOUCHING / TANGENT POINT
        # ====================================================

        absolute_difference = np.abs(
            differences
        )

        for index in range(
            1,
            len(
                segment_x
            )
            - 1,
        ):

            x_value = float(
                segment_x[
                    index
                ]
            )

            if (
                x_value
                < minimum_valid_x
            ):

                continue

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
                <= float(
                    absolute_difference[
                        index - 1
                    ]
                )
                and
                current_distance
                <= float(
                    absolute_difference[
                        index + 1
                    ]
                )
            )

            if (
                same_side
                and local_minimum
            ):

                crossings.append(
                    x_value
                )

    crossings = (
        merge_close_values(
            crossings,
            minimum_distance=(
                MERGE_DISTANCE_PIXELS
            ),
        )
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

def make_result_table(
    result,
):

    peak_ratio = (
        result.get(
            "peak_height_ratio"
        )
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
                "PLT distribution peak"
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
                "–"
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
# RESULT IMAGE
# ============================================================

def annotate_result(
    image_rgb,
    probability_mask,
    plot_bounds,
    trace_result,
    peak_result,
    crossings_px,
    crossings_fl,
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

    # ========================================================
    # RAW SEGMENTATION
    # ========================================================

    overlay = (
        annotated.copy()
    )

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

    annotated = (
        cv2.addWeighted(
            annotated,
            0.86,
            overlay,
            0.14,
            0,
        )
    )

    # ========================================================
    # CLEAN TRACE
    #
    # Cyan line = distribution actually used for measurement.
    # ========================================================

    trace_x = np.asarray(
        trace_result.get(
            "x",
            [],
        ),
        dtype=np.float32,
    )

    trace_y = np.asarray(
        trace_result.get(
            "y",
            [],
        ),
        dtype=np.float32,
    )

    if (
        len(trace_x)
        == len(trace_y)
    ):

        finite = np.isfinite(
            trace_y
        )

        points = [
            (
                int(
                    round(x)
                ),
                int(
                    round(y)
                ),
            )
            for x, y in zip(
                trace_x[
                    finite
                ],
                trace_y[
                    finite
                ],
            )
        ]

        for (
            point_1,
            point_2,
        ) in zip(
            points[:-1],
            points[1:],
        ):

            if (
                abs(
                    point_2[0]
                    - point_1[0]
                )
                <= 3
            ):

                cv2.line(
                    annotated,
                    point_1,
                    point_2,
                    (
                        255,
                        180,
                        0,
                    ),
                    1,
                    cv2.LINE_AA,
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

    curve_half_y = (
        peak_result.get(
            "curve_half_y"
        )
    )

    peak_x = (
        peak_result.get(
            "peak_x"
        )
    )

    peak_y = (
        peak_result.get(
            "peak_y"
        )
    )

    # ========================================================
    # REAL DISTRIBUTION PEAK
    # ========================================================

    if (
        peak_x is not None
        and peak_y is not None
    ):

        cv2.circle(
            annotated,
            (
                int(
                    round(
                        peak_x
                    )
                ),
                int(
                    round(
                        peak_y
                    )
                ),
            ),
            4,
            (
                255,
                0,
                255,
            ),
            -1,
            cv2.LINE_AA,
        )

    # ========================================================
    # 50% OF ACTUAL DISTRIBUTION
    # ========================================================

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
            "50% of curve",
            (
                min(
                    width - 90,
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

        # ====================================================
        # ALL VALID INTERSECTIONS
        # ====================================================

        for (
            index,
            (
                x_pixel,
                x_value_fl,
            ),
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

    # ========================================================
    # STATUS
    # ========================================================

    if (
        reason
        == "peak_too_low"
    ):

        summary = (
            "Status: no_intersection "
            "| Distribution too low"
        )

    else:

        summary = (
            f"Status: {status}"
        )

        if (
            width_50_fl
            is not None
        ):

            summary += (
                f" | W50: "
                f"{width_50_fl:.2f} fL"
            )

    cv2.rectangle(
        annotated,
        (
            0,
            0,
        ),
        (
            min(
                width - 1,
                max(
                    225,
                    len(
                        summary
                    )
                    * 6,
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
# MAIN FUNCTION
# ============================================================

@torch.inference_mode()
def analyze_plt_graph(
    input_image,
    model,
    device,
    x_axis_max_fl=40.0,
    segmentation_threshold=0.35,
):

    # ========================================================
    # PREPARE
    # ========================================================

    (
        original_image,
        image_tensor,
        transform_info,
    ) = prepare_inference_image(
        input_image,
        device,
    )

    # ========================================================
    # U-NET CURVE SEGMENTATION
    # ========================================================

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

    # ========================================================
    # GRAPH BOUNDARIES
    # ========================================================

    plot_bounds = (
        detect_plot_bounds(
            original_image
        )
    )

    # ========================================================
    # CLEAN DISTRIBUTION TRACE
    # ========================================================

    trace_result = (
        trace_curve_centerline(
            probability_mask,
            plot_bounds,
            float(
                segmentation_threshold
            ),
        )
    )

    # ========================================================
    # ACTUAL DISTRIBUTION PEAK
    # ========================================================

    peak_result = (
        calculate_distribution_peak(
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

    reason = (
        peak_result.get(
            "reason"
        )
    )

    # ========================================================
    # LOW DISTRIBUTION CHECK
    # ========================================================

    if not peak_result.get(
        "valid",
        False,
    ):

        status = (
            "reject"
        )

    elif not peak_result.get(
        "peak_is_high_enough",
        False,
    ):

        # IMPORTANT:
        #
        # LOW GRAPH:
        #
        # no orange line
        # no red point
        # no width
        status = (
            "no_intersection"
        )

    else:

        # ====================================================
        # DETECT 50% OF ACTUAL CURVE
        # ====================================================

        crossing_result = (
            detect_distribution_half_crossings(
                trace_result,
                plot_bounds,
                peak_result[
                    "curve_half_y"
                ],
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

        if (
            len(
                crossings_fl
            )
            == 0
        ):

            status = (
                "no_intersection"
            )

        elif (
            len(
                crossings_fl
            )
            == 1
        ):

            status = (
                "single_intersection"
            )

        else:

            status = (
                "measure_width"
            )

    # ========================================================
    # WIDTH
    # ========================================================

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

    if (
        number_of_crossings
        >= 2
    ):

        width_50 = float(
            maximum_intersection
            - minimum_intersection
        )

    else:

        width_50 = None

    # ========================================================
    # WARNINGS
    # ========================================================

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
            "The PLT distribution "
            "could not be traced reliably."
        )

    elif (
        reason
        == "peak_too_low"
    ):

        warnings.append(
            "The detected PLT distribution "
            "is too low for half-peak measurement, "
            "so no 50% point was generated."
        )

    elif (
        status
        == "no_intersection"
    ):

        warnings.append(
            "No genuine intersection with "
            "50% of the detected distribution "
            "peak was found."
        )

    elif (
        status
        == "single_intersection"
    ):

        warnings.append(
            "Only one genuine 50% intersection "
            "was detected, so width cannot "
            "be calculated."
        )

    elif (
        number_of_crossings
        > 2
    ):

        warnings.append(
            "More than two genuine intersections "
            "were detected. Width uses maximum X "
            "minus minimum X."
        )

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

    # ========================================================
    # RESULT JSON
    # ========================================================

    result = {
        "status":
            status,

        "reason":
            (
                reason
                if reason
                is not None
                else "None"
            ),

        "measurement_definition":
            (
                "50_percent_of_detected_"
                "distribution_peak"
            ),

        "minimum_peak_height_ratio":
            MIN_PEAK_HEIGHT_RATIO,

        "peak_height_ratio":
            (
                None
                if peak_height_ratio
                is None
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
                if peak_height_ratio
                is None
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
                if width_50
                is None
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
                if peak_result.get(
                    "curve_half_y"
                )
                is None
                else round(
                    float(
                        peak_result[
                            "curve_half_y"
                        ]
                    ),
                    2,
                )
            ),

        "peak_x_pixel":
            (
                None
                if peak_result.get(
                    "peak_x"
                )
                is None
                else round(
                    float(
                        peak_result[
                            "peak_x"
                        ]
                    ),
                    2,
                )
            ),

        "peak_y_pixel":
            (
                None
                if peak_result.get(
                    "peak_y"
                )
                is None
                else round(
                    float(
                        peak_result[
                            "peak_y"
                        ]
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
                    dict.fromkeys(
                        warnings
                    )
                )
            ),
    }

    # ========================================================
    # RESULT IMAGE
    # ========================================================

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

            trace_result=(
                trace_result
            ),

            peak_result=(
                peak_result
            ),

            crossings_px=(
                crossings_px
            ),

            crossings_fl=(
                crossings_fl
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