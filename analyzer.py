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
            ) / 2.0
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
    minimum_distance=2.5,
):

    if not values:
        return []

    values = sorted(
        float(value)
        for value
        in values
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
        for group
        in groups
    ]


def detect_fixed_50_crossings(
    trace_result,
    plot_bounds,
    tangent_tolerance_pixels=1.25,
):

    if not trace_result.get(
        "valid",
        False,
    ):

        return {
            "valid":
                False,

            "reason":
                "insufficient_curve_support",

            "crossings_px":
                [],
        }

    x_values = np.asarray(
        trace_result["x"],
        dtype=np.float32,
    )

    y_values = np.asarray(
        trace_result["y"],
        dtype=np.float32,
    )

    target_y = float(
        plot_bounds[
            "fixed_y50"
        ]
    )

    finite = np.isfinite(
        y_values
    )

    crossings = []

    continuous_segments = (
        find_true_runs(
            finite
        )
    )

    for start, end in (
        continuous_segments
    ):

        if (
            end - start + 1
            < 4
        ):
            continue

        segment_x = x_values[
            start:
            end + 1
        ]

        segment_y = y_values[
            start:
            end + 1
        ]

        differences = (
            segment_y
            - target_y
        )

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

            if d1 * d2 < 0:

                crossing = (
                    interpolate_crossing(
                        x1,
                        y1,
                        x2,
                        y2,
                        target_y,
                    )
                )

                crossings.append(
                    crossing
                )

        # Detect a curve that touches
        # the fixed 50% line without
        # actually crossing through it.
        absolute_difference = (
            np.abs(
                differences
            )
        )

        for index in range(
            1,
            len(segment_x) - 1,
        ):

            if (
                absolute_difference[
                    index
                ]
                <= tangent_tolerance_pixels
            ):

                left_difference = (
                    absolute_difference[
                        index - 1
                    ]
                )

                right_difference = (
                    absolute_difference[
                        index + 1
                    ]
                )

                same_side = (
                    differences[
                        index - 1
                    ]
                    * differences[
                        index + 1
                    ]
                    > 0
                )

                local_minimum = (
                    absolute_difference[
                        index
                    ]
                    <= left_difference
                    and
                    absolute_difference[
                        index
                    ]
                    <= right_difference
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

    crossings = (
        merge_close_values(
            crossings,
            minimum_distance=2.5,
        )
    )

    if len(crossings) == 0:

        status = (
            "no_intersection"
        )

    elif len(crossings) == 1:

        status = (
            "single_intersection"
        )

    else:

        status = (
            "measure_width"
        )

    return {
        "valid":
            True,

        "reason":
            None,

        "crossings_px":
            crossings,

        "status":
            status,

        "fixed_y50":
            target_y,
    }


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

    normalized = (
        float(x_pixel)
        - float(x_left)
    ) / denominator

    normalized = float(
        np.clip(
            normalized,
            0.0,
            1.0,
        )
    )

    return float(
        x_min_fl
        + normalized
        * (
            x_max_fl
            - x_min_fl
        )
    )


def format_fl(
    value,
):

    if value is None:
        return "Not available"

    return (
        f"{float(value):.3f} fL"
    )


def make_result_table(
    result,
):

    rows = [
        [
            "Status",
            result["status"],
        ],

        [
            "Measurement",
            "Fixed 50% of graph height",
        ],

        [
            "X-axis range",
            (
                f'{result["x_min_fl"]:.0f}'
                f'–'
                f'{result["x_max_fl"]:.0f}'
                f' fL'
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
                f'{result["curve_support"]:.1f}%'
            ),
        ],

        [
            "Warning",
            result["warning"],
        ],
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Measurement",
            "Result",
        ],
    )


def annotate_result(
    image_rgb,
    probability_mask,
    plot_bounds,
    crossings_px,
    crossings_fl,
    width_50_fl,
    status,
    segmentation_threshold,
):

    annotated = cv2.cvtColor(
        image_rgb.copy(),
        cv2.COLOR_RGB2BGR,
    )

    height, width = (
        annotated.shape[:2]
    )

    # Light visualization of
    # segmentation result.
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

    y_top = int(
        plot_bounds[
            "y_top"
        ]
    )

    baseline_y = int(
        plot_bounds[
            "baseline_y"
        ]
    )

    y50 = int(
        round(
            plot_bounds[
                "fixed_y50"
            ]
        )
    )

    # Detected Y-axis.
    cv2.line(
        annotated,
        (
            x_left,
            y_top,
        ),
        (
            x_left,
            baseline_y,
        ),
        (
            255,
            90,
            0,
        ),
        1,
        cv2.LINE_AA,
    )

    # Detected X-axis.
    cv2.line(
        annotated,
        (
            x_left,
            baseline_y,
        ),
        (
            x_right,
            baseline_y,
        ),
        (
            255,
            90,
            0,
        ),
        1,
        cv2.LINE_AA,
    )

    # Fixed 50% line.
    cv2.line(
        annotated,
        (
            x_left,
            y50,
        ),
        (
            x_right,
            y50,
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
        "Fixed 50%",
        (
            min(
                width - 70,
                x_left + 3,
            ),
            max(
                13,
                y50 - 5,
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
                y50,
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
            y50 - 8
            if index % 2 == 0
            else y50 + 17
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

    summary = (
        f"Status: {status}"
    )

    if width_50_fl is not None:

        summary += (
            f" | Width: "
            f"{width_50_fl:.2f} fL"
        )

    cv2.rectangle(
        annotated,
        (0, 0),
        (
            min(
                width - 1,
                max(
                    220,
                    len(summary) * 7,
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
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
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


@torch.inference_mode()
def analyze_plt_graph(
    input_image,
    model,
    device,
    x_axis_max_fl=40.0,
    segmentation_threshold=0.35,
):

    (
        original_image,
        image_tensor,
        transform_info,
    ) = prepare_inference_image(
        input_image,
        device,
    )

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

    plot_bounds = (
        detect_plot_bounds(
            original_image
        )
    )

    trace_result = (
        trace_curve_centerline(
            probability_mask,
            plot_bounds,
            float(
                segmentation_threshold
            ),
        )
    )

    crossing_result = (
        detect_fixed_50_crossings(
            trace_result,
            plot_bounds,
        )
    )

    x_min_fl = 0.0

    x_max_fl = float(
        x_axis_max_fl
    )

    if crossing_result.get(
        "valid",
        False,
    ):

        crossings_px = (
            crossing_result.get(
                "crossings_px",
                [],
            )
        )

    else:

        crossings_px = []

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
        key=lambda item: item[0],
    )

    crossings_px = [
        float(item[0])
        for item
        in pairs
    ]

    crossings_fl = [
        float(item[1])
        for item
        in pairs
    ]

    number_of_crossings = len(
        crossings_fl
    )

    if not crossing_result.get(
        "valid",
        False,
    ):

        status = "reject"

    elif number_of_crossings == 0:

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

    if crossings_fl:

        minimum_intersection = (
            min(
                crossings_fl
            )
        )

        maximum_intersection = (
            max(
                crossings_fl
            )
        )

    else:

        minimum_intersection = None
        maximum_intersection = None

    if number_of_crossings >= 2:

        width_50 = (
            maximum_intersection
            - minimum_intersection
        )

    else:

        width_50 = None

    warnings = []

    if not plot_bounds.get(
        "automatic",
        False,
    ):

        warnings.append(
            "Graph boundaries were "
            "estimated using fallback "
            "logic. Check the annotation."
        )

    if status == "reject":

        warnings.append(
            "The curve could not be "
            "traced reliably."
        )

    elif status == (
        "no_intersection"
    ):

        warnings.append(
            "The graph curve did not "
            "reach the fixed 50% level."
        )

    elif status == (
        "single_intersection"
    ):

        warnings.append(
            "Only one genuine 50% "
            "intersection was detected, "
            "so width cannot be calculated."
        )

    elif (
        number_of_crossings > 2
    ):

        warnings.append(
            "More than two intersections "
            "were detected. Width is "
            "calculated using maximum X "
            "minus minimum X."
        )

    support_ratio = float(
        trace_result.get(
            "support_ratio",
            0.0,
        )
    )

    result = {
        "status":
            status,

        "measurement_definition":
            (
                "fixed_50_percent_"
                "of_plot_height"
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

        "fixed_y50_pixel":
            round(
                float(
                    plot_bounds[
                        "fixed_y50"
                    ]
                ),
                2,
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

    annotated = annotate_result(
        image_rgb=original_image,
        probability_mask=(
            probability_mask
        ),
        plot_bounds=plot_bounds,
        crossings_px=(
            crossings_px
        ),
        crossings_fl=(
            crossings_fl
        ),
        width_50_fl=(
            width_50
        ),
        status=status,
        segmentation_threshold=float(
            segmentation_threshold
        ),
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