# ============================================================
# analyzer.py
# PLT ROI Peak 50% V4 Analyzer
# ============================================================

import cv2
import numpy as np
import pandas as pd
import torch

from preprocessing import (
    read_rgb_image,
    prepare_input,
    restore_probability,
    probability_to_mask
)


# ============================================================
# Sensitivity conversion
# ============================================================

def sensitivity_to_threshold(
    sensitivity
):

    """
    Sensitivity range:
        0   = Low
        100 = High

    Higher sensitivity means a lower segmentation
    threshold and allows weaker curve probabilities.
    """

    sensitivity = float(
        np.clip(
            sensitivity,
            0,
            100
        )
    )

    threshold = (
        0.65
        -
        (
            sensitivity / 100.0
        )
        * 0.50
    )

    return float(
        np.clip(
            threshold,
            0.15,
            0.65
        )
    )


# ============================================================
# Reference line detection
# ============================================================

def detect_reference_lines(rgb):

    height, width = rgb.shape[:2]

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(
            18,
            int(0.035 * height)
        ),
        minLineLength=max(
            20,
            int(0.16 * height)
        ),
        maxLineGap=max(
            5,
            int(0.025 * height)
        )
    )

    candidates = []

    if lines is not None:

        # Critical shape fix:
        # Supports (N, 1, 4) and (N, 4)
        lines = np.asarray(
            lines
        ).reshape(
            -1,
            4
        )

        for x1, y1, x2, y2 in lines:

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            dx = abs(
                x2 - x1
            )

            dy = abs(
                y2 - y1
            )

            # Approximately vertical
            if (

                dy >= max(
                    20,
                    int(0.10 * height)
                )

                and

                dx <= max(
                    4,
                    int(0.025 * width)
                )

            ):

                x = (
                    x1 + x2
                ) / 2.0

                if (

                    0.06 * width
                    < x
                    < 0.96 * width

                ):

                    candidates.append(
                        [
                            x,
                            dy
                        ]
                    )

    # --------------------------------------------------------
    # Merge nearby line segments
    # --------------------------------------------------------

    merged = []

    merge_distance = max(
        6,
        int(0.015 * width)
    )

    for x, score in sorted(
        candidates,
        key=lambda item: item[0]
    ):

        if not merged:

            merged.append(
                [
                    x,
                    score
                ]
            )

        elif abs(
            x - merged[-1][0]
        ) <= merge_distance:

            merged[-1][1] += score

        else:

            merged.append(
                [
                    x,
                    score
                ]
            )

    # --------------------------------------------------------
    # Dark-pixel projection
    # --------------------------------------------------------

    dark = (
        gray < 165
    ).astype(
        np.uint8
    )

    y0 = int(
        0.10 * height
    )

    y1 = int(
        0.92 * height
    )

    projection = dark[
        y0:y1
    ].sum(
        axis=0
    ).astype(
        np.float32
    )

    projection_max = max(
        float(
            projection.max()
        ),
        1.0
    )

    scored = []

    for x, score in merged:

        xi = int(
            np.clip(
                round(x),
                0,
                width - 1
            )
        )

        local_projection = projection[
            max(
                0,
                xi - 3
            ):
            min(
                width,
                xi + 4
            )
        ]

        projection_score = (
            float(
                local_projection.max()
            )
            /
            projection_max
        )

        combined_score = (
            score
            *
            (
                0.5
                +
                0.5
                *
                projection_score
            )
        )

        scored.append(
            [
                x,
                combined_score
            ]
        )

    scored.sort(
        key=lambda item: item[1],
        reverse=True
    )

    # --------------------------------------------------------
    # Select the strongest separated pair
    # --------------------------------------------------------

    best_pair = None

    best_score = -1

    minimum_separation = max(
        30,
        int(0.20 * width)
    )

    for i in range(
        len(scored)
    ):

        for j in range(
            i + 1,
            len(scored)
        ):

            a, score_a = scored[i]

            b, score_b = scored[j]

            left, right = sorted(
                [
                    a,
                    b
                ]
            )

            separation = (
                right - left
            )

            if (

                separation
                >= minimum_separation

                and

                0.04 * width
                < left

                and

                right
                < 0.98 * width

            ):

                pair_score = (
                    score_a
                    +
                    score_b
                )

                if pair_score > best_score:

                    best_score = (
                        pair_score
                    )

                    best_pair = (
                        int(left),
                        int(right)
                    )

    # --------------------------------------------------------
    # Fallback ROI
    # --------------------------------------------------------

    if best_pair is None:

        return {

            "left": int(
                0.15 * width
            ),

            "right": int(
                0.85 * width
            ),

            "confidence": 0.15,

            "method": "fallback"

        }

    return {

        "left": best_pair[0],

        "right": best_pair[1],

        "confidence": float(
            np.clip(
                best_score
                /
                max(
                    height,
                    1
                ),
                0,
                1
            )
        ),

        "method": "hough_projection"

    }


# ============================================================
# Model prediction
# ============================================================

@torch.no_grad()
def predict_curve(
    model,
    device,
    rgb,
    image_size,
    threshold
):

    height, width = rgb.shape[:2]

    tensor = prepare_input(
        rgb,
        image_size
    ).to(
        device
    )

    logits = model(
        tensor
    )

    probability = torch.sigmoid(
        logits
    )[0, 0].cpu().numpy()

    probability = restore_probability(
        probability,
        width,
        height
    )

    mask = probability_to_mask(
        probability,
        threshold
    )

    return (
        probability,
        mask
    )


# ============================================================
# Baseline detection
# ============================================================

def estimate_baseline(rgb):

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray.shape

    y0 = int(
        0.70 * height
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    scores = edges[
        y0:
        int(0.97 * height)
    ].sum(
        axis=1
    )

    baseline = (
        y0
        +
        int(
            np.argmax(
                scores
            )
        )
    )

    baseline = int(
        np.clip(
            baseline,
            int(0.72 * height),
            int(0.96 * height)
        )
    )

    return baseline


# ============================================================
# ROI centerline
# ============================================================

def build_centerline(
    mask,
    left,
    right,
    baseline
):

    height, width = mask.shape

    xs = np.arange(
        max(
            0,
            left
        ),
        min(
            width,
            right + 1
        )
    )

    ys = np.full(
        len(xs),
        np.nan,
        dtype=np.float32
    )

    for index, x in enumerate(xs):

        yy = np.where(
            mask[
                :baseline + 1,
                int(x)
            ] > 0
        )[0]

        if len(yy):

            ys[index] = float(
                np.median(
                    yy
                )
            )

    # --------------------------------------------------------
    # Interpolate only short gaps
    # --------------------------------------------------------

    good = np.isfinite(
        ys
    )

    if good.sum() >= 2:

        interpolated = np.interp(
            xs,
            xs[good],
            ys[good]
        )

        max_gap = max(
            12,
            int(0.03 * width)
        )

        for i in np.where(
            ~good
        )[0]:

            left_indices = np.where(
                good[:i]
            )[0]

            right_indices = np.where(
                good[i + 1:]
            )[0]

            if (

                len(left_indices)
                and
                len(right_indices)

            ):

                left_index = (
                    left_indices[-1]
                )

                right_index = (
                    right_indices[0]
                    +
                    i
                    +
                    1
                )

                gap = (
                    right_index
                    -
                    left_index
                )

                if gap <= max_gap:

                    ys[i] = (
                        interpolated[i]
                    )

    return (
        xs,
        ys
    )


# ============================================================
# Peak and 50% intersections
# ============================================================

def peak_and_intersections(
    xs,
    ys,
    baseline
):

    good = np.isfinite(
        ys
    )

    if good.sum() < 3:

        return None

    x = xs[
        good
    ].astype(
        float
    )

    y = ys[
        good
    ].astype(
        float
    )

    curve_height = np.maximum(
        baseline - y,
        0
    )

    peak_index = int(
        np.argmax(
            curve_height
        )
    )

    peak_height = float(
        curve_height[
            peak_index
        ]
    )

    if peak_height <= 2:

        return None

    peak_x = float(
        x[
            peak_index
        ]
    )

    peak_y = float(
        y[
            peak_index
        ]
    )

    y50 = float(
        baseline
        -
        0.5
        *
        peak_height
    )

    differences = (
        y50 - y
    )

    intersections = []

    # --------------------------------------------------------
    # Detect crossings
    # --------------------------------------------------------

    for i in range(
        len(x) - 1
    ):

        a = differences[i]

        b = differences[
            i + 1
        ]

        if not (

            np.isfinite(a)
            and
            np.isfinite(b)

        ):

            continue

        # Exact crossing
        if abs(a) < 1e-6:

            intersections.append(
                float(
                    x[i]
                )
            )

        # Sign change
        if a * b < 0:

            t = (
                abs(a)
                /
                (
                    abs(a)
                    +
                    abs(b)
                    +
                    1e-12
                )
            )

            crossing_x = (
                x[i]
                +
                t
                *
                (
                    x[i + 1]
                    -
                    x[i]
                )
            )

            intersections.append(
                float(
                    crossing_x
                )
            )

    # --------------------------------------------------------
    # Remove nearby duplicates
    # --------------------------------------------------------

    intersections = sorted(
        intersections
    )

    cleaned = []

    merge_distance = max(
        3,
        int(
            0.008
            *
            max(
                1,
                x.max() - x.min()
            )
        )
    )

    for value in intersections:

        if (

            not cleaned
            or
            abs(
                value
                -
                cleaned[-1]
            )
            >
            merge_distance

        ):

            cleaned.append(
                value
            )

    return {

        "peak_x": peak_x,

        "peak_y": peak_y,

        "peak_height": peak_height,

        "y50": y50,

        "intersections": cleaned

    }


# ============================================================
# Pixel-to-X-axis conversion
# ============================================================

def pixel_to_x_value(
    x,
    left,
    right,
    x_min,
    x_max
):

    if right <= left:

        return np.nan

    ratio = (
        x - left
    ) / (
        right - left
    )

    ratio = np.clip(
        ratio,
        0,
        1
    )

    return float(
        x_min
        +
        ratio
        *
        (
            x_max
            -
            x_min
        )
    )


# ============================================================
# Pixel-to-Y-axis conversion
# ============================================================

def pixel_to_y_value(
    y,
    baseline,
    y_max
):

    """
    Assumption:
        Baseline represents Y = 0.
        The top of the image represents Y = y_max.

    This value is provided as a visual vertical calibration.
    """

    if y_max is None:

        return None

    try:

        y_max = float(
            y_max
        )

    except (
        TypeError,
        ValueError
    ):

        return None

    if y_max <= 0:

        return None

    if baseline <= 0:

        return None

    value = (
        baseline - y
    ) / baseline * y_max

    return float(
        max(
            0,
            value
        )
    )


# ============================================================
# Complete analysis
# ============================================================

def analyze_plt_image(
    model,
    device,
    image,
    x_min,
    x_max,
    threshold=0.35,
    y_max=None,
    sensitivity=None,
    image_size=512
):

    rgb = read_rgb_image(
        image
    )

    height, width = rgb.shape[:2]

    # --------------------------------------------------------
    # Sensitivity override
    # --------------------------------------------------------

    if sensitivity is not None:

        threshold = sensitivity_to_threshold(
            sensitivity
        )

    threshold = float(
        np.clip(
            threshold,
            0.15,
            0.65
        )
    )

    # --------------------------------------------------------
    # Detect ROI
    # --------------------------------------------------------

    reference = detect_reference_lines(
        rgb
    )

    left = reference[
        "left"
    ]

    right = reference[
        "right"
    ]

    # --------------------------------------------------------
    # Segment curve
    # --------------------------------------------------------

    probability, mask = predict_curve(
        model,
        device,
        rgb,
        image_size,
        threshold
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = estimate_baseline(
        rgb
    )

    # --------------------------------------------------------
    # Extract centerline only inside ROI
    # --------------------------------------------------------

    xs, ys = build_centerline(
        mask,
        left,
        right,
        baseline
    )

    measurement = peak_and_intersections(
        xs,
        ys,
        baseline
    )

    result = {

        "status": "peak_not_found",

        "warning": None,

        "image_rgb": rgb,

        "curve_probability": probability,

        "curve_mask": mask,

        "reference_left_px": left,

        "reference_right_px": right,

        "reference_confidence":
            reference["confidence"],

        "reference_method":
            reference["method"],

        "baseline_y_px": baseline,

        "threshold": threshold,

        "sensitivity": sensitivity,

        "peak_x_px": None,

        "peak_y_px": None,

        "peak_height_px": None,

        "peak_y_fl": None,

        "y50_px": None,

        "y50_fl": None,

        "intersections_px": [],

        "intersections_fl": [],

        "width_50_fl": None

    }

    if measurement is None:

        result["warning"] = (
            "A valid curve peak could not be detected "
            "inside the ROI."
        )

        return result

    peak_x = measurement[
        "peak_x"
    ]

    peak_y = measurement[
        "peak_y"
    ]

    peak_height = measurement[
        "peak_height"
    ]

    y50 = measurement[
        "y50"
    ]

    intersections = [

        x

        for x in measurement[
            "intersections"
        ]

        if left <= x <= right

    ]

    result.update({

        "peak_x_px": peak_x,

        "peak_y_px": peak_y,

        "peak_height_px":
            peak_height,

        "y50_px": y50,

        "intersections_px":
            intersections,

        "peak_y_fl":
            pixel_to_y_value(
                peak_y,
                baseline,
                y_max
            ),

        "y50_fl":
            pixel_to_y_value(
                y50,
                baseline,
                y_max
            )

    })

    # --------------------------------------------------------
    # X-axis calibration
    # --------------------------------------------------------

    try:

        x_min = float(
            x_min
        )

        x_max = float(
            x_max
        )

        if x_max <= x_min:

            raise ValueError(
                "X-axis maximum must be greater "
                "than X-axis minimum."
            )

        intersections_fl = [

            pixel_to_x_value(
                x,
                left,
                right,
                x_min,
                x_max
            )

            for x in intersections

        ]

        result[
            "intersections_fl"
        ] = intersections_fl

        # Width only when at least two crossings exist
        if len(
            intersections_fl
        ) >= 2:

            result[
                "width_50_fl"
            ] = (

                max(
                    intersections_fl
                )
                -
                min(
                    intersections_fl
                )

            )

    except (
        TypeError,
        ValueError
    ):

        result["warning"] = (
            "Invalid X-axis calibration."
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if len(
        intersections
    ) == 0:

        result["status"] = (
            "no_intersection"
        )

    elif len(
        intersections
    ) == 1:

        result["status"] = (
            "single_intersection"
        )

    else:

        result["status"] = (
            "measure_width"
        )

    return result


# ============================================================
# Annotation
# ============================================================

def annotate_result(
    result
):

    image = result[
        "image_rgb"
    ].copy()

    height, width = image.shape[:2]

    left = int(
        result[
            "reference_left_px"
        ]
    )

    right = int(
        result[
            "reference_right_px"
        ]
    )

    # ROI boundaries
    cv2.line(
        image,
        (
            left,
            0
        ),
        (
            left,
            height - 1
        ),
        (
            0,
            255,
            0
        ),
        2
    )

    cv2.line(
        image,
        (
            right,
            0
        ),
        (
            right,
            height - 1
        ),
        (
            0,
            255,
            0
        ),
        2
    )

    # 50% horizontal line
    if result[
        "y50_px"
    ] is not None:

        y50 = int(
            result[
                "y50_px"
            ]
        )

        cv2.line(
            image,
            (
                left,
                y50
            ),
            (
                right,
                y50
            ),
            (
                255,
                165,
                0
            ),
            2
        )

    # Peak
    if (

        result[
            "peak_x_px"
        ] is not None

        and

        result[
            "peak_y_px"
        ] is not None

    ):

        cv2.circle(
            image,
            (
                int(
                    result[
                        "peak_x_px"
                    ]
                ),
                int(
                    result[
                        "peak_y_px"
                    ]
                )
            ),
            7,
            (
                255,
                0,
                255
            ),
            -1
        )

    # Intersections
    if result[
        "y50_px"
    ] is not None:

        y50 = int(
            result[
                "y50_px"
            ]
        )

        for x in result[
            "intersections_px"
        ]:

            cv2.circle(
                image,
                (
                    int(x),
                    y50
                ),
                7,
                (
                    255,
                    0,
                    0
                ),
                -1
            )

    return image


# ============================================================
# Results table
# ============================================================

def make_result_table(
    result
):

    intersections_fl = result.get(
        "intersections_fl",
        []
    )

    return pd.DataFrame({

        "Measurement": [

            "Status",

            "Sensitivity",

            "Segmentation threshold",

            "Reference ROI left (px)",

            "Reference ROI right (px)",

            "Peak X (px)",

            "Peak Y (px)",

            "Peak Y (fL)",

            "50% level (px)",

            "50% level (fL)",

            "Intersections (fL)",

            "Width at 50% (fL)"

        ],

        "Value": [

            result.get(
                "status"
            ),

            result.get(
                "sensitivity"
            ),

            round(
                result.get(
                    "threshold",
                    np.nan
                ),
                4
            ),

            result.get(
                "reference_left_px"
            ),

            result.get(
                "reference_right_px"
            ),

            result.get(
                "peak_x_px"
            ),

            result.get(
                "peak_y_px"
            ),

            result.get(
                "peak_y_fl"
            ),

            result.get(
                "y50_px"
            ),

            result.get(
                "y50_fl"
            ),

            [
                round(
                    value,
                    4
                )

                for value in intersections_fl
            ],

            result.get(
                "width_50_fl"
            )

        ]

    })