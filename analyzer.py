from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from preprocessing import (
    normalize_tensor,
    read_rgb,
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 512

DEFAULT_THRESHOLD = 0.28


# ============================================================
# SENSITIVITY DESCRIPTION
# ============================================================

def get_sensitivity_description(
    threshold: float,
) -> str:

    threshold = float(
        threshold
    )

    if threshold <= 0.12:

        return (
            "Very high sensitivity"
        )

    if threshold <= 0.18:

        return (
            "High sensitivity"
        )

    if threshold <= 0.28:

        return (
            "Medium sensitivity"
        )

    if threshold <= 0.36:

        return (
            "Low sensitivity"
        )

    return (
        "Very conservative sensitivity"
    )


# ============================================================
# NORMALIZE
# ============================================================

def _norm(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.size == 0:

        return np.zeros_like(
            values
        )

    maximum = float(
        values.max()
    )

    if maximum <= 1e-9:

        return np.zeros_like(
            values
        )

    return (
        values
        / maximum
    )


# ============================================================
# VERTICAL RUN
# ============================================================

def _vertical_run_stats(
    binary_col: np.ndarray,
):

    binary = (
        np.asarray(
            binary_col,
            dtype=np.uint8,
        )
        > 0
    )

    indices = np.flatnonzero(
        binary
    )

    if indices.size == 0:

        return 0, 0, 0.0

    starts = indices[
        np.r_[
            True,
            np.diff(indices)
            > 1,
        ]
    ]

    ends = indices[
        np.r_[
            np.diff(indices)
            > 1,
            True,
        ]
    ]

    lengths = (
        ends
        - starts
        + 1
    )

    return (
        int(
            len(lengths)
        ),
        int(
            lengths.max()
        ),
        float(
            lengths.sum()
            / len(binary_col)
        ),
    )


# ============================================================
# ROI DETECTION
# ============================================================

def detect_reference_lines(
    rgb: np.ndarray,
) -> dict:

    h, w = rgb.shape[:2]

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    y0 = int(
        0.04 * h
    )

    y1 = int(
        0.92 * h
    )

    x0 = int(
        0.025 * w
    )

    x1 = int(
        0.975 * w
    )

    crop = gray[
        y0:y1,
        x0:x1,
    ]

    q = np.percentile(
        crop,
        35,
    )

    dark = (
        crop
        <= min(
            205,
            q + 15,
        )
    ).astype(
        np.uint8
    )

    raw = np.zeros(
        crop.shape[1],
        np.float32,
    )

    runs = np.zeros_like(
        raw
    )

    hough_score = np.zeros_like(
        raw
    )

    for x in range(
        crop.shape[1]
    ):

        nr, mr, cov = (
            _vertical_run_stats(
                dark[:, x]
            )
        )

        raw[x] = cov

        runs[x] = (
            min(
                nr,
                12,
            )
            / 12.0
            + 0.25
            * min(
                mr
                / max(
                    1,
                    crop.shape[0],
                ),
                0.35,
            )
        )

    edges = cv2.Canny(
        crop,
        40,
        140,
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(
            12,
            int(
                0.018 * h
            ),
        ),
        minLineLength=max(
            10,
            int(
                0.03 * h
            ),
        ),
        maxLineGap=max(
            5,
            int(
                0.015 * h
            ),
        ),
    )

    if lines is not None:

        records = (
            np.asarray(
                lines
            ).reshape(
                -1,
                4,
            )
        )

        for (
            x_a,
            y_a,
            x_b,
            y_b,
        ) in records:

            dx = abs(
                int(x_b)
                - int(x_a)
            )

            dy = abs(
                int(y_b)
                - int(y_a)
            )

            if (
                dx
                <= max(
                    3,
                    int(
                        0.01 * w
                    ),
                )
                and dy
                >= max(
                    8,
                    int(
                        0.025 * h
                    ),
                )
            ):

                x_mid = int(
                    round(
                        (
                            x_a
                            + x_b
                        )
                        / 2
                    )
                )

                if (
                    0
                    <= x_mid
                    < crop.shape[1]
                ):

                    hough_score[
                        x_mid
                    ] += dy

    score = (
        0.35
        * _norm(
            raw
        )
        + 0.40
        * _norm(
            runs
        )
        + 0.25
        * _norm(
            hough_score
        )
    )

    score = cv2.GaussianBlur(
        score.reshape(
            1,
            -1,
        ),
        (
            0,
            0,
        ),
        max(
            1,
            0.003 * w,
        ),
    ).ravel()

    edge = int(
        0.02 * w
    )

    score[
        :edge
    ] = 0

    score[
        -edge:
    ] = 0

    order = np.argsort(
        score
    )[::-1]

    candidates = []

    minimum_distance = max(
        8,
        int(
            0.025 * w
        ),
    )

    for candidate in order:

        x = int(
            candidate
        )

        if score[
            x
        ] <= 0:

            break

        if all(
            abs(
                x
                - old
            )
            >= minimum_distance
            for old in candidates
        ):

            candidates.append(
                x
            )

        if len(
            candidates
        ) >= 40:

            break

    best = None
    best_score = -1.0

    minimum_separation = max(
        30,
        int(
            0.30
            * crop.shape[1]
        ),
    )

    for i, a in enumerate(
        candidates
    ):

        for b in candidates[
            i + 1:
        ]:

            left, right = sorted(
                (
                    a,
                    b,
                )
            )

            separation = (
                right
                - left
            )

            if (
                separation
                < minimum_separation
            ):

                continue

            pair_score = float(
                score[
                    left
                ]
                + score[
                    right
                ]
            )

            if (
                left
                < 0.45
                * crop.shape[1]
                and right
                > 0.55
                * crop.shape[1]
            ):

                pair_score += 0.25

            if pair_score > best_score:

                best_score = (
                    pair_score
                )

                best = (
                    left,
                    right,
                )

    if best is None:

        return {
            "ok": False,
            "left_x": None,
            "right_x": None,
            "roi_width": None,
            "confidence": 0.0,
            "method": "no_reliable_pair",
        }

    left, right = best

    left += x0
    right += x0

    return {
        "ok": True,
        "left_x": int(left),
        "right_x": int(right),
        "roi_width": int(
            right - left
        ),
        "confidence": float(
            np.clip(
                best_score
                / 2.0,
                0,
                1,
            )
        ),
        "method": (
            "dashed-runs+hough"
        ),
    }


# ============================================================
# HORIZONTAL X-AXIS DETECTION
# ============================================================

def detect_horizontal_x_axis(
    rgb: np.ndarray,
) -> dict:
    """
    Detect the actual horizontal X-axis.

    IMPORTANT:
    This function does NOT detect numerical labels.

    It looks for a long horizontal line in the lower
    portion of the graph.
    """

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    h, w = gray.shape

    # --------------------------------------------------------
    # Improve visibility
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(
            8,
            8,
        ),
    )

    enhanced = clahe.apply(
        gray
    )

    # --------------------------------------------------------
    # Canny
    # --------------------------------------------------------

    edges = cv2.Canny(
        enhanced,
        30,
        110,
    )

    # Only search in lower graph area.
    y_start = int(
        0.55 * h
    )

    y_end = int(
        0.90 * h
    )

    lower_edges = np.zeros_like(
        edges
    )

    lower_edges[
        y_start:y_end
    ] = edges[
        y_start:y_end
    ]

    # --------------------------------------------------------
    # Hough candidates
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        lower_edges,
        1,
        np.pi / 180,
        threshold=max(
            20,
            int(
                0.025 * w
            ),
        ),
        minLineLength=max(
            40,
            int(
                0.25 * w
            ),
        ),
        maxLineGap=max(
            8,
            int(
                0.02 * w
            ),
        ),
    )

    candidates = []

    if lines is not None:

        records = (
            np.asarray(
                lines
            ).reshape(
                -1,
                4,
            )
        )

        for (
            x1,
            y1,
            x2,
            y2,
        ) in records:

            dx = abs(
                int(x2)
                - int(x1)
            )

            dy = abs(
                int(y2)
                - int(y1)
            )

            if dx <= 0:
                continue

            angle = abs(
                np.degrees(
                    np.arctan2(
                        dy,
                        dx,
                    )
                )
            )

            if angle > 2.5:
                continue

            y_mid = (
                int(y1)
                + int(y2)
            ) // 2

            length = float(
                np.hypot(
                    dx,
                    dy,
                )
            )

            x_left = min(
                int(x1),
                int(x2),
            )

            x_right = max(
                int(x1),
                int(x2),
            )

            # Darkness directly around the line.
            y0 = max(
                0,
                y_mid - 2,
            )

            y1b = min(
                h,
                y_mid + 3,
            )

            band = enhanced[
                y0:y1b,
                x_left:x_right + 1,
            ]

            darkness = (
                float(
                    np.mean(
                        band < 190
                    )
                )
                if band.size
                else 0.0
            )

            # Prefer long lines low in the graph.
            vertical_position = (
                1.0
                - abs(
                    (
                        y_mid
                        / max(
                            1,
                            h,
                        )
                    )
                    - 0.79
                )
                / 0.30
            )

            vertical_position = float(
                np.clip(
                    vertical_position,
                    0,
                    1,
                )
            )

            score = (
                length
                * (
                    0.65
                    + 0.25
                    * darkness
                    + 0.10
                    * vertical_position
                )
            )

            candidates.append(
                {
                    "start_x": x_left,
                    "end_x": x_right,
                    "y": y_mid,
                    "length": length,
                    "darkness": darkness,
                    "score": score,
                }
            )

    # --------------------------------------------------------
    # Morphological backup
    # --------------------------------------------------------

    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    lower_binary = np.zeros_like(
        binary
    )

    lower_binary[
        y_start:y_end
    ] = binary[
        y_start:y_end
    ]

    kernel_width = max(
        20,
        int(
            0.05 * w
        ),
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            kernel_width,
            1,
        ),
    )

    horizontal = cv2.morphologyEx(
        lower_binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    horizontal = cv2.morphologyEx(
        horizontal,
        cv2.MORPH_CLOSE,
        horizontal_kernel,
    )

    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:

        x, y, cw, ch = (
            cv2.boundingRect(
                contour
            )
        )

        if (
            cw
            < max(
                100,
                int(
                    0.30 * w
                ),
            )
        ):

            continue

        if ch > 8:
            continue

        candidates.append(
            {
                "start_x": int(
                    x
                ),
                "end_x": int(
                    x + cw - 1
                ),
                "y": int(
                    y + ch // 2
                ),
                "length": float(
                    cw
                ),
                "darkness": 0.50,
                "score": float(
                    cw * 0.80
                ),
            }
        )

    # --------------------------------------------------------
    # No candidate
    # --------------------------------------------------------

    if not candidates:

        return {
            "ok": False,
            "start_x": None,
            "end_x": None,
            "y": None,
            "confidence": 0.0,
            "method": "not_found",
        }

    # --------------------------------------------------------
    # Remove tiny duplicates / pick best
    # --------------------------------------------------------

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True,
    )

    best = candidates[0]

    # If there are several overlapping pieces at nearly the
    # same Y, merge them into a longer horizontal axis.
    same_y = [
        c
        for c in candidates
        if abs(
            c["y"]
            - best["y"]
        )
        <= 6
    ]

    if len(
        same_y
    ) > 1:

        start = min(
            c["start_x"]
            for c in same_y
        )

        end = max(
            c["end_x"]
            for c in same_y
        )

        merged_length = (
            end
            - start
            + 1
        )

        # Only accept merging when the resulting line is
        # reasonably continuous.
        union_area = np.zeros(
            (
                1,
                w,
            ),
            dtype=np.uint8,
        )

        for c in same_y:

            cv2.line(
                union_area,
                (
                    c[
                        "start_x"
                    ],
                    0,
                ),
                (
                    c[
                        "end_x"
                    ],
                    0,
                ),
                255,
                1,
            )

        # Longest connected run.
        positions = np.flatnonzero(
            union_area[0] > 0
        )

        if positions.size:

            starts = positions[
                np.r_[
                    True,
                    np.diff(
                        positions
                    ) > 1,
                ]
            ]

            ends = positions[
                np.r_[
                    np.diff(
                        positions
                    ) > 1,
                    True,
                ]
            ]

            lengths = (
                ends
                - starts
                + 1
            )

            largest = int(
                np.argmax(
                    lengths
                )
            )

            start = int(
                starts[
                    largest
                ]
            )

            end = int(
                ends[
                    largest
                ]
            )

        best = dict(
            best,
            start_x=start,
            end_x=end,
            length=float(
                end
                - start
                + 1
            ),
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    axis_length = (
        best["end_x"]
        - best["start_x"]
        + 1
    )

    minimum_axis_length = max(
        120,
        int(
            0.35 * w
        ),
    )

    if (
        axis_length
        < minimum_axis_length
    ):

        return {
            "ok": False,
            "start_x": None,
            "end_x": None,
            "y": int(
                best["y"]
            ),
            "confidence": float(
                np.clip(
                    axis_length
                    / max(
                        1,
                        minimum_axis_length,
                    ),
                    0,
                    1,
                )
            ),
            "method": "axis_too_short",
        }

    confidence = float(
        np.clip(
            0.65
            * (
                axis_length
                / max(
                    1,
                    0.85 * w,
                )
            )
            + 0.20
            * best.get(
                "darkness",
                0.5,
            )
            + 0.15,
            0,
            1,
        )
    )

    return {
        "ok": True,
        "start_x": int(
            best["start_x"]
        ),
        "end_x": int(
            best["end_x"]
        ),
        "y": int(
            best["y"]
        ),
        "length": int(
            axis_length
        ),
        "confidence": confidence,
        "method": (
            "horizontal-axis-line"
        ),
    }


# ============================================================
# MODEL PREDICTION
# ============================================================

@torch.inference_mode()
def predict_curve(
    model,
    device,
    rgb: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:

    tensor = normalize_tensor(
        rgb,
        image_size,
    ).to(
        device
    )

    probability = torch.sigmoid(
        model(
            tensor
        )
    )[0, 0].detach().cpu().numpy()

    return cv2.resize(
        probability,
        (
            rgb.shape[1],
            rgb.shape[0],
        ),
        interpolation=cv2.INTER_LINEAR,
    )


# ============================================================
# VISUAL FAINТ-CURVE SCORE
# ============================================================

def build_visual_curve_score(
    rgb: np.ndarray,
    left: int,
    right: int,
    baseline: int,
) -> np.ndarray:

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(
            8,
            8,
        ),
    )

    enhanced = clahe.apply(
        gray
    )

    dark_score = np.clip(
        (
            220.0
            - enhanced.astype(
                np.float32
            )
        )
        / 170.0,
        0.0,
        1.0,
    )

    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            13,
            13,
        ),
    )

    blackhat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        blackhat_kernel,
    ).astype(
        np.float32
    )

    blackhat = _norm(
        blackhat
    )

    edges = cv2.Canny(
        enhanced,
        20,
        100,
    ).astype(
        np.float32
    )

    edges = _norm(
        edges
    )

    visual = (
        0.45
        * dark_score
        + 0.35
        * blackhat
        + 0.20
        * edges
    )

    visual = cv2.GaussianBlur(
        visual.astype(
            np.float32
        ),
        (
            0,
            0,
        ),
        1.0,
    )

    result = np.zeros_like(
        visual,
        dtype=np.float32,
    )

    top = int(
        0.02 * rgb.shape[0]
    )

    bottom = max(
        top + 1,
        int(
            baseline
        ) - 2,
    )

    result[
        top:bottom,
        left:right + 1,
    ] = visual[
        top:bottom,
        left:right + 1,
    ]

    return np.clip(
        result,
        0,
        1,
    )


# ============================================================
# MASK
# ============================================================

def clean_curve_probability(
    probability: np.ndarray,
    left: int,
    right: int,
    threshold: float,
) -> np.ndarray:

    mask = (
        probability
        >= float(
            threshold
        )
    ).astype(
        np.uint8
    )

    result = np.zeros_like(
        mask
    )

    result[
        :,
        left:right + 1,
    ] = mask[
        :,
        left:right + 1,
    ]

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        np.ones(
            (
                3,
                3,
            ),
            np.uint8,
        ),
    )

    return result


# ============================================================
# BASELINE
# ============================================================

def estimate_baseline(
    rgb: np.ndarray,
    mask: np.ndarray,
    left: int,
    right: int,
) -> int:

    h, w = mask.shape

    l = max(
        0,
        int(left),
    )

    r = min(
        w - 1,
        int(right),
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    ylo = int(
        0.58 * h
    )

    yhi = int(
        0.985 * h
    )

    edges = cv2.Canny(
        gray,
        40,
        140,
    )

    row_edges = (
        edges[
            ylo:yhi,
            l:r + 1,
        ]
        .sum(
            1
        )
        .astype(
            np.float32
        )
    )

    row_dark = (
        (
            gray[
                ylo:yhi,
                l:r + 1,
            ]
            < 180
        )
        .mean(
            1
        )
        .astype(
            np.float32
        )
    )

    score = (
        0.70
        * _norm(
            row_edges
        )
        + 0.30
        * _norm(
            row_dark
        )
    )

    axis_y = (
        ylo
        + int(
            np.argmax(
                score
            )
        )
        if len(score)
        else int(
            0.90 * h
        )
    )

    bottoms = []

    step = max(
        1,
        int(
            (
                r
                - l
                + 1
            )
            / 300
        ),
    )

    for x in range(
        l,
        r + 1,
        step,
    ):

        ys = np.where(
            mask[
                :,
                x
            ]
            > 0
        )[0]

        ys = ys[
            ys
            < 0.97 * h
        ]

        if ys.size:

            bottoms.append(
                int(
                    ys.max()
                )
            )

    curve_y = (
        float(
            np.percentile(
                bottoms,
                97,
            )
        )
        if len(
            bottoms
        ) >= 20
        else None
    )

    baseline = (
        axis_y
        if (
            curve_y is None
            or abs(
                curve_y
                - axis_y
            )
            > 0.07 * h
        )
        else (
            0.55 * axis_y
            + 0.45 * curve_y
        )
    )

    return int(
        np.clip(
            round(
                baseline
            ),
            int(
                0.55 * h
            ),
            int(
                0.99 * h
            ),
        )
    )


# ============================================================
# PLOT TOP
# ============================================================

def estimate_plot_top(
    rgb: np.ndarray,
    left: int,
    right: int,
    baseline: int,
) -> int:

    h, w = rgb.shape[:2]

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    half = max(
        1,
        int(
            0.008 * w
        ),
    )

    values = []

    for x in (
        int(left),
        int(right),
    ):

        a = max(
            0,
            x - half,
        )

        b = min(
            w,
            x + half + 1,
        )

        strip = gray[
            :baseline + 1,
            a:b,
        ]

        if strip.size == 0:
            continue

        count = (
            strip
            < 210
        ).sum(
            axis=1
        )

        ys = np.where(
            count > 0
        )[0]

        if ys.size:

            values.append(
                int(
                    ys[0]
                )
            )

    if values:

        top = min(
            values
        )

    else:

        top = int(
            0.08 * h
        )

    return int(
        np.clip(
            top,
            0,
            max(
                0,
                baseline - 10,
            ),
        )
    )


# ============================================================
# CURVE TRACKER
# ============================================================

def _walk_track(
    probability,
    visual,
    xs,
    ys,
    confidence,
    seed_index,
    seed_y,
    lower,
    direction,
    threshold,
):

    h = probability.shape[0]

    sensitivity = float(
        np.clip(
            (
                0.32
                - threshold
            )
            / 0.27,
            0,
            1,
        )
    )

    max_step = int(
        h
        * (
            0.030
            + 0.032
            * sensitivity
        )
    )

    max_step = max(
        5,
        min(
            40,
            max_step,
        ),
    )

    max_gap = int(
        8
        + 16
        * sensitivity
    )

    visual_weight = (
        0.10
        + 0.22
        * sensitivity
    )

    track_threshold = max(
        0.025,
        threshold
        * (
            0.70
            - 0.25
            * sensitivity
        ),
    )

    previous_y = float(
        seed_y
    )

    gap_count = 0

    iterator = (
        range(
            seed_index + 1,
            len(xs),
        )
        if direction > 0
        else range(
            seed_index - 1,
            -1,
            -1,
        )
    )

    for j in iterator:

        if gap_count > max_gap:

            break

        x = int(
            xs[j]
        )

        y0 = max(
            0,
            int(
                round(
                    previous_y
                )
            )
            - max_step,
        )

        y1 = min(
            lower,
            int(
                round(
                    previous_y
                )
            )
            + max_step,
        )

        if y1 < y0:

            gap_count += 1

            continue

        probabilities = probability[
            y0:y1 + 1,
            x,
        ]

        visual_values = visual[
            y0:y1 + 1,
            x,
        ]

        candidate_y = np.arange(
            y0,
            y1 + 1,
            dtype=np.float32,
        )

        hybrid = (
            (
                1.0
                - visual_weight
            )
            * probabilities
            + visual_weight
            * visual_values
        )

        continuity = (
            0.014
            + 0.012
            * sensitivity
        )

        score = (
            hybrid
            - continuity
            * np.abs(
                candidate_y
                - previous_y
            )
        )

        index = int(
            np.argmax(
                score
            )
        )

        chosen_probability = float(
            probabilities[
                index
            ]
        )

        chosen_score = float(
            score[
                index
            ]
        )

        chosen_visual = float(
            visual_values[
                index
            ]
        )

        accept = (
            chosen_probability
            >= track_threshold
            or (
                chosen_score
                >= track_threshold
                * 0.65
                and chosen_visual
                >= 0.25
            )
        )

        if not accept:

            gap_count += 1

            continue

        chosen_y = int(
            candidate_y[
                index
            ]
        )

        a = max(
            y0,
            chosen_y - 2,
        )

        b = min(
            y1,
            chosen_y + 2,
        )

        p_local = probability[
            a:b + 1,
            x,
        ]

        v_local = visual[
            a:b + 1,
            x,
        ]

        local = (
            (
                1.0
                - visual_weight
            )
            * p_local
            + visual_weight
            * v_local
        )

        weights = np.maximum(
            local,
            0,
        )

        total = float(
            weights.sum()
        )

        if total <= 1e-8:

            center = float(
                chosen_y
            )

        else:

            center = float(
                (
                    np.arange(
                        a,
                        b + 1,
                    )
                    * weights
                ).sum()
                / total
            )

        ys[j] = center

        confidence[j] = max(
            chosen_probability,
            chosen_score,
        )

        previous_y = center

        gap_count = 0


def build_curve_track(
    probability,
    visual,
    left,
    right,
    baseline,
    threshold,
):

    h, _ = probability.shape

    xs = np.arange(
        int(left),
        int(right) + 1,
        dtype=np.int32,
    )

    ys = np.full(
        len(xs),
        np.nan,
        dtype=np.float32,
    )

    confidence = np.zeros(
        len(xs),
        dtype=np.float32,
    )

    lower = min(
        h - 2,
        int(
            baseline
        ) - 1,
    )

    if lower < 5:

        return (
            xs,
            ys,
            confidence,
        )

    sensitivity = float(
        np.clip(
            (
                0.32
                - threshold
            )
            / 0.27,
            0,
            1,
        )
    )

    visual_weight = (
        0.10
        + 0.22
        * sensitivity
    )

    hybrid = (
        (
            1
            - visual_weight
        )
        * probability
        + visual_weight
        * visual
    )

    crop = hybrid[
        :lower + 1,
        int(left):int(right) + 1,
    ]

    if crop.size == 0:

        return (
            xs,
            ys,
            confidence,
        )

    py, px = np.unravel_index(
        int(
            np.argmax(
                crop
            )
        ),
        crop.shape,
    )

    seed_score = float(
        crop[
            py,
            px
        ]
    )

    seed_threshold = max(
        0.07,
        threshold
        * (
            1.10
            - 0.35
            * sensitivity
        ),
    )

    if (
        seed_score
        < seed_threshold
    ):

        return (
            xs,
            ys,
            confidence,
        )

    ys[
        px
    ] = float(
        py
    )

    confidence[
        px
    ] = seed_score

    _walk_track(
        probability,
        visual,
        xs,
        ys,
        confidence,
        int(
            px
        ),
        float(
            py
        ),
        lower,
        +1,
        threshold,
    )

    _walk_track(
        probability,
        visual,
        xs,
        ys,
        confidence,
        int(
            px
        ),
        float(
            py
        ),
        lower,
        -1,
        threshold,
    )

    return (
        xs,
        ys,
        confidence,
    )


# ============================================================
# SAFE SMOOTHING
# ============================================================

def smooth_array(
    values: np.ndarray,
) -> np.ndarray:
    """
    Safe float32 smoothing.

    Does not call cv2.medianBlur on float32 data.
    """

    result = np.full_like(
        values,
        np.nan,
        dtype=np.float32,
    )

    finite = np.isfinite(
        values
    )

    if finite.sum() < 5:

        result[
            finite
        ] = values[
            finite
        ]

        return result

    indices = np.flatnonzero(
        finite
    )

    breaks = (
        np.where(
            np.diff(
                indices
            )
            > 1
        )[0]
        + 1
    )

    segments = np.split(
        indices,
        breaks,
    )

    for segment in segments:

        if len(
            segment
        ) < 3:

            result[
                segment
            ] = values[
                segment
            ]

            continue

        segment_values = (
            values[
                segment
            ].astype(
                np.float32
            )
        )

        kernel = min(
            15,
            max(
                5,
                (
                    len(
                        segment
                    )
                    // 50
                )
                * 2
                + 1,
            ),
        )

        radius = kernel // 2

        padded = np.pad(
            segment_values,
            (
                radius,
                radius,
            ),
            mode="edge",
        )

        windows = (
            np.lib.stride_tricks
            .sliding_window_view(
                padded,
                kernel,
            )
        )

        result[
            segment
        ] = np.median(
            windows,
            axis=-1,
        ).astype(
            np.float32
        )

    return result


# ============================================================
# GENUINE INTERSECTIONS
# ============================================================

def genuine_intersections(
    xs,
    ys,
    y50,
    left,
    right,
):

    valid = (
        np.isfinite(
            xs
        )
        & np.isfinite(
            ys
        )
        & (
            xs >= left
        )
        & (
            xs <= right
        )
    )

    x = xs[
        valid
    ].astype(
        float
    )

    y = ys[
        valid
    ].astype(
        float
    )

    if len(x) < 3:

        return []

    difference = (
        y
        - float(
            y50
        )
    )

    intersections = []

    for i in range(
        len(x) - 1
    ):

        # Never bridge a large missing segment.
        if (
            x[i + 1]
            - x[i]
            > 12
        ):

            continue

        if (
            abs(
                difference[i]
            )
            <= 0.45
        ):

            intersections.append(
                float(
                    x[i]
                )
            )

        if (
            difference[i]
            * difference[i + 1]
            < 0
        ):

            denominator = (
                abs(
                    difference[i]
                )
                + abs(
                    difference[i + 1]
                )
                + 1e-12
            )

            t = (
                abs(
                    difference[i]
                )
                / denominator
            )

            intersections.append(
                float(
                    x[i]
                    + t
                    * (
                        x[i + 1]
                        - x[i]
                    )
                )
            )

    if (
        len(difference)
        and abs(
            difference[-1]
        )
        <= 0.45
    ):

        intersections.append(
            float(
                x[-1]
            )
        )

    intersections.sort()

    output = []

    merge_distance = max(
        2.0,
        0.004
        * max(
            1,
            right - left,
        ),
    )

    for value in intersections:

        if (
            not output
            or value
            - output[-1]
            > merge_distance
        ):

            output.append(
                value
            )

    return output


# ============================================================
# PIXEL TO F-L
# ============================================================

def px_to_fl(
    x,
    axis_start_px,
    axis_end_px,
    xmin,
    xmax,
):

    denominator = float(
        axis_end_px
        - axis_start_px
    )

    if denominator <= 0:

        return float(
            "nan"
        )

    normalized = (
        float(x)
        - float(
            axis_start_px
        )
    ) / denominator

    value = (
        xmin
        + normalized
        * (
            xmax
            - xmin
        )
    )

    return float(
        np.clip(
            value,
            xmin,
            xmax,
        )
    )


# ============================================================
# PIXEL TO Y
# ============================================================

def py_to_y_fl(
    y,
    top,
    baseline,
    y_max,
):

    denominator = float(
        baseline
        - top
    )

    if denominator <= 0:

        return float(
            "nan"
        )

    value = (
        (
            baseline
            - float(y)
        )
        / denominator
        * float(
            y_max
        )
    )

    return float(
        np.clip(
            value,
            0,
            y_max,
        )
    )


# ============================================================
# RECONSTRUCTED X TICKS
# ============================================================

def build_reconstructed_ticks(
    xmin,
    xmax,
):

    start = int(
        np.ceil(
            xmin
            / 10.0
        )
        * 10
    )

    end = int(
        np.floor(
            xmax
            / 10.0
        )
        * 10
    )

    if end < start:

        return []

    return [
        float(
            value
        )
        for value in range(
            start,
            end + 1,
            10,
        )
    ]


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_plt_image(
    model,
    device,
    image_input: Any,
    x_min_fl: float,
    x_max_fl: float,
    y_max_fl: float,
    threshold: float = DEFAULT_THRESHOLD,
    manual_axis_start_px: int | None = None,
    manual_axis_end_px: int | None = None,
) -> dict:

    rgb = read_rgb(
        image_input
    )

    xmin = float(
        x_min_fl
    )

    xmax = float(
        x_max_fl
    )

    ymax = float(
        y_max_fl
    )

    threshold = float(
        threshold
    )

    if (
        xmax <= xmin
    ):

        raise ValueError(
            "X-axis maximum must be greater "
            "than X-axis minimum."
        )

    if (
        ymax <= 0
    ):

        raise ValueError(
            "Y-axis maximum must be greater than 0."
        )

    h, w = rgb.shape[:2]

    if (
        manual_axis_start_px is None
        or manual_axis_end_px is None
    ):

        raise ValueError(
            "Manual/automatic X-axis calibration "
            "positions are required."
        )

    axis_start = int(
        manual_axis_start_px
    )

    axis_end = int(
        manual_axis_end_px
    )

    if (
        axis_start < 0
        or axis_start >= w
        or axis_end < 0
        or axis_end >= w
        or axis_end <= axis_start
    ):

        raise ValueError(
            "Invalid X-axis pixel calibration."
        )

    pixels_per_fl = (
        (
            axis_end
            - axis_start
        )
        / (
            xmax
            - xmin
        )
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "status": "error",

        "measurement_source": (
            "V8 U-Net + faint-curve enhancement "
            "+ manual X-axis calibration"
        ),

        "axis_detection": (
            "ROI automatic; X-axis automatic "
            "with manual pixel correction"
        ),

        "x_min_fl": xmin,
        "x_max_fl": xmax,

        "y_min_fl": 0.0,
        "y_max_fl": ymax,

        "segmentation_threshold": (
            threshold
        ),

        "sensitivity": (
            get_sensitivity_description(
                threshold
            )
        ),

        "reference_left_px": None,
        "reference_right_px": None,
        "reference_confidence": 0.0,
        "reference_method": None,

        "x_axis_start_px": axis_start,
        "x_axis_end_px": axis_end,
        "x_axis_y_px": None,

        "pixels_per_fl": pixels_per_fl,

        "x_axis_method": (
            "Automatic horizontal axis "
            "+ manual correction"
        ),

        "x_axis_reconstructed_ticks": (
            build_reconstructed_ticks(
                xmin,
                xmax,
            )
        ),

        "baseline_y_px": None,
        "plot_top_y_px": None,

        "peak_x_px": None,
        "peak_y_px": None,

        "peak_x_fl": None,
        "peak_y_fl": None,

        "peak_height_px": None,

        "y50_px": None,
        "y50_fl": None,

        "intersections_px": [],
        "intersections_fl": [],

        "intersection_count": 0,

        "width_50_fl": None,

        "confidence": 0.0,

        "warning": None,

        "image_rgb": rgb,

        "curve_probability": None,
        "visual_curve_score": None,
        "curve_mask": None,

        "centerline_x": None,
        "centerline_y": None,
        "centerline_confidence": None,
    }

    # ========================================================
    # ROI
    # ========================================================

    reference = detect_reference_lines(
        rgb
    )

    if not reference.get(
        "ok"
    ):

        result.update(
            {
                "status": (
                    "roi_not_found"
                ),
                "warning": (
                    "Two reliable vertical dashed "
                    "ROI boundaries were not detected."
                ),
            }
        )

        return result

    roi_left = int(
        reference[
            "left_x"
        ]
    )

    roi_right = int(
        reference[
            "right_x"
        ]
    )

    result.update(
        {
            "reference_left_px": roi_left,
            "reference_right_px": roi_right,
            "reference_confidence": float(
                reference.get(
                    "confidence",
                    0.0,
                )
            ),
            "reference_method": reference.get(
                "method"
            ),
        }
    )

    if (
        roi_right
        - roi_left
        < max(
            40,
            int(
                0.15 * w
            ),
        )
    ):

        result.update(
            {
                "status": (
                    "roi_invalid"
                ),
                "warning": (
                    "Detected ROI is too narrow."
                ),
            }
        )

        return result

    # ========================================================
    # MODEL
    # ========================================================

    probability = predict_curve(
        model,
        device,
        rgb,
    )

    # ========================================================
    # MASK + BASELINE
    # ========================================================

    mask = clean_curve_probability(
        probability,
        roi_left,
        roi_right,
        threshold,
    )

    baseline = estimate_baseline(
        rgb,
        mask,
        roi_left,
        roi_right,
    )

    result[
        "baseline_y_px"
    ] = baseline

    # ========================================================
    # X-AXIS Y POSITION
    # ========================================================

    axis_detected = detect_horizontal_x_axis(
        rgb
    )

    if axis_detected.get(
        "ok"
    ):

        result[
            "x_axis_y_px"
        ] = axis_detected.get(
            "y"
        )

    else:

        result[
            "x_axis_y_px"
        ] = baseline

    # ========================================================
    # PLOT TOP
    # ========================================================

    plot_top = estimate_plot_top(
        rgb,
        roi_left,
        roi_right,
        baseline,
    )

    result[
        "plot_top_y_px"
    ] = plot_top

    # ========================================================
    # VISUAL FAINТ CURVE
    # ========================================================

    visual = build_visual_curve_score(
        rgb,
        roi_left,
        roi_right,
        baseline,
    )

    # ========================================================
    # TRACK
    # ========================================================

    (
        xs,
        ys,
        confidence,
    ) = build_curve_track(
        probability,
        visual,
        roi_left,
        roi_right,
        baseline,
        threshold,
    )

    result.update(
        {
            "curve_probability": probability,
            "visual_curve_score": visual,
            "curve_mask": mask,
            "centerline_x": xs,
            "centerline_y": ys,
            "centerline_confidence": confidence,
        }
    )

    # ========================================================
    # VALID TRACE
    # ========================================================

    valid = (
        np.isfinite(
            xs
        )
        & np.isfinite(
            ys
        )
        & (
            ys
            < baseline - 1
        )
        & (
            xs >= roi_left
        )
        & (
            xs <= roi_right
        )
    )

    if valid.sum() < 10:

        result.update(
            {
                "status": (
                    "curve_not_found"
                ),
                "warning": (
                    "The curve could not be reliably "
                    "traced. Lower the sensitivity "
                    "threshold for faint graphs."
                ),
            }
        )

        return result

    # ========================================================
    # PEAK
    # ========================================================

    x_valid = xs[
        valid
    ].astype(
        float
    )

    y_valid = ys[
        valid
    ].astype(
        float
    )

    confidence_valid = confidence[
        valid
    ].astype(
        float
    )

    heights = (
        baseline
        - y_valid
    )

    smoothed = smooth_array(
        heights
    )

    if not np.isfinite(
        smoothed
    ).any():

        result.update(
            {
                "status": (
                    "peak_not_found"
                ),
                "warning": (
                    "The peak could not be calculated."
                ),
            }
        )

        return result

    peak_index = int(
        np.nanargmax(
            smoothed
        )
    )

    peak_x_px = float(
        x_valid[
            peak_index
        ]
    )

    peak_y_px = float(
        y_valid[
            peak_index
        ]
    )

    peak_height = float(
        baseline
        - peak_y_px
    )

    peak_x_fl = px_to_fl(
        peak_x_px,
        axis_start,
        axis_end,
        xmin,
        xmax,
    )

    peak_y_fl = py_to_y_fl(
        peak_y_px,
        plot_top,
        baseline,
        ymax,
    )

    result.update(
        {
            "peak_x_px": peak_x_px,
            "peak_y_px": peak_y_px,

            "peak_x_fl": peak_x_fl,
            "peak_y_fl": peak_y_fl,

            "peak_height_px": peak_height,
        }
    )

    if peak_height < 3:

        result.update(
            {
                "status": (
                    "peak_not_found"
                ),
                "warning": (
                    "Detected curve height is too small "
                    "for a reliable 50% measurement."
                ),
            }
        )

        return result

    # ========================================================
    # 50%
    # ========================================================

    y50_px = (
        baseline
        - 0.5
        * peak_height
    )

    y50_fl = py_to_y_fl(
        y50_px,
        plot_top,
        baseline,
        ymax,
    )

    result[
        "y50_px"
    ] = y50_px

    result[
        "y50_fl"
    ] = y50_fl

    # ========================================================
    # GENUINE INTERSECTIONS
    # ========================================================

    intersections_px = (
        genuine_intersections(
            xs,
            ys,
            y50_px,
            roi_left,
            roi_right,
        )
    )

    intersections_fl = [

        px_to_fl(
            px,
            axis_start,
            axis_end,
            xmin,
            xmax,
        )

        for px in intersections_px
    ]

    # --------------------------------------------------------
    # Validate every intersection against the trace.
    # --------------------------------------------------------

    verified = []

    for px, value in zip(
        intersections_px,
        intersections_fl,
    ):

        index = int(
            np.argmin(
                np.abs(
                    xs
                    - px
                )
            )
        )

        if (
            np.isfinite(
                ys[index]
            )
            and abs(
                float(
                    ys[index]
                )
                - y50_px
            )
            <= 3
        ):

            verified.append(
                (
                    float(
                        px
                    ),
                    float(
                        value
                    ),
                )
            )

    verified.sort(
        key=lambda item: item[1]
    )

    intersections_px = [
        item[0]
        for item in verified
    ]

    intersections_fl = [
        round(
            item[1],
            6,
        )
        for item in verified
    ]

    # ========================================================
    # STATUS + WIDTH
    # ========================================================

    count = len(
        intersections_fl
    )

    if count == 0:

        status = (
            "no_intersection"
        )

    elif count == 1:

        status = (
            "single_intersection"
        )

    else:

        status = (
            "measure_width"
        )

    width = (
        max(
            intersections_fl
        )
        - min(
            intersections_fl
        )
        if count >= 2
        else None
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    trace_conf = (
        float(
            np.nanmedian(
                confidence_valid
            )
        )
        if np.isfinite(
            confidence_valid
        ).any()
        else 0.0
    )

    roi_conf = float(
        reference.get(
            "confidence",
            0.0,
        )
    )

    confidence_score = float(
        np.clip(
            0.30
            * roi_conf
            + 0.70
            * trace_conf,
            0,
            1,
        )
    )

    # ========================================================
    # WARNINGS
    # ========================================================

    warnings = []

    if threshold <= 0.18:

        warnings.append(
            (
                "High sensitivity selected. "
                "Check the curve guides for noise."
            )
        )

    if count == 0:

        warnings.append(
            (
                "The curve does not genuinely cross "
                "the 50% level inside the ROI."
            )
        )

    elif count == 1:

        warnings.append(
            (
                "Only one genuine 50% intersection "
                "was detected; width is not calculated."
            )
        )

    # ========================================================
    # FINAL
    # ========================================================

    result.update(
        {
            "status": status,

            "intersections_px": (
                intersections_px
            ),

            "intersections_fl": (
                intersections_fl
            ),

            "intersection_count": count,

            "width_50_fl": width,

            "confidence": confidence_score,

            "warning": (
                " ".join(
                    dict.fromkeys(
                        warnings
                    )
                )
                if warnings
                else None
            ),
        }
    )

    return result


# ============================================================
# ANNOTATION
# ============================================================

def annotate_result(
    result: dict,
    show_guides: bool = True,
) -> np.ndarray:

    image = result[
        "image_rgb"
    ].copy()

    h, w = image.shape[:2]

    roi_left = result.get(
        "reference_left_px"
    )

    roi_right = result.get(
        "reference_right_px"
    )

    axis_start = result.get(
        "x_axis_start_px"
    )

    axis_end = result.get(
        "x_axis_end_px"
    )

    axis_y = result.get(
        "x_axis_y_px"
    )

    baseline = result.get(
        "baseline_y_px"
    )

    xmin = result.get(
        "x_min_fl"
    )

    xmax = result.get(
        "x_max_fl"
    )

    # ========================================================
    # ROI BOUNDARIES
    # ========================================================

    if roi_left is not None:

        cv2.line(
            image,
            (
                int(
                    roi_left
                ),
                0,
            ),
            (
                int(
                    roi_left
                ),
                h - 1,
            ),
            (
                0,
                255,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

    if roi_right is not None:

        cv2.line(
            image,
            (
                int(
                    roi_right
                ),
                0,
            ),
            (
                int(
                    roi_right
                ),
                h - 1,
            ),
            (
                0,
                255,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # ACTUAL X AXIS
    # ========================================================

    if (
        axis_start is not None
        and axis_end is not None
    ):

        y = int(
            axis_y
            if axis_y is not None
            else (
                baseline
                if baseline is not None
                else int(
                    0.80 * h
                )
            )
        )

        # Main axis
        cv2.line(
            image,
            (
                int(
                    axis_start
                ),
                y,
            ),
            (
                int(
                    axis_end
                ),
                y,
            ),
            (
                0,
                220,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        # End markers -- short lines, NOT dots.
        cv2.line(
            image,
            (
                int(
                    axis_start
                ),
                max(
                    0,
                    y - 8,
                ),
            ),
            (
                int(
                    axis_start
                ),
                min(
                    h - 1,
                    y + 8,
                ),
            ),
            (
                0,
                220,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        cv2.line(
            image,
            (
                int(
                    axis_end
                ),
                max(
                    0,
                    y - 8,
                ),
            ),
            (
                int(
                    axis_end
                ),
                min(
                    h - 1,
                    y + 8,
                ),
            ),
            (
                0,
                220,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        # ====================================================
        # RECONSTRUCT TICKS MATHEMATICALLY
        # ====================================================

        ticks = result.get(
            "x_axis_reconstructed_ticks",
            [],
        )

        axis_range = (
            xmax
            - xmin
        )

        if axis_range > 0:

            for value in ticks:

                ratio = (
                    value
                    - xmin
                ) / axis_range

                px = int(
                    round(
                        axis_start
                        + ratio
                        * (
                            axis_end
                            - axis_start
                        )
                    )
                )

                # Tick line
                cv2.line(
                    image,
                    (
                        px,
                        y - 4,
                    ),
                    (
                        px,
                        y + 10,
                    ),
                    (
                        0,
                        220,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                # Label
                cv2.putText(
                    image,
                    f"{value:g}",
                    (
                        max(
                            1,
                            px - 10,
                        ),
                        min(
                            h - 4,
                            y + 28,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (
                        0,
                        150,
                        180,
                    ),
                    1,
                    cv2.LINE_AA,
                )

    # ========================================================
    # 50% HORIZONTAL LINE
    # ========================================================

    y50 = result.get(
        "y50_px"
    )

    if (
        y50 is not None
        and roi_left is not None
        and roi_right is not None
    ):

        y = int(
            round(
                y50
            )
        )

        cv2.line(
            image,
            (
                int(
                    roi_left
                ),
                y,
            ),
            (
                int(
                    roi_right
                ),
                y,
            ),
            (
                255,
                165,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

        y50_value = result.get(
            "y50_fl"
        )

        if y50_value is not None:

            cv2.putText(
                image,
                (
                    f"50% = "
                    f"{y50_value:.3f} fL"
                ),
                (
                    max(
                        2,
                        int(
                            roi_left
                        ) + 5,
                    ),
                    max(
                        15,
                        y - 7,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (
                    255,
                    165,
                    0,
                ),
                1,
                cv2.LINE_AA,
            )

    # ========================================================
    # PEAK CROSSHAIR
    # ========================================================

    peak_x = result.get(
        "peak_x_px"
    )

    peak_y = result.get(
        "peak_y_px"
    )

    if (
        show_guides
        and peak_x is not None
        and peak_y is not None
    ):

        px = int(
            round(
                peak_x
            )
        )

        py = int(
            round(
                peak_y
            )
        )

        # Horizontal guide
        cv2.line(
            image,
            (
                int(
                    roi_left
                ),
                py,
            ),
            (
                px,
                py,
            ),
            (
                255,
                0,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        # Vertical guide
        cv2.line(
            image,
            (
                px,
                py,
            ),
            (
                px,
                int(
                    axis_y
                    if axis_y is not None
                    else baseline
                ),
            ),
            (
                255,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            (
                f"Peak "
                f"X={result.get('peak_x_fl', 0):.3f} fL"
                f"  Y={result.get('peak_y_fl', 0):.3f} fL"
            ),
            (
                max(
                    2,
                    min(
                        w - 210,
                        px + 8,
                    ),
                ),
                max(
                    15,
                    py - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (
                255,
                0,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

    # ========================================================
    # INTERSECTION GUIDE LINES
    # ========================================================

    if (
        show_guides
        and y50 is not None
    ):

        intersection_pixels = result.get(
            "intersections_px",
            [],
        )

        intersection_values = result.get(
            "intersections_fl",
            [],
        )

        axis_bottom = int(
            axis_y
            if axis_y is not None
            else (
                baseline
                if baseline is not None
                else y50
            )
        )

        for index, intersection_x in enumerate(
            intersection_pixels
        ):

            px = int(
                round(
                    intersection_x
                )
            )

            # Vertical guide from 50% intersection
            # down to the actual X-axis.
            cv2.line(
                image,
                (
                    px,
                    int(
                        round(
                            y50
                        )
                    ),
                ),
                (
                    px,
                    axis_bottom,
                ),
                (
                    255,
                    0,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

            # Small cross line at intersection.
            cv2.line(
                image,
                (
                    px - 7,
                    int(
                        round(
                            y50
                        )
                    ),
                ),
                (
                    px + 7,
                    int(
                        round(
                            y50
                        )
                    ),
                ),
                (
                    255,
                    0,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

            if (
                index
                < len(
                    intersection_values
                )
            ):

                value = (
                    intersection_values[
                        index
                    ]
                )

                label_y = (
                    int(
                        round(
                            y50
                        )
                    )
                    - 10
                    if index % 2 == 0
                    else int(
                        round(
                            y50
                        )
                    )
                    + 20
                )

                cv2.putText(
                    image,
                    (
                        f"{value:.3f} fL"
                    ),
                    (
                        max(
                            1,
                            min(
                                w - 100,
                                px - 35,
                            ),
                        ),
                        max(
                            14,
                            min(
                                h - 4,
                                label_y,
                            ),
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (
                        220,
                        0,
                        0,
                    ),
                    1,
                    cv2.LINE_AA,
                )

    return image


# ============================================================
# RESULT TABLE
# ============================================================

def make_result_table(
    result: dict,
) -> pd.DataFrame:

    intersections = result.get(
        "intersections_fl",
        [],
    )

    rows = [

        (
            "Status",
            result.get(
                "status"
            ),
        ),

        (
            "Measurement source",
            result.get(
                "measurement_source"
            ),
        ),

        (
            "ROI boundaries",
            (
                f"{result.get('reference_left_px')}"
                f"–"
                f"{result.get('reference_right_px')}"
                " px"
            ),
        ),

        (
            "X-axis calibration",
            result.get(
                "x_axis_method"
            ),
        ),

        (
            "X-axis pixel range",
            (
                f"{result.get('x_axis_start_px')}"
                f"–"
                f"{result.get('x_axis_end_px')}"
                " px"
            ),
        ),

        (
            "X-axis numerical range",
            (
                f"{result.get('x_min_fl'):g}"
                f"–"
                f"{result.get('x_max_fl'):g}"
                " fL"
            ),
        ),

        (
            "Y-axis range",
            (
                f"0–"
                f"{result.get('y_max_fl'):g}"
                " fL"
            ),
        ),

        (
            "Sensitivity",
            (
                f"{result.get('sensitivity')} "
                f"(threshold "
                f"{result.get('segmentation_threshold'):.2f})"
            ),
        ),

        (
            "Peak position",
            (
                "Not available"
                if result.get(
                    "peak_x_fl"
                ) is None
                else (
                    f"{result['peak_x_fl']:.3f} fL"
                )
            ),
        ),

        (
            "Peak Y value",
            (
                "Not available"
                if result.get(
                    "peak_y_fl"
                ) is None
                else (
                    f"{result['peak_y_fl']:.3f} fL"
                )
            ),
        ),

        (
            "Selected Y level",
            (
                "Not available"
                if result.get(
                    "y50_fl"
                ) is None
                else (
                    f"{result['y50_fl']:.3f} fL "
                    "(50% of detected peak)"
                )
            ),
        ),

        (
            "Left 50% intersection",
            (
                "Not available"
                if len(
                    intersections
                ) < 1
                else (
                    f"{intersections[0]:.3f} fL"
                )
            ),
        ),

        (
            "Right 50% intersection",
            (
                "Not available"
                if len(
                    intersections
                ) < 2
                else (
                    f"{intersections[-1]:.3f} fL"
                )
            ),
        ),

        (
            "Width at 50%",
            (
                "Not available"
                if result.get(
                    "width_50_fl"
                ) is None
                else (
                    f"{result['width_50_fl']:.3f} fL"
                )
            ),
        ),

        (
            "Number of intersections",
            len(
                intersections
            ),
        ),

        (
            "All intersections",
            (
                "None"
                if not intersections
                else (
                    ", ".join(
                        f"{v:.3f}"
                        for v in intersections
                    )
                    + " fL"
                )
            ),
        ),

        (
            "Confidence",
            (
                f"{100 * float(result.get('confidence', 0)):.1f}%"
            ),
        ),

        (
            "Warning",
            result.get(
                "warning"
            )
            or "",
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Measurement",
            "Result",
        ],
    )