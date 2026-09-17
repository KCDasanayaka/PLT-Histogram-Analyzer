from __future__ import annotations

from pathlib import Path
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
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 512

DEFAULT_THRESHOLD = 0.28


# ============================================================
# SENSITIVITY PRESETS
# ============================================================
#
# Higher sensitivity -> lower segmentation threshold.
#
# This lets the user select the behavior before analysis
# without directly manipulating the model threshold.
#
# Low:
#     More conservative
#     Rejects weaker/noisier responses
#
# Medium:
#     Balanced default
#
# High:
#     Retains weaker curve responses
#
# ============================================================

SENSITIVITY_PRESETS = {
    "Low": 0.36,
    "Medium": 0.28,
    "High": 0.22,
}


# ============================================================
# SMALL HELPERS
# ============================================================

def _vertical_run_stats(
    binary_col: np.ndarray,
):

    b = (
        np.asarray(
            binary_col,
            dtype=np.uint8,
        )
        > 0
    )

    idx = np.flatnonzero(b)

    if idx.size == 0:

        return 0, 0, 0.0

    starts = idx[
        np.r_[
            True,
            np.diff(idx) > 1,
        ]
    ]

    ends = idx[
        np.r_[
            np.diff(idx) > 1,
            True,
        ]
    ]

    lengths = (
        ends - starts + 1
    )

    return (
        int(len(lengths)),
        int(lengths.max()),
        float(
            lengths.sum()
            / len(binary_col)
        ),
    )


def _norm(
    v: np.ndarray,
) -> np.ndarray:

    v = np.asarray(
        v,
        dtype=np.float32,
    )

    m = (
        float(v.max())
        if v.size
        else 0.0
    )

    if m <= 1e-9:

        return np.zeros_like(v)

    return v / m


# ============================================================
# REFERENCE LINE DETECTION
# ============================================================

def detect_reference_lines(
    rgb: np.ndarray,
) -> dict:
    """
    Detect the two vertical dashed reference boundaries.

    Returns one dictionary:
        ref["left_x"]
        ref["right_x"]
    """

    h, w = rgb.shape[:2]

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    y0 = int(0.04 * h)
    y1 = int(0.92 * h)

    x0 = int(0.025 * w)
    x1 = int(0.975 * w)

    g = gray[
        y0:y1,
        x0:x1,
    ]

    q = np.percentile(
        g,
        35,
    )

    dark = (
        g
        <= min(
            205,
            q + 15,
        )
    ).astype(np.uint8)

    raw = np.zeros(
        g.shape[1],
        np.float32,
    )

    runs = np.zeros_like(raw)

    hough = np.zeros_like(raw)

    # --------------------------------------------------------
    # Vertical run statistics
    # --------------------------------------------------------

    for x in range(g.shape[1]):

        nr, mr, cov = (
            _vertical_run_stats(
                dark[:, x]
            )
        )

        raw[x] = cov

        runs[x] = (
            min(nr, 12) / 12.0
            + 0.25
            * min(
                mr
                / max(
                    1,
                    g.shape[0],
                ),
                0.35,
            )
        )

    # --------------------------------------------------------
    # Hough support
    # --------------------------------------------------------

    edges = cv2.Canny(
        g,
        40,
        140,
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(
            12,
            int(0.018 * h),
        ),
        minLineLength=max(
            10,
            int(0.03 * h),
        ),
        maxLineGap=max(
            5,
            int(0.015 * h),
        ),
    )

    if lines is not None:

        line_records = (
            np.asarray(lines)
            .reshape(-1, 4)
        )

        for (
            x_a,
            y_a,
            x_b,
            y_b,
        ) in line_records:

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
                    int(0.01 * w),
                )
                and dy
                >= max(
                    8,
                    int(0.025 * h),
                )
            ):

                xm = int(
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
                    <= xm
                    < g.shape[1]
                ):

                    hough[xm] += dy

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    score = (
        0.35 * _norm(raw)
        + 0.40 * _norm(runs)
        + 0.25 * _norm(hough)
    )

    score = cv2.GaussianBlur(
        score.reshape(1, -1),
        (0, 0),
        max(
            1,
            0.003 * w,
        ),
    ).ravel()

    edge = int(
        0.02 * w
    )

    score[:edge] = 0
    score[-edge:] = 0

    order = np.argsort(
        score
    )[::-1]

    candidates = []

    min_dist = max(
        8,
        int(0.025 * w),
    )

    for qx in order:

        xx = int(qx)

        if score[xx] <= 0:

            break

        if all(
            abs(xx - c)
            >= min_dist
            for c in candidates
        ):

            candidates.append(
                xx
            )

        if len(candidates) >= 40:

            break

    # --------------------------------------------------------
    # Select best pair
    # --------------------------------------------------------

    best = None
    best_pair = -1.0

    min_sep = max(
        30,
        int(
            0.30
            * g.shape[1]
        ),
    )

    for i, a in enumerate(
        candidates
    ):

        for b in candidates[
            i + 1:
        ]:

            left0, right0 = sorted(
                (a, b)
            )

            sep = (
                right0
                - left0
            )

            if sep < min_sep:

                continue

            pair = float(
                score[left0]
                + score[right0]
            )

            if (
                left0
                < 0.45
                * g.shape[1]
                and right0
                > 0.55
                * g.shape[1]
            ):

                pair += 0.25

            pair += (
                0.15
                * min(
                    left0
                    / max(
                        1,
                        0.18
                        * g.shape[1],
                    ),
                    1,
                )
            )

            pair += (
                0.15
                * min(
                    (
                        g.shape[1]
                        - right0
                    )
                    / max(
                        1,
                        0.18
                        * g.shape[1],
                    ),
                    1,
                )
            )

            if pair > best_pair:

                best_pair = pair

                best = (
                    left0,
                    right0,
                )

    if best is None:

        return {
            "ok": False,
            "left_x": None,
            "right_x": None,
            "roi_width": None,
            "confidence": 0.0,
            "method": "no_reliable_pair",
            "candidates": [
                int(c + x0)
                for c in candidates
            ],
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
                best_pair / 2.0,
                0,
                1,
            )
        ),
        "method": (
            "dashed-runs+hough"
        ),
        "candidates": [
            int(c + x0)
            for c in candidates
        ],
    }


# ============================================================
# PREDICT CURVE
# ============================================================

@torch.inference_mode()
def predict_curve(
    model,
    device,
    rgb: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:

    x = normalize_tensor(
        rgb,
        image_size,
    ).to(device)

    p = torch.sigmoid(
        model(x)
    )[0, 0].detach().cpu().numpy()

    return cv2.resize(
        p,
        (
            rgb.shape[1],
            rgb.shape[0],
        ),
        interpolation=cv2.INTER_LINEAR,
    )


# ============================================================
# CLEAN CURVE PROBABILITY
# ============================================================

def clean_curve_probability(
    prob: np.ndarray,
    left: int,
    right: int,
    threshold: float,
):

    mask = (
        prob
        >= float(threshold)
    ).astype(np.uint8)

    clean = np.zeros_like(
        mask
    )

    clean[
        :,
        left:right + 1,
    ] = mask[
        :,
        left:right + 1,
    ]

    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        np.ones(
            (3, 3),
            np.uint8,
        ),
    )

    return clean


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

    row_edge = (
        edges[
            ylo:yhi,
            l:r + 1,
        ]
        .sum(1)
        .astype(np.float32)
    )

    row_dark = (
        (
            gray[
                ylo:yhi,
                l:r + 1,
            ]
            < 180
        )
        .mean(1)
        .astype(np.float32)
    )

    score = (
        0.70 * _norm(row_edge)
        + 0.30 * _norm(row_dark)
    )

    axis_y = (
        ylo
        + int(
            np.argmax(score)
        )
        if len(score)
        else int(0.90 * h)
    )

    bottoms = []

    step = max(
        1,
        int(
            (r - l + 1)
            / 300
        ),
    )

    for x in range(
        l,
        r + 1,
        step,
    ):

        yy = np.where(
            mask[:, x] > 0
        )[0]

        yy = yy[
            yy < 0.97 * h
        ]

        if yy.size:

            bottoms.append(
                int(yy.max())
            )

    curve_y = (
        float(
            np.percentile(
                bottoms,
                97,
            )
        )
        if len(bottoms) >= 20
        else None
    )

    base = (
        axis_y
        if (
            curve_y is None
            or abs(
                curve_y
                - axis_y
            )
            > 0.07 * h
        )
        else
        0.55 * axis_y
        + 0.45 * curve_y
    )

    return int(
        np.clip(
            round(base),
            int(0.55 * h),
            int(0.99 * h),
        )
    )


# ============================================================
# ESTIMATE PLOT TOP
# ============================================================

def estimate_plot_top(
    rgb: np.ndarray,
    left: int,
    right: int,
    baseline_y: int,
) -> int:
    """
    Estimate the top of the plotting area from the
    vertical dashed reference boundaries.

    This is used only for converting pixel Y positions
    into the user-provided Y-axis scale.
    """

    h, w = rgb.shape[:2]

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    # Narrow strips around both reference lines.
    strip_half_width = max(
        1,
        int(0.008 * w),
    )

    left_a = max(
        0,
        int(left)
        - strip_half_width,
    )

    left_b = min(
        w - 1,
        int(left)
        + strip_half_width,
    )

    right_a = max(
        0,
        int(right)
        - strip_half_width,
    )

    right_b = min(
        w - 1,
        int(right)
        + strip_half_width,
    )

    left_strip = gray[
        :baseline_y + 1,
        left_a:left_b + 1,
    ]

    right_strip = gray[
        :baseline_y + 1,
        right_a:right_b + 1,
    ]

    dark_left = (
        left_strip < 210
    ).sum(axis=1)

    dark_right = (
        right_strip < 210
    ).sum(axis=1)

    combined = (
        dark_left
        + dark_right
    )

    # Ignore very top image margin.
    start_y = int(
        0.03 * h
    )

    candidates = np.where(
        combined[
            start_y:
            baseline_y + 1
        ]
        > 0
    )[0]

    if candidates.size:

        top = (
            start_y
            + int(candidates[0])
        )

    else:

        # Safe fallback.
        top = int(
            0.08 * h
        )

    # Keep top safely above baseline.
    top = min(
        top,
        baseline_y - 10,
    )

    top = max(
        0,
        top,
    )

    return int(top)


# ============================================================
# CURVE TRACKING
# ============================================================

def _walk_track(
    prob,
    xs,
    ys,
    conf,
    seed_k,
    seed_y,
    lower,
    direction,
):

    max_step = max(
        6,
        int(
            0.035
            * prob.shape[0]
        ),
    )

    prev = seed_y

    gaps = 0

    if direction > 0:

        rng = range(
            seed_k + 1,
            len(xs),
        )

    else:

        rng = range(
            seed_k - 1,
            -1,
            -1,
        )

    for j in rng:

        if gaps > 12:

            break

        x = xs[j]

        y0 = max(
            0,
            int(round(prev))
            - max_step,
        )

        y1 = min(
            lower,
            int(round(prev))
            + max_step,
        )

        vals = prob[
            y0:y1 + 1,
            x,
        ]

        if vals.size == 0:

            gaps += 1

            continue

        yy = np.arange(
            y0,
            y1 + 1,
        )

        score = (
            vals.astype(
                np.float32
            )
            - 0.018
            * np.abs(
                yy - prev
            )
        )

        k = int(
            np.argmax(score)
        )

        p = float(
            vals[k]
        )

        if p < 0.22:

            gaps += 1

            continue

        chosen = int(
            yy[k]
        )

        a = max(
            y0,
            chosen - 2,
        )

        b = min(
            y1,
            chosen + 2,
        )

        vv = prob[
            a:b + 1,
            x,
        ]

        ww = np.maximum(
            vv,
            0,
        )

        chosen_f = float(
            (
                np.arange(
                    a,
                    b + 1,
                )
                * ww
            ).sum()
            / (
                ww.sum()
                + 1e-6
            )
        )

        ys[j] = chosen_f

        conf[j] = p

        prev = chosen_f

        gaps = 0


def build_curve_track(
    prob: np.ndarray,
    left: int,
    right: int,
    baseline: int,
):

    h, _ = prob.shape

    left = int(left)
    right = int(right)

    xs = np.arange(
        left,
        right + 1,
        dtype=np.int32,
    )

    ys = np.full(
        len(xs),
        np.nan,
        np.float32,
    )

    conf = np.zeros(
        len(xs),
        np.float32,
    )

    lower = min(
        h - 2,
        int(baseline) - 1,
    )

    if lower < 5:

        return (
            xs,
            ys,
            conf,
        )

    crop = prob[
        :lower + 1,
        left:right + 1,
    ]

    py, px = np.unravel_index(
        int(
            np.argmax(crop)
        ),
        crop.shape,
    )

    peak_p = float(
        crop[py, px]
    )

    if peak_p < 0.45:

        return (
            xs,
            ys,
            conf,
        )

    ys[px] = float(py)

    conf[px] = peak_p

    _walk_track(
        prob,
        xs,
        ys,
        conf,
        int(px),
        float(py),
        lower,
        +1,
    )

    _walk_track(
        prob,
        xs,
        ys,
        conf,
        int(px),
        float(py),
        lower,
        -1,
    )

    return (
        xs,
        ys,
        conf,
    )


# ============================================================
# SMOOTHING
# ============================================================

def smooth_array(
    v: np.ndarray,
) -> np.ndarray:

    finite = np.isfinite(
        v
    )

    if finite.sum() < 5:

        return v.copy()

    x = np.arange(
        len(v)
    )

    a = np.interp(
        x,
        x[finite],
        v[finite],
    ).astype(np.float32)

    k = min(
        15,
        max(
            5,
            (
                len(v)
                // 50
            )
            * 2
            + 1,
        ),
    )

    return cv2.medianBlur(
        a,
        k,
    ).astype(np.float32)


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
        np.isfinite(ys)
        & np.isfinite(xs)
        & (xs >= left)
        & (xs <= right)
    )

    x = xs[
        valid
    ].astype(float)

    y = ys[
        valid
    ].astype(float)

    if len(x) < 3:

        return []

    d = (
        y
        - float(y50)
    )

    ints = []

    for i in range(
        len(x) - 1
    ):

        if (
            x[i + 1]
            - x[i]
            > 12
        ):

            continue

        # Very close to line
        if (
            abs(d[i])
            <= 0.45
        ):

            ints.append(
                float(x[i])
            )

        # Actual sign crossing
        if (
            d[i]
            * d[i + 1]
            < 0
        ):

            t = (
                abs(d[i])
                / (
                    abs(d[i])
                    + abs(
                        d[i + 1]
                    )
                    + 1e-12
                )
            )

            ints.append(
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
        len(d)
        and abs(d[-1])
        <= 0.45
    ):

        ints.append(
            float(x[-1])
        )

    ints.sort()

    out = []

    merge = max(
        2.0,
        0.004
        * max(
            1,
            right - left,
        ),
    )

    for value in ints:

        if (
            not out
            or value - out[-1]
            > merge
        ):

            out.append(
                value
            )

    return out


# ============================================================
# PIXEL -> X VALUE
# ============================================================

def px_to_fl(
    x: float,
    left: int,
    right: int,
    xmin: float,
    xmax: float,
) -> float:

    if right <= left:

        return float("nan")

    return float(
        xmin
        + (
            float(x)
            - left
        )
        / (
            right
            - left
        )
        * (
            xmax
            - xmin
        )
    )


# ============================================================
# PIXEL -> Y VALUE
# ============================================================

def py_to_y_fl(
    y: float,
    top: int,
    baseline: int,
    y_max: float,
) -> float:
    """
    Convert image Y coordinate into the user-provided
    numerical Y-axis range.

    top      -> y_max
    baseline -> 0

    Image coordinates increase downward, therefore:
        y_value = (baseline - y) / (baseline - top) * y_max
    """

    denominator = (
        float(baseline - top)
    )

    if denominator <= 0:

        return float("nan")

    value = (
        (
            float(baseline)
            - float(y)
        )
        / denominator
        * float(y_max)
    )

    return float(
        np.clip(
            value,
            0.0,
            float(y_max),
        )
    )


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

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if (
        not np.isfinite(xmin)
        or not np.isfinite(xmax)
        or xmax <= xmin
    ):

        raise ValueError(
            "X-axis maximum must be greater "
            "than X-axis minimum."
        )

    if (
        not np.isfinite(ymax)
        or ymax <= 0
    ):

        raise ValueError(
            "Y-axis maximum must be greater than 0."
        )

    # --------------------------------------------------------
    # Detect ROI
    # --------------------------------------------------------

    ref = detect_reference_lines(
        rgb
    )

    result = {

        "status": "error",

        "measurement_source": (
            "V8 U-Net curve segmentation "
            "+ geometric measurement"
        ),

        "axis_detection": (
            "Automatic dashed ROI; "
            "X/Y-axis values supplied by user"
        ),

        "x_min_fl": xmin,
        "x_max_fl": xmax,
        "y_min_fl": 0.0,
        "y_max_fl": ymax,

        "segmentation_threshold": threshold,

        "reference_left_px": ref.get(
            "left_x"
        ),

        "reference_right_px": ref.get(
            "right_x"
        ),

        "reference_confidence": float(
            ref.get(
                "confidence",
                0.0,
            )
        ),

        "reference_method": ref.get(
            "method"
        ),

        "plot_top_y_px": None,

        "baseline_y_px": None,

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
        "curve_mask": None,

        "centerline_x": None,
        "centerline_y": None,
        "centerline_confidence": None,
    }

    # --------------------------------------------------------
    # ROI validation
    # --------------------------------------------------------

    if not ref.get("ok"):

        result.update(
            {
                "status": "roi_not_found",
                "warning": (
                    "Two reliable vertical dashed "
                    "reference boundaries were not detected."
                ),
            }
        )

        return result

    try:

        left = int(
            ref["left_x"]
        )

        right = int(
            ref["right_x"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:

        result.update(
            {
                "status": "roi_not_found",
                "warning": (
                    "Invalid detected ROI coordinates: "
                    f"{exc}"
                ),
            }
        )

        return result

    if (
        right - left
        < max(
            40,
            int(
                0.15
                * rgb.shape[1]
            ),
        )
    ):

        result.update(
            {
                "status": "roi_invalid",
                "warning": (
                    "Detected ROI is too narrow."
                ),
            }
        )

        return result

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prob = predict_curve(
        model,
        device,
        rgb,
    )

    mask = clean_curve_probability(
        prob,
        left,
        right,
        threshold,
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    base = estimate_baseline(
        rgb,
        mask,
        left,
        right,
    )

    # --------------------------------------------------------
    # Plot top
    # --------------------------------------------------------

    plot_top = estimate_plot_top(
        rgb,
        left,
        right,
        base,
    )

    # --------------------------------------------------------
    # Curve trace
    # --------------------------------------------------------

    (
        xs,
        ys,
        cc,
    ) = build_curve_track(
        prob,
        left,
        right,
        base,
    )

    result.update(
        {
            "plot_top_y_px": plot_top,

            "baseline_y_px": base,

            "curve_probability": prob,

            "curve_mask": mask,

            "centerline_x": xs,

            "centerline_y": ys,

            "centerline_confidence": cc,
        }
    )

    # --------------------------------------------------------
    # Valid curve points
    # --------------------------------------------------------

    valid = (
        np.isfinite(ys)
        & (
            ys
            < base - 1
        )
        & (
            xs >= left
        )
        & (
            xs <= right
        )
    )

    if valid.sum() < 10:

        result.update(
            {
                "status": "curve_not_found",
                "warning": (
                    "A reliable curve trace was not found "
                    "inside the ROI."
                ),
            }
        )

        return result

    x = xs[
        valid
    ].astype(float)

    y = ys[
        valid
    ].astype(float)

    c = cc[
        valid
    ].astype(float)

    # --------------------------------------------------------
    # Detect peak
    # --------------------------------------------------------

    heights = (
        base - y
    )

    sm = smooth_array(
        heights
    )

    pi = int(
        np.nanargmax(sm)
    )

    peak_x = float(
        x[pi]
    )

    peak_y = float(
        y[pi]
    )

    peak_h = float(
        base - peak_y
    )

    # Numerical Y of peak
    peak_y_fl = py_to_y_fl(
        peak_y,
        plot_top,
        base,
        ymax,
    )

    result.update(
        {
            "peak_x_px": peak_x,
            "peak_y_px": peak_y,

            "peak_x_fl": px_to_fl(
                peak_x,
                left,
                right,
                xmin,
                xmax,
            ),

            "peak_y_fl": peak_y_fl,

            "peak_height_px": peak_h,
        }
    )

    if peak_h < 3:

        result.update(
            {
                "status": "peak_not_found",
                "warning": (
                    "Detected curve height is too small "
                    "for a reliable 50% measurement."
                ),
            }
        )

        return result

    # --------------------------------------------------------
    # Calculate 50% of actual detected peak height
    # --------------------------------------------------------

    y50 = float(
        base
        - 0.5 * peak_h
    )

    y50_fl = py_to_y_fl(
        y50,
        plot_top,
        base,
        ymax,
    )

    # --------------------------------------------------------
    # Genuine intersections only
    # --------------------------------------------------------

    ints_px = genuine_intersections(
        xs,
        ys,
        y50,
        left,
        right,
    )

    ints_fl = []

    for px in ints_px:

        f = px_to_fl(
            px,
            left,
            right,
            xmin,
            xmax,
        )

        if np.isfinite(f):

            ints_fl.append(
                float(f)
            )

    # --------------------------------------------------------
    # Final hard geometry validation
    # --------------------------------------------------------
    #
    # Prevent isolated model pixels from being treated
    # as genuine curve intersections.
    #
    # --------------------------------------------------------

    checked = []

    for f in ints_fl:

        px = (
            left
            + (
                f - xmin
            )
            / (
                xmax - xmin
            )
            * (
                right - left
            )
        )

        k = int(
            np.argmin(
                np.abs(
                    xs - px
                )
            )
        )

        if (
            np.isfinite(
                ys[k]
            )
            and abs(
                float(
                    ys[k]
                )
                - y50
            )
            <= 3
        ):

            checked.append(
                f
            )

    ints_fl = sorted(
        set(
            round(
                value,
                6,
            )
            for value in checked
        )
    )

    # --------------------------------------------------------
    # Determine status
    # --------------------------------------------------------

    if len(ints_fl) == 0:

        status = "no_intersection"

    elif len(ints_fl) == 1:

        status = "single_intersection"

    else:

        status = "measure_width"

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    curve_conf = (
        float(
            np.nanmedian(c)
        )
        if np.isfinite(c).any()
        else 0.0
    )

    confidence = float(
        np.clip(
            0.45
            * ref[
                "confidence"
            ]
            + 0.55
            * curve_conf,
            0,
            1,
        )
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result.update(
        {
            "status": status,

            "y50_px": y50,

            "y50_fl": y50_fl,

            "intersections_px": ints_px,

            "intersections_fl": ints_fl,

            "intersection_count": len(
                ints_fl
            ),

            "width_50_fl": (
                max(ints_fl)
                - min(ints_fl)
                if len(ints_fl) >= 2
                else None
            ),

            "confidence": confidence,
        }
    )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    if len(ints_fl) == 0:

        result["warning"] = (
            "The traced curve does not genuinely cross "
            "the 50% level inside the ROI."
        )

    elif len(ints_fl) == 1:

        result["warning"] = (
            "Only one genuine 50% intersection was detected; "
            "width is not calculated."
        )

    return result


# ============================================================
# ANNOTATION
# ============================================================

def annotate_result(
    result: dict,
) -> np.ndarray:

    im = result[
        "image_rgb"
    ].copy()

    h, _ = im.shape[:2]

    left = result.get(
        "reference_left_px"
    )

    right = result.get(
        "reference_right_px"
    )

    plot_top = result.get(
        "plot_top_y_px"
    )

    baseline = result.get(
        "baseline_y_px"
    )

    # --------------------------------------------------------
    # ROI boundaries
    # --------------------------------------------------------

    if left is not None:

        cv2.line(
            im,
            (
                int(left),
                0,
            ),
            (
                int(left),
                h - 1,
            ),
            (
                0,
                255,
                0,
            ),
            2,
        )

    if right is not None:

        cv2.line(
            im,
            (
                int(right),
                0,
            ),
            (
                int(right),
                h - 1,
            ),
            (
                0,
                255,
                0,
            ),
            2,
        )

    # --------------------------------------------------------
    # Plot top guide
    # --------------------------------------------------------

    if plot_top is not None:

        cv2.line(
            im,
            (
                int(left)
                if left is not None
                else 0,
                int(plot_top),
            ),
            (
                int(right)
                if right is not None
                else im.shape[1] - 1,
                int(plot_top),
            ),
            (
                150,
                150,
                150,
            ),
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Baseline guide
    # --------------------------------------------------------

    if baseline is not None:

        cv2.line(
            im,
            (
                int(left)
                if left is not None
                else 0,
                int(baseline),
            ),
            (
                int(right)
                if right is not None
                else im.shape[1] - 1,
                int(baseline),
            ),
            (
                0,
                180,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # 50% line
    # --------------------------------------------------------

    if (
        result.get("y50_px")
        is not None
        and left is not None
        and right is not None
    ):

        y50 = int(
            round(
                result[
                    "y50_px"
                ]
            )
        )

        cv2.line(
            im,
            (
                int(left),
                y50,
            ),
            (
                int(right),
                y50,
            ),
            (
                255,
                165,
                0,
            ),
            2,
        )

        cv2.putText(
            im,
            (
                "50% "
                f"({result.get('y50_fl', 0):.3f} fL)"
            ),
            (
                max(
                    2,
                    int(left) + 3,
                ),
                max(
                    15,
                    y50 - 6,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                255,
                165,
                0,
            ),
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Peak point
    # --------------------------------------------------------

    if (
        result.get("peak_x_px")
        is not None
        and result.get("peak_y_px")
        is not None
    ):

        px = int(
            round(
                result[
                    "peak_x_px"
                ]
            )
        )

        py = int(
            round(
                result[
                    "peak_y_px"
                ]
            )
        )

        cv2.circle(
            im,
            (
                px,
                py,
            ),
            7,
            (
                255,
                0,
                255,
            ),
            -1,
        )

        peak_y_fl = result.get(
            "peak_y_fl"
        )

        if (
            peak_y_fl
            is not None
        ):

            cv2.putText(
                im,
                f"Peak Y: {peak_y_fl:.3f} fL",
                (
                    max(
                        2,
                        px + 8,
                    ),
                    max(
                        15,
                        py - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (
                    255,
                    0,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Intersections
    # --------------------------------------------------------

    if result.get(
        "y50_px"
    ) is not None:

        y50 = int(
            round(
                result[
                    "y50_px"
                ]
            )
        )

        for index, xx in enumerate(
            result.get(
                "intersections_px",
                [],
            )
        ):

            x = int(
                round(xx)
            )

            cv2.circle(
                im,
                (
                    x,
                    y50,
                ),
                7,
                (
                    255,
                    0,
                    0,
                ),
                -1,
            )

            ints = result.get(
                "intersections_fl",
                [],
            )

            if index < len(ints):

                label = (
                    f"{ints[index]:.3f} fL"
                )

                text_y = (
                    y50 - 10
                    if index % 2 == 0
                    else y50 + 18
                )

                cv2.putText(
                    im,
                    label,
                    (
                        max(
                            2,
                            min(
                                im.shape[1]
                                - 85,
                                x - 30,
                            ),
                        ),
                        max(
                            14,
                            min(
                                im.shape[0]
                                - 4,
                                text_y,
                            ),
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (
                        255,
                        0,
                        0,
                    ),
                    1,
                    cv2.LINE_AA,
                )

    return im


# ============================================================
# RESULT TABLE
# ============================================================

def make_result_table(
    result: dict,
) -> pd.DataFrame:

    ints = result.get(
        "intersections_fl",
        [],
    )

    xmin = result.get(
        "x_min_fl"
    )

    xmax = result.get(
        "x_max_fl"
    )

    ymax = result.get(
        "y_max_fl"
    )

    peak_x_fl = result.get(
        "peak_x_fl"
    )

    peak_y_fl = result.get(
        "peak_y_fl"
    )

    y50_fl = result.get(
        "y50_fl"
    )

    def fmt(
        value,
        suffix="",
    ):

        if value is None:

            return "Not available"

        try:

            return f"{float(value):.3f}{suffix}"

        except (
            TypeError,
            ValueError,
        ):

            return str(value)

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
            "Axis detection",
            result.get(
                "axis_detection"
            ),
        ),

        (
            "X-axis range",
            (
                f"{xmin:g}–"
                f"{xmax:g} fL"
            ),
        ),

        (
            "Y-axis range",
            (
                f"0–"
                f"{ymax:g} fL"
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
            "Detection sensitivity",
            (
                f"Threshold "
                f"{result.get('segmentation_threshold', DEFAULT_THRESHOLD):.2f}"
            ),
        ),

        (
            "Peak position",
            fmt(
                peak_x_fl,
                " fL",
            ),
        ),

        (
            "Peak Y value",
            fmt(
                peak_y_fl,
                " fL",
            ),
        ),

        (
            "Selected Y level",
            (
                "50% of detected peak"
                if y50_fl is None
                else (
                    f"{y50_fl:.3f} fL "
                    "(50% of detected peak)"
                )
            ),
        ),

        (
            "Left 50% intersection",
            (
                "Not available"
                if len(ints) < 1
                else f"{ints[0]:.3f} fL"
            ),
        ),

        (
            "Right 50% intersection",
            (
                "Not available"
                if len(ints) < 2
                else f"{ints[-1]:.3f} fL"
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
            len(ints),
        ),

        (
            "All intersections",
            (
                "None"
                if not ints
                else ", ".join(
                    f"{value:.3f}"
                    for value in ints
                )
                + " fL"
            ),
        ),

        (
            "Confidence",
            (
                f"{100 * result.get('confidence', 0):.1f}%"
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