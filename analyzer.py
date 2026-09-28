from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from preprocessing import normalize_tensor, read_rgb


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 512
DEFAULT_THRESHOLD = 0.28


# ============================================================
# SENSITIVITY
# ============================================================

SENSITIVITY_PRESETS = {
    "Low": {
        "threshold": 0.36,
        "track_threshold": 0.30,
        "seed_threshold": 0.42,
        "visual_weight": 0.08,
        "max_gap": 8,
        "max_step_fraction": 0.030,
    },

    "Medium": {
        "threshold": 0.28,
        "track_threshold": 0.21,
        "seed_threshold": 0.34,
        "visual_weight": 0.12,
        "max_gap": 12,
        "max_step_fraction": 0.035,
    },

    "High": {
        "threshold": 0.20,
        "track_threshold": 0.14,
        "seed_threshold": 0.25,
        "visual_weight": 0.18,
        "max_gap": 16,
        "max_step_fraction": 0.045,
    },

    "Very High": {
        "threshold": 0.14,
        "track_threshold": 0.09,
        "seed_threshold": 0.18,
        "visual_weight": 0.24,
        "max_gap": 20,
        "max_step_fraction": 0.055,
    },
}


# ============================================================
# HELPERS
# ============================================================

def _vertical_run_stats(binary_col: np.ndarray):

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

    lengths = ends - starts + 1

    return (
        int(len(lengths)),
        int(lengths.max()),
        float(
            lengths.sum()
            / len(binary_col)
        ),
    )


def _norm(v: np.ndarray) -> np.ndarray:

    v = np.asarray(
        v,
        dtype=np.float32,
    )

    if not v.size:
        return np.zeros_like(v)

    m = float(v.max())

    if m <= 1e-9:
        return np.zeros_like(v)

    return v / m


# ============================================================
# REFERENCE ROI DETECTION
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
    ).astype(
        np.uint8
    )

    raw = np.zeros(
        g.shape[1],
        np.float32,
    )

    runs = np.zeros_like(
        raw
    )

    hough = np.zeros_like(
        raw
    )

    # --------------------------------------------------------
    # Vertical run score
    # --------------------------------------------------------

    for x in range(
        g.shape[1]
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
                    g.shape[0],
                ),
                0.35,
            )
        )

    # --------------------------------------------------------
    # Hough vertical support
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

        for (
            x_a,
            y_a,
            x_b,
            y_b,
        ) in np.asarray(
            lines
        ).reshape(
            -1,
            4,
        ):

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
    # Score
    # --------------------------------------------------------

    score = (
        0.35 * _norm(raw)
        + 0.40 * _norm(runs)
        + 0.25 * _norm(hough)
    )

    score = cv2.GaussianBlur(
        score.reshape(
            1,
            -1,
        ),
        (0, 0),
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

    min_dist = max(
        8,
        int(
            0.025 * w
        ),
    )

    for qx in order:

        xx = int(qx)

        if score[xx] <= 0:
            break

        if all(
            abs(
                xx - c
            )
            >= min_dist
            for c in candidates
        ):

            candidates.append(
                xx
            )

        if len(
            candidates
        ) >= 40:

            break

    # --------------------------------------------------------
    # Best pair
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
                (
                    a,
                    b,
                )
            )

            if (
                right0
                - left0
                < min_sep
            ):

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
                int(
                    c + x0
                )
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
            int(
                c + x0
            )
            for c in candidates
        ],
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
# VISUAL CURVE ENHANCEMENT
# ============================================================

def build_visual_curve_score(
    rgb: np.ndarray,
    left: int,
    right: int,
    baseline: int,
) -> np.ndarray:
    """
    Additional image-based curve cue.

    This does NOT replace the U-Net.

    It helps when the curve is visibly present but the
    segmentation probability is weak.
    """

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    # CLAHE improves faint grayscale strokes.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(
        gray
    )

    # Dark-pixel cue.
    dark_score = np.clip(
        (
            185.0
            - enhanced.astype(
                np.float32
            )
        )
        / 125.0,
        0.0,
        1.0,
    )

    # Black-hat highlights dark thin curves
    # against brighter surroundings.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            11,
            11,
        ),
    )

    blackhat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        kernel,
    ).astype(
        np.float32
    )

    blackhat = _norm(
        blackhat
    )

    visual = (
        0.55 * dark_score
        + 0.45 * blackhat
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

    # Only graph ROI.
    output = np.zeros_like(
        visual,
        dtype=np.float32,
    )

    upper = max(
        0,
        int(
            0.02
            * rgb.shape[0]
        ),
    )

    lower = max(
        upper + 1,
        int(baseline) - 2,
    )

    output[
        upper:lower,
        int(left):int(right) + 1,
    ] = visual[
        upper:lower,
        int(left):int(right) + 1,
    ]

    return np.clip(
        output,
        0.0,
        1.0,
    )


# ============================================================
# CURVE MASK
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
    ).astype(
        np.uint8
    )

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
            (
                3,
                3,
            ),
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
        .mean(1)
        .astype(
            np.float32
        )
    )

    score = (
        0.70 * _norm(
            row_edge
        )
        + 0.30 * _norm(
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
                int(
                    yy.max()
                )
            )

    curve_y = (
        float(
            np.percentile(
                bottoms,
                97,
            )
        )
        if len(bottoms)
        >= 20
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
        else (
            0.55 * axis_y
            + 0.45 * curve_y
        )
    )

    return int(
        np.clip(
            round(base),
            int(
                0.55 * h
            ),
            int(
                0.99 * h
            ),
        )
    )


# ============================================================
# ACTUAL HORIZONTAL X-AXIS DETECTION
# ============================================================

def detect_x_axis_span(
    rgb: np.ndarray,
    baseline_y: int,
) -> dict:
    """
    Detect the actual horizontal X-axis line.

    We intentionally do NOT detect 10/20/30/40 labels or ticks.

    The user's entered X-axis range is mapped onto this actual
    horizontal axis.
    """

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    h, w = gray.shape

    center = int(
        baseline_y
    )

    row_min = max(
        0,
        center - 6,
    )

    row_max = min(
        h - 1,
        center + 6,
    )

    best = None

    for y in range(
        row_min,
        row_max + 1,
    ):

        row = gray[
            y:y + 1,
            :,
        ]

        threshold = np.percentile(
            gray[
                max(
                    0,
                    center - 30,
                ):
                min(
                    h,
                    center + 30,
                ),
                :,
            ],
            35,
        )

        dark = (
            row[0]
            <= min(
                180,
                threshold + 35,
            )
        ).astype(
            np.uint8
        )

        # Fill small gaps caused by ticks/compression.
        closed = cv2.morphologyEx(
            dark.reshape(
                1,
                -1,
            ),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (
                    17,
                    1,
                ),
            ),
        )[0]

        runs = np.flatnonzero(
            closed
        )

        if runs.size == 0:
            continue

        starts = runs[
            np.r_[
                True,
                np.diff(runs) > 1,
            ]
        ]

        ends = runs[
            np.r_[
                np.diff(runs) > 1,
                True,
            ]
        ]

        lengths = (
            ends
            - starts
            + 1
        )

        idx = int(
            np.argmax(
                lengths
            )
        )

        start = int(
            starts[idx]
        )

        end = int(
            ends[idx]
        )

        length = (
            end
            - start
            + 1
        )

        if best is None or length > best["length"]:

            best = {
                "y": int(y),
                "start": start,
                "end": end,
                "length": length,
            }

    # --------------------------------------------------------
    # Hough fallback
    # --------------------------------------------------------

    if best is None:

        edges = cv2.Canny(
            gray,
            40,
            140,
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(
                20,
                int(
                    0.03 * w
                ),
            ),
            minLineLength=max(
                30,
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

        horizontal = []

        if lines is not None:

            for (
                x1,
                y1,
                x2,
                y2,
            ) in np.asarray(
                lines
            ).reshape(
                -1,
                4,
            ):

                dx = abs(
                    int(x2)
                    - int(x1)
                )

                dy = abs(
                    int(y2)
                    - int(y1)
                )

                if (
                    dx > 0
                    and dy
                    <= max(
                        3,
                        int(
                            0.01 * w
                        ),
                    )
                    and abs(
                        (
                            y1
                            + y2
                        )
                        / 2
                        - center
                    )
                    <= 12
                ):

                    horizontal.append(
                        (
                            dx,
                            min(
                                x1,
                                x2,
                            ),
                            max(
                                x1,
                                x2,
                            ),
                        )
                    )

        if horizontal:

            horizontal.sort(
                reverse=True
            )

            dx, start, end = (
                horizontal[0]
            )

            best = {
                "y": center,
                "start": int(start),
                "end": int(end),
                "length": int(
                    end - start + 1
                ),
            }

    # --------------------------------------------------------
    # No axis
    # --------------------------------------------------------

    if best is None:

        return {
            "ok": False,
            "start_x": None,
            "end_x": None,
            "y": int(
                baseline_y
            ),
            "confidence": 0.0,
            "method": "not_found",
        }

    # Basic quality requirement.
    min_length = max(
        100,
        int(
            0.35 * w
        ),
    )

    confidence = float(
        np.clip(
            best["length"]
            / max(
                1,
                0.75 * w,
            ),
            0,
            1,
        )
    )

    if best["length"] < min_length:

        return {
            "ok": False,
            "start_x": None,
            "end_x": None,
            "y": int(
                best["y"]
            ),
            "confidence": confidence,
            "method": "too_short",
        }

    return {
        "ok": True,

        "start_x": int(
            best["start"]
        ),

        "end_x": int(
            best["end"]
        ),

        "y": int(
            best["y"]
        ),

        "length": int(
            best["length"]
        ),

        "confidence": confidence,

        "method": (
            "horizontal-axis-line"
        ),
    }


# ============================================================
# X PIXEL -> fL
# ============================================================

def px_to_fl(
    x: float,
    axis_start: int,
    axis_end: int,
    xmin: float,
    xmax: float,
) -> float:

    denominator = float(
        axis_end
        - axis_start
    )

    if denominator <= 0:
        return float(
            "nan"
        )

    normalized = (
        float(x)
        - float(axis_start)
    ) / denominator

    return float(
        xmin
        + normalized
        * (
            xmax
            - xmin
        )
    )


# ============================================================
# Y PIXEL -> fL
# ============================================================

def py_to_y_fl(
    y: float,
    top: int,
    baseline: int,
    y_max: float,
) -> float:

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
# PLOT TOP
# ============================================================

def estimate_plot_top(
    rgb: np.ndarray,
    left: int,
    right: int,
    baseline_y: int,
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

    candidates = []

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
            :baseline_y + 1,
            a:b,
        ]

        if strip.size == 0:
            continue

        dark = (
            strip < 210
        ).sum(
            axis=1
        )

        ys = np.where(
            dark > 0
        )[0]

        if ys.size:

            candidates.append(
                int(
                    ys[0]
                )
            )

    if candidates:

        top = min(
            candidates
        )

    else:

        top = int(
            0.08 * h
        )

    top = min(
        top,
        baseline_y - 10,
    )

    return int(
        max(
            0,
            top,
        )
    )


# ============================================================
# CURVE TRACKING
# ============================================================

def _walk_track(
    prob,
    visual,
    xs,
    ys,
    conf,
    seed_k,
    seed_y,
    lower,
    direction,
    settings,
):

    h = prob.shape[0]

    max_step = max(
        5,
        int(
            settings[
                "max_step_fraction"
            ]
            * h
        ),
    )

    max_gap = int(
        settings[
            "max_gap"
        ]
    )

    visual_weight = float(
        settings[
            "visual_weight"
        ]
    )

    prev = float(
        seed_y
    )

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

        if gaps > max_gap:

            break

        x = int(
            xs[j]
        )

        y0 = max(
            0,
            int(
                round(
                    prev
                )
            )
            - max_step,
        )

        y1 = min(
            lower,
            int(
                round(
                    prev
                )
            )
            + max_step,
        )

        if y1 < y0:

            gaps += 1
            continue

        pvals = prob[
            y0:y1 + 1,
            x,
        ]

        vvals = visual[
            y0:y1 + 1,
            x,
        ]

        yy = np.arange(
            y0,
            y1 + 1,
            dtype=np.float32,
        )

        hybrid = (
            (
                1.0
                - visual_weight
            )
            * pvals
            + visual_weight
            * vvals
        )

        # Continuity penalty.
        score = (
            hybrid
            - 0.018
            * np.abs(
                yy
                - prev
            )
        )

        k = int(
            np.argmax(
                score
            )
        )

        chosen_score = float(
            score[k]
        )

        chosen_probability = float(
            pvals[k]
        )

        # Dynamic sensitivity acceptance.
        accept_threshold = max(
            0.05,
            min(
                0.40,
                float(
                    settings[
                        "track_threshold"
                    ]
                ),
            ),
        )

        if (
            chosen_probability
            < accept_threshold
            and chosen_score
            < (
                accept_threshold
                * 0.90
            )
        ):

            gaps += 1
            continue

        chosen = int(
            yy[k]
        )

        # Weighted center around selected point.
        a = max(
            y0,
            chosen - 2,
        )

        b = min(
            y1,
            chosen + 2,
        )

        vv = (
            (
                1.0
                - visual_weight
            )
            * prob[
                a:b + 1,
                x,
            ]
            + visual_weight
            * visual[
                a:b + 1,
                x,
            ]
        )

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

        conf[j] = max(
            chosen_probability,
            chosen_score,
        )

        prev = chosen_f

        gaps = 0


def build_curve_track(
    prob: np.ndarray,
    visual: np.ndarray,
    left: int,
    right: int,
    baseline: int,
    settings: dict,
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

        return xs, ys, conf

    visual_weight = float(
        settings[
            "visual_weight"
        ]
    )

    hybrid = (
        (
            1.0
            - visual_weight
        )
        * prob
        + visual_weight
        * visual
    )

    crop = hybrid[
        :lower + 1,
        left:right + 1,
    ]

    if crop.size == 0:

        return xs, ys, conf

    py, px = np.unravel_index(
        int(
            np.argmax(
                crop
            )
        ),
        crop.shape,
    )

    seed_score = float(
        crop[py, px]
    )

    if (
        seed_score
        < float(
            settings[
                "seed_threshold"
            ]
        )
    ):

        return xs, ys, conf

    ys[px] = float(
        py
    )

    conf[px] = seed_score

    _walk_track(
        prob,
        visual,
        xs,
        ys,
        conf,
        int(px),
        float(py),
        lower,
        +1,
        settings,
    )

    _walk_track(
        prob,
        visual,
        xs,
        ys,
        conf,
        int(px),
        float(py),
        lower,
        -1,
        settings,
    )

    return (
        xs,
        ys,
        conf,
    )


# ============================================================
# SAFE SMOOTHING
# ============================================================

def _median_1d(
    values: np.ndarray,
    kernel_size: int,
) -> np.ndarray:

    if len(values) < 3:

        return values.copy()

    k = int(
        kernel_size
    )

    if k < 3:
        return values.copy()

    if k % 2 == 0:
        k += 1

    radius = k // 2

    padded = np.pad(
        values,
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
            k,
        )
    )

    return np.median(
        windows,
        axis=-1,
    ).astype(
        np.float32
    )


def smooth_array(
    v: np.ndarray,
) -> np.ndarray:
    """
    Safe smoothing that preserves NaN gaps.

    The old implementation used:
        cv2.medianBlur(float32_array, ...)

    which caused OpenCV 4.14 to fail.

    This implementation also avoids interpolating across
    large missing curve sections.
    """

    result = np.full_like(
        v,
        np.nan,
        dtype=np.float32,
    )

    finite = np.isfinite(
        v
    )

    if finite.sum() < 5:

        result[
            finite
        ] = v[
            finite
        ]

        return result

    indices = np.flatnonzero(
        finite
    )

    # Split into contiguous finite segments.
    split_points = (
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
        split_points,
    )

    for seg in segments:

        if len(seg) < 3:

            result[
                seg
            ] = v[
                seg
            ]

            continue

        segment_values = (
            v[
                seg
            ].astype(
                np.float32
            )
        )

        kernel = min(
            15,
            max(
                5,
                (
                    len(seg)
                    // 50
                )
                * 2
                + 1,
            ),
        )

        result[
            seg
        ] = _median_1d(
            segment_values,
            kernel,
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
            ys
        )
        & np.isfinite(
            xs
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

        if (
            abs(
                d[i]
            )
            <= 0.45
        ):

            ints.append(
                float(
                    x[i]
                )
            )

        if (
            d[i]
            * d[i + 1]
            < 0
        ):

            t = (
                abs(
                    d[i]
                )
                / (
                    abs(
                        d[i]
                    )
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
        and abs(
            d[-1]
        )
        <= 0.45
    ):

        ints.append(
            float(
                x[-1]
            )
        )

    ints.sort()

    output = []

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
            not output
            or value
            - output[-1]
            > merge
        ):

            output.append(
                value
            )

    return output


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
    sensitivity_name: str = "Medium",
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
    # Base sensitivity settings
    # --------------------------------------------------------

    sensitivity_name = (
        sensitivity_name
        if sensitivity_name
        in SENSITIVITY_PRESETS
        else "Medium"
    )

    settings = dict(
        SENSITIVITY_PRESETS[
            sensitivity_name
        ]
    )

    # User fine-tuning modifies the model threshold.
    # Keep other controls coupled to it.
    delta = (
        threshold
        - settings[
            "threshold"
        ]
    )

    settings[
        "threshold"
    ] = threshold

    settings[
        "track_threshold"
    ] = float(
        np.clip(
            settings[
                "track_threshold"
            ]
            + delta
            * 0.80,
            0.05,
            0.40,
        )
    )

    settings[
        "seed_threshold"
    ] = float(
        np.clip(
            settings[
                "seed_threshold"
            ]
            + delta
            * 0.50,
            0.10,
            0.50,
        )
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if (
        not np.isfinite(
            xmin
        )
        or not np.isfinite(
            xmax
        )
        or xmax <= xmin
    ):

        raise ValueError(
            "X-axis maximum must be greater "
            "than X-axis minimum."
        )

    if (
        not np.isfinite(
            ymax
        )
        or ymax <= 0
    ):

        raise ValueError(
            "Y-axis maximum must be greater than 0."
        )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    ref = detect_reference_lines(
        rgb
    )

    result = {

        "status": "error",

        "measurement_source": (
            "V8 U-Net + enhanced "
            "curve tracking + actual "
            "horizontal-axis calibration"
        ),

        "axis_detection": (
            "Automatic dashed ROI + "
            "automatic horizontal-axis geometry; "
            "X/Y values supplied manually"
        ),

        "x_min_fl": xmin,
        "x_max_fl": xmax,

        "y_min_fl": 0.0,
        "y_max_fl": ymax,

        "sensitivity": (
            sensitivity_name
        ),

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

        "x_axis_start_px": None,
        "x_axis_end_px": None,
        "x_axis_y_px": None,
        "x_axis_confidence": 0.0,
        "x_axis_method": None,

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

    # --------------------------------------------------------
    # ROI validation
    # --------------------------------------------------------

    if not ref.get(
        "ok"
    ):

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

    left = int(
        ref["left_x"]
    )

    right = int(
        ref["right_x"]
    )

    if (
        right
        - left
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
    # U-Net
    # --------------------------------------------------------

    prob = predict_curve(
        model,
        device,
        rgb,
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    mask = clean_curve_probability(
        prob,
        left,
        right,
        threshold,
    )

    baseline = estimate_baseline(
        rgb,
        mask,
        left,
        right,
    )

    result[
        "baseline_y_px"
    ] = baseline

    # --------------------------------------------------------
    # Actual horizontal X-axis
    # --------------------------------------------------------

    axis = detect_x_axis_span(
        rgb,
        baseline,
    )

    if axis.get(
        "ok"
    ):

        axis_start = int(
            axis[
                "start_x"
            ]
        )

        axis_end = int(
            axis[
                "end_x"
            ]
        )

        result.update(
            {
                "x_axis_start_px": axis_start,
                "x_axis_end_px": axis_end,
                "x_axis_y_px": axis.get(
                    "y"
                ),
                "x_axis_confidence": float(
                    axis.get(
                        "confidence",
                        0.0,
                    )
                ),
                "x_axis_method": axis.get(
                    "method"
                ),
            }
        )

    else:

        # Important fallback:
        # only used if actual axis cannot be found.
        axis_start = left
        axis_end = right

        result.update(
            {
                "x_axis_start_px": axis_start,
                "x_axis_end_px": axis_end,
                "x_axis_y_px": baseline,
                "x_axis_confidence": 0.25,
                "x_axis_method": (
                    "ROI fallback"
                ),
            }
        )

    # --------------------------------------------------------
    # Plot top
    # --------------------------------------------------------

    plot_top = estimate_plot_top(
        rgb,
        left,
        right,
        baseline,
    )

    result[
        "plot_top_y_px"
    ] = plot_top

    # --------------------------------------------------------
    # Enhanced visual cue
    # --------------------------------------------------------

    visual = build_visual_curve_score(
        rgb,
        left,
        right,
        baseline,
    )

    # --------------------------------------------------------
    # Hybrid curve tracking
    # --------------------------------------------------------

    xs, ys, cc = build_curve_track(
        prob,
        visual,
        left,
        right,
        baseline,
        settings,
    )

    result.update(
        {
            "curve_probability": prob,
            "visual_curve_score": visual,
            "curve_mask": mask,
            "centerline_x": xs,
            "centerline_y": ys,
            "centerline_confidence": cc,
        }
    )

    # --------------------------------------------------------
    # Valid trace
    # --------------------------------------------------------

    valid = (
        np.isfinite(
            ys
        )
        & (
            ys
            < baseline - 1
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
                    "inside the ROI. Increase sensitivity "
                    "and analyze again."
                ),
            }
        )

        return result

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

    c = cc[
        valid
    ].astype(
        float
    )

    # --------------------------------------------------------
    # Peak
    # --------------------------------------------------------

    heights = (
        baseline
        - y
    )

    smoothed = smooth_array(
        heights
    )

    valid_smoothed = np.isfinite(
        smoothed
    )

    if not valid_smoothed.any():

        result.update(
            {
                "status": "peak_not_found",
                "warning": (
                    "A reliable peak could not be "
                    "calculated from the curve trace."
                ),
            }
        )

        return result

    # x/y arrays and smoothing arrays have the same indexing
    pi = int(
        np.nanargmax(
            smoothed
        )
    )

    peak_x = float(
        x[pi]
    )

    peak_y = float(
        y[pi]
    )

    peak_height = float(
        baseline
        - peak_y
    )

    peak_x_fl = px_to_fl(
        peak_x,
        axis_start,
        axis_end,
        xmin,
        xmax,
    )

    peak_y_fl = py_to_y_fl(
        peak_y,
        plot_top,
        baseline,
        ymax,
    )

    result.update(
        {
            "peak_x_px": peak_x,
            "peak_y_px": peak_y,
            "peak_x_fl": peak_x_fl,
            "peak_y_fl": peak_y_fl,
            "peak_height_px": peak_height,
        }
    )

    if peak_height < 3:

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
    # 50% level
    # --------------------------------------------------------

    y50 = float(
        baseline
        - 0.5
        * peak_height
    )

    y50_fl = py_to_y_fl(
        y50,
        plot_top,
        baseline,
        ymax,
    )

    # --------------------------------------------------------
    # Genuine intersections
    # --------------------------------------------------------

    ints_px = genuine_intersections(
        xs,
        ys,
        y50,
        left,
        right,
    )

    # --------------------------------------------------------
    # Convert with ACTUAL HORIZONTAL AXIS
    # --------------------------------------------------------

    ints_fl = [

        px_to_fl(
            px,
            axis_start,
            axis_end,
            xmin,
            xmax,
        )

        for px in ints_px
    ]

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    pairs = []

    for px, fl in zip(
        ints_px,
        ints_fl,
    ):

        if not np.isfinite(
            fl
        ):

            continue

        if (
            fl < xmin
            or fl > xmax
        ):

            continue

        k = int(
            np.argmin(
                np.abs(
                    xs
                    - px
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

            pairs.append(
                (
                    float(fl),
                    float(px),
                )
            )

    pairs.sort(
        key=lambda z: z[0]
    )

    ints_fl = [
        round(
            p[0],
            6,
        )
        for p in pairs
    ]

    ints_px = [
        p[1]
        for p in pairs
    ]

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if len(ints_fl) == 0:

        status = (
            "no_intersection"
        )

    elif len(ints_fl) == 1:

        status = (
            "single_intersection"
        )

    else:

        status = (
            "measure_width"
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    curve_conf = (
        float(
            np.nanmedian(
                c
            )
        )
        if np.isfinite(
            c
        ).any()
        else 0.0
    )

    ref_conf = float(
        ref.get(
            "confidence",
            0.0,
        )
    )

    axis_conf = float(
        result[
            "x_axis_confidence"
        ]
    )

    confidence = float(
        np.clip(
            0.25 * ref_conf
            + 0.50 * curve_conf
            + 0.25 * axis_conf,
            0,
            1,
        )
    )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    warnings = []

    if not axis.get(
        "ok"
    ):

        warnings.append(
            (
                "Actual horizontal X-axis could not be "
                "detected reliably; ROI mapping was used "
                "as fallback."
            )
        )

    if sensitivity_name in {
        "High",
        "Very High",
    }:

        warnings.append(
            (
                f"{sensitivity_name} sensitivity selected. "
                "Review the annotated curve for noise "
                "before accepting the measurement."
            )
        )

    if len(ints_fl) == 0:

        warnings.append(
            (
                "The traced curve does not genuinely "
                "cross the 50% level inside the ROI."
            )
        )

    elif len(ints_fl) == 1:

        warnings.append(
            (
                "Only one genuine 50% intersection was "
                "detected; width is not calculated."
            )
        )

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
) -> np.ndarray:

    im = result[
        "image_rgb"
    ].copy()

    h, w = im.shape[:2]

    left = result.get(
        "reference_left_px"
    )

    right = result.get(
        "reference_right_px"
    )

    baseline = result.get(
        "baseline_y_px"
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

    # --------------------------------------------------------
    # ROI boundaries = GREEN
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
    # ACTUAL X AXIS = BLUE/CYAN
    # --------------------------------------------------------

    if (
        axis_start is not None
        and axis_end is not None
    ):

        y = int(
            axis_y
            if axis_y is not None
            else baseline
        )

        cv2.circle(
            im,
            (
                int(axis_start),
                y,
            ),
            6,
            (
                255,
                255,
                0,
            ),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            im,
            (
                int(axis_end),
                y,
            ),
            6,
            (
                255,
                255,
                0,
            ),
            -1,
            cv2.LINE_AA,
        )

        xmin = result.get(
            "x_min_fl"
        )

        xmax = result.get(
            "x_max_fl"
        )

        cv2.putText(
            im,
            f"{xmin:g} fL",
            (
                max(
                    2,
                    int(axis_start) - 10,
                ),
                min(
                    h - 5,
                    y + 25,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                180,
                180,
                0,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            im,
            f"{xmax:g} fL",
            (
                max(
                    2,
                    int(axis_end) - 35,
                ),
                min(
                    h - 5,
                    y + 25,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                180,
                180,
                0,
            ),
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # 50% line = ORANGE
    # --------------------------------------------------------

    if (
        result.get(
            "y50_px"
        ) is not None
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
            cv2.LINE_AA,
        )

        y50_fl = result.get(
            "y50_fl"
        )

        if y50_fl is not None:

            cv2.putText(
                im,
                (
                    f"50% = "
                    f"{y50_fl:.3f} fL"
                ),
                (
                    max(
                        2,
                        int(left) + 5,
                    ),
                    max(
                        15,
                        y50 - 7,
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

    # --------------------------------------------------------
    # Peak = MAGENTA
    # --------------------------------------------------------

    if (
        result.get(
            "peak_x_px"
        ) is not None
        and result.get(
            "peak_y_px"
        ) is not None
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
            cv2.LINE_AA,
        )

        peak_y = result.get(
            "peak_y_fl"
        )

        if peak_y is not None:

            cv2.putText(
                im,
                (
                    f"Peak Y: "
                    f"{peak_y:.3f} fL"
                ),
                (
                    min(
                        w - 150,
                        px + 8,
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

    # --------------------------------------------------------
    # Intersections = RED
    # --------------------------------------------------------

    y50 = result.get(
        "y50_px"
    )

    values = result.get(
        "intersections_fl",
        [],
    )

    if y50 is not None:

        y = int(
            round(y50)
        )

        for i, xx in enumerate(
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
                    y,
                ),
                7,
                (
                    255,
                    0,
                    0,
                ),
                -1,
                cv2.LINE_AA,
            )

            if i < len(
                values
            ):

                text_y = (
                    y - 9
                    if i % 2 == 0
                    else y + 18
                )

                cv2.putText(
                    im,
                    (
                        f"{values[i]:.3f} fL"
                    ),
                    (
                        max(
                            1,
                            min(
                                w - 90,
                                x - 30,
                            ),
                        ),
                        max(
                            13,
                            min(
                                h - 4,
                                text_y,
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
                f"{result['x_min_fl']:g}–"
                f"{result['x_max_fl']:g} fL"
            ),
        ),

        (
            "Y-axis range",
            (
                f"0–"
                f"{result['y_max_fl']:g} fL"
            ),
        ),

        (
            "Sensitivity",
            (
                f"{result.get('sensitivity', 'Medium')} "
                f"(threshold "
                f"{result.get('segmentation_threshold', DEFAULT_THRESHOLD):.2f})"
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
            "Actual X-axis",
            (
                f"{result.get('x_axis_start_px')}"
                f"–"
                f"{result.get('x_axis_end_px')}"
                " px"
            ),
        ),

        (
            "X-axis method",
            result.get(
                "x_axis_method"
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
                if len(ints) < 1
                else (
                    f"{ints[0]:.3f} fL"
                )
            ),
        ),

        (
            "Right 50% intersection",
            (
                "Not available"
                if len(ints) < 2
                else (
                    f"{ints[-1]:.3f} fL"
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
                ints
            ),
        ),

        (
            "All intersections",
            (
                "None"
                if not ints
                else (
                    ", ".join(
                        f"{v:.3f}"
                        for v in ints
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