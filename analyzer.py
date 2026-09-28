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
            "Very high sensitivity — useful for extremely faint "
            "curves, but more background noise may be included."
        )

    if threshold <= 0.18:

        return (
            "High sensitivity — recommended for weak-visibility "
            "curves."
        )

    if threshold <= 0.28:

        return (
            "Balanced sensitivity — recommended starting point."
        )

    if threshold <= 0.36:

        return (
            "Conservative sensitivity — useful for cleaner graphs."
        )

    return (
        "Very conservative sensitivity — weak curve pixels "
        "may be rejected."
    )


# ============================================================
# BASIC HELPERS
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

    idx = np.flatnonzero(
        b
    )

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


def _norm(
    v: np.ndarray,
) -> np.ndarray:

    v = np.asarray(
        v,
        dtype=np.float32,
    )

    if not v.size:

        return np.zeros_like(
            v
        )

    m = float(
        v.max()
    )

    if m <= 1e-9:

        return np.zeros_like(
            v
        )

    return (
        v / m
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
    # Vertical runs
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
    # Hough
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

        line_records = (
            np.asarray(
                lines
            )
            .reshape(
                -1,
                4,
            )
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
    # Combined score
    # --------------------------------------------------------

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
            hough
        )
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

        xx = int(
            qx
        )

        if score[
            xx
        ] <= 0:

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

            sep = (
                right0
                - left0
            )

            if sep < min_sep:

                continue

            pair = float(
                score[
                    left0
                ]
                + score[
                    right0
                ]
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
            "method": (
                "no_reliable_pair"
            ),
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
        "left_x": int(
            left
        ),
        "right_x": int(
            right
        ),
        "roi_width": int(
            right - left
        ),
        "confidence": float(
            np.clip(
                best_pair
                / 2.0,
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
# FAINT-CURVE VISUAL ENHANCEMENT
# ============================================================

def build_visual_curve_score(
    rgb: np.ndarray,
    left: int,
    right: int,
    baseline: int,
) -> np.ndarray:
    """
    Generates an additional image-based cue for faint curves.

    U-Net remains the main model signal.
    """

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    # --------------------------------------------------------
    # CLAHE
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
    # Dark-line cue
    # --------------------------------------------------------

    dark_score = np.clip(
        (
            210.0
            - enhanced.astype(
                np.float32
            )
        )
        / 160.0,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Black-hat cue
    # --------------------------------------------------------

    blackhat_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                13,
                13,
            ),
        )
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

    # --------------------------------------------------------
    # Edge cue
    # --------------------------------------------------------

    edges = cv2.Canny(
        enhanced,
        25,
        100,
    ).astype(
        np.float32
    )

    edges = _norm(
        edges
    )

    # --------------------------------------------------------
    # Combined
    # --------------------------------------------------------

    visual = (
        0.45 * dark_score
        + 0.35 * blackhat
        + 0.20 * edges
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

    # Restrict to ROI and above baseline.
    output = np.zeros_like(
        visual,
        dtype=np.float32,
    )

    top = max(
        0,
        int(
            0.02
            * rgb.shape[0]
        ),
    )

    bottom = max(
        top + 1,
        int(
            baseline
        )
        - 2,
    )

    output[
        top:bottom,
        left:right + 1,
    ] = visual[
        top:bottom,
        left:right + 1,
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
) -> np.ndarray:

    mask = (
        prob
        >= float(
            threshold
        )
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
            row_edge
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
            mask[
                :,
                x
            ]
            > 0
        )[0]

        yy = yy[
            yy
            < 0.97 * h
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
        if len(
            bottoms
        ) >= 20
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
            round(
                base
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
            strip
            < 210
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
    probability: np.ndarray,
    visual: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    confidence: np.ndarray,
    seed_index: int,
    seed_y: float,
    lower: int,
    direction: int,
    threshold: float,
):

    image_height = (
        probability.shape[0]
    )

    # --------------------------------------------------------
    # Lower threshold => more movement tolerance.
    # --------------------------------------------------------

    sensitivity_factor = float(
        np.clip(
            (
                0.32
                - threshold
            )
            / 0.27,
            0.0,
            1.0,
        )
    )

    max_step = int(
        image_height
        * (
            0.030
            + 0.030
            * sensitivity_factor
        )
    )

    max_step = max(
        5,
        min(
            36,
            max_step,
        ),
    )

    max_gap = int(
        8
        + 14
        * sensitivity_factor
    )

    # Combined signal weight.
    visual_weight = (
        0.10
        + 0.20
        * sensitivity_factor
    )

    # Probability acceptance.
    track_threshold = max(
        0.035,
        min(
            0.35,
            threshold
            * (
                0.65
                - 0.20
                * sensitivity_factor
            ),
        ),
    )

    prev = float(
        seed_y
    )

    gaps = 0

    if direction > 0:

        iterator = range(
            seed_index + 1,
            len(xs),
        )

    else:

        iterator = range(
            seed_index - 1,
            -1,
            -1,
        )

    for j in iterator:

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

        pvals = probability[
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
        continuity_penalty = (
            0.014
            + 0.010
            * sensitivity_factor
        )

        score = (
            hybrid
            - continuity_penalty
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

        selected_probability = float(
            pvals[k]
        )

        selected_score = float(
            score[k]
        )

        # ----------------------------------------------------
        # Accept weak pixels when visual cue is strong.
        # ----------------------------------------------------

        acceptance = (
            selected_probability
            >= track_threshold
            or (
                selected_score
                >= track_threshold
                * 0.75
                and float(
                    vvals[k]
                )
                >= 0.28
            )
        )

        if not acceptance:

            gaps += 1
            continue

        chosen_y = int(
            yy[k]
        )

        # ----------------------------------------------------
        # Local weighted centre
        # ----------------------------------------------------

        a = max(
            y0,
            chosen_y - 2,
        )

        b = min(
            y1,
            chosen_y + 2,
        )

        local_probability = probability[
            a:b + 1,
            x,
        ]

        local_visual = visual[
            a:b + 1,
            x,
        ]

        local_signal = (
            (
                1.0
                - visual_weight
            )
            * local_probability
            + visual_weight
            * local_visual
        )

        weights = np.maximum(
            local_signal,
            0.0,
        )

        denominator = float(
            weights.sum()
        )

        if denominator <= 1e-8:

            chosen_center = float(
                chosen_y
            )

        else:

            chosen_center = float(
                (
                    np.arange(
                        a,
                        b + 1,
                    )
                    * weights
                ).sum()
                / denominator
            )

        ys[j] = chosen_center

        confidence[j] = max(
            selected_probability,
            selected_score,
        )

        prev = chosen_center

        gaps = 0


def build_curve_track(
    probability: np.ndarray,
    visual: np.ndarray,
    left: int,
    right: int,
    baseline: int,
    threshold: float,
):

    h, _ = probability.shape

    left = int(
        left
    )

    right = int(
        right
    )

    xs = np.arange(
        left,
        right + 1,
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

    # --------------------------------------------------------
    # Sensitivity-dependent hybrid weight.
    # --------------------------------------------------------

    sensitivity_factor = float(
        np.clip(
            (
                0.32
                - threshold
            )
            / 0.27,
            0.0,
            1.0,
        )
    )

    visual_weight = (
        0.10
        + 0.20
        * sensitivity_factor
    )

    hybrid = (
        (
            1.0
            - visual_weight
        )
        * probability
        + visual_weight
        * visual
    )

    crop = hybrid[
        :lower + 1,
        left:right + 1,
    ]

    if crop.size == 0:

        return (
            xs,
            ys,
            confidence,
        )

    # --------------------------------------------------------
    # Find seed peak.
    # --------------------------------------------------------

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

    # Lower sensitivity threshold permits lower seed.
    seed_threshold = max(
        0.08,
        threshold
        * 1.10,
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

    # --------------------------------------------------------
    # Track right
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Track left
    # --------------------------------------------------------

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
# SAFE 1D MEDIAN
# ============================================================

def _median_1d(
    values: np.ndarray,
    kernel_size: int,
) -> np.ndarray:

    if len(
        values
    ) < 3:

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


# ============================================================
# SAFE CURVE SMOOTHING
# ============================================================

def smooth_array(
    v: np.ndarray,
) -> np.ndarray:
    """
    Safe float32 smoothing.

    IMPORTANT:
    This intentionally does NOT use cv2.medianBlur().
    OpenCV 4.14 rejects the float32 array used by this pipeline.
    """

    output = np.full_like(
        v,
        np.nan,
        dtype=np.float32,
    )

    finite = np.isfinite(
        v
    )

    if finite.sum() < 5:

        output[
            finite
        ] = v[
            finite
        ]

        return output

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

            output[
                segment
            ] = v[
                segment
            ]

            continue

        values = (
            v[
                segment
            ]
            .astype(
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

        output[
            segment
        ] = _median_1d(
            values,
            kernel,
        )

    return output


# ============================================================
# GENUINE INTERSECTIONS
# ============================================================

def genuine_intersections(
    xs: np.ndarray,
    ys: np.ndarray,
    y50: float,
    left: int,
    right: int,
) -> list[float]:

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
        - float(
            y50
        )
    )

    intersections = []

    for i in range(
        len(x) - 1
    ):

        # Do not bridge large missing curve gaps.
        if (
            x[i + 1]
            - x[i]
            > 12
        ):

            continue

        # Direct close intersection.
        if (
            abs(
                d[i]
            )
            <= 0.45
        ):

            intersections.append(
                float(
                    x[i]
                )
            )

        # True sign crossing.
        if (
            d[i]
            * d[i + 1]
            < 0
        ):

            denominator = (
                abs(
                    d[i]
                )
                + abs(
                    d[i + 1]
                )
                + 1e-12
            )

            t = (
                abs(
                    d[i]
                )
                / denominator
            )

            x_cross = (
                x[i]
                + t
                * (
                    x[i + 1]
                    - x[i]
                )
            )

            intersections.append(
                float(
                    x_cross
                )
            )

    if (
        len(d)
        and abs(
            d[-1]
        )
        <= 0.45
    ):

        intersections.append(
            float(
                x[-1]
            )
        )

    intersections.sort()

    # Merge duplicates.
    output = []

    minimum_distance = max(
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
            or (
                value
                - output[-1]
                > minimum_distance
            )
        ):

            output.append(
                value
            )

    return output


# ============================================================
# PIXEL -> X VALUE
# ============================================================

def px_to_fl(
    x: float,
    axis_start_px: int,
    axis_end_px: int,
    xmin: float,
    xmax: float,
) -> float:

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
        float(
            xmin
        )
        + normalized
        * (
            float(xmax)
            - float(xmin)
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
# PIXEL -> Y VALUE
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
            float(
                baseline
            )
            - float(
                y
            )
        )
        / denominator
        * float(
            y_max
        )
    )

    return float(
        np.clip(
            value,
            0.0,
            float(
                y_max
            ),
        )
    )


# ============================================================
# EXPECTED RECONSTRUCTED TICKS
# ============================================================

def build_reconstructed_ticks(
    xmin: float,
    xmax: float,
) -> list[float]:

    start = int(
        np.ceil(
            xmin / 10.0
            - 1e-9
        )
        * 10
    )

    end = int(
        np.floor(
            xmax / 10.0
            + 1e-9
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

    # --------------------------------------------------------
    # Validate numerical ranges.
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

    image_height, image_width = (
        rgb.shape[:2]
    )

    # --------------------------------------------------------
    # Manual axis validation.
    # --------------------------------------------------------

    if (
        manual_axis_start_px is None
        or manual_axis_end_px is None
    ):

        raise ValueError(
            "Manual X-axis start and end positions "
            "are required."
        )

    axis_start_px = int(
        manual_axis_start_px
    )

    axis_end_px = int(
        manual_axis_end_px
    )

    if (
        axis_start_px < 0
        or axis_start_px
        >= image_width
        or axis_end_px < 0
        or axis_end_px
        >= image_width
    ):

        raise ValueError(
            "Manual X-axis pixel positions must "
            "be inside the uploaded image."
        )

    if (
        axis_end_px
        <= axis_start_px
    ):

        raise ValueError(
            "Manual X-axis maximum position must "
            "be greater than the minimum position."
        )

    pixels_per_fl = float(
        (
            axis_end_px
            - axis_start_px
        )
        / (
            xmax
            - xmin
        )
    )

    # ========================================================
    # RESULT INITIALIZATION
    # ========================================================

    result = {

        "status": "error",

        "measurement_source": (
            "V8 U-Net + faint-curve enhancement "
            "+ manual X-axis calibration"
        ),

        "axis_detection": (
            "ROI automatic; X-axis calibration manual"
        ),

        "x_min_fl": xmin,
        "x_max_fl": xmax,

        "y_min_fl": 0.0,
        "y_max_fl": ymax,

        "segmentation_threshold": threshold,

        "sensitivity": (
            get_sensitivity_description(
                threshold
            )
        ),

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        "reference_left_px": None,
        "reference_right_px": None,
        "reference_confidence": 0.0,
        "reference_method": None,

        # ----------------------------------------------------
        # Manual X axis
        # ----------------------------------------------------

        "x_axis_start_px": axis_start_px,
        "x_axis_end_px": axis_end_px,

        "x_axis_y_px": None,

        "pixels_per_fl": pixels_per_fl,

        "x_axis_method": (
            "Manual calibration"
        ),

        "x_axis_reconstructed_ticks": (
            build_reconstructed_ticks(
                xmin,
                xmax,
            )
        ),

        # ----------------------------------------------------
        # Y geometry
        # ----------------------------------------------------

        "baseline_y_px": None,
        "plot_top_y_px": None,

        # ----------------------------------------------------
        # Peak
        # ----------------------------------------------------

        "peak_x_px": None,
        "peak_y_px": None,

        "peak_x_fl": None,
        "peak_y_fl": None,

        "peak_height_px": None,

        # ----------------------------------------------------
        # 50%
        # ----------------------------------------------------

        "y50_px": None,
        "y50_fl": None,

        # ----------------------------------------------------
        # Intersections
        # ----------------------------------------------------

        "intersections_px": [],
        "intersections_fl": [],

        "intersection_count": 0,

        "width_50_fl": None,

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        "confidence": 0.0,

        # ----------------------------------------------------
        # Warning
        # ----------------------------------------------------

        "warning": None,

        # ----------------------------------------------------
        # Internal data
        # ----------------------------------------------------

        "image_rgb": rgb,

        "curve_probability": None,
        "visual_curve_score": None,
        "curve_mask": None,

        "centerline_x": None,
        "centerline_y": None,
        "centerline_confidence": None,
    }

    # ========================================================
    # DETECT ROI
    # ========================================================

    reference = (
        detect_reference_lines(
            rgb
        )
    )

    result[
        "reference_left_px"
    ] = reference.get(
        "left_x"
    )

    result[
        "reference_right_px"
    ] = reference.get(
        "right_x"
    )

    result[
        "reference_confidence"
    ] = float(
        reference.get(
            "confidence",
            0.0,
        )
    )

    result[
        "reference_method"
    ] = reference.get(
        "method"
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
                    "The two vertical dashed ROI "
                    "boundaries could not be detected."
                ),
            }
        )

        return result

    left = int(
        reference[
            "left_x"
        ]
    )

    right = int(
        reference[
            "right_x"
        ]
    )

    if (
        right
        - left
        < max(
            40,
            int(
                0.15
                * image_width
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
    # MODEL CURVE PROBABILITY
    # ========================================================

    probability = predict_curve(
        model,
        device,
        rgb,
    )

    # ========================================================
    # CLEAN MASK
    # ========================================================

    curve_mask = (
        clean_curve_probability(
            probability,
            left,
            right,
            threshold,
        )
    )

    # ========================================================
    # BASELINE
    # ========================================================

    baseline = estimate_baseline(
        rgb,
        curve_mask,
        left,
        right,
    )

    result[
        "baseline_y_px"
    ] = baseline

    # ========================================================
    # PLOT TOP
    # ========================================================

    plot_top = estimate_plot_top(
        rgb,
        left,
        right,
        baseline,
    )

    result[
        "plot_top_y_px"
    ] = plot_top

    # ========================================================
    # FAINT-CURVE VISUAL ENHANCEMENT
    # ========================================================

    visual = build_visual_curve_score(
        rgb,
        left,
        right,
        baseline,
    )

    # ========================================================
    # TRACK CURVE
    # ========================================================

    (
        xs,
        ys,
        curve_confidence,
    ) = build_curve_track(
        probability,
        visual,
        left,
        right,
        baseline,
        threshold,
    )

    result.update(
        {
            "curve_probability": probability,
            "visual_curve_score": visual,
            "curve_mask": curve_mask,

            "centerline_x": xs,
            "centerline_y": ys,

            "centerline_confidence": (
                curve_confidence
            ),
        }
    )

    # ========================================================
    # VALID CURVE POINTS
    # ========================================================

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

    if (
        valid.sum()
        < 10
    ):

        result.update(
            {
                "status": (
                    "curve_not_found"
                ),
                "warning": (
                    "A reliable curve trace could not "
                    "be established. Try lowering the "
                    "sensitivity threshold."
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

    conf = curve_confidence[
        valid
    ].astype(
        float
    )

    # ========================================================
    # PEAK
    # ========================================================

    heights = (
        baseline
        - y
    )

    smoothed = smooth_array(
        heights
    )

    finite_smoothed = np.isfinite(
        smoothed
    )

    if not finite_smoothed.any():

        result.update(
            {
                "status": (
                    "peak_not_found"
                ),
                "warning": (
                    "The curve peak could not "
                    "be calculated."
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
        x[
            peak_index
        ]
    )

    peak_y_px = float(
        y[
            peak_index
        ]
    )

    peak_height_px = float(
        baseline
        - peak_y_px
    )

    # --------------------------------------------------------
    # Peak X uses MANUAL axis calibration.
    # --------------------------------------------------------

    peak_x_fl = px_to_fl(
        peak_x_px,
        axis_start_px,
        axis_end_px,
        xmin,
        xmax,
    )

    # --------------------------------------------------------
    # Peak Y.
    # --------------------------------------------------------

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

            "peak_height_px": peak_height_px,
        }
    )

    if (
        peak_height_px
        < 3
    ):

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
    # 50% OF DETECTED PEAK
    # ========================================================

    y50_px = float(
        baseline
        - 0.5
        * peak_height_px
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
            left,
            right,
        )
    )

    # ========================================================
    # CONVERT X VALUES USING MANUAL AXIS
    # ========================================================

    intersections_fl = []

    for px in intersections_px:

        value = px_to_fl(
            px,
            axis_start_px,
            axis_end_px,
            xmin,
            xmax,
        )

        if np.isfinite(
            value
        ):

            intersections_fl.append(
                float(
                    value
                )
            )

    # --------------------------------------------------------
    # Final physical/curve consistency validation.
    # --------------------------------------------------------

    checked_pairs = []

    for px, value in zip(
        intersections_px,
        intersections_fl,
    ):

        nearest = int(
            np.argmin(
                np.abs(
                    xs
                    - px
                )
            )
        )

        if (
            np.isfinite(
                ys[
                    nearest
                ]
            )
            and abs(
                float(
                    ys[
                        nearest
                    ]
                )
                - y50_px
            )
            <= 3
        ):

            checked_pairs.append(
                (
                    float(
                        value
                    ),
                    float(
                        px
                    ),
                )
            )

    checked_pairs.sort(
        key=lambda item: item[0]
    )

    intersections_fl = [
        round(
            item[0],
            6,
        )
        for item in checked_pairs
    ]

    intersections_px = [
        item[1]
        for item in checked_pairs
    ]

    # ========================================================
    # STATUS
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

    # ========================================================
    # WIDTH
    # ========================================================

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

    curve_conf = (
        float(
            np.nanmedian(
                conf
            )
        )
        if np.isfinite(
            conf
        ).any()
        else 0.0
    )

    roi_conf = float(
        reference.get(
            "confidence",
            0.0,
        )
    )

    axis_conf = 1.0

    confidence = float(
        np.clip(
            0.25 * roi_conf
            + 0.55 * curve_conf
            + 0.20 * axis_conf,
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
                "High sensitivity selected; inspect the "
                "curve trace for possible noise."
            )
        )

    if count == 0:

        warnings.append(
            (
                "The traced curve does not genuinely "
                "cross the 50% level inside the ROI."
            )
        )

    elif count == 1:

        warnings.append(
            (
                "Only one genuine 50% intersection was "
                "detected; width is not calculated."
            )
        )

    # ========================================================
    # FINAL RESULT
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

    image = result[
        "image_rgb"
    ].copy()

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    roi_left = result.get(
        "reference_left_px"
    )

    roi_right = result.get(
        "reference_right_px"
    )

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
        )

    # --------------------------------------------------------
    # Manual X axis.
    # --------------------------------------------------------

    axis_start = result.get(
        "x_axis_start_px"
    )

    axis_end = result.get(
        "x_axis_end_px"
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

    if (
        axis_start is not None
        and axis_end is not None
        and baseline is not None
    ):

        axis_start = int(
            axis_start
        )

        axis_end = int(
            axis_end
        )

        axis_y = int(
            baseline
        )

        # ----------------------------------------------------
        # Main reconstructed axis.
        # ----------------------------------------------------

        cv2.line(
            image,
            (
                axis_start,
                axis_y,
            ),
            (
                axis_end,
                axis_y,
            ),
            (
                0,
                220,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Endpoints.
        # ----------------------------------------------------

        cv2.circle(
            image,
            (
                axis_start,
                axis_y,
            ),
            6,
            (
                0,
                255,
                255,
            ),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            image,
            (
                axis_end,
                axis_y,
            ),
            6,
            (
                0,
                255,
                255,
            ),
            -1,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Endpoint labels.
        # ----------------------------------------------------

        cv2.putText(
            image,
            (
                f"{xmin:g} fL"
            ),
            (
                max(
                    2,
                    axis_start - 12,
                ),
                min(
                    h - 5,
                    axis_y + 27,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                0,
                160,
                180,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            (
                f"{xmax:g} fL"
            ),
            (
                max(
                    2,
                    axis_end - 40,
                ),
                min(
                    h - 5,
                    axis_y + 27,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                0,
                160,
                180,
            ),
            1,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Reconstructed major ticks.
        #
        # These are mathematical ticks, NOT detected ticks.
        # ----------------------------------------------------

        tick_values = (
            result.get(
                "x_axis_reconstructed_ticks",
                [],
            )
        )

        axis_range = (
            xmax - xmin
        )

        if axis_range > 0:

            for value in tick_values:

                normalized = (
                    float(
                        value
                    )
                    - xmin
                ) / axis_range

                px = int(
                    round(
                        axis_start
                        + normalized
                        * (
                            axis_end
                            - axis_start
                        )
                    )
                )

                cv2.line(
                    image,
                    (
                        px,
                        axis_y - 4,
                    ),
                    (
                        px,
                        axis_y + 10,
                    ),
                    (
                        0,
                        220,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                # Do not label min/end twice.
                if (
                    value != xmin
                    and value != xmax
                ):

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
                                axis_y + 27,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (
                            0,
                            150,
                            180,
                        ),
                        1,
                        cv2.LINE_AA,
                    )

    # --------------------------------------------------------
    # 50% line.
    # --------------------------------------------------------

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

        y50_fl = result.get(
            "y50_fl"
        )

        if y50_fl is not None:

            cv2.putText(
                image,
                (
                    f"50% = "
                    f"{y50_fl:.3f} fL"
                ),
                (
                    max(
                        2,
                        int(
                            roi_left
                        ) + 5,
                    ),
                    max(
                        14,
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

    # --------------------------------------------------------
    # Peak.
    # --------------------------------------------------------

    peak_x = result.get(
        "peak_x_px"
    )

    peak_y = result.get(
        "peak_y_px"
    )

    if (
        peak_x is not None
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

        cv2.circle(
            image,
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

        peak_y_fl = result.get(
            "peak_y_fl"
        )

        if peak_y_fl is not None:

            cv2.putText(
                image,
                (
                    f"Peak Y: "
                    f"{peak_y_fl:.3f} fL"
                ),
                (
                    min(
                        w - 160,
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
    # Genuine intersections.
    # --------------------------------------------------------

    intersections_px = result.get(
        "intersections_px",
        [],
    )

    intersections_fl = result.get(
        "intersections_fl",
        [],
    )

    if y50 is not None:

        y = int(
            round(
                y50
            )
        )

        for i, px_value in enumerate(
            intersections_px
        ):

            px = int(
                round(
                    px_value
                )
            )

            cv2.circle(
                image,
                (
                    px,
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

            if (
                i
                < len(
                    intersections_fl
                )
            ):

                label_y = (
                    y - 9
                    if i % 2 == 0
                    else y + 18
                )

                cv2.putText(
                    image,
                    (
                        f"{intersections_fl[i]:.3f} fL"
                    ),
                    (
                        max(
                            2,
                            min(
                                w - 95,
                                px - 32,
                            ),
                        ),
                        max(
                            13,
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
            "ROI detection",
            (
                f"{result.get('reference_left_px')}"
                f"–"
                f"{result.get('reference_right_px')} px"
            ),
        ),

        (
            "X-axis calibration",
            result.get(
                "x_axis_method"
            ),
        ),

        (
            "X-axis range",
            (
                f"{result.get('x_min_fl'):g}"
                f"–"
                f"{result.get('x_max_fl'):g}"
                f" fL"
            ),
        ),

        (
            "Y-axis range",
            (
                f"0–"
                f"{result.get('y_max_fl'):g}"
                f" fL"
            ),
        ),

        (
            "Manual X-axis start",
            (
                f"{result.get('x_axis_start_px')} px"
                f" → "
                f"{result.get('x_min_fl'):g} fL"
            ),
        ),

        (
            "Manual X-axis end",
            (
                f"{result.get('x_axis_end_px')} px"
                f" → "
                f"{result.get('x_max_fl'):g} fL"
            ),
        ),

        (
            "Pixels per fL",
            (
                f"{result.get('pixels_per_fl'):.3f}"
            ),
        ),

        (
            "Sensitivity threshold",
            (
                f"{result.get('segmentation_threshold'):.2f}"
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