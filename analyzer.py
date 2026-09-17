from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from preprocessing import normalize_tensor, read_rgb


IMAGE_SIZE = 512
DEFAULT_THRESHOLD = 0.28


# ============================================================
# USER-FACING SENSITIVITY
# ============================================================

SENSITIVITY_PRESETS = {
    "Low": 0.36,
    "Medium": 0.28,
    "High": 0.22,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def _vertical_run_stats(binary_col: np.ndarray):
    b = np.asarray(binary_col, dtype=np.uint8) > 0
    idx = np.flatnonzero(b)

    if idx.size == 0:
        return 0, 0, 0.0

    starts = idx[np.r_[True, np.diff(idx) > 1]]
    ends = idx[np.r_[np.diff(idx) > 1, True]]
    lengths = ends - starts + 1

    return (
        int(len(lengths)),
        int(lengths.max()),
        float(lengths.sum() / len(binary_col)),
    )


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)

    if not v.size:
        return np.zeros_like(v)

    m = float(v.max())

    if m <= 1e-9:
        return np.zeros_like(v)

    return v / m


# ============================================================
# DETECT TWO VERTICAL DASHED ROI BOUNDARIES
# ============================================================

def detect_reference_lines(rgb: np.ndarray) -> dict:

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
        g <= min(
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
    # Vertical dashed-run score
    # --------------------------------------------------------

    for x in range(g.shape[1]):

        nr, mr, cov = _vertical_run_stats(
            dark[:, x]
        )

        raw[x] = cov

        runs[x] = (
            min(nr, 12) / 12.0
            + 0.25
            * min(
                mr / max(
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

        line_records = np.asarray(
            lines
        ).reshape(
            -1,
            4,
        )

        for x_a, y_a, x_b, y_b in line_records:

            dx = abs(
                int(x_b)
                - int(x_a)
            )

            dy = abs(
                int(y_b)
                - int(y_a)
            )

            if (
                dx <= max(
                    3,
                    int(0.01 * w),
                )
                and dy >= max(
                    8,
                    int(0.025 * h),
                )
            ):

                xm = int(
                    round(
                        (
                            x_a + x_b
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
        "method": "dashed-runs+hough",
        "candidates": [
            int(c + x0)
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
        0.70
        * _norm(row_edge)
        + 0.30
        * _norm(row_dark)
    )

    axis_y = (
        ylo
        + int(
            np.argmax(score)
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
# X-AXIS TICK DETECTION
# ============================================================

def _tick_start_run(
    col: np.ndarray,
) -> int:

    b = (
        np.asarray(
            col,
            dtype=np.uint8,
        )
        > 0
    )

    if b.size == 0:
        return 0

    best = 0

    # Tick should be attached to/very close to the axis.
    for start in range(
        min(
            3,
            len(b),
        )
    ):

        if not b[start]:
            continue

        n = 0

        for j in range(
            start,
            len(b),
        ):

            if b[j]:
                n += 1
            else:
                break

        best = max(
            best,
            n,
        )

    return best


def _cluster_centers(
    xs: list[int],
) -> list[int]:

    if not xs:
        return []

    xs = sorted(
        set(
            int(x)
            for x in xs
        )
    )

    clusters = [
        [xs[0]]
    ]

    for x in xs[1:]:

        if (
            x
            - clusters[-1][-1]
            <= 3
        ):

            clusters[-1].append(
                x
            )

        else:

            clusters.append(
                [x]
            )

    return [
        int(
            round(
                float(
                    np.mean(
                        cluster
                    )
                )
            )
        )
        for cluster in clusters
    ]


def detect_x_tick_candidates(
    rgb: np.ndarray,
    baseline_y: int,
    exclude_x: tuple[int, int] | None = None,
) -> list[dict]:
    """
    Detect short vertical tick marks attached to the X-axis.

    Searches the full image width. It does NOT assume the two
    dashed ROI boundaries are the numerical axis endpoints.
    """

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    h, w = gray.shape

    y0 = max(
        0,
        int(baseline_y) - 1,
    )

    y1 = min(
        h,
        int(baseline_y)
        + max(
            12,
            int(0.08 * h),
        ),
    )

    band = gray[
        y0:y1
    ]

    # Dark-pixel threshold.
    dark = (
        band < 165
    ).astype(np.uint8)

    raw = []

    for x in range(w):

        # Do not let the dashed ROI boundaries become X ticks.
        if (
            exclude_x is not None
            and min(
                abs(
                    x
                    - exclude_x[0]
                ),
                abs(
                    x
                    - exclude_x[1]
                ),
            )
            <= 5
        ):
            continue

        run = _tick_start_run(
            dark[:, x]
        )

        if run >= 3:

            raw.append(
                x
            )

    centers = _cluster_centers(
        raw
    )

    results = []

    for center in centers:

        start = max(
            0,
            center - 3,
        )

        end = min(
            w,
            center + 4,
        )

        local_runs = [
            _tick_start_run(
                dark[:, x]
            )
            for x in range(
                start,
                end,
            )
        ]

        results.append(
            {
                "x": int(
                    center
                ),
                "strength": int(
                    max(
                        local_runs
                    )
                    if local_runs
                    else 0
                ),
            }
        )

    return results


# ============================================================
# EXPECTED 10 fL MAJOR TICKS
# ============================================================

def _expected_major_ticks(
    xmin: float,
    xmax: float,
) -> list[float]:

    start = (
        int(
            np.ceil(
                xmin / 10.0
                - 1e-9
            )
        )
        * 10
    )

    end = (
        int(
            np.floor(
                xmax / 10.0
                + 1e-9
            )
        )
        * 10
    )

    if end < start:
        return []

    return [
        float(v)
        for v in range(
            start,
            end + 1,
            10,
        )
    ]


# ============================================================
# BUILD X-AXIS CALIBRATION
# ============================================================

def calibrate_x_axis(
    rgb: np.ndarray,
    baseline_y: int,
    xmin: float,
    xmax: float,
    roi_left: int,
    roi_right: int,
) -> dict:
    """
    Fit actual 10 fL tick locations.

    Example:
        detected pixels:
            52 -> 0 fL
            78 -> 10 fL
            105 -> 20 fL
            132 -> 30 fL
            159 -> 40 fL

    The resulting pixel/fL relationship is then used for every
    detected curve intersection.
    """

    candidates = detect_x_tick_candidates(
        rgb,
        baseline_y,
        exclude_x=(
            roi_left,
            roi_right,
        ),
    )

    tick_values = _expected_major_ticks(
        xmin,
        xmax,
    )

    fallback = {
        "ok": False,
        "method": "roi_fallback",
        "tick_positions_px": [],
        "tick_values_fl": [],
        "pixels_per_10fl": None,
        "origin_px": None,
        "first_tick_value_fl": None,
        "confidence": 0.0,
        "tick_candidates_px": [
            int(c["x"])
            for c in candidates
        ],
    }

    # Need at least 2 values and 2 candidate tick marks.
    if (
        len(candidates) < 2
        or len(tick_values) < 2
    ):

        fallback["warning"] = (
            "Major X-axis tick marks could not be "
            "detected reliably; ROI mapping was used "
            "as fallback."
        )

        return fallback

    candidate_x = [
        int(c["x"])
        for c in candidates
    ]

    strength_by_x = {
        int(c["x"]): float(
            c["strength"]
        )
        for c in candidates
    }

    n = len(
        tick_values
    )

    best = None
    best_score = -1e9

    # --------------------------------------------------------
    # Search for a regular 10 fL pixel sequence.
    # --------------------------------------------------------

    for i in range(
        len(candidate_x)
    ):

        for j in range(
            i + 1,
            len(candidate_x),
        ):

            dx = float(
                candidate_x[j]
                - candidate_x[i]
            )

            if dx < 8:
                continue

            for k0 in range(n):

                for k1 in range(
                    k0 + 1,
                    n,
                ):

                    step = (
                        dx
                        / float(
                            k1 - k0
                        )
                    )

                    if (
                        step < 8
                        or step
                        > 0.6
                        * rgb.shape[1]
                    ):
                        continue

                    origin = (
                        candidate_x[i]
                        - k0 * step
                    )

                    predicted = [
                        origin
                        + k * step
                        for k in range(n)
                    ]

                    tolerance = max(
                        4.0,
                        min(
                            0.20
                            * step,
                            12.0,
                        ),
                    )

                    used = set()

                    matches = []

                    # Match predicted tick positions
                    # to actual image candidates.
                    for tick_index, px in enumerate(
                        predicted
                    ):

                        best_candidate = None
                        best_distance = (
                            tolerance
                            + 1
                        )

                        for ci, cx in enumerate(
                            candidate_x
                        ):

                            if ci in used:
                                continue

                            distance = abs(
                                float(cx)
                                - px
                            )

                            if (
                                distance
                                < best_distance
                            ):

                                best_distance = distance
                                best_candidate = ci

                        if (
                            best_candidate
                            is not None
                            and best_distance
                            <= tolerance
                        ):

                            used.add(
                                best_candidate
                            )

                            matches.append(
                                (
                                    tick_index,
                                    candidate_x[
                                        best_candidate
                                    ],
                                    best_distance,
                                )
                            )

                    if len(matches) < 2:
                        continue

                    residual = float(
                        np.mean(
                            [
                                m[2]
                                for m in matches
                            ]
                        )
                    )

                    strength = float(
                        np.mean(
                            [
                                strength_by_x[
                                    m[1]
                                ]
                                for m in matches
                            ]
                        )
                    )

                    score = (
                        12.0
                        * len(matches)
                        - 1.5
                        * residual
                        + 0.5
                        * strength
                    )

                    if (
                        len(matches)
                        == n
                    ):
                        score += 6.0

                    if score > best_score:

                        best_score = score

                        best = {
                            "origin": origin,
                            "step": step,
                            "matches": matches,
                        }

    if (
        best is None
        or len(
            best["matches"]
        ) < 2
    ):

        fallback["warning"] = (
            "Major X-axis tick marks could not be "
            "fit to the expected 10 fL sequence; "
            "ROI mapping was used as fallback."
        )

        return fallback

    observed = []

    for (
        tick_index,
        px,
        residual,
    ) in best[
        "matches"
    ]:

        observed.append(
            {
                "value_fl": float(
                    tick_values[
                        tick_index
                    ]
                ),
                "x_px": int(
                    px
                ),
                "residual_px": float(
                    residual
                ),
            }
        )

    residual = float(
        np.mean(
            [
                x["residual_px"]
                for x in observed
            ]
        )
    )

    coverage = (
        len(observed)
        / float(n)
    )

    confidence = float(
        np.clip(
            0.45
            * coverage
            + 0.55
            * np.exp(
                -residual / 6.0
            ),
            0,
            1,
        )
    )

    return {
        "ok": True,

        "method": (
            "actual X-axis 10 fL "
            "tick calibration"
        ),

        "tick_positions_px": [
            int(
                x["x_px"]
            )
            for x in observed
        ],

        "tick_values_fl": [
            float(
                x["value_fl"]
            )
            for x in observed
        ],

        "tick_observations": observed,

        "pixels_per_10fl": float(
            best["step"]
        ),

        "origin_px": float(
            best["origin"]
        ),

        "first_tick_value_fl": float(
            tick_values[0]
        ),

        "confidence": confidence,

        "tick_candidates_px": candidate_x,
    }


# ============================================================
# PIXEL X -> fL USING REAL TICK CALIBRATION
# ============================================================

def axis_px_to_fl(
    x: float,
    calibration: dict,
    xmin: float,
    xmax: float,
    roi_left: int,
    roi_right: int,
) -> float:

    # --------------------------------------------------------
    # Preferred method: real axis ticks
    # --------------------------------------------------------

    if (
        calibration.get(
            "ok"
        )
        and calibration.get(
            "pixels_per_10fl"
        )
    ):

        first_value = float(
            calibration[
                "first_tick_value_fl"
            ]
        )

        first_px = float(
            calibration[
                "origin_px"
            ]
        )

        step = float(
            calibration[
                "pixels_per_10fl"
            ]
        )

        value = (
            first_value
            + (
                float(x)
                - first_px
            )
            / step
            * 10.0
        )

        return float(
            np.clip(
                value,
                xmin,
                xmax,
            )
        )

    # --------------------------------------------------------
    # Explicit fallback only
    # --------------------------------------------------------

    if roi_right <= roi_left:

        return float(
            "nan"
        )

    value = (
        xmin
        + (
            float(x)
            - roi_left
        )
        / float(
            roi_right
            - roi_left
        )
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
# CURVE TRACK
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

    rng = (
        range(
            seed_k + 1,
            len(xs),
        )
        if direction > 0
        else range(
            seed_k - 1,
            -1,
            -1,
        )
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

        return xs, ys, conf

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

        return xs, ys, conf

    ys[px] = float(
        py
    )

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

    return xs, ys, conf


# ============================================================
# SAFE FLOAT SMOOTHING
# ============================================================

def smooth_array(
    v: np.ndarray,
) -> np.ndarray:
    """
    Safe 1-D median smoothing.

    IMPORTANT:
    Do not use cv2.medianBlur here.

    OpenCV 4.14 rejects the float32 input that the previous
    V8 implementation supplied, causing:

        (-215:Assertion failed)
        src.depth() == CV_8U ...

    NumPy median filtering avoids the problem completely.
    """

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
    ).astype(
        np.float32
    )

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

    if k % 2 == 0:
        k += 1

    radius = k // 2

    padded = np.pad(
        a,
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
        & (
            xs >= left
        )
        & (
            xs <= right
        )
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

        # Close-to-line point
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

        # Genuine sign crossing
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

    out = []

    merge = max(
        2.0,
        0.004
        * max(
            1,
            right - left,
        ),
    )

    for v in ints:

        if (
            not out
            or v
            - out[-1]
            > merge
        ):

            out.append(
                v
            )

    return out


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
# ESTIMATE PLOT TOP
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

    strips = [

        gray[
            :baseline_y + 1,
            max(
                0,
                left - half,
            ):
            min(
                w,
                left + half + 1,
            ),
        ],

        gray[
            :baseline_y + 1,
            max(
                0,
                right - half,
            ):
            min(
                w,
                right + half + 1,
            ),
        ],
    ]

    candidates = []

    for strip in strips:

        if strip.size == 0:
            continue

        dark_count = (
            strip < 210
        ).sum(
            axis=1
        )

        ys = np.where(
            dark_count > 0
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
        np.clip(
            top,
            0,
            max(
                0,
                baseline_y - 10,
            ),
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
    y_max_fl: float = 10.0,
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
    # Validation
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
    # Detect dashed ROI
    # --------------------------------------------------------

    ref = detect_reference_lines(
        rgb
    )

    result = {

        "status": "error",

        "measurement_source": (
            "V8 U-Net curve segmentation "
            "+ tick-calibrated geometry"
        ),

        "axis_detection": (
            "Automatic dashed ROI + actual "
            "X-axis tick calibration; "
            "X/Y values supplied by user"
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

        "baseline_y_px": None,

        "plot_top_y_px": None,

        "x_axis_calibration": None,

        "x_tick_positions_px": [],
        "x_tick_values_fl": [],

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
    # ROI failure
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

    try:

        left = int(
            ref[
                "left_x"
            ]
        )

        right = int(
            ref[
                "right_x"
            ]
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
    # Curve segmentation
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
    # Plot top for numerical Y-axis
    # --------------------------------------------------------

    plot_top = estimate_plot_top(
        rgb,
        left,
        right,
        base,
    )

    # --------------------------------------------------------
    # NEW:
    # X-axis calibration from actual ticks
    # --------------------------------------------------------

    x_cal = calibrate_x_axis(
        rgb,
        base,
        xmin,
        xmax,
        left,
        right,
    )

    # --------------------------------------------------------
    # Track curve
    # --------------------------------------------------------

    xs, ys, cc = build_curve_track(
        prob,
        left,
        right,
        base,
    )

    result.update(
        {
            "baseline_y_px": base,

            "plot_top_y_px": plot_top,

            "x_axis_calibration": x_cal,

            "x_tick_positions_px": x_cal.get(
                "tick_positions_px",
                [],
            ),

            "x_tick_values_fl": x_cal.get(
                "tick_values_fl",
                [],
            ),

            "curve_probability": prob,

            "curve_mask": mask,

            "centerline_x": xs,

            "centerline_y": ys,

            "centerline_confidence": cc,
        }
    )

    # --------------------------------------------------------
    # Valid curve
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
        np.nanargmax(
            sm
        )
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

    # --------------------------------------------------------
    # IMPORTANT:
    # Peak X now uses REAL X-axis calibration.
    # --------------------------------------------------------

    peak_x_fl = axis_px_to_fl(
        peak_x,
        x_cal,
        xmin,
        xmax,
        left,
        right,
    )

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

            "peak_x_fl": peak_x_fl,
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
    # 50% level
    # --------------------------------------------------------

    y50 = float(
        base
        - 0.5
        * peak_h
    )

    y50_fl = py_to_y_fl(
        y50,
        plot_top,
        base,
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
    # NEW:
    # Convert each intersection using the actual
    # detected X-axis tick calibration.
    # --------------------------------------------------------

    ints_fl = [

        axis_px_to_fl(
            px,
            x_cal,
            xmin,
            xmax,
            left,
            right,
        )

        for px in ints_px
    ]

    # --------------------------------------------------------
    # Final geometry verification
    # --------------------------------------------------------

    checked = []

    checked_px = []

    for px, f in zip(
        ints_px,
        ints_fl,
    ):

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
                float(f)
            )

            checked_px.append(
                float(px)
            )

    # --------------------------------------------------------
    # Sort using numerical X value
    # --------------------------------------------------------

    pairs = sorted(
        zip(
            checked,
            checked_px,
        ),
        key=lambda z: z[0],
    )

    ints_fl = sorted(
        set(
            round(
                pair[0],
                6,
            )
            for pair in pairs
        )
    )

    ints_px = [
        pair[1]
        for pair in pairs
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
            np.nanmedian(c)
        )

        if np.isfinite(c).any()

        else 0.0
    )

    axis_conf = (

        float(
            x_cal.get(
                "confidence",
                0.35,
            )
        )

        if x_cal.get(
            "ok"
        )

        else 0.35
    )

    confidence = float(
        np.clip(
            0.30
            * ref[
                "confidence"
            ]
            + 0.45
            * curve_conf
            + 0.25
            * axis_conf,
            0,
            1,
        )
    )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    warnings = []

    if not x_cal.get(
        "ok"
    ):

        warnings.append(
            x_cal.get(
                "warning",
                (
                    "X-axis tick calibration "
                    "was unavailable; ROI mapping "
                    "fallback was used."
                ),
            )
        )

    else:

        expected_count = len(
            _expected_major_ticks(
                xmin,
                xmax,
            )
        )

        detected_count = len(
            x_cal.get(
                "tick_positions_px",
                [],
            )
        )

        if (
            detected_count
            < expected_count
        ):

            warnings.append(
                (
                    "Some major X-axis ticks were "
                    "not visible; calibration used "
                    "the detected subset."
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
                "Only one genuine 50% intersection "
                "was detected; width is not calculated."
            )
        )

    # --------------------------------------------------------
    # Final result
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

    # --------------------------------------------------------
    # ROI lines
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
    # NEW:
    # Draw detected real X-axis ticks.
    # Cyan = calibration points.
    # --------------------------------------------------------

    if baseline is not None:

        tick_positions = result.get(
            "x_tick_positions_px",
            [],
        )

        tick_values = result.get(
            "x_tick_values_fl",
            [],
        )

        for px, value in zip(
            tick_positions,
            tick_values,
        ):

            px = int(
                round(px)
            )

            cv2.line(
                im,
                (
                    px,
                    max(
                        0,
                        int(baseline)
                        - 4,
                    ),
                ),
                (
                    px,
                    min(
                        h - 1,
                        int(baseline)
                        + 8,
                    ),
                ),
                (
                    0,
                    210,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.circle(
                im,
                (
                    px,
                    int(baseline),
                ),
                3,
                (
                    0,
                    210,
                    255,
                ),
                -1,
                cv2.LINE_AA,
            )

            cv2.putText(
                im,
                f"{value:g}",
                (
                    max(
                        1,
                        px - 10,
                    ),
                    min(
                        h - 4,
                        int(baseline)
                        + 22,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (
                    0,
                    160,
                    200,
                ),
                1,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # 50% line
    # --------------------------------------------------------

    if (
        result.get(
            "y50_px"
        )
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

        y50_fl = result.get(
            "y50_fl"
        )

        label = (
            "50%"
            if y50_fl is None
            else (
                f"50% = "
                f"{y50_fl:.3f} fL"
            )
        )

        cv2.putText(
            im,
            label,
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
    # Peak
    # --------------------------------------------------------

    if (
        result.get(
            "peak_x_px"
        )
        is not None
        and result.get(
            "peak_y_px"
        )
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
            cv2.LINE_AA,
        )

        peak_y_fl = result.get(
            "peak_y_fl"
        )

        if peak_y_fl is not None:

            cv2.putText(
                im,
                (
                    f"Peak Y: "
                    f"{peak_y_fl:.3f} fL"
                ),
                (
                    max(
                        2,
                        px + 7,
                    ),
                    max(
                        14,
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

        intersection_values = (
            result.get(
                "intersections_fl",
                [],
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
                cv2.LINE_AA,
            )

            if (
                index
                < len(
                    intersection_values
                )
            ):

                text_y = (
                    y50 - 9
                    if index % 2 == 0
                    else y50 + 18
                )

                cv2.putText(
                    im,
                    (
                        f"{intersection_values[index]:.3f} fL"
                    ),
                    (
                        max(
                            1,
                            min(
                                w - 88,
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

    xmin = result.get(
        "x_min_fl"
    )

    xmax = result.get(
        "x_max_fl"
    )

    ymax = result.get(
        "y_max_fl"
    )

    calibration = result.get(
        "x_axis_calibration"
    ) or {}

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
            "X-axis calibration",
            calibration.get(
                "method",
                "Not available",
            ),
        ),

        (
            "Detected X ticks",
            (
                "None"
                if not result.get(
                    "x_tick_values_fl"
                )
                else (
                    ", ".join(
                        f"{v:g}"
                        for v in result[
                            "x_tick_values_fl"
                        ]
                    )
                    + " fL"
                )
            ),
        ),

        (
            "Detection sensitivity",
            (
                f"Threshold "
                f"{float(result.get('segmentation_threshold', DEFAULT_THRESHOLD)):.2f}"
            ),
        ),

        (
            "Peak position",
            (
                "Not available"
                if result.get(
                    "peak_x_fl"
                )
                is None
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
                )
                is None
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
                )
                is None
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
                )
                is None
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