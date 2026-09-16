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


def _vertical_run_stats(binary_col: np.ndarray):
    b = np.asarray(binary_col, dtype=np.uint8) > 0
    idx = np.flatnonzero(b)
    if idx.size == 0:
        return 0, 0, 0.0
    starts = idx[np.r_[True, np.diff(idx) > 1]]
    ends = idx[np.r_[np.diff(idx) > 1, True]]
    lengths = ends - starts + 1
    return int(len(lengths)), int(lengths.max()), float(lengths.sum() / len(binary_col))


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    m = float(v.max()) if v.size else 0.0
    return v / m if m > 1e-9 else np.zeros_like(v)


def detect_reference_lines(rgb: np.ndarray) -> dict:
    """Detect the two vertical dashed reference boundaries.

    IMPORTANT: this function returns one dictionary only. The caller must use
    ref['left_x'], ref['right_x']; it must never unpack the return value.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    y0, y1 = int(0.04 * h), int(0.92 * h)
    x0, x1 = int(0.025 * w), int(0.975 * w)
    g = gray[y0:y1, x0:x1]

    q = np.percentile(g, 35)
    dark = (g <= min(205, q + 15)).astype(np.uint8)
    raw = np.zeros(g.shape[1], np.float32)
    runs = np.zeros_like(raw)
    hough = np.zeros_like(raw)

    for x in range(g.shape[1]):
        nr, mr, cov = _vertical_run_stats(dark[:, x])
        raw[x] = cov
        runs[x] = min(nr, 12) / 12.0 + 0.25 * min(mr / max(1, g.shape[0]), 0.35)

    edges = cv2.Canny(g, 40, 140)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(12, int(0.018 * h)),
        minLineLength=max(10, int(0.03 * h)),
        maxLineGap=max(5, int(0.015 * h)),
    )

    if lines is not None:
        # Supports both OpenCV shapes: N x 1 x 4 and N x 4.
        line_records = np.asarray(lines).reshape(-1, 4)
        for x_a, y_a, x_b, y_b in line_records:
            dx = abs(int(x_b) - int(x_a))
            dy = abs(int(y_b) - int(y_a))
            if dx <= max(3, int(0.01 * w)) and dy >= max(8, int(0.025 * h)):
                xm = int(round((x_a + x_b) / 2))
                if 0 <= xm < g.shape[1]:
                    hough[xm] += dy

    score = 0.35 * _norm(raw) + 0.40 * _norm(runs) + 0.25 * _norm(hough)
    score = cv2.GaussianBlur(score.reshape(1, -1), (0, 0), max(1, 0.003 * w)).ravel()
    edge = int(0.02 * w)
    score[:edge] = 0
    score[-edge:] = 0

    order = np.argsort(score)[::-1]
    candidates = []
    min_dist = max(8, int(0.025 * w))
    for qx in order:
        xx = int(qx)
        if score[xx] <= 0:
            break
        if all(abs(xx - c) >= min_dist for c in candidates):
            candidates.append(xx)
        if len(candidates) >= 40:
            break

    best = None
    best_pair = -1.0
    min_sep = max(30, int(0.30 * g.shape[1]))

    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            left0, right0 = sorted((a, b))
            sep = right0 - left0
            if sep < min_sep:
                continue
            pair = float(score[left0] + score[right0])
            # Prefer a pair spanning most of the plotting area, while keeping
            # a modest margin from the image edges.
            if left0 < 0.45 * g.shape[1] and right0 > 0.55 * g.shape[1]:
                pair += 0.25
            pair += 0.15 * min(left0 / max(1, 0.18 * g.shape[1]), 1)
            pair += 0.15 * min((g.shape[1] - right0) / max(1, 0.18 * g.shape[1]), 1)
            if pair > best_pair:
                best_pair = pair
                best = (left0, right0)

    if best is None:
        return {
            "ok": False,
            "left_x": None,
            "right_x": None,
            "roi_width": None,
            "confidence": 0.0,
            "method": "no_reliable_pair",
            "candidates": [int(c + x0) for c in candidates],
        }

    left, right = best
    left += x0
    right += x0
    return {
        "ok": True,
        "left_x": int(left),
        "right_x": int(right),
        "roi_width": int(right - left),
        "confidence": float(np.clip(best_pair / 2.0, 0, 1)),
        "method": "dashed-runs+hough",
        "candidates": [int(c + x0) for c in candidates],
    }


@torch.inference_mode()
def predict_curve(model, device, rgb: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    x = normalize_tensor(rgb, image_size).to(device)
    p = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
    return cv2.resize(p, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def clean_curve_probability(prob: np.ndarray, left: int, right: int, threshold: float):
    mask = (prob >= float(threshold)).astype(np.uint8)
    clean = np.zeros_like(mask)
    clean[:, left:right + 1] = mask[:, left:right + 1]
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return clean


def estimate_baseline(rgb: np.ndarray, mask: np.ndarray, left: int, right: int) -> int:
    h, w = mask.shape
    l, r = max(0, int(left)), min(w - 1, int(right))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    ylo, yhi = int(0.58 * h), int(0.985 * h)
    edges = cv2.Canny(gray, 40, 140)
    row_edge = edges[ylo:yhi, l:r + 1].sum(1).astype(np.float32)
    row_dark = (gray[ylo:yhi, l:r + 1] < 180).mean(1).astype(np.float32)
    score = 0.70 * _norm(row_edge) + 0.30 * _norm(row_dark)
    axis_y = ylo + int(np.argmax(score)) if len(score) else int(0.90 * h)

    bottoms = []
    step = max(1, int((r - l + 1) / 300))
    for x in range(l, r + 1, step):
        yy = np.where(mask[:, x] > 0)[0]
        yy = yy[yy < 0.97 * h]
        if yy.size:
            bottoms.append(int(yy.max()))

    curve_y = float(np.percentile(bottoms, 97)) if len(bottoms) >= 20 else None
    base = axis_y if curve_y is None or abs(curve_y - axis_y) > 0.07 * h else 0.55 * axis_y + 0.45 * curve_y
    return int(np.clip(round(base), int(0.55 * h), int(0.99 * h)))


def _walk_track(prob, xs, ys, conf, seed_k, seed_y, lower, direction):
    max_step = max(6, int(0.035 * prob.shape[0]))
    prev = seed_y
    gaps = 0
    rng = range(seed_k + 1, len(xs)) if direction > 0 else range(seed_k - 1, -1, -1)

    for j in rng:
        if gaps > 12:
            break
        x = xs[j]
        y0 = max(0, int(round(prev)) - max_step)
        y1 = min(lower, int(round(prev)) + max_step)
        vals = prob[y0:y1 + 1, x]
        if vals.size == 0:
            gaps += 1
            continue
        yy = np.arange(y0, y1 + 1)
        score = vals.astype(np.float32) - 0.018 * np.abs(yy - prev)
        k = int(np.argmax(score))
        p = float(vals[k])
        if p < 0.22:
            gaps += 1
            continue
        chosen = int(yy[k])
        a, b = max(y0, chosen - 2), min(y1, chosen + 2)
        vv = prob[a:b + 1, x]
        ww = np.maximum(vv, 0)
        chosen_f = float((np.arange(a, b + 1) * ww).sum() / (ww.sum() + 1e-6))
        ys[j] = chosen_f
        conf[j] = p
        prev = chosen_f
        gaps = 0


def build_curve_track(prob: np.ndarray, left: int, right: int, baseline: int):
    h, _ = prob.shape
    left, right = int(left), int(right)
    xs = np.arange(left, right + 1, dtype=np.int32)
    ys = np.full(len(xs), np.nan, np.float32)
    conf = np.zeros(len(xs), np.float32)
    lower = min(h - 2, int(baseline) - 1)
    if lower < 5:
        return xs, ys, conf

    crop = prob[:lower + 1, left:right + 1]
    py, px = np.unravel_index(int(np.argmax(crop)), crop.shape)
    peak_p = float(crop[py, px])
    if peak_p < 0.45:
        return xs, ys, conf

    ys[px] = float(py)
    conf[px] = peak_p
    _walk_track(prob, xs, ys, conf, int(px), float(py), lower, +1)
    _walk_track(prob, xs, ys, conf, int(px), float(py), lower, -1)
    return xs, ys, conf


def smooth_array(v: np.ndarray) -> np.ndarray:
    finite = np.isfinite(v)
    if finite.sum() < 5:
        return v.copy()
    x = np.arange(len(v))
    a = np.interp(x, x[finite], v[finite]).astype(np.float32)
    k = min(15, max(5, (len(v) // 50) * 2 + 1))
    return cv2.medianBlur(a, k).astype(np.float32)


def genuine_intersections(xs, ys, y50, left, right):
    valid = np.isfinite(ys) & np.isfinite(xs) & (xs >= left) & (xs <= right)
    x = xs[valid].astype(float)
    y = ys[valid].astype(float)
    if len(x) < 3:
        return []

    d = y - float(y50)
    ints = []
    for i in range(len(x) - 1):
        if x[i + 1] - x[i] > 12:
            continue
        if abs(d[i]) <= 0.45:
            ints.append(float(x[i]))
        if d[i] * d[i + 1] < 0:
            t = abs(d[i]) / (abs(d[i]) + abs(d[i + 1]) + 1e-12)
            ints.append(float(x[i] + t * (x[i + 1] - x[i])))
    if len(d) and abs(d[-1]) <= 0.45:
        ints.append(float(x[-1]))

    ints.sort()
    out = []
    merge = max(2.0, 0.004 * max(1, right - left))
    for v in ints:
        if not out or v - out[-1] > merge:
            out.append(v)
    return out


def px_to_fl(x: float, left: int, right: int, xmin: float, xmax: float) -> float:
    if right <= left:
        return float("nan")
    return float(xmin + (float(x) - left) / (right - left) * (xmax - xmin))


def analyze_plt_image(model, device, image_input: Any, x_min_fl: float, x_max_fl: float, threshold: float = DEFAULT_THRESHOLD) -> dict:
    rgb = read_rgb(image_input)
    xmin, xmax = float(x_min_fl), float(x_max_fl)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        raise ValueError("X-axis maximum must be greater than X-axis minimum.")

    ref = detect_reference_lines(rgb)
    result = {
        "status": "error",
        "measurement_source": "V8 U-Net curve segmentation + geometric measurement",
        "axis_detection": "Automatic dashed ROI; X-axis values supplied by user",
        "x_min_fl": xmin,
        "x_max_fl": xmax,
        "reference_left_px": ref.get("left_x"),
        "reference_right_px": ref.get("right_x"),
        "reference_confidence": float(ref.get("confidence", 0.0)),
        "reference_method": ref.get("method"),
        "baseline_y_px": None,
        "peak_x_px": None,
        "peak_y_px": None,
        "peak_height_px": None,
        "y50_px": None,
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

    if not ref.get("ok"):
        result.update({"status": "roi_not_found", "warning": "Two reliable vertical dashed reference boundaries were not detected."})
        return result

    # BUG-PROOF: numeric conversion is applied to the actual dictionary values,
    # never to dictionary keys such as 'left_x'.
    try:
        left = int(ref["left_x"])
        right = int(ref["right_x"])
    except (KeyError, TypeError, ValueError) as exc:
        result.update({"status": "roi_not_found", "warning": f"Invalid detected ROI coordinates: {exc}"})
        return result

    if right - left < max(40, int(0.15 * rgb.shape[1])):
        result.update({"status": "roi_invalid", "warning": "Detected ROI is too narrow."})
        return result

    prob = predict_curve(model, device, rgb)
    mask = clean_curve_probability(prob, left, right, threshold)
    base = estimate_baseline(rgb, mask, left, right)
    xs, ys, cc = build_curve_track(prob, left, right, base)

    result.update({
        "baseline_y_px": base,
        "curve_probability": prob,
        "curve_mask": mask,
        "centerline_x": xs,
        "centerline_y": ys,
        "centerline_confidence": cc,
    })

    valid = np.isfinite(ys) & (ys < base - 1) & (xs >= left) & (xs <= right)
    if valid.sum() < 10:
        result.update({"status": "curve_not_found", "warning": "A reliable curve trace was not found inside the ROI."})
        return result

    x = xs[valid].astype(float)
    y = ys[valid].astype(float)
    c = cc[valid].astype(float)
    heights = base - y
    sm = smooth_array(heights)
    pi = int(np.nanargmax(sm))
    peak_x = float(x[pi])
    peak_y = float(y[pi])
    peak_h = float(base - peak_y)

    if peak_h < 3:
        result.update({"status": "peak_not_found", "warning": "Detected curve height is too small for a reliable 50% measurement."})
        return result

    y50 = float(base - 0.5 * peak_h)
    ints_px = genuine_intersections(xs, ys, y50, left, right)
    ints_fl = []
    for px in ints_px:
        f = px_to_fl(px, left, right, xmin, xmax)
        if np.isfinite(f):
            ints_fl.append(float(f))

    # Final hard geometry check: a reported x must coincide with the tracked
    # curve close to the 50% level. This prevents isolated segmentation pixels
    # from becoming fake intersections.
    checked = []
    for f in ints_fl:
        px = left + (f - xmin) / (xmax - xmin) * (right - left)
        k = int(np.argmin(np.abs(xs - px)))
        if np.isfinite(ys[k]) and abs(float(ys[k]) - y50) <= 3:
            checked.append(f)

    ints_fl = sorted(set(round(v, 6) for v in checked))
    status = "no_intersection" if len(ints_fl) == 0 else "single_intersection" if len(ints_fl) == 1 else "measure_width"
    curve_conf = float(np.nanmedian(c)) if np.isfinite(c).any() else 0.0

    result.update({
        "status": status,
        "peak_x_px": peak_x,
        "peak_y_px": peak_y,
        "peak_height_px": peak_h,
        "y50_px": y50,
        "intersections_px": ints_px,
        "intersections_fl": ints_fl,
        "intersection_count": len(ints_fl),
        "width_50_fl": (max(ints_fl) - min(ints_fl)) if len(ints_fl) >= 2 else None,
        "confidence": float(np.clip(0.45 * ref["confidence"] + 0.55 * curve_conf, 0, 1)),
    })

    if len(ints_fl) == 0:
        result["warning"] = "The traced curve does not genuinely cross the 50% level inside the ROI."
    elif len(ints_fl) == 1:
        result["warning"] = "Only one genuine 50% intersection was detected; width is not calculated."

    return result


def annotate_result(result: dict) -> np.ndarray:
    im = result["image_rgb"].copy()
    h, _ = im.shape[:2]
    left = result.get("reference_left_px")
    right = result.get("reference_right_px")

    if left is not None:
        cv2.line(im, (int(left), 0), (int(left), h - 1), (0, 255, 0), 2)
    if right is not None:
        cv2.line(im, (int(right), 0), (int(right), h - 1), (0, 255, 0), 2)
    if result.get("y50_px") is not None and left is not None and right is not None:
        y50 = int(round(result["y50_px"]))
        cv2.line(im, (int(left), y50), (int(right), y50), (255, 165, 0), 2)
    if result.get("peak_x_px") is not None:
        cv2.circle(im, (int(round(result["peak_x_px"])), int(round(result["peak_y_px"]))), 7, (255, 0, 255), -1)
    if result.get("y50_px") is not None:
        y50 = int(round(result["y50_px"]))
        for xx in result.get("intersections_px", []):
            cv2.circle(im, (int(round(xx)), y50), 7, (255, 0, 0), -1)
    return im


def make_result_table(result: dict) -> pd.DataFrame:
    ints = result.get("intersections_fl", [])
    xmin = result.get("x_min_fl")
    xmax = result.get("x_max_fl")
    peak_fl = None
    if result.get("peak_x_px") is not None and result.get("reference_left_px") is not None:
        peak_fl = px_to_fl(result["peak_x_px"], result["reference_left_px"], result["reference_right_px"], xmin, xmax)

    rows = [
        ("Status", result.get("status")),
        ("Measurement source", result.get("measurement_source")),
        ("Axis detection", result.get("axis_detection")),
        ("X-axis range", f"{xmin:g}–{xmax:g} fL"),
        ("ROI boundaries", f"{result.get('reference_left_px')}–{result.get('reference_right_px')} px"),
        ("Peak position", None if peak_fl is None else round(peak_fl, 3)),
        ("Selected Y level", "50% of detected peak height" if result.get("y50_px") is not None else None),
        ("Left 50% intersection", None if len(ints) < 1 else round(ints[0], 3)),
        ("Right 50% intersection", None if len(ints) < 2 else round(ints[-1], 3)),
        ("Width at 50%", None if result.get("width_50_fl") is None else round(result["width_50_fl"], 3)),
        ("Number of intersections", len(ints)),
        ("All intersections", "None" if not ints else ", ".join(f"{v:.3f}" for v in ints)),
        ("Confidence", f"{100 * result.get('confidence', 0):.1f}%"),
        ("Warning", result.get("warning") or ""),
    ]
    return pd.DataFrame(rows, columns=["Measurement", "Result"])
