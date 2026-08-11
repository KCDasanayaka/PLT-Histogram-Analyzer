import cv2
import numpy as np
import torch

from PIL import Image

from scipy.signal import (
    savgol_filter,
)

from scipy.ndimage import (
    median_filter,
)


IMAGE_HEIGHT = 192
IMAGE_WIDTH = 288


def resize_and_pad_image(
    image,
    output_width=IMAGE_WIDTH,
    output_height=IMAGE_HEIGHT,
):

    original_height, original_width = (
        image.shape[:2]
    )

    scale = min(
        output_width
        / original_width,
        output_height
        / original_height,
    )

    resized_width = max(
        1,
        int(
            round(
                original_width
                * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                original_height
                * scale
            )
        ),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_CUBIC
    )

    resized_image = cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=interpolation,
    )

    canvas = np.full(
        (
            output_height,
            output_width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    x_offset = (
        output_width
        - resized_width
    ) // 2

    y_offset = (
        output_height
        - resized_height
    ) // 2

    canvas[
        y_offset:
        y_offset + resized_height,

        x_offset:
        x_offset + resized_width,
    ] = resized_image

    transform_info = {
        "x_offset":
            x_offset,

        "y_offset":
            y_offset,

        "resized_width":
            resized_width,

        "resized_height":
            resized_height,

        "original_width":
            original_width,

        "original_height":
            original_height,
    }

    return (
        canvas,
        transform_info,
    )


def prepare_inference_image(
    input_image,
    device,
):

    if input_image is None:
        raise ValueError(
            "Please upload a PLT graph."
        )

    if isinstance(
        input_image,
        Image.Image,
    ):

        image_rgb = np.array(
            input_image.convert(
                "RGB"
            )
        )

    else:

        image_rgb = np.asarray(
            input_image
        )

        if image_rgb.ndim == 2:

            image_rgb = (
                cv2.cvtColor(
                    image_rgb,
                    cv2.COLOR_GRAY2RGB,
                )
            )

        if (
            image_rgb.ndim == 3
            and image_rgb.shape[2]
            == 4
        ):

            image_rgb = (
                cv2.cvtColor(
                    image_rgb,
                    cv2.COLOR_RGBA2RGB,
                )
            )

        image_rgb = (
            image_rgb.astype(
                np.uint8
            )
        )

    height, width = (
        image_rgb.shape[:2]
    )

    if (
        height < 60
        or width < 90
    ):

        raise ValueError(
            "The uploaded image is "
            "too small. Upload a "
            "clear cropped PLT graph."
        )

    resized_image, transform_info = (
        resize_and_pad_image(
            image_rgb
        )
    )

    tensor = (
        torch.from_numpy(
            resized_image.transpose(
                2,
                0,
                1,
            )
        )
        .float()
        / 255.0
    )

    tensor = (
        tensor - 0.5
    ) / 0.5

    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )

    return (
        image_rgb,
        tensor,
        transform_info,
    )


def restore_probability_mask(
    padded_mask,
    transform_info,
):

    x_offset = (
        transform_info[
            "x_offset"
        ]
    )

    y_offset = (
        transform_info[
            "y_offset"
        ]
    )

    resized_width = (
        transform_info[
            "resized_width"
        ]
    )

    resized_height = (
        transform_info[
            "resized_height"
        ]
    )

    original_width = (
        transform_info[
            "original_width"
        ]
    )

    original_height = (
        transform_info[
            "original_height"
        ]
    )

    cropped = padded_mask[
        y_offset:
        y_offset + resized_height,

        x_offset:
        x_offset + resized_width,
    ]

    restored = cv2.resize(
        cropped,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return np.clip(
        restored,
        0.0,
        1.0,
    )


def longest_horizontal_segment(
    binary_row,
):

    best_start = None
    best_end = None
    current_start = None

    for index, value in enumerate(
        binary_row
    ):

        if (
            value
            and current_start
            is None
        ):
            current_start = index

        if (
            (
                not value
                or index
                == len(binary_row) - 1
            )
            and current_start
            is not None
        ):

            if (
                value
                and index
                == len(binary_row) - 1
            ):
                end = index
            else:
                end = index - 1

            if (
                best_start is None
                or (
                    end
                    - current_start
                    > best_end
                    - best_start
                )
            ):

                best_start = (
                    current_start
                )

                best_end = end

            current_start = None

    return (
        best_start,
        best_end,
    )


def detect_plot_bounds(
    image_rgb,
):

    height, width = (
        image_rgb.shape[:2]
    )

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    inverted = (
        cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            9,
        )
    )

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    25,
                    width // 4,
                ),
                1,
            ),
        )
    )

    horizontal = (
        cv2.morphologyEx(
            inverted,
            cv2.MORPH_OPEN,
            horizontal_kernel,
        )
    )

    row_scores = (
        horizontal.sum(
            axis=1
        )
    )

    candidate_rows = np.arange(
        int(
            height * 0.48
        ),
        height,
    )

    baseline_y = int(
        candidate_rows[
            np.argmax(
                row_scores[
                    candidate_rows
                ]
            )
        ]
    )

    row_binary = (
        horizontal[
            max(
                0,
                baseline_y - 1,
            ):
            min(
                height,
                baseline_y + 2,
            )
        ]
        .max(axis=0)
        > 0
    )

    (
        x_left_horizontal,
        x_right_horizontal,
    ) = longest_horizontal_segment(
        row_binary
    )

    if (
        x_left_horizontal is None
        or x_right_horizontal is None
        or (
            x_right_horizontal
            - x_left_horizontal
            < width * 0.30
        )
    ):

        x_left_horizontal = int(
            width * 0.06
        )

        x_right_horizontal = int(
            width * 0.92
        )

        baseline_ok = False

    else:

        baseline_ok = True

    vertical_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                max(
                    20,
                    height // 4,
                ),
            ),
        )
    )

    vertical = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    search_left = max(
        0,
        int(
            x_left_horizontal
            - width * 0.08
        ),
    )

    search_right = min(
        width,
        int(
            x_left_horizontal
            + width * 0.12
        ),
    )

    column_scores = (
        vertical[
            :,
            search_left:
            search_right,
        ]
        .sum(axis=0)
    )

    if (
        column_scores.size > 0
        and np.max(
            column_scores
        ) > 0
    ):

        x_left = (
            search_left
            + int(
                np.argmax(
                    column_scores
                )
            )
        )

        column_binary = (
            vertical[
                :baseline_y + 1,
                x_left,
            ]
            > 0
        )

        y_positions = np.where(
            column_binary
        )[0]

        if len(
            y_positions
        ) >= 5:

            y_top = int(
                y_positions.min()
            )

            y_axis_ok = True

        else:

            y_top = int(
                height * 0.18
            )

            y_axis_ok = False

    else:

        x_left = int(
            x_left_horizontal
        )

        y_top = int(
            height * 0.18
        )

        y_axis_ok = False

    x_right = int(
        x_right_horizontal
    )

    if (
        baseline_y
        - y_top
        < height * 0.25
    ):

        y_top = max(
            2,
            int(
                baseline_y
                - height * 0.55
            ),
        )

        y_axis_ok = False

    if (
        x_right
        - x_left
        < width * 0.35
    ):

        x_left = int(
            width * 0.06
        )

        x_right = int(
            width * 0.92
        )

        baseline_ok = False

    fixed_y50 = (
        y_top
        + baseline_y
    ) / 2.0

    confidence = float(
        np.mean(
            [
                baseline_ok,
                y_axis_ok,
            ]
        )
    )

    return {
        "x_left":
            int(x_left),

        "x_right":
            int(x_right),

        "y_top":
            int(y_top),

        "baseline_y":
            int(baseline_y),

        "fixed_y50":
            float(fixed_y50),

        "automatic":
            bool(
                baseline_ok
                and y_axis_ok
            ),

        "confidence":
            confidence,
    }


def find_true_runs(
    boolean_array,
):

    boolean_array = np.asarray(
        boolean_array,
        dtype=bool,
    )

    if len(
        boolean_array
    ) == 0:

        return []

    padded = np.concatenate(
        [
            [False],
            boolean_array,
            [False],
        ]
    )

    differences = np.diff(
        padded.astype(
            np.int8
        )
    )

    starts = np.where(
        differences == 1
    )[0]

    ends = (
        np.where(
            differences == -1
        )[0]
        - 1
    )

    return list(
        zip(
            starts.tolist(),
            ends.tolist(),
        )
    )


def remove_vertical_artifacts(
    binary_mask,
    plot_bounds,
):

    cleaned = (
        binary_mask
        .copy()
        .astype(
            np.uint8
        )
    )

    x_left = (
        plot_bounds[
            "x_left"
        ]
    )

    x_right = (
        plot_bounds[
            "x_right"
        ]
    )

    y_top = (
        plot_bounds[
            "y_top"
        ]
    )

    baseline_y = (
        plot_bounds[
            "baseline_y"
        ]
    )

    plot_height = max(
        1,
        baseline_y - y_top,
    )

    cleaned[
        :,
        max(
            0,
            x_left - 3,
        ):
        min(
            cleaned.shape[1],
            x_left + 5,
        ),
    ] = 0

    cleaned[
        max(
            0,
            baseline_y - 2,
        ):
        ,
        :
    ] = 0

    vertical_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                max(
                    8,
                    int(
                        plot_height
                        * 0.28
                    ),
                ),
            ),
        )
    )

    vertical_lines = (
        cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            vertical_kernel,
        )
    )

    vertical_lines = cv2.dilate(
        vertical_lines,
        np.ones(
            (3, 3),
            np.uint8,
        ),
        iterations=1,
    )

    cleaned[
        vertical_lines > 0
    ] = 0

    region_mask = np.zeros_like(
        cleaned
    )

    region_mask[
        y_top:
        baseline_y,

        x_left:
        x_right + 1,
    ] = 1

    cleaned *= region_mask

    return cleaned


def fill_small_nan_gaps(
    values,
    maximum_gap,
):

    values = values.copy()

    valid = np.where(
        np.isfinite(
            values
        )
    )[0]

    if len(valid) < 2:
        return values

    for left, right in zip(
        valid[:-1],
        valid[1:],
    ):

        gap = (
            right
            - left
            - 1
        )

        if (
            gap > 0
            and gap
            <= maximum_gap
        ):

            values[
                left:
                right + 1
            ] = np.linspace(
                values[left],
                values[right],
                gap + 2,
            )

    return values


def trace_curve_centerline(
    probability_mask,
    plot_bounds,
    segmentation_threshold,
):

    x_left = (
        plot_bounds[
            "x_left"
        ]
    )

    x_right = (
        plot_bounds[
            "x_right"
        ]
    )

    y_top = (
        plot_bounds[
            "y_top"
        ]
    )

    baseline_y = (
        plot_bounds[
            "baseline_y"
        ]
    )

    binary = (
        probability_mask
        >= segmentation_threshold
    ).astype(
        np.uint8
    )

    binary = (
        remove_vertical_artifacts(
            binary,
            plot_bounds,
        )
    )

    x_values = np.arange(
        x_left,
        x_right + 1,
        dtype=np.float32,
    )

    centerline = np.full(
        len(x_values),
        np.nan,
        dtype=np.float32,
    )

    support = np.zeros(
        len(x_values),
        dtype=np.float32,
    )

    for local_index, x_pixel in enumerate(
        range(
            x_left,
            x_right + 1,
        )
    ):

        y_candidates = (
            np.where(
                binary[
                    y_top:
                    baseline_y,
                    x_pixel,
                ]
                > 0
            )[0]
            + y_top
        )

        if len(
            y_candidates
        ) == 0:
            continue

        probabilities = (
            probability_mask[
                y_candidates,
                x_pixel,
            ]
        )

        total_probability = float(
            probabilities.sum()
        )

        if total_probability <= 0:
            continue

        cumulative = np.cumsum(
            probabilities
        )

        median_index = int(
            np.searchsorted(
                cumulative,
                cumulative[-1]
                / 2.0,
            )
        )

        median_index = int(
            np.clip(
                median_index,
                0,
                len(
                    y_candidates
                ) - 1,
            )
        )

        centerline[
            local_index
        ] = float(
            y_candidates[
                median_index
            ]
        )

        support[
            local_index
        ] = float(
            probabilities.mean()
        )

    maximum_gap = max(
        2,
        int(
            round(
                len(centerline)
                * 0.025
            )
        ),
    )

    centerline = (
        fill_small_nan_gaps(
            centerline,
            maximum_gap,
        )
    )

    valid_runs = (
        find_true_runs(
            np.isfinite(
                centerline
            )
        )
    )

    minimum_run = max(
        5,
        int(
            round(
                len(centerline)
                * 0.025
            )
        ),
    )

    for start, end in valid_runs:

        run_length = (
            end
            - start
            + 1
        )

        if (
            run_length
            < minimum_run
        ):

            centerline[
                start:
                end + 1
            ] = np.nan

            continue

        run = centerline[
            start:
            end + 1
        ]

        if run_length >= 5:

            run = median_filter(
                run,
                size=3,
                mode="nearest",
            )

        if run_length >= 9:

            window = min(
                15,
                (
                    run_length
                    if run_length % 2 == 1
                    else run_length - 1
                ),
            )

            if window >= 5:

                run = savgol_filter(
                    run,
                    window_length=window,
                    polyorder=2,
                    mode="interp",
                )

        centerline[
            start:
            end + 1
        ] = run

    valid_count = int(
        np.isfinite(
            centerline
        ).sum()
    )

    support_ratio = (
        valid_count
        / max(
            len(centerline),
            1,
        )
    )

    valid = (
        valid_count
        >= max(
            8,
            int(
                len(centerline)
                * 0.08
            ),
        )
    )

    return {
        "valid":
            valid,

        "x":
            x_values,

        "y":
            centerline,

        "support":
            support,

        "support_ratio":
            support_ratio,

        "cleaned_mask":
            binary,
    }