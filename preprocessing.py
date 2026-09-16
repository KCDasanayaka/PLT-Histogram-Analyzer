# ============================================================
# preprocessing.py
# ============================================================

import cv2
import numpy as np
import torch


# ============================================================
# Convert input to RGB NumPy image
# ============================================================

def read_rgb_image(image):

    if image is None:

        raise ValueError(
            "No image was provided."
        )

    # PIL Image
    if hasattr(image, "convert"):

        image = image.convert(
            "RGB"
        )

        rgb = np.asarray(
            image
        ).copy()

    # NumPy image
    else:

        rgb = np.asarray(
            image
        ).copy()

        if rgb.ndim != 3:

            raise ValueError(
                "Expected an RGB image."
            )

        if rgb.shape[2] == 4:

            rgb = rgb[:, :, :3]

    if rgb.dtype != np.uint8:

        rgb = np.clip(
            rgb,
            0,
            255
        ).astype(
            np.uint8
        )

    return rgb


# ============================================================
# Prepare input tensor
# ============================================================

def prepare_input(
    rgb,
    image_size=512
):

    resized = cv2.resize(
        rgb,
        (
            image_size,
            image_size
        ),
        interpolation=cv2.INTER_AREA
    )

    normalized = (
        resized.astype(
            np.float32
        ) / 255.0
    )

    tensor = torch.from_numpy(
        normalized.transpose(
            2,
            0,
            1
        )
    ).unsqueeze(
        0
    ).float()

    return tensor


# ============================================================
# Restore prediction to original dimensions
# ============================================================

def restore_probability(
    probability,
    width,
    height
):

    return cv2.resize(
        probability,
        (
            width,
            height
        ),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# Convert probability to mask
# ============================================================

def probability_to_mask(
    probability,
    threshold=0.35
):

    mask = (
        probability >= threshold
    ).astype(
        np.uint8
    )

    kernel = np.ones(
        (
            3,
            3
        ),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask