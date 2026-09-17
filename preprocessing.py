from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


# ============================================================
# IMAGE READING
# ============================================================

def read_rgb(
    image_input: Any,
) -> np.ndarray:
    """
    Convert supported image input formats into
    RGB uint8 HxWx3.
    """

    # --------------------------------------------------------
    # File path
    # --------------------------------------------------------

    if isinstance(
        image_input,
        (str, Path),
    ):

        bgr = cv2.imread(
            str(image_input),
            cv2.IMREAD_COLOR,
        )

        if bgr is None:

            raise FileNotFoundError(
                str(image_input)
            )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

    # --------------------------------------------------------
    # PIL image
    # --------------------------------------------------------

    elif isinstance(
        image_input,
        Image.Image,
    ):

        rgb = np.asarray(
            image_input.convert("RGB")
        )

    # --------------------------------------------------------
    # Raw bytes
    # --------------------------------------------------------

    elif isinstance(
        image_input,
        (bytes, bytearray),
    ):

        arr = np.frombuffer(
            image_input,
            dtype=np.uint8,
        )

        bgr = cv2.imdecode(
            arr,
            cv2.IMREAD_COLOR,
        )

        if bgr is None:

            raise ValueError(
                "The uploaded file could not "
                "be decoded as an image."
            )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

    # --------------------------------------------------------
    # File-like object
    # --------------------------------------------------------

    elif hasattr(
        image_input,
        "read",
    ):

        data = image_input.read()

        if hasattr(
            image_input,
            "seek",
        ):

            try:

                image_input.seek(0)

            except Exception:

                pass

        return read_rgb(data)

    # --------------------------------------------------------
    # NumPy
    # --------------------------------------------------------

    else:

        rgb = np.asarray(
            image_input
        )

        if rgb.ndim == 2:

            rgb = cv2.cvtColor(
                rgb.astype(np.uint8),
                cv2.COLOR_GRAY2RGB,
            )

        elif (
            rgb.ndim == 3
            and rgb.shape[2] == 4
        ):

            rgb = rgb[:, :, :3]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
    ):

        raise ValueError(
            "Expected an RGB image, "
            f"received shape {rgb.shape}."
        )

    return np.clip(
        rgb,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# RESIZE
# ============================================================

def resize_for_model(
    rgb: np.ndarray,
    image_size: int = 512,
) -> np.ndarray:

    return cv2.resize(
        rgb,
        (
            image_size,
            image_size,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# NORMALIZE
# ============================================================

def normalize_tensor(
    rgb: np.ndarray,
    image_size: int = 512,
):

    import torch

    resized = resize_for_model(
        rgb,
        image_size,
    )

    x = (
        resized.astype(np.float32)
        / 255.0
    )

    tensor = torch.from_numpy(
        x.transpose(2, 0, 1)
    ).unsqueeze(0).float()

    return tensor