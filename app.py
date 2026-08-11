from pathlib import Path

import gradio as gr
import pandas as pd
import torch

from model import (
    load_plt_model,
)

from analyzer import (
    analyze_plt_graph,
)


# -------------------------------------------------
# Configuration
# -------------------------------------------------

BASE_DIR = Path(
    __file__
).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "plt_curve_unet_fixed50_best.pt"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# -------------------------------------------------
# Load the trained model ONCE
# -------------------------------------------------

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model checkpoint was not found: "
        f"{MODEL_PATH}"
    )


print(
    "Loading PLT model..."
)

print(
    "Device:",
    DEVICE,
)

(
    model,
    checkpoint,
    DEFAULT_THRESHOLD,
) = load_plt_model(
    MODEL_PATH,
    DEVICE,
)

print(
    "Model loaded successfully."
)

print(
    "Default threshold:",
    DEFAULT_THRESHOLD,
)


# -------------------------------------------------
# Gradio prediction function
# -------------------------------------------------

def predict(
    image,
    x_axis_max,
    curve_sensitivity,
):

    if image is None:

        empty_table = (
            pd.DataFrame(
                [
                    [
                        "Status",
                        "No image uploaded",
                    ]
                ],
                columns=[
                    "Measurement",
                    "Result",
                ],
            )
        )

        return (
            None,
            empty_table,
            {
                "status":
                    "no_image"
            },
        )

    try:

        (
            annotated_image,
            result_table,
            result_json,
        ) = analyze_plt_graph(
            input_image=image,
            model=model,
            device=DEVICE,
            x_axis_max_fl=float(
                x_axis_max
            ),
            segmentation_threshold=float(
                curve_sensitivity
            ),
        )

        return (
            annotated_image,
            result_table,
            result_json,
        )

    except Exception as error:

        print(
            "Analysis error:",
            error,
        )

        error_table = (
            pd.DataFrame(
                [
                    [
                        "Status",
                        "Failed",
                    ],
                    [
                        "Error",
                        str(error),
                    ],
                ],
                columns=[
                    "Measurement",
                    "Result",
                ],
            )
        )

        return (
            None,
            error_table,
            {
                "status":
                    "failed",

                "error":
                    str(error),
            },
        )


# -------------------------------------------------
# Interface
# -------------------------------------------------

with gr.Blocks(
    title=(
        "PLT Histogram Analyzer"
    )
) as demo:

    gr.Markdown(
        """
# PLT Histogram Analyzer

Upload a cropped **PLT histogram graph** to detect where
the curve reaches the fixed **50% graph height**.

The system detects all genuine 50% intersections and
calculates the width when at least two intersections exist.

**Research prototype — not for medical diagnosis.**
"""
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label=(
                    "Upload cropped "
                    "PLT graph"
                ),
            )

            x_axis_max = (
                gr.Dropdown(
                    choices=[
                        "40",
                        "30",
                    ],
                    value="40",
                    label=(
                        "X-axis maximum "
                        "(fL)"
                    ),
                    info=(
                        "Choose the maximum "
                        "value shown on the "
                        "graph's X-axis."
                    ),
                )
            )

            curve_sensitivity = (
                gr.Slider(
                    minimum=0.15,
                    maximum=0.75,
                    value=(
                        DEFAULT_THRESHOLD
                    ),
                    step=0.01,
                    label=(
                        "Curve Detection "
                        "Sensitivity"
                    ),
                    info=(
                        "Lower values detect "
                        "fainter curve lines. "
                        "Higher values require "
                        "clearer curve lines."
                    ),
                )
            )

            analyze_button = (
                gr.Button(
                    "Analyze Graph",
                    variant="primary",
                )
            )

            clear_button = (
                gr.ClearButton(
                    [
                        input_image,
                    ]
                )
            )

        with gr.Column():

            annotated_output = (
                gr.Image(
                    type="pil",
                    label=(
                        "Detected Result"
                    ),
                )
            )

            result_table = (
                gr.Dataframe(
                    headers=[
                        "Measurement",
                        "Result",
                    ],
                    datatype=[
                        "str",
                        "str",
                    ],
                    interactive=False,
                    label=(
                        "Measurements"
                    ),
                )
            )

            result_json = gr.JSON(
                label=(
                    "Detailed Result"
                )
            )

    analyze_button.click(
        fn=predict,
        inputs=[
            input_image,
            x_axis_max,
            curve_sensitivity,
        ],
        outputs=[
            annotated_output,
            result_table,
            result_json,
        ],
        api_name="analyze",
    )


if __name__ == "__main__":

    demo.launch()