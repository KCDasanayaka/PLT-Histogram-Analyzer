---
title: PLT Histogram Analyzer
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# PLT Histogram Analyzer

A deep-learning-based PLT histogram curve analysis system.

The application detects the PLT histogram curve and identifies
the X-axis locations where the curve reaches a fixed 50% of the
graph plotting height.

## Features

- PLT curve segmentation using a trained U-Net
- Fixed 50% graph-height measurement
- Automatic graph boundary detection
- Detection of multiple 50% intersections
- X-coordinate conversion to fL
- Width calculation when two or more intersections exist
- Support for 0–40 fL and 0–30 fL graphs
- Visual annotated output
- JSON result output

## Measurement Logic

The vertical plotting region is interpreted as:

```text
100% ───────────────────────── Graph top
 |
 |
 50% ───────────────────────── Fixed measurement level
 |
 |
 0%  ───────────────────────── X-axis baseline
```

The 50% measurement level does not depend on the highest point
of the detected curve.

### No intersection

If the curve never reaches the fixed 50% level:

```text
Status: no_intersection
Width: Not available
```

### One intersection

If the curve crosses the fixed 50% level only once:

```text
Status: single_intersection
Width: Not available
```

### Two intersections

If two intersections are detected:

```text
Width = Right Intersection - Left Intersection
```

### More than two intersections

Every genuine intersection is reported.

The final width is:

```text
Width = Maximum X Intersection - Minimum X Intersection
```

## Project Structure

```text
PLT-Histogram-Analyzer/
│
├── app.py
├── model.py
├── analyzer.py
├── preprocessing.py
├── requirements.txt
├── README.md
│
└── models/
    └── plt_curve_unet_fixed50_best.pt
```

## Model

The application uses a U-Net segmentation model trained to
separate the actual PLT histogram curve from graph axes,
reference lines, text and background noise.

The trained checkpoint is stored at:

```text
models/plt_curve_unet_fixed50_best.pt
```

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open the local Gradio URL shown in the terminal.

## Hugging Face Spaces

This repository can be deployed directly as a Gradio
Hugging Face Space.

Upload all project files and the trained checkpoint.

The Space will automatically build the environment and start
`app.py`.

## Input

Upload only the cropped PLT histogram region.

For the best results:

- Keep the X-axis visible.
- Keep the Y-axis visible.
- Keep the complete graph region.
- Avoid cropping through the graph boundaries.
- Use the clearest available scan or screenshot.

## Output

The application returns:

- Detection status
- Fixed 50% line
- Number of intersections
- All X intersection values
- Minimum X
- Maximum X
- Width at 50%
- Annotated graph
- Detailed JSON output

## Disclaimer

This application is a research prototype intended for
graph-analysis experiments.

It is not intended to provide medical diagnosis,
treatment recommendations, or clinical decisions.