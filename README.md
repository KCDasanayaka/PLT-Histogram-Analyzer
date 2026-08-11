# 📈 PLT Histogram Analyzer

A deep-learning-based application for analyzing PLT histogram
graphs and detecting where the graph curve reaches a fixed
50% of the plotting height.

The application uses a trained U-Net segmentation model to
identify the PLT histogram curve and deterministic geometry
to calculate intersection positions.

---

## Live Application

This project is designed to run using
**Streamlit Community Cloud**.

The application allows users to:

1. Upload a cropped PLT histogram graph.
2. Detect the histogram curve.
3. Locate the fixed 50% graph-height level.
4. Detect all genuine curve intersections.
5. Calculate X-axis intersection values in fL.
6. Calculate the width when two or more intersections exist.

---

## Measurement Logic

The vertical plotting area is interpreted as:

```text
100% ───────────────────────── Graph Top
 |
 |
 50% ───────────────────────── Fixed Measurement Level
 |
 |
 0%  ───────────────────────── X-axis Baseline
```

The 50% level is based on the plotting area.

It does **not** depend on the maximum height of the detected
curve.

---

## Result Logic

### No Intersection

If the curve does not reach the fixed 50% level:

```text
Status: no_intersection
Intersections: 0
Width: Not available
```

---

### One Intersection

If the curve reaches the fixed 50% level only once:

```text
Status: single_intersection
Intersections: 1
Width: Not available
```

---

### Two Intersections

If two intersections exist:

```text
Width =
Maximum X Intersection
-
Minimum X Intersection
```

---

### More Than Two Intersections

All genuine intersections are returned.

For example:

```text
Point 1 = 5.32 fL
Point 2 = 11.47 fL
Point 3 = 29.84 fL
```

The final width is:

```text
29.84 - 5.32
=
24.52 fL
```

Therefore:

```text
Width =
Maximum Intersection
-
Minimum Intersection
```

---

## Model

The application uses a U-Net-based semantic segmentation
network trained to identify the PLT histogram curve.

The trained model checkpoint is:

```text
models/plt_curve_unet_fixed50_best.pt
```

The neural network is responsible for curve segmentation.

The following operations are performed using deterministic
geometry:

- Graph boundary detection
- Fixed 50% level calculation
- Intersection detection
- X-coordinate conversion
- Width calculation

---

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

---

## Technologies

- Python
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Pandas
- SciPy
- Pillow

---

## Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project:

```bash
cd PLT-Histogram-Analyzer
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application should open in your browser.

Usually the local address is:

```text
http://localhost:8501
```

---

## Streamlit Community Cloud Deployment

The application can be deployed directly from GitHub.

### Step 1

Push the complete project to GitHub.

Make sure the repository contains:

```text
app.py
model.py
analyzer.py
preprocessing.py
requirements.txt
models/plt_curve_unet_fixed50_best.pt
```

### Step 2

Open Streamlit Community Cloud.

Sign in using your GitHub account.

### Step 3

Choose:

```text
Create app
```

Select the GitHub repository containing this project.

### Step 4

Set the main file path to:

```text
app.py
```

### Step 5

Deploy the application.

Streamlit will install the packages listed in:

```text
requirements.txt
```

and start the application automatically.

---

## Input Requirements

For better results, upload only the cropped PLT histogram
area.

The graph should preferably contain:

- Complete X-axis
- Complete Y-axis
- Full histogram curve
- Clear graph boundaries
- Minimal surrounding report text

Supported image formats include:

```text
PNG
JPG
JPEG
WEBP
```

---

## X-Axis Range

The application currently supports:

```text
0–40 fL
```

and

```text
0–30 fL
```

The correct range should be selected before analysis.

---

## Curve Detection Sensitivity

The Curve Detection Sensitivity controls the minimum
segmentation probability accepted as part of the curve.

Lower values:

```text
Detect faint lines
+
May detect additional noise
```

Higher values:

```text
Detect clearer curve pixels
+
May miss very faint parts
```

The default value is loaded from the trained model
checkpoint.

---

## Output

The application provides:

- Analysis status
- Annotated histogram
- Fixed 50% line
- Number of intersections
- Individual X intersection values
- Minimum intersection
- Maximum intersection
- Width at 50%
- Curve support information
- Detailed JSON result

---

## Model Loading

The trained PyTorch model is loaded using Streamlit's
resource caching.

This means the model is loaded when the application starts
and reused for subsequent predictions.

The model is **not retrained** when a user uploads a graph.

---

## Disclaimer

This project is a research prototype developed for automated
PLT histogram graph analysis.

It is not intended to provide medical diagnosis, treatment
recommendations, or clinical decisions.