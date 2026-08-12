# 📈 PLT Histogram Analyzer

A deep-learning-based application for analyzing
PLT histogram graphs.

The system uses a trained U-Net model to detect
the actual PLT distribution curve and then applies
geometric analysis to identify where the curve reaches
50% of its own peak height.

---

## Measurement Method

The current system does **not** use a fixed 50%
position on the graph Y-axis.

Instead, it first detects the PLT distribution.

```text
Graph Top
────────────────────────────

          Curve Peak
             ▲
             │
             │
Curve 50% ──────────────────
             │
             │
────────────────────────────
X-axis / Baseline
```

The curve peak represents 100% of the detected
distribution height.

The measurement level is:

```text
50% of detected curve peak height
```

---

## Low Distribution Rejection

Before calculating the curve's 50% level,
the system checks how high the distribution
peak is compared with the complete graph height.

The current minimum requirement is:

```text
Peak Height ≥ 50% of graph plotting height
```

If the peak is lower than this threshold:

```text
Status: no_intersection
50% Line: Not generated
Intersections: 0
Width: Not available
```

This prevents very low PLT distributions from
producing misleading half-peak measurements.

---

## Intersection Logic

### No Intersection

```text
Intersections = 0
Width = Not available
```

### One Intersection

```text
Intersections = 1
Width = Not available
```

### Two Intersections

```text
Width =
Right X - Left X
```

### More Than Two Intersections

All genuine intersections are reported.

For example:

```text
Point 1 = 4.82 fL
Point 2 = 10.75 fL
Point 3 = 31.40 fL
```

The width is:

```text
31.40 - 4.82
=
26.58 fL
```

Therefore:

```text
Width =
Maximum X intersection
-
Minimum X intersection
```

---

## Model Architecture

The application uses a U-Net semantic
segmentation model.

The neural network is responsible only for:

```text
Input PLT Graph
        ↓
Curve Segmentation
        ↓
Detected PLT Curve
```

The model does not directly predict the
50% intersection values.

---

## Post-processing

After segmentation, deterministic geometry
is used for:

- Graph boundary detection
- Curve tracing
- Peak detection
- Low-distribution rejection
- Half-peak calculation
- Intersection detection
- X-axis conversion to fL
- Width calculation

---

## Current Measurement Pipeline

```text
Upload PLT Graph
        ↓
U-Net Curve Segmentation
        ↓
Remove Axes / Reference Lines
        ↓
Trace PLT Distribution Curve
        ↓
Detect Distribution Peak
        ↓
Calculate Peak Height Relative to Graph
        ↓
Is Peak ≥ 50% of Graph Height?
       / \
     NO   YES
     ↓      ↓
No         Calculate 50%
Measurement of Curve Peak
            ↓
       Find All Genuine
       Intersections
            ↓
       Convert X → fL
            ↓
       Calculate Width
       if 2+ points
```

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

## Important Note About Model Filename

The checkpoint currently retains the filename:

```text
plt_curve_unet_fixed50_best.pt
```

However, the U-Net checkpoint is used only
for curve segmentation.

The current 50% measurement calculation is
performed inside:

```text
analyzer.py
```

and uses:

```text
50% of detected curve peak
```

rather than a fixed 50% Y-axis position.

Therefore retraining the segmentation model
is not required for this measurement-logic change.

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

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run app.py
```

---

## Streamlit Deployment

The project can be deployed using
Streamlit Community Cloud.

Connect the GitHub repository and choose:

```text
app.py
```

as the main application file.

Streamlit will automatically install
the packages listed in:

```text
requirements.txt
```

and start the application.

---

## Input Requirements

Upload a cropped PLT histogram.

For better detection, the image should contain:

- Complete X-axis
- Complete Y-axis
- Complete histogram distribution
- Clear curve
- Minimal surrounding report content

Supported formats:

```text
PNG
JPG
JPEG
WEBP
```

---

## X-Axis Support

The application currently supports:

```text
0–40 fL
```

and:

```text
0–30 fL
```

The correct range should be selected before
performing analysis.

---

## Curve Detection Sensitivity

The Curve Detection Sensitivity controls how
confident the U-Net must be before a pixel is
accepted as part of the PLT curve.

Lower values detect fainter curves but may
include additional noise.

Higher values require clearer curve pixels
but may miss faint curve sections.

---

## Output

The system provides:

- Distribution status
- Detected peak height
- Half-peak measurement level
- Number of genuine intersections
- Individual intersection X values
- Minimum X
- Maximum X
- Width at 50%
- Annotated graph
- Curve support information
- Detailed JSON result

---

## Disclaimer

This application is a research prototype for
automated PLT histogram graph analysis.

It is not intended for medical diagnosis,
treatment recommendations, or clinical decisions.