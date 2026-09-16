# PLT Histogram Analyzer - V8 Streamlit

This project updates the deployed Streamlit application to use the V8 PLT ROI Peak-50% pipeline.

## V8 measurement flow

1. Upload only the PLT graph image.
2. Detect the two vertical dashed reference boundaries.
3. Analyze only the image region between those boundaries.
4. Run the V8 U-Net curve segmentation model.
5. Estimate the plotting baseline.
6. Track one continuous curve from its strongest model response.
7. Select the highest point actually reached by that tracked curve inside the ROI.
8. Calculate the 50% level from that detected peak height.
9. Keep only genuine crossings of the traced curve with the 50% level.
10. Calculate width only when at least two genuine intersections exist.

There is no artificial left/right intersection creation.

## Project structure

```text
PLT-Histogram-Analyzer-V8/
├── app.py
├── model.py
├── analyzer.py
├── preprocessing.py
├── requirements.txt
├── README.md
└── models/
    └── plt_roi_peak50_unet_v8_best.pt
```

## 1. Replace the old project files

Copy these files into your existing Streamlit repository and replace the old versions:

- `app.py`
- `model.py`
- `analyzer.py`
- `preprocessing.py`
- `requirements.txt`

Create or update the `models/` folder.

## 2. Add the downloaded V8 checkpoint

Put your downloaded V8 best checkpoint here:

```text
models/plt_roi_peak50_unet_v8_best.pt
```

The loader also accepts the legacy filename `plt_curve_unet_fixed50_best.pt` as a fallback, but using the explicit V8 filename is recommended.

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Deploy on Streamlit

Commit/push the updated Python files, `requirements.txt`, and the checkpoint under `models/`.

In Streamlit deployment settings, use:

```text
app.py
```

No Colab/Google Drive code is required in the deployed application.

## 5. X-axis input

The current V8 model does not perform OCR for the numerical x-axis ticks. The user therefore enters the graph's x-axis minimum and maximum in the sidebar.

The detected dashed ROI is mapped linearly to that supplied range.

## 6. Important debugging behavior

The old ROI bug came from inconsistent reference-line return handling. V8 uses a single dictionary return structure:

```python
ref["left_x"]
ref["right_x"]
```

The code never attempts to convert the string key `"left_x"` into an integer. This prevents the former:

```text
ERROR: invalid literal for int() with base 10: 'left_x'
```

error.
