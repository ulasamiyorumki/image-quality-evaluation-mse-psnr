# Image Quality Evaluation Using MSE and PSNR

## Project Overview

This project evaluates image quality by applying controlled degradation methods to built-in sample images and comparing the processed images with the originals using Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR).

The project includes an interactive Streamlit dashboard. Users can select one of six sample images provided by `scikit-image`, choose a degradation method, adjust its parameters, and view the processed image and quality metrics dynamically.

## Objectives

- Implement common image degradation methods.
- Calculate MSE and PSNR between original and processed images.
- Run experiments at different degradation levels.
- Display image comparisons, metrics, tables, and charts in an interactive dashboard.

## Image Degradation Methods

The system supports three degradation methods:

1. **Gaussian Noise** — adds adjustable random noise to an image.
2. **Gaussian Blur** — applies adjustable blur to reduce fine details.
3. **JPEG Compression** — compresses an image at a selected quality level.

Each processed image is generated from the original image. Degradation methods are not compounded unless a future project requirement explicitly adds that feature.

## Image Quality Metrics

- **Mean Squared Error (MSE):** Measures the average squared pixel difference between the original and processed images. Lower values generally indicate smaller pixel-level differences.
- **Peak Signal-to-Noise Ratio (PSNR):** Expresses image fidelity in decibels (dB). Higher values generally indicate smaller pixel-level errors. Identical images have MSE = 0 and PSNR = infinity.

These metrics measure pixel-level differences and do not fully represent every aspect of perceived visual quality.

## Interactive Dashboard

The Streamlit dashboard allows users to:

- Choose from six built-in sample images provided by `scikit-image`.
- Select Gaussian Noise, Gaussian Blur, or JPEG Compression.
- Adjust degradation parameters interactively.
- View original and processed images side by side.
- View MSE and PSNR values.
- Review experiment results and charts that update based on the selected image and settings.

## Project Structure

```text
image-quality-evaluation-mse-psnr/
├── app.py
├── requirements.txt
├── README.md
├── INTERFACE_CONTRACT.md
└── src/
    ├── __init__.py
    ├── image_degradation.py
    ├── metrics.py
    └── evaluation.py
```

The sample images are provided by `scikit-image` at runtime, so a separate image data folder is not required.

## Technologies

- Python
- Streamlit
- NumPy
- OpenCV
- Pandas
- Matplotlib
- scikit-image

## Installation and Setup

### macOS / Linux

```bash
git clone https://github.com/YOUR_USERNAME/image-quality-evaluation-mse-psnr.git
cd image-quality-evaluation-mse-psnr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows — Command Prompt

```bat
git clone https://github.com/YOUR_USERNAME/image-quality-evaluation-mse-psnr.git
cd image-quality-evaluation-mse-psnr
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Windows — PowerShell

```powershell
git clone https://github.com/YOUR_USERNAME/image-quality-evaluation-mse-psnr.git
cd image-quality-evaluation-mse-psnr
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Replace `YOUR_USERNAME` with the GitHub account that owns the repository.

## Running the Dashboard

With the virtual environment activated, run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL to open the dashboard in a browser.

## Experimental Workflow

1. Select one of the built-in sample images.
2. Select a degradation method.
3. Adjust the relevant parameters.
4. Generate a processed image from the original.
5. Calculate MSE and PSNR.
6. Display the original and processed images side by side with their metric values.
7. Use the evaluation module to compare multiple parameter settings in a Pandas DataFrame and display the results in the dashboard.

## Team Responsibilities

- **Person 1 — Image Degradation:** Implement Gaussian noise, Gaussian blur, and JPEG compression in `src/image_degradation.py`.
- **Person 2 — Metrics:** Implement MSE and PSNR in `src/metrics.py`.
- **Person 3 — Evaluation:** Integrate degradation and metric functions, run experiments across parameter levels, and organize results in a Pandas DataFrame. CSV export is out of scope.
- **Person 4 — Dashboard:** Develop the Streamlit interface, built-in sample image selection, interactive controls, metric display, tables, and charts.

## Scope and Limitations

The project focuses on controlled image degradation and pixel-based quality evaluation. MSE and PSNR are useful quantitative measures but may not always match human perception. Dashboard results depend on the selected sample image and degradation parameters.

The current version uses built-in `scikit-image` sample images, displays results dynamically, and does not implement user image upload or CSV export.
