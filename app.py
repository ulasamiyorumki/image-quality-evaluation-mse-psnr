# Implements an interactive Streamlit dashboard for image quality evaluation.
# Displays original and processed images side by side.
# Provides sliders to select degradation methods and adjust their intensity.
# Displays MSE and PSNR metric cards and comparison charts.
# Integrates the image degradation, metrics, and evaluation modules.
# Updates images and metrics dynamically based on user inputs.
# Includes explanations of every section, parameter, metric, and chart.

import streamlit as st
import numpy as np
import altair as alt

from skimage import data
from skimage.color import gray2rgb
from skimage.util import img_as_ubyte

from src.image_degradation import (
    add_gaussian_noise,
    apply_gaussian_blur,
    apply_jpeg_compression,
)
from src.metrics import calculate_mse, calculate_psnr
from src.evaluation import run_evaluation


st.set_page_config(
    page_title="Image Quality Evaluation",
    page_icon="🖼️",
    layout="wide",
)

st.title("Image Quality Evaluation")
st.write(
    "Evaluate image degradation using MSE and PSNR metrics."
)

with st.expander("ℹ️ How to use this dashboard"):
    st.markdown(
        """
This dashboard shows how different types of image degradation affect
measurable image quality.

1. **Choose a sample image** in the sidebar. All images are built into
   scikit-image, so no upload is needed.
2. **Choose a degradation method**: Gaussian Noise, Gaussian Blur, or
   JPEG Compression.
3. **Adjust the parameters** with the sliders. The processed image is
   always created from the original image, so effects are never stacked.
4. **Compare the results**: the original and processed images are shown
   side by side, followed by the MSE and PSNR values and an analysis of
   several degradation levels.

Every section has its own ℹ️ panel explaining what it shows.
        """
    )

st.divider()


# Image preparation

def prepare_image(image):
    image = np.asarray(image)

    if image.ndim == 2:
        image = gray2rgb(image)

    if image.dtype != np.uint8:
        image = img_as_ubyte(image)

    return np.ascontiguousarray(image)


# Sample images from scikit-image

SAMPLE_IMAGES = {
    "Astronaut": data.astronaut,
    "Coffee": data.coffee,
    "Chelsea the Cat": data.chelsea,
    "Rocket": data.rocket,
    "Hubble Deep Field": data.hubble_deep_field,
    "Immunohistochemistry": data.immunohistochemistry,
}


@st.cache_data
def load_sample_image(name):
    return prepare_image(SAMPLE_IMAGES[name]())


# Sidebar controls

st.sidebar.header("Experiment Settings")

image_name = st.sidebar.selectbox(
    "Select a sample image",
    list(SAMPLE_IMAGES.keys()),
    help=(
        "Built-in test images from scikit-image. Images with many fine "
        "details and edges react more strongly to blur and compression "
        "than smooth images, so the same settings can give different "
        "MSE and PSNR values on different images."
    ),
)

original_image = load_sample_image(image_name)

method_label = st.sidebar.selectbox(
    "Degradation Method",
    ["Gaussian Noise", "Gaussian Blur", "JPEG Compression"],
    help=(
        "The type of distortion applied to the original image. "
        "See the method explanation under the image comparison for details."
    ),
)

if method_label == "Gaussian Noise":
    method = "gaussian_noise"

    sigma = st.sidebar.slider(
        "Noise Intensity (Sigma)",
        min_value=0,
        max_value=80,
        value=20,
        step=5,
        help=(
            "Standard deviation (σ) of the added noise. Higher values "
            "produce stronger, grainier noise. With σ = 0 and mean = 0, "
            "the image is unchanged."
        ),
    )

    mean = st.sidebar.slider(
        "Noise Mean",
        min_value=-20,
        max_value=20,
        value=0,
        help=(
            "Average value (μ) of the added noise. A positive mean makes the "
            "image brighter overall, a negative mean makes it darker. "
            "Non-zero values increase MSE even when σ is 0."
        ),
    )

    parameters = {"mean": mean, "sigma": sigma}

elif method_label == "Gaussian Blur":
    method = "gaussian_blur"

    kernel_size = st.sidebar.select_slider(
        "Blur Kernel Size",
        options=[3, 5, 7, 9, 11, 15, 21],
        value=5,
        help=(
            "Width and height of the blur window in pixels. It must be an "
            "odd number so the window has a center pixel. A larger window "
            "lets the blur reach further, but only if sigma is large enough."
        ),
    )

    sigma = st.sidebar.slider(
        "Blur Sigma",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help=(
            "Spread (σ) of the Gaussian weights. Higher values give more "
            "weight to distant neighbours and make the image softer."
        ),
    )

    parameters = {
        "kernel_size": kernel_size,
        "sigma": sigma,
    }

else:
    method = "jpeg_compression"

    quality = st.sidebar.slider(
        "JPEG Quality",
        min_value=1,
        max_value=100,
        value=75,
        help=(
            "JPEG quality factor from 1 to 100. Lower values compress the "
            "image more strongly, which reduces file size but removes more "
            "detail and causes visible block artifacts."
        ),
    )

    parameters = {"quality": quality}


# Process the original image and calculate metrics

if method == "gaussian_noise":
    processed_image = add_gaussian_noise(
        original_image.copy(),
        **parameters,
    )

elif method == "gaussian_blur":
    processed_image = apply_gaussian_blur(
        original_image.copy(),
        **parameters,
    )

else:
    processed_image = apply_jpeg_compression(
        original_image.copy(),
        **parameters,
    )

mse_value = calculate_mse(original_image, processed_image)
psnr_value = calculate_psnr(original_image, processed_image)


# Original and processed images

st.subheader("Image Comparison")

original_column, processed_column = st.columns(2)

with original_column:
    st.markdown("**Original Image**")
    st.image(original_image, width="stretch")

with processed_column:
    st.markdown("**Processed Image**")
    st.image(processed_image, width="stretch")

st.caption(f"Selected method: {method_label} | Parameters: {parameters}")

with st.expander(f"ℹ️ What does {method_label} do?"):
    if method == "gaussian_noise":
        st.markdown(
            """
**Gaussian noise** adds a random value to every pixel. The random values
follow a normal (Gaussian) distribution. This simulates sensor noise,
for example from a camera in low light or a poor transmission signal.
            """
        )
        st.latex(r"I_{\text{noisy}}(x, y) = I(x, y) + n(x, y), \qquad n(x, y) \sim \mathcal{N}(\mu, \sigma^2)")
        st.markdown(
            """
- $I(x, y)$: original pixel value at position $(x, y)$
- $n(x, y)$: random noise value for that pixel
- $\\mu$ (**Noise Mean**): average of the noise; shifts overall brightness
- $\\sigma$ (**Noise Intensity**): standard deviation; controls how strong the noise is

After adding the noise, pixel values are kept in the valid 0–255 range.
Because the noise is random, MSE and PSNR can change slightly each time
the image is processed with the same settings.
            """
        )

    elif method == "gaussian_blur":
        st.markdown(
            """
**Gaussian blur** replaces every pixel with a weighted average of its
neighbours. Closer neighbours get higher weights. This removes fine
details and softens edges, similar to an out-of-focus camera.
The weights come from the 2D Gaussian function:
            """
        )
        st.latex(r"G(x, y) = \frac{1}{2\pi\sigma^2} \, e^{-\frac{x^2 + y^2}{2\sigma^2}}")
        st.markdown("The blurred image is the convolution of the image with this kernel:")
        st.latex(r"I_{\text{blur}} = I * G")
        st.markdown(
            """
- **Kernel Size**: the size of the window ($k \\times k$ pixels) that is averaged
- **Blur Sigma** ($\\sigma$): how quickly the weights decrease with distance

Both parameters work together. Almost all of the Gaussian's weight lies
within about $3\\sigma$ of the center, so with a small sigma, increasing
the kernel size has little visible effect.
            """
        )

    else:
        st.markdown(
            """
**JPEG compression** is a lossy compression method used for most photos.
The image is split into 8×8 pixel blocks, each block is transformed with
the Discrete Cosine Transform (DCT), and the resulting frequency values
are divided by a quantization table and rounded:
            """
        )
        st.latex(r"F_Q(u, v) = \operatorname{round}\left(\frac{F(u, v)}{Q(u, v)}\right)")
        st.markdown(
            """
- $F(u, v)$: DCT coefficient (frequency content of the block)
- $Q(u, v)$: quantization step; larger when **JPEG Quality** is lower

Rounding throws information away, mostly fine details (high frequencies)
that the human eye notices less. Low quality settings cause visible
**block artifacts** and color banding. Even at quality 100 the result is
usually not identical to the original, because rounding still occurs.
            """
        )


# MSE and PSNR metric cards

st.divider()
st.subheader("Image Quality Metrics")

mse_column, psnr_column = st.columns(2)

with mse_column:
    st.metric("Mean Squared Error (MSE)", f"{mse_value:.2f}")

    with st.expander("ℹ️ What is MSE?"):
        st.markdown(
            """
**Mean Squared Error** is the average of the squared differences between
the pixels of the original and the processed image:
            """
        )
        st.latex(r"\text{MSE} = \frac{1}{M \cdot N \cdot C} \sum_{c=1}^{C} \sum_{i=1}^{M} \sum_{j=1}^{N} \left( I(i, j, c) - K(i, j, c) \right)^2")
        st.markdown(
            """
- $I$: original image, $K$: processed image
- $M \\times N$: image height and width in pixels
- $C$: number of color channels (3 for RGB)

**How to read it:**
- **MSE = 0** means the two images are identical.
- **Lower is better.** Larger values mean larger pixel differences.
- Squaring makes large errors count much more than small ones.
- For 8-bit images the maximum possible value is $255^2 = 65025$.
            """
        )

with psnr_column:
    psnr_display = (
        "∞ dB" if np.isinf(psnr_value) else f"{psnr_value:.2f} dB"
    )
    st.metric("Peak Signal-to-Noise Ratio (PSNR)", psnr_display)

    with st.expander("ℹ️ What is PSNR?"):
        st.markdown(
            """
**Peak Signal-to-Noise Ratio** compares the largest possible pixel value
with the size of the error, on a logarithmic decibel (dB) scale:
            """
        )
        st.latex(r"\text{PSNR} = 10 \cdot \log_{10}\left( \frac{\text{MAX}_I^2}{\text{MSE}} \right) \ \text{dB}")
        st.markdown(
            """
- $\\text{MAX}_I$: maximum pixel value, **255** for 8-bit images
- MSE: the value shown on the left

**How to read it:**
- **Higher is better.** PSNR goes down when MSE goes up.
- **∞ dB** means MSE = 0, so the images are identical.
- Example: MSE = 100 gives PSNR ≈ 28.13 dB.

Rough guide for 8-bit images (not a strict standard):

| PSNR | Typical quality |
|---|---|
| above 40 dB | differences are very hard to see |
| 30–40 dB | good quality, small visible changes |
| 20–30 dB | clearly visible degradation |
| below 20 dB | heavy degradation |
            """
        )

st.info(
    "MSE and PSNR measure pixel-by-pixel differences. They do not always "
    "match human perception: two images with the same PSNR can look very "
    "different, for example noise versus blur at the same error level."
)


# Comparison experiments and charts

st.divider()
st.subheader("Degradation Intensity Analysis")

with st.expander("ℹ️ What is this analysis for?"):
    st.markdown(
        """
The metric cards above show the result for **one** setting. This section
applies the selected method at **several fixed intensity levels** to the
same image, so you can see how image quality changes as the degradation
gets stronger. Your current sidebar setting is added too, marked
**(current)** in the table and shown as a red point in the charts.

**Levels tested:**

| Method | Varied parameter | Levels | Fixed parameter |
|---|---|---|---|
| Gaussian Noise | sigma | 5, 15, 30, 50 | mean = 0 |
| Gaussian Blur | kernel size | 3, 5, 7, 9 | sigma = 1.0 |
| JPEG Compression | quality | 90, 70, 40, 10 | – |

Each level is applied to the **original** image, not to the previous
result, so every row is an independent experiment.

**Table columns:**
- **Intensity**: the parameter level used in that experiment
- **MSE**: pixel error for that level (lower is better)
- **PSNR (dB)**: quality for that level (higher is better)

**How to read the charts:**
- The horizontal axis shows the intensity levels ordered from the weakest
  to the strongest degradation; the vertical axis shows the metric value.
- Stronger degradation means **MSE rises** and **PSNR falls**. The two
  charts should mirror each other.
- For JPEG, a **lower** quality number means **stronger** compression,
  so the axis goes from high quality (left) to low quality (right).
- For blur, sigma is fixed at 1.0, so the curve flattens at larger
  kernel sizes: the extra pixels at the edges of the window get almost
  no weight.
- A steep part of a curve shows where small parameter changes have a big
  effect on quality.
        """
    )

if method == "gaussian_noise":
    experiments = [
        {"method": method, "parameters": {"mean": 0, "sigma": level}}
        for level in [5, 15, 30, 50]
    ]
    experiments.append({"method": method, "parameters": parameters.copy()})

elif method == "gaussian_blur":
    experiments = [
        {
            "method": method,
            "parameters": {"kernel_size": level, "sigma": 1.0},
        }
        for level in [3, 5, 7, 9]
    ]
    experiments.append({"method": method, "parameters": parameters.copy()})

else:
    experiments = [
        {"method": method, "parameters": {"quality": level}}
        for level in [90, 70, 40, 10]
    ]
    experiments.append({"method": method, "parameters": parameters.copy()})

results_df = run_evaluation(original_image, experiments)


def get_intensity_label(row):
    params = row["parameters"]

    if row["method"] == "gaussian_noise":
        label = f"Sigma {params['sigma']}"
        if params.get("mean", 0) != 0:
            label += f", mean {params['mean']}"
        return label

    if row["method"] == "gaussian_blur":
        return f"Kernel {params['kernel_size']}, σ {params['sigma']}"

    return f"Quality {params['quality']}"


def get_severity(row):
    # Numeric value used to order rows from weakest to strongest degradation.
    params = row["parameters"]

    if row["method"] == "gaussian_noise":
        return params["sigma"]

    if row["method"] == "gaussian_blur":
        return params["kernel_size"] + params["sigma"] / 100

    # Lower JPEG quality means stronger degradation.
    return 100 - params["quality"]


results_df["Intensity"] = results_df.apply(get_intensity_label, axis=1)
results_df["Severity"] = results_df.apply(get_severity, axis=1)

# The last experiment is the user's current sidebar setting.
results_df["Setting"] = "Fixed level"
results_df.loc[results_df.index[-1], "Setting"] = "Current setting"
results_df.loc[results_df.index[-1], "Intensity"] += " (current)"

# Order rows from weakest to strongest degradation.
# Text labels would otherwise be sorted alphabetically (e.g. "Sigma 5" after "Sigma 30").
results_df = results_df.sort_values("Severity", kind="stable").reset_index(drop=True)

# Keep the table focused on metric results.
display_df = results_df[
    ["Intensity", "mse_value", "psnr_value"]
].rename(
    columns={
        "mse_value": "MSE",
        "psnr_value": "PSNR (dB)",
    }
)

st.dataframe(display_df, hide_index=True, width="stretch")

chart_df = results_df[["Intensity", "Setting", "mse_value", "psnr_value"]].copy()
intensity_order = chart_df["Intensity"].tolist()


def build_metric_chart(column, title):
    base = alt.Chart(chart_df).encode(
        x=alt.X(
            "Intensity:N",
            sort=intensity_order,
            title="Degradation level (weak → strong)",
            axis=alt.Axis(labelAngle=-45, labelOverlap=False),
        ),
        y=alt.Y(f"{column}:Q", title=title, scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("Intensity:N"),
            alt.Tooltip(f"{column}:Q", title=title, format=".2f"),
        ],
    )

    line = base.mark_line()
    points = base.mark_point(filled=True, size=90).encode(
        color=alt.Color(
            "Setting:N",
            scale=alt.Scale(
                domain=["Fixed level", "Current setting"],
                range=["#6CA6E0", "#F05A5A"],
            ),
            legend=alt.Legend(title=None, orient="bottom"),
        )
    )

    return line + points


chart_column_1, chart_column_2 = st.columns(2)

with chart_column_1:
    st.markdown("**MSE Comparison**")
    st.altair_chart(build_metric_chart("mse_value", "MSE"), width="stretch")
    st.caption("Higher MSE means larger pixel errors (worse quality).")

with chart_column_2:
    st.markdown("**PSNR Comparison**")
    st.altair_chart(build_metric_chart("psnr_value", "PSNR (dB)"), width="stretch")
    st.caption("Higher PSNR means the image is closer to the original (better quality).")

st.caption(
    "Results update automatically when the image, degradation method, "
    "or intensity settings change."
)
