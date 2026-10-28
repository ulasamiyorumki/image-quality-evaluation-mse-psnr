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
st.write("Evaluate image degradation using MSE and PSNR metrics.")

with st.expander("ℹ️ How to use this dashboard", expanded=False):
    st.markdown(
        """
        This dashboard demonstrates how different types of image
        degradation affect measurable image quality.

        1. **Choose a sample image** in the sidebar.
        2. **Choose a degradation method:** Gaussian Noise, Gaussian Blur,
           or JPEG Compression.
        3. **Adjust the parameters** using the sliders.
        4. **Compare the images** and review the MSE and PSNR metrics.
        5. **Explore Degradation Intensity Analysis** to compare several
           degradation levels.

        All processed images are generated from the original image.
        Degradation effects are never stacked.
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


# Built-in sample images

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
        "Choose a built-in test image. Images with many fine details "
        "and edges may react differently to blur and compression "
        "than smoother images."
    ),
)

original_image = load_sample_image(image_name)

method_label = st.sidebar.selectbox(
    "Degradation Method",
    ["Gaussian Noise", "Gaussian Blur", "JPEG Compression"],
    help="Choose the type of distortion to apply to the original image.",
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
            "generally produce stronger noise."
        ),
    )

    mean = st.sidebar.slider(
        "Noise Mean",
        min_value=-20,
        max_value=20,
        value=0,
        help=(
            "Average value (μ) of the noise. Positive values tend to "
            "brighten the image, while negative values tend to darken it."
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
            "Width and height of the blur window in pixels. "
            "The kernel size must be an odd number."
        ),
    )

    sigma = st.sidebar.slider(
        "Blur Sigma",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help=(
            "Controls the spread of the Gaussian weights. Higher values "
            "generally produce stronger blur."
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
            "JPEG quality factor. Lower values mean stronger compression "
            "and usually more visible artifacts."
        ),
    )

    parameters = {"quality": quality}


# Process the original image

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


# Image comparison

st.subheader("Image Comparison")

original_column, processed_column = st.columns(2)

with original_column:
    st.markdown("**Original Image**")
    st.image(original_image, width="stretch")

with processed_column:
    st.markdown("**Processed Image**")
    st.image(processed_image, width="stretch")

st.caption(
    f"Selected method: {method_label} | Parameters: {parameters}"
)


# Degradation method explanation

with st.expander(
    f"ℹ️ What does {method_label} do?",
    expanded=False,
):
    if method == "gaussian_noise":
        st.markdown(
            """
            **Gaussian noise** adds random values to image pixels.
            These values follow a normal distribution. This can simulate
            sensor noise or interference during image transmission.

            The noise model is:
            """
        )

        st.latex(
            r"I_{\text{noisy}}(x,y)"
            r"= I(x,y)+n(x,y), \qquad "
            r"n(x,y)\sim\mathcal{N}(\mu,\sigma^2)"
        )

        st.markdown(
            r"""
            - \(I(x,y)\): original pixel value.
            - \(n(x,y)\): random noise value.
            - \(\mu\): mean of the added noise.
            - \(\sigma\): standard deviation controlling noise intensity.

            Pixel values are clipped to the valid 0–255 range.
            With random noise, results may change between runs unless
            the random seed is fixed.
            """
        )

    elif method == "gaussian_blur":
        st.markdown(
            """
            **Gaussian blur** replaces each pixel with a weighted average
            of its neighbours. Nearby pixels receive greater weights.
            This reduces fine details and softens edges.

            The two-dimensional Gaussian function is:
            """
        )

        st.latex(
            r"G(x,y)=\frac{1}{2\pi\sigma^2}"
            r"\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)"
        )

        st.markdown("The blurred image is obtained by convolution:")

        st.latex(
            r"I_{\text{blur}}=I\ast G"
        )

        st.markdown(
            r"""
            - **Kernel Size:** dimensions of the filtering window,
              \(k\times k\) pixels.
            - **Blur Sigma (\(\sigma\)):** controls the spread of the
              Gaussian weights.

            These parameters work together. When sigma is small,
            increasing kernel size may have little visible effect.
            """
        )

    else:
        st.markdown(
            """
            **JPEG compression** reduces image data using a lossy
            compression process. The image is processed in blocks,
            transformed using the Discrete Cosine Transform (DCT),
            and quantized. Quantization discards some information.
            """
        )

        st.latex(
            r"F_Q(u,v)="
            r"\operatorname{round}\left(\frac{F(u,v)}{Q(u,v)}\right)"
        )

        st.markdown(
            r"""
            - \(F(u,v)\): DCT coefficient representing frequency content.
            - \(Q(u,v)\): quantization step.

            Lower JPEG quality generally means stronger quantization,
            more lost detail, and more visible compression artifacts.
            Even at quality 100, the compressed image may differ from
            the original.
            """
        )


# Image quality metrics

st.divider()
st.subheader("Image Quality Metrics")

mse_column, psnr_column = st.columns(2)

with mse_column:
    st.metric(
        "Mean Squared Error (MSE)",
        f"{mse_value:.2f}",
    )

    with st.expander("ℹ️ What is MSE?", expanded=False):
        st.markdown(
            """
            **Mean Squared Error (MSE)** measures the average squared
            difference between corresponding pixel values in the
            original and processed images.
            """
        )

        st.latex(
            r"\mathrm{MSE}=\frac{1}{MNC}"
            r"\sum_{c=1}^{C}\sum_{i=1}^{M}\sum_{j=1}^{N}"
            r"\left(I(i,j,c)-K(i,j,c)\right)^2"
        )

        st.markdown(
            r"""
            - \(I\): original image.
            - \(K\): processed image.
            - \(M\), \(N\): image height and width.
            - \(C\): number of color channels (3 for RGB images).

            **How to interpret MSE:**

            - **MSE = 0:** the images are identical.
            - **Lower MSE:** smaller average pixel differences.
            - **Higher MSE:** larger average pixel differences.

            Squaring gives larger errors more influence on the result.
            For 8-bit images, the maximum possible MSE is \(255^2=65025\).
            """
        )


with psnr_column:
    psnr_display = (
        "∞ dB" if np.isinf(psnr_value) else f"{psnr_value:.2f} dB"
    )

    st.metric(
        "Peak Signal-to-Noise Ratio (PSNR)",
        psnr_display,
    )

    with st.expander("ℹ️ What is PSNR?", expanded=False):
        st.markdown(
            """
            **Peak Signal-to-Noise Ratio (PSNR)** expresses the ratio
            between the maximum possible pixel value and the error
            between the original and processed images, using a
            logarithmic decibel (dB) scale.
            """
        )

        st.latex(
            r"\mathrm{PSNR}=10\log_{10}"
            r"\left(\frac{\mathrm{MAX}_I^2}{\mathrm{MSE}}\right)"
            r"\ \mathrm{dB}"
        )

        st.markdown(
            r"""
            - $\mathrm{MAX}_I$: maximum pixel value, 255 for 8-bit images.
            - $\mathrm{MSE}$: mean squared error shown beside this metric.
            

            **How to interpret PSNR:**

            - **Higher PSNR:** generally smaller differences from the original.
            - **∞ dB:** the images are identical because MSE is zero.
            - **Lower PSNR:** generally greater distortion.

            Approximate guidance for 8-bit images (not a strict standard):

            | PSNR | Typical interpretation |
            |---|---|
            | Above 40 dB | Differences may be difficult to see |
            | 30–40 dB | Relatively high similarity |
            | 20–30 dB | More noticeable degradation |
            | Below 20 dB | Potentially substantial degradation |
            """
        )


st.info(
    "MSE and PSNR measure pixel-level differences. They do not always "
    "match human perception: noise and blur can look different even "
    "when their metric values are similar."
)


# Degradation intensity analysis

st.divider()
st.subheader("Degradation Intensity Analysis")

with st.expander(
    "ℹ️ What is this analysis for?",
    expanded=False,
):
    st.markdown(
        """
        The metric cards above show the result for one selected setting.
        This section compares that result with several fixed degradation
        levels applied independently to the same original image.

        The current sidebar setting is marked **(current)** in the table
        and highlighted in red in the charts.

        **Levels tested:**

        | Method | Varied parameter | Test levels | Fixed parameter |
        |---|---|---|---|
        | Gaussian Noise | Sigma | 5, 15, 30, 50 | Mean = 0 |
        | Gaussian Blur | Kernel size | 3, 5, 7, 9 | Sigma = 1.0 |
        | JPEG Compression | Quality | 90, 70, 40, 10 | None |

        Every experiment starts from the original image, not from a
        previously degraded result.

        **Table columns:**

        - **Intensity:** parameter values used in the experiment.
        - **MSE:** average squared pixel error; lower is better.
        - **PSNR (dB):** logarithmic image similarity metric; higher is better.

        **How to read the charts:**

        - The horizontal axis orders the experiments from weaker to stronger
          degradation.
        - Stronger degradation generally increases MSE and decreases PSNR.
        - For JPEG, lower quality means stronger compression.
        - For blur, sigma is fixed at 1.0, so increasing kernel size may
          have a limited effect once the window is sufficiently large.
        - A steep curve indicates that a parameter change has a relatively
          large effect on the metric.
        """
    )


if method == "gaussian_noise":
    experiments = [
        {
            "method": method,
            "parameters": {"mean": 0, "sigma": level},
        }
        for level in [5, 15, 30, 50]
    ]

    experiments.append(
        {"method": method, "parameters": parameters.copy()}
    )

elif method == "gaussian_blur":
    experiments = [
        {
            "method": method,
            "parameters": {"kernel_size": level, "sigma": 1.0},
        }
        for level in [3, 5, 7, 9]
    ]

    experiments.append(
        {"method": method, "parameters": parameters.copy()}
    )

else:
    experiments = [
        {
            "method": method,
            "parameters": {"quality": level},
        }
        for level in [90, 70, 40, 10]
    ]

    experiments.append(
        {"method": method, "parameters": parameters.copy()}
    )


results_df = run_evaluation(original_image, experiments)


def get_intensity_label(row):
    params = row["parameters"]

    if row["method"] == "gaussian_noise":
        label = f"Sigma {params['sigma']}"

        if params.get("mean", 0) != 0:
            label += f", mean {params['mean']}"

        return label

    if row["method"] == "gaussian_blur":
        return (
            f"Kernel {params['kernel_size']}, "
            f"sigma {params['sigma']}"
        )

    return f"Quality {params['quality']}"


def get_severity(row):
    params = row["parameters"]

    if row["method"] == "gaussian_noise":
        return params["sigma"]

    if row["method"] == "gaussian_blur":
        return params["kernel_size"] + params["sigma"] / 100

    # Lower JPEG quality means stronger degradation.
    return 100 - params["quality"]


results_df["Intensity"] = results_df.apply(
    get_intensity_label,
    axis=1,
)

results_df["Severity"] = results_df.apply(
    get_severity,
    axis=1,
)

# The last experiment represents the current sidebar setting.
results_df["Setting"] = "Fixed level"
results_df.loc[
    results_df.index[-1],
    "Setting",
] = "Current setting"

results_df.loc[
    results_df.index[-1],
    "Intensity",
] += " (current)"

# Sort from weaker to stronger degradation.
results_df = results_df.sort_values(
    "Severity",
    kind="stable",
).reset_index(drop=True)


# Results table

display_df = results_df[
    ["Intensity", "mse_value", "psnr_value"]
].rename(
    columns={
        "mse_value": "MSE",
        "psnr_value": "PSNR (dB)",
    }
)

st.dataframe(
    display_df,
    hide_index=True,
    width="stretch",
)


# Comparison charts

chart_df = results_df[
    ["Intensity", "Setting", "mse_value", "psnr_value"]
].copy()

intensity_order = chart_df["Intensity"].tolist()


def build_metric_chart(column, title):
    base = alt.Chart(chart_df).encode(
        x=alt.X(
            "Intensity:N",
            sort=intensity_order,
            title="Degradation level (weak → strong)",
            axis=alt.Axis(
                labelAngle=-45,
                labelOverlap=False,
            ),
        ),
        y=alt.Y(
            f"{column}:Q",
            title=title,
            scale=alt.Scale(zero=False),
        ),
        tooltip=[
            alt.Tooltip("Intensity:N"),
            alt.Tooltip(
                f"{column}:Q",
                title=title,
                format=".2f",
            ),
        ],
    )

    line = base.mark_line()

    points = base.mark_point(
        filled=True,
        size=90,
    ).encode(
        color=alt.Color(
            "Setting:N",
            scale=alt.Scale(
                domain=["Fixed level", "Current setting"],
                range=["#6CA6E0", "#F05A5A"],
            ),
            legend=alt.Legend(
                title=None,
                orient="bottom",
            ),
        )
    )

    return line + points


chart_column_1, chart_column_2 = st.columns(2)

with chart_column_1:
    st.markdown("**MSE Comparison**")

    st.altair_chart(
        build_metric_chart("mse_value", "MSE"),
        width="stretch",
    )

    st.caption(
        "Higher MSE means larger average pixel errors (worse similarity)."
    )


with chart_column_2:
    st.markdown("**PSNR Comparison**")

    st.altair_chart(
        build_metric_chart("psnr_value", "PSNR (dB)"),
        width="stretch",
    )

    st.caption(
        "Higher PSNR generally means the processed image is closer "
        "to the original."
    )


st.caption(
    "Results update automatically when the image, degradation method, "
    "or intensity settings change."
)
