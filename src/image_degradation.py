# Implements Gaussian noise, Gaussian blur, and JPEG compression.
# Accepts adjustable parameters with default values to control degradation intensity.
# Generates each degraded image from the original image.
# Provides three independently testable image degradation functions.

import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=20, seed=42):
    """Add Gaussian noise to an image."""

    rng = np.random.default_rng(seed)
    image_float = image.astype(np.float32)

    noise = rng.normal(mean, sigma, image.shape)
    noisy_image = image_float + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image.astype(np.uint8)


def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to an image."""

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    blurred_image = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
    )

    return blurred_image


def apply_jpeg_compression(image, quality=75):
    """Apply JPEG compression to an image."""

    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100.")

    # Convert RGB to BGR because OpenCV uses BGR for color encoding.
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success, encoded_image = cv2.imencode(
        ".jpg",
        bgr_image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise RuntimeError("JPEG encoding failed.")

    decoded_image = cv2.imdecode(
        encoded_image,
        cv2.IMREAD_COLOR,
    )

    if decoded_image is None:
        raise RuntimeError("JPEG decoding failed.")

    # Convert BGR back to RGB for consistency across project modules.
    processed_image = cv2.cvtColor(
        decoded_image,
        cv2.COLOR_BGR2RGB,
    )

    return processed_image
