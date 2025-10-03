#noise_filter.py
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, denoise_wavelet

def apply_filter(channel: np.ndarray, method: str = "bilateral", params: dict = None) -> np.ndarray:
    if params is None:
        params = {}

    match method:
        case "bilateral":
            d = params.get("d", 5)
            sigmaColor = params.get("sigmaColor", 0.1)  # już w skali [0,1]
            sigmaSpace = params.get("sigmaSpace", 5)
            result = cv2.bilateralFilter(channel, d, sigmaColor, sigmaSpace)

        case "gaussian":
            ksize = params.get("ksize", 5)
            sigma = params.get("sigma", 1.0)
            result = cv2.GaussianBlur(channel, (ksize, ksize), sigma)

        case "median":
            ksize = params.get("ksize", 3)
            # upewniamy się, że kanał jest 2D
            if channel.ndim != 2:
                raise ValueError("median filter works only on 2D single channel")
            tmp = (channel * 255).clip(0, 255).astype(np.uint8)
            result = cv2.medianBlur(tmp, ksize).astype(np.float32) / 255.0

        case "nl_means":
            patch_kw = dict(patch_size=5, patch_distance=6, h=0.1, fast_mode=True)
            patch_kw.update(params)
            result = denoise_nl_means(channel, **patch_kw)

        case "wavelet":
            result = denoise_wavelet(
                channel,
                sigma=params.get("sigma", None),
                mode=params.get("mode", "soft"),
                rescale_sigma=True
            )

        case _:
            raise ValueError(f"Nieznany filtr: {method}")

    return result

"""
Przykładowe wywołanie funkcji apply_filter dla każdego rodzaju filtra

channel = img[:,:,0]  # np. kanał R

# Bilateral filter
f_bilateral = apply_filter(channel, "bilateral", {"d": 7, "sigmaColor": 0.1, "sigmaSpace": 5})

# Gaussian blur
f_gaussian = apply_filter(channel, "gaussian", {"ksize": 5, "sigma": 1.5})

# Median blur
f_median = apply_filter(channel, "median", {"ksize": 3})

# Non-local Means (NL-means)
f_nlmeans = apply_filter(channel, "nl_means", {"h": 0.1, "patch_size": 5, "patch_distance": 6})

# Wavelet denoising
f_wavelet = apply_filter(channel, "wavelet", {"sigma": 0.05, "mode": "soft"})
"""