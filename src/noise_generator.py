#noise_generator.py
import numpy as np

def add_noise(img: np.ndarray, method: str, params: dict | None = None, seed: int | None = None) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if params is None:
        params = {}

    noisy = None
    match method.lower():
        case "gaussian":
            sigma = params.get("sigma", 0.0)
            mean = params.get("mean", 0.05)
            noise = rng.normal(loc=mean, scale=sigma, size=img.shape)
            noisy = img + noise

        case "poisson":
            peak = params.get("peak", 50.0)   # im większy peak, tym mniej względnego szumu
            lam = np.clip(img * peak, 0, None)
            noisy = rng.poisson(lam) / peak

        case "shot":  # alias na poisson
            lam = params.get("lam", 50.0)
            lam = np.clip(img * lam, 0, None)
            noisy = rng.poisson(lam) / lam.max()  # normalizacja

        case "read":  # czysto Gaussowski szum odczytu
            sigma = params.get("sigma", 0.01)
            noise = rng.normal(loc=0.0, scale=sigma, size=img.shape)
            noisy = img + noise

        case "shot_read":  # Poisson + Gaussian
            gain = params.get("gain", 100.0)
            read_noise_std = params.get("read_noise_std", 1.0)
            lam = np.clip(img * gain, 0, None)
            shot = rng.poisson(lam)
            read = rng.normal(loc=0.0, scale=read_noise_std, size=img.shape)
            noisy = (shot + read) / gain

        case _:
            raise ValueError(f"Nieznany typ szumu: {method}")

    return np.clip(noisy, 0.0, 1.0)

"""
Przykładowe wywołanie funkcji add_noise() dla każdego rodzaju szumu

# Gaussian noise
noisy_gauss = add_noise(img, "gaussian", {"mean": 0.0, "sigma": 0.05}, seed=42)

# Poisson noise
noisy_poisson = add_noise(img, "poisson", {"peak": 100.0}, seed=42)

# Shot noise (alias na Poisson, ale trochę inny zapis)
noisy_shot = add_noise(img, "shot", {"lam": 80.0}, seed=42)

# Read noise (czysto Gaussowski szum odczytu)
noisy_read = add_noise(img, "read", {"sigma": 0.01}, seed=42)

# Shot + Read noise
noisy_shot_read = add_noise(img, "shot_read", {"gain": 200.0, "read_noise_std": 2.0}, seed=42)
"""