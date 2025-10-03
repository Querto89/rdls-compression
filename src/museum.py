import imageio.v3 as iio
from color_space_transforms import color_space_transform
from img_compares import compare_images
from utils import _to_float01
from utils import _from_float01
from noise_generator import add_noise
from database import initialize, import_to_excel
from entropy import get_entropy
import numpy as np
import sqlite3
import os
import pandas as pd
from openpyxl import load_workbook
from collections import Counter
from pathlib import Path
import math
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as PSNR
import matplotlib.gridspec as gridspec
import hashlib

#Funkcja importowania danych z bazy danych do Excela

def import_to_excel(conn=None, file_path="dane.xlsx", sheet_name="Data", rowstart=0, colstart=0):
    if conn is None:
        conn, _ = initialize()
    
    df = pd.read_sql_query("SELECT * FROM entropy_results", conn)

    if os.path.exists(file_path):
        # Plik istnieje → nadpisujemy tylko wskazany arkusz
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=rowstart, startcol=colstart)
    else:
        # Plik nie istnieje → tworzymy nowy
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=rowstart, startcol=colstart)

    print(f"[OK] Dane zapisane w arkuszu '{sheet_name}' pliku {file_path}")

def compare_images(img_orig: np.ndarray, img_rec: np.ndarray, title=""):
    # --- Test PSNR
    data_range = img_orig.max() - img_orig.min()
    psnr_value = PSNR(img_orig, img_rec, data_range=data_range)
    psnr_result = f"PSNR = {psnr_value:.2f} dB"

    # --- Test zgodności bitowej
    bit_acc_result = ""
    if(img_orig.shape == img_rec.shape):
        bit_acc_result = "Bit-acc: Obrazy mają takie same rozmiary"
    else:
        bit_acc_result = "Bit-acc: Obrazy mają różne rozmiary"
    
    # --- Bit-perfect check
    bit_equal = np.array_equal(img_orig,img_rec)
    bit_equal_result = f"Bit-perfect: {bit_equal}"

    # --- Hash check
    h1 = hashlib.sha256(img_orig.tobytes()).hexdigest()
    h2 = hashlib.sha256(img_rec.tobytes()).hexdigest()
    hash_test = (h1==h2)
    hash_result = f"Hash-equal: {hash_test}"

    # --- Różnice pixeli
    diff = img_rec.astype(np.int32) - img_orig.astype(np.int32)
    max_diff = np.abs(diff).max()
    nonzero = np.count_nonzero(diff)
    max_diff_result = f"Maksymalna różnica: {max_diff}"
    nonzero_result = f"Liczba różniących się pikseli: {nonzero}"

    fig = plt.figure(figsize=(12,10))
    fig.subplots_adjust(bottom=0.18, hspace=0.3)

    gs = gridspec.GridSpec(2, 2, figure=fig)

    # --- górny rząd: obrazy
    ax0 = fig.add_subplot(gs[0,0])
    ax0.imshow(img_orig)
    ax0.set_title("Oryginalny")
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0,1])
    ax1.imshow(img_rec)
    ax1.set_title("Zrekonstruowany")
    ax1.axis('off')

    # --- dolny rząd: histogram + mapa błędu
    ax2 = fig.add_subplot(gs[1,0])
    ax2.hist(diff.ravel(), bins=513, range=(-256, 256), color="blue")
    ax2.set_title("Histogram różnic")
    ax2.set_xlabel("Różnica (rec - orig)")
    ax2.set_ylabel("Liczba pikseli")
    ax2.set_aspect('auto')  # wymusza równy układ w hist

    ax3 = fig.add_subplot(gs[1,1])
    im = ax3.imshow(np.abs(diff).sum(axis=2), cmap="hot")
    ax3.set_title("Mapa błędu")
    ax3.set_aspect('auto')  # wymusza proporcje
    plt.colorbar(im, ax=ax3)

    # Dodanie wyników testów pod obrazami
    results = [bit_acc_result, bit_equal_result, max_diff_result, nonzero_result, psnr_result]
    for i, text in enumerate(results):
        fig.text(0.5, 0.01 + i*0.03, text, ha='center', fontsize=12)

    plt.show()
    return 0

"""case "gaussian":

                case "median":

                case "nl_means":

                case "wavelet":"""

for filter_name in filters_names_list:
            match(filter_name):
                case "bilateral":
                    for sigma_color in bilateral_sigmaColor_values:
                        print(sigma_color)
                case _:
                    raise ValueError(f"Nieznany filtr: {method}")


def add_noise(img: np.ndarray, method: str = "gaussian", params: dict | None = None, seed: int | None = None) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if params is None:
        params = {}

    noisy = None
    match method.lower():
        case "gaussian":
            mean = params.get("mean", 0.0)
            sigma = params.get("sigma", 0.05)
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