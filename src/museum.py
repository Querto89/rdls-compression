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

#basic_methods.py
#basic_methods.py
import imageio.v3 as iio
import numpy as np
import json
from color_space_transforms import color_space_transform
from img_compares import compare_images
from utils import _to_float01, _from_float01
from noise_generator import add_noise
from database import initialize, db_save_image
from entropy import get_entropy, get_channel_entropy
from compressor import make_compress

#Inicjalizacja bazy danych
conn, cursor = initialize()

#Baza obrazów testowych
images_set = [
    'kodim01.png',
    'kodim02.png',
    'kodim03.png',
    'kodim04.png',
    'kodim05.png',
    'kodim06.png',
    'kodim07.png',
    'kodim08.png',
    'kodim09.png',
    'kodim10.png'
]

#Lista transformacji
color_transformation_list = [
    'RCT',
    'LDgEb',
    'YCoCg-R',
    'RDgDb' 
]

#Baza parametrów filtrów
filters_names_list = [
    'bilateral',
    'gaussian',
    'median',
    'nl_means',
    'wavelet'
] 
default_filter_params = {
    'bilateral_d': 7,
    'bilateral_sigmaColor': 0.1, #Parametr iterowany w zakresie 0.05-0.3 z krokiem 0.05
    'bilateral_sigmaSpace': 5,
    'gaussian_ksize': 5,
    'gaussian_sigma': 5, #Parametr iterowany w zakresie 0.5-3.0 z krokiem 0.5
    'median_ksize': 3, #Parametr iterowany w zakresie 3-9 z krokiem 2
    'nl_means_h': 0.1, #Parametr iterowany w zakresie 0.05-0.3 z krokiem 0.05
    'nl_means_patch_size': 5,
    'nl_means_patch_distance': 6,
    'wavelet_sigma': 0.05, #Parametr iterowany w zakresie 0.01-0.1 z krokiem 0.01
    'wavelet_mode': "soft"
}

bilateral_sigmaColor_values = np.arange(0.05, 0.35, 0.05).tolist()
gaussian_sigma_values = np.arange(0.5,3.5,0.5).tolist()
median_ksize_values = list(range(3,11,2))
nl_means_h_values = np.arange(0.05,0.35,0.05).tolist()
wavelet_sigma = np.arange(0.01,0.11,0.01).tolist()

#Baza parametrów szumów
seed = 42
noises_names_list = [
    'gaussian',
    'poisson',
    'shot',
    'read',
    'shot_read'
]

default_noises_params = {
    'gaussian_mean': 0.0,
    'gaussian_sigma': 0.05, #Parametr iterowany w zakresie 0.01-0.1 z krokiem 0.01
    'poisson_peak': 100.0, #Parametr iterowany w zakresie 50-150 z krokiem 25
    'shot_lam': 80.0, #Parametr iterowany w zakresie 40-120 z krokiem 20
    'read_sigma': 0.01, #Parametr iterowany w zakresie 0.005-0.02 z krokiem 0.005
    'shot_read_gain': 200.0, 
    'shot_read_noise_std': 2.0, #Parametr iterowany w zakresie 1-4 z krokiem 1
}

gaussian_sigma_values = np.arange(0.01,0.1,0.01).tolist()
poisson_peak_values = list(range(50,150,25))
shot_lam_values = np.arange(40,120,20).tolist()
read_sigma_values = np.arange(0.005,0.02,0.005).tolist()
shot_read_sigma_values = list(range(1,4,1))

for image_name in images_set:
    cursor.execute('SELECT MAX(id) FROM images')
    row = cursor.fetchone()   # zwróci np. (5,) jeśli ostatni ID = 5
    current_max_id = row[0] if row[0] is not None else 0
    for transformation_name in color_transformation_list:
        NO_RDLS_transform = transformation_name
        RDLS_transform = "RDLS-"+transformation_name 
        img_org = iio.imread('data/natural/'+image_name)
        img_org_f, orig_dtype, orig_max = _to_float01(img_org)
        for noise_name in noises_names_list:
            match(noise_name):
                case "gaussian":
                    noise_param_values_list = gaussian_sigma_values
                    noise_params = {
                        "first_param": None,
                        "second_param": default_noises_params['gaussian_mean']
                    }
                case "poisson":
                    noise_param_values_list = poisson_peak_values
                    noise_params = {
                        "first_param": None
                    }
                case "shot":
                    noise_param_values_list = shot_lam_values
                    noise_params = {
                        "first_param": None
                    }
                case "read":
                    noise_param_values_list = read_sigma_values
                    noise_params = {
                        "first_param": None
                    }
                case "shot_read":
                    noise_param_values_list = shot_read_sigma_values
                    noise_params = {
                        "first_param": None,
                        "second_param": default_noises_params['shot_read_gain']
                    }
            for param in noise_param_values_list:
                noise_params['first_param']=param
                img_noise = add_noise(img_org_f, noise_name, noise_params, seed)