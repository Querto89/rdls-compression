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