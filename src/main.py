#main.py
import imageio.v3 as iio
import json
from color_space_transforms import color_space_transform
from img_compares import compare_images
from utils import _to_float01, _from_float01
from noise_generator import add_noise
from database import initialize, import_to_excel
from entropy import get_entropy, get_channel_entropy
from compressor import make_compress

images = [
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

color_spaces = [
    "RCT",
    "LDgEb",
    "YCoCg-R",
    "RDgDb" 
]

#Inicjalizacja bazy danych
conn, cursor = initialize()
cursor.execute('SELECT MAX(id) FRPOM images')
row = cursor.fetchone()   # zwróci np. (5,) jeśli ostatni ID = 5
id = row[0] if row[0] is not None else 0
id=id+1

#Wybór obrazu testowego i przestrzenii barw
img_num = 2
color_space = "RDLS-RCT"

#Parametry szumu
noise_method = "gaussian"
noise_params = {"mean": 0.0, "sigma": 0.05}
seed = 42

#Parametry filtra
filter_method = "median"
filter_params = {"ksize": 3}

#Przygotowywanie obrazu testowego (wczytanie obrazu >> konwersja do FLOAT[0,1] >> zaszumienie)
img_org = iio.imread('data/natural/'+images[img_num-1])
cursor.execute("INSERT INTO images (id,images,transform) VALUES (?, ?, ?)", (id,images[img_num-1],color_space))#Zapisanie wybranego obrazu do tablicy images
img_org_f, orig_dtype, orig_max = _to_float01(img_org)
img_noise = add_noise(img_org_f, noise_method, noise_params, seed)
cursor.execute("INSERT INTO noises (id,name,params) VALUES (?, ?, ?)", (id,noise_method,json.dumps(noise_params)))#Zapisanie wybranego szumu do tablicy noises

#Przeprowadzenie eksperymentu (transformacja do przestrzenii barw >> rekonstrukcja obrazu)
img_rec = color_space_transform(img_noise, color_space, filter_method, filter_params)
cursor.execute("INSERT INTO filters (id,name,params) VALUES (?, ?, ?)", (id,filter_method,json.dumps(filter_params)))#Zapisanie wybranego szumu do tablicy noises
cursor.execute("INSERT INTO images (id,images) VALUES (?, ?)", (id,images[img_num-1]))#Zapisanie wybranego obrazu do tablicy images

#Konwersja obrazów zrekonstruowanego i zaszumionego do u8
img_noise_u8 = _from_float01(img_noise,orig_dtype,orig_max)
img_rec_u8 = _from_float01(img_rec,orig_dtype,orig_max)

#Obliczanie entropii
h0,h1 = get_entropy(img_rec_u8, color_space, images[img_num-1])
channel_h0 = get_channel_entropy(img_rec_u8)

#Porównywanie obrazów
comp_results = compare_images(img_noise_u8, img_rec_u8)
cursor.execute("INSERT INTO entropy_results (id,H0,H1,H0_R,H0_G,H0_B,psnr,bit_perfect,hash_equal) VALUES (?, ?)", (id,h0,h1,channel_h0("R_H0"),channel_h0("G_H0"),channel_h0("B_H0"),comp_results("psnr"),comp_results("bit_perfect"),comp_results("hash_check")))

#Zapisanie obliczeń do tablicy


conn.close()
"""
Wzorce filtrów:

filter_method = "bilateral"
filter_params = {"d": 7, "sigmaColor": 0.1, "sigmaSpace": 5}

filter_method = "gaussian"
filter_params = {"ksize": 5, "sigma": 1.5}

filter_method = "median"
filter_params = {"ksize": 1}

filter_method = "nl_means"
filter_params = {"h": 0.1, "patch_size": 5, "patch_distance": 6}

filter_method = "wavelet"
filter_params = {"sigma": 0.05, "mode": "soft"}
"""

"""
Wzorce szumów:

noise_method = "gaussian"
noise_params = {"mean": 0.0, "sigma": 0.05}

noise_method = "poisson"
noise_params = {"peak": 100.0}

noise_method = "shot"
noise_params = {"lam": 80.0}

noise_method = "read"
noise_params = {"sigma": 0.01}

noise_method = "shot_read"
noise_params = {"gain": 200.0, "read_noise_std": 2.0}
"""