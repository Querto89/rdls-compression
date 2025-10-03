#basic_methods.py
import imageio.v3 as iio
import numpy as np
import json
from color_space_transforms import color_space_transform
from img_compares import compare_images
from utils import _to_float01, _from_float01
from noise_generator import add_noise
from database import initialize, db_save_image, db_save_noise, db_save_filter, db_save_measurments_results
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
    'RDgDb',
    'RDLS-RCT',
    'RDLS-LDgEb',
    'RDLS-YCoCg-R',
    'RDLS-RDgDb' 
]

compression_formats_list = [
    'JPEG 2000',
    'JPEG XL',
    'HEVC',
    'VVC'
]

#Baza parametrów filtrów
filter_configs = {
    "bilateral": {
        "values": np.arange(0.05, 0.35, 0.05).tolist(),
        "params": {"d": 7, "sigmaColor": None, "sigmaSpace": 5}
    },
    "gaussian": {
        "values": np.arange(0.5,3.5,0.5).tolist(),
        "params": {"ksize": 5, "sigma": None}
    },
    "median": {
        "values": np.arange(3,11,2).tolist(),
        "params": {"ksize": None}
    },
    "nl_means": {
        "values": np.arange(0.05,0.35,0.05).tolist(),
        "params": {"h": None, 'patch_size': 5, 'patch_distance':6}
    },
    "wavelet": {
        "values": np.arange(0.01,0.11,0.01).tolist(),
        "params": {"sigma": None, "mode": "soft"}
    }
}

#Baza parametrów szumów
seed = 42
noise_configs = {
    "gaussian": {
        "values": np.arange(0.01,0.1,0.01).tolist(),
        "params": {"sigma": None, "mean": 0.0}
    },
    "poisson": {
        "values": np.arange(50.0,150.0,25.0).tolist(),
        "params": {"peak": None}
    },
    "shot": {
        "values": np.arange(40.0,120.0,20.0).tolist(),
        "params": {"lam": None}
    },
    "read": {
        "values": np.arange(0.005,0.02,0.005).tolist(),
        "params": {"sigma": None}
    },
    "shot_read": {
        "values": np.arange(1.0,4.0,1.0).tolist(),
        "params": {"gain": 200.0, "read_noise_std": None}
    }
}

for image_name in images_set:
    img_org = iio.imread('data/natural/'+image_name)
    img_org_f, orig_dtype, orig_max = _to_float01(img_org)
    for transformation_name in color_transformation_list:
        for noise_name, noise_cfg in noise_configs.items():
            for val in noise_cfg["values"]:
                noise_params = noise_cfg["params"].copy()
                for noise_param_name, noise_param_value in noise_params.items():
                    if noise_param_value is None:
                        noise_params[noise_param_name] = val
                        img_noise = add_noise(img_org_f, noise_name, noise_params, seed)
                        img_noise_u8 = _from_float01(img_noise,orig_dtype,orig_max)
                        for filter_name, filter_cfg in filter_configs.items():
                            for val in filter_cfg["values"]:
                                filter_params = filter_cfg["params"].copy()
                                for filter_param_name, filter_param_value in filter_params.items():
                                    if filter_param_value is None:
                                        filter_params[filter_param_name] = val
                                        img_rec = color_space_transform(img_noise, transformation_name, filter_name, filter_params)
                                        img_rec_u8 = _from_float01(img_rec,orig_dtype,orig_max)
                                        entropy = get_entropy(img_rec_u8, transformation_name, image_name)
                                        channel_h0 = get_channel_entropy(img_rec_u8)
                                        for compression_format in compression_formats_list:
                                            comp_results = make_compress(img_rec, image_name, compression_format)
                                            cursor.execute('SELECT MAX(id) FROM images')
                                            row = cursor.fetchone()
                                            current_max_id = row[0] if row[0] is not None else 0
                                            id = current_max_id + 1
                                            db_save_image(cursor,id,image_name,transformation_name,filter_name,noise_name,compression_format)
                                            db_save_noise(cursor,id,noise_name,json.dumps(noise_params))
                                            db_save_filter(cursor,id,filter_name,json.dumps(filter_params))
                                            db_save_measurments_results(cursor,id,entropy['H0'],entropy['H1'],channel_h0['R_H0'],channel_h0['G_H0'],channel_h0['B_H0'],comp_results['psnr'],comp_results['bit_perfect'],comp_results['hash_equal'],compression_format,comp_results['size'],comp_results['compression_ratio'],comp_results['bpp'])
                                            conn.commit()
conn.close()