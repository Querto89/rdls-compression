import numpy as np
import imageio.v3 as iio
from color_space_transforms import color_space_transform
from img_compares import compare_images
from utils import _to_float01
from utils import _from_float01
from noise_generator import add_noise

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
    'kodim10.png',
    'kodim11.png',
    'kodim12.png',
    'kodim13.png',
    'kodim14.png',
    'kodim15.png',
    'kodim16.png',
    'kodim17.png',
    'kodim18.png',
    'kodim19.png',
    'kodim20.png',
    'kodim21.png',
    'kodim22.png',
    'kodim23.png',
    'kodim24.png'
]

img_num = 1

color_space = "RDLS-RCT"

filter_method = "median"
filter_params = {"ksize": 3}

noise_method = "gaussian"
noise_params = {"mean": 0.0, "sigma": 0.05}
seed = 42

img_org = iio.imread('data/natural/'+images[img_num-1])
img_org_f,orig_dtype, orig_max = _to_float01(img_org)
img_noise = add_noise(img_org_f, noise_method, noise_params, seed)

img_rec = color_space_transform(img_noise, color_space, filter_method, filter_params)
img_rec_u8 = _from_float01(img_rec,orig_dtype,orig_max)
img_noise_u8 = _from_float01(img_noise,orig_dtype,orig_max)
compare_images(img_noise_u8, img_rec_u8)


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