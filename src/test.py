import numpy as np
import imageio.v3 as iio
from img_compares import compare_images
from utils import _to_float01
from color_space_transforms import color_space_transform

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
img_org = iio.imread('data/natural/'+images[img_num-1])
img_org_f,orig_dtype, orig_max = _to_float01(img_org)
img_RCT = color_space_transform(img_org_f)
compare_images(img_org_f,img_RCT)