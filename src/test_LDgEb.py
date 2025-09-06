import numpy as np
import imageio.v3 as iio
from tester import compare_images

def forward_LDgEb(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)
    
    Dg = - G + R
    t = np.floor( Dg >> 2)
    L = R - t
    Eb = B - L
    
    return np.stack((L, Dg, Eb), axis=2).astype(np.int32)

def inverse_LDgEb(rct_img):
    L = rct_img[:,:,0].astype(np.int32)
    Dg = rct_img[:,:,1].astype(np.int32)
    Eb = rct_img[:,:,2].astype(np.int32)

    B = Eb + L
    t = np.floor( Dg >> 2)
    R = L + t
    G = - Dg + R

    return np.stack((R, G, B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
ldgeb = forward_LDgEb(img)
rec = inverse_LDgEb(ldgeb)
compare_images(img,rec)