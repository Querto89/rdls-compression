import numpy as np
import cv2
import imageio.v3 as iio
from tester import compare_images

def median_filter(channel,n):
    """Zwraca kanał po medianie 3x3"""
    return cv2.medianBlur(channel.astype(np.uint8), n).astype(np.int32)

def forward_LDgEb(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)
    
    Rd = median_filter(R,3)
    Dg = - G + Rd
    Dgd = median_filter(Dg,3)
    t = np.floor( Dgd >> 2)
    L = R - t
    Ld = median_filter(L,3)
    Eb = B - Ld
    
    return np.stack((L, Dg, Eb), axis=2).astype(np.int32)

def inverse_LDgEb(rct_img):
    L = rct_img[:,:,0].astype(np.int32)
    Dg = rct_img[:,:,1].astype(np.int32)
    Eb = rct_img[:,:,2].astype(np.int32)
    
    Ld = median_filter(L,3)
    B = Eb + Ld
    Dgd = median_filter(Dg,3)
    t = np.floor( Dgd >> 2)
    R = L + t
    Rd = median_filter(R,3)
    G = - Dg + Rd

    return np.stack((R, G, B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
ldgeb = forward_LDgEb(img)
rec = inverse_LDgEb(ldgeb)
compare_images(img,rec)