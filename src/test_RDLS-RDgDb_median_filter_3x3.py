import numpy as np
import cv2
import imageio.v3 as iio
from tester import compare_images

def median_filter(channel,n):
    """Zwraca kanał po medianie 3x3"""
    return cv2.medianBlur(channel.astype(np.uint8), n).astype(np.int32)

def forward_RDgDb(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)

    Gd = median_filter(G,3)
    Db = - B + Gd
    Rd = median_filter(R,3)
    Dg = - G + Rd

    return np.stack((R, Dg, Db), axis=2).astype(np.int32)

def inverse_RDgDb(rdgd_img):
    R = rdgd_img[:,:,0].astype(np.int32)
    Dg = rdgd_img[:,:,1].astype(np.int32)
    Db = rdgd_img[:,:,2].astype(np.int32)
    
    Rd = median_filter(R,3)
    G = - Dg + Rd
    Gd = median_filter(G,3)
    B = - Db + Gd
    return np.stack((B, G, R), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
rdgdb = forward_RDgDb(img)
rec = inverse_RDgDb(rdgdb)
compare_images(img,rec)