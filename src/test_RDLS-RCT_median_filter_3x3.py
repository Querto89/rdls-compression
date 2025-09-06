import numpy as np
import cv2
import imageio.v3 as iio
from tester import compare_images

def median_filter(channel,n):
    """Zwraca kanał po medianie 3x3"""
    return cv2.medianBlur(channel.astype(np.uint8), n).astype(np.int32)

def forward_RCT_RDLS(img):
    R = img[:,:,2].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,0].astype(np.int32)

    Gd = median_filter(G,3)
    Cv = R - Gd
    Cu = B - Gd
    Cvd = median_filter(Cv,3)
    Cud = median_filter(Cu,3)
    t = ( Cvd + Cud ) >> 2
    Y = G + np.floor(t)

    return np.stack((Y, Cu, Cv), axis=2)

def inverse_RCT_RDLS(rct_img):
    Y  = rct_img[:,:,0].astype(np.int32)
    Cv = rct_img[:,:,1].astype(np.int32)
    Cu = rct_img[:,:,2].astype(np.int32)

    Cvd = median_filter(Cv,3)
    Cud = median_filter(Cu,3)

    t = ( Cvd + Cud ) >> 2
    G = Y - t
    Gd = median_filter(G,3)
    B = Cu + Gd
    R = Cv + Gd

    return np.stack((R,G,B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
rct = forward_RCT_RDLS(img)
rec = inverse_RCT_RDLS(rct)
compare_images(img,rec)