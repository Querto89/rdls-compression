import numpy as np
import cv2
import imageio.v3 as iio
from tester import compare_images

def median_filter(channel,n):
    """Zwraca kanał po medianie 3x3"""
    return cv2.medianBlur(channel.astype(np.uint8), n).astype(np.int32)

def forward_YCoCgR(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)
    
    Bd = median_filter(B,3)
    Co = R - Bd
    Cod = median_filter(Co,3)
    Gd = median_filter(G,3)
    Cg = - B - np.floor(Cod/2) + Gd
    Cgd = median_filter(Cg,3)
    Y = G - np.ceil(Cgd/2)
    
    return np.stack((Y, Co, Cg), axis=2).astype(np.int32)

def inverse_YCoCgR(rct_img):
    Y  = rct_img[:,:,0].astype(np.int32)
    Co = rct_img[:,:,1].astype(np.int32)
    Cg = rct_img[:,:,2].astype(np.int32)
    
    Cgd = median_filter(Cg,3)
    G = Y + np.ceil(Cgd/2)
    Cod = median_filter(Co,3)
    Gd = median_filter(G,3)
    B = - Cg - np.floor(Cod/2) + Gd
    Bd = median_filter(B,3)
    R = Co + Bd

    return np.stack((R, G, B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
ycocgr = forward_YCoCgR(img)
rec = inverse_YCoCgR(ycocgr)
compare_images(img,rec)