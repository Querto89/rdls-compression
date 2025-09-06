import numpy as np
import cv2
import imageio.v3 as iio
from tester import compare_images

def bilateral_filter(channel, d=5, sigmaColor=75, sigmaSpace=75):
    """
    Zwraca kanał po filtrze bilateralnym.
    d - średnica sąsiedztwa
    sigmaColor - wpływ różnic kolorów
    sigmaSpace - zasięg w pikselach
    """
    return cv2.bilateralFilter(channel.astype(np.uint8), d, sigmaColor, sigmaSpace).astype(np.int32)

def forward_RDgDb(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)

    Gd = bilateral_filter(G, d=5, sigmaColor=50, sigmaSpace=50)
    Db = - B + Gd
    Rd = bilateral_filter(R, d=5, sigmaColor=50, sigmaSpace=50)
    Dg = - G + Rd

    return np.stack((R, Dg, Db), axis=2).astype(np.int32)

def inverse_RDgDb(rdgd_img):
    R = rdgd_img[:,:,0].astype(np.int32)
    Dg = rdgd_img[:,:,1].astype(np.int32)
    Db = rdgd_img[:,:,2].astype(np.int32)
    
    Rd = bilateral_filter(R, d=5, sigmaColor=50, sigmaSpace=50)
    G = - Dg + Rd
    Gd = bilateral_filter(G, d=5, sigmaColor=50, sigmaSpace=50)
    B = - Db + Gd
    return np.stack((R, G, B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
rdgdb = forward_RDgDb(img)
rec = inverse_RDgDb(rdgdb)
compare_images(img,rec)