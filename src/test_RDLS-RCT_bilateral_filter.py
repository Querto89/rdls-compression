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

def forward_RCT_RDLS(img):
    R = img[:,:,2].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,0].astype(np.int32)

    Gd = bilateral_filter(G, d=5, sigmaColor=50, sigmaSpace=50)
    Cv = R - Gd
    Cu = B - Gd
    Cvd = bilateral_filter(Cv, d=5, sigmaColor=50, sigmaSpace=50)
    Cud = bilateral_filter(Cu, d=5, sigmaColor=50, sigmaSpace=50)
    t = ( Cvd + Cud ) >> 2
    Y = G + np.floor(t)

    return np.stack((Y, Cu, Cv), axis=2)

def inverse_RCT_RDLS(rct_img):
    Y  = rct_img[:,:,0].astype(np.int32)
    Cv = rct_img[:,:,1].astype(np.int32)
    Cu = rct_img[:,:,2].astype(np.int32)

    Cvd = bilateral_filter(Cv, d=5, sigmaColor=50, sigmaSpace=50)
    Cud = bilateral_filter(Cu, d=5, sigmaColor=50, sigmaSpace=50)
    t = ( Cvd + Cud ) >> 2
    G = Y - t
    Gd = bilateral_filter(G, d=5, sigmaColor=50, sigmaSpace=50)
    B = Cu + Gd
    R = Cv + Gd

    return np.stack((R,G,B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
rct = forward_RCT_RDLS(img)
rec = inverse_RCT_RDLS(rct)
compare_images(img,rec)