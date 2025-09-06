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

def forward_YCoCgR(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)
    
    Bd = bilateral_filter(B, d=5, sigmaColor=50, sigmaSpace=50)
    Co = R - Bd
    Cod = bilateral_filter(Co, d=5, sigmaColor=50, sigmaSpace=50)
    Gd = bilateral_filter(G, d=5, sigmaColor=50, sigmaSpace=50)
    Cg = - B - np.floor(Cod/2) + Gd
    Cgd = bilateral_filter(Cg, d=5, sigmaColor=50, sigmaSpace=50)
    Y = G - np.ceil(Cgd/2)
    
    return np.stack((Y, Co, Cg), axis=2).astype(np.int32)

def inverse_YCoCgR(rct_img):
    Y  = rct_img[:,:,0].astype(np.int32)
    Co = rct_img[:,:,1].astype(np.int32)
    Cg = rct_img[:,:,2].astype(np.int32)
    
    Cgd = bilateral_filter(Cg, d=5, sigmaColor=50, sigmaSpace=50)
    G = Y + np.ceil(Cgd/2)
    Cod = bilateral_filter(Co, d=5, sigmaColor=50, sigmaSpace=50)
    Gd = bilateral_filter(G, d=5, sigmaColor=50, sigmaSpace=50)
    B = - Cg - np.floor(Cod/2) + Gd
    Bd = bilateral_filter(B, d=5, sigmaColor=50, sigmaSpace=50)
    R = Co + Bd

    return np.stack((R, G, B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
ycocgr = forward_YCoCgR(img)
rec = inverse_YCoCgR(ycocgr)
compare_images(img,rec)