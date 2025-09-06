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

def forward_LDgEb(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)
    
    Rd = bilateral_filter(R, d=5, sigmaColor=50, sigmaSpace=50)
    Dg = - G + Rd
    Dgd = bilateral_filter(Dg, d=5, sigmaColor=50, sigmaSpace=50)
    t = np.floor( Dgd >> 2)
    L = R - t
    Ld = bilateral_filter(L, d=5, sigmaColor=50, sigmaSpace=50)
    Eb = B - Ld
    
    return np.stack((L, Dg, Eb), axis=2).astype(np.int32)

def inverse_LDgEb(rct_img):
    L = rct_img[:,:,0].astype(np.int32)
    Dg = rct_img[:,:,1].astype(np.int32)
    Eb = rct_img[:,:,2].astype(np.int32)
    
    Ld = bilateral_filter(L, d=5, sigmaColor=50, sigmaSpace=50)
    B = Eb + Ld
    Dgd = bilateral_filter(Dg, d=5, sigmaColor=50, sigmaSpace=50)
    t = np.floor( Dgd >> 2)
    R = L + t
    Rd = bilateral_filter(R, d=5, sigmaColor=50, sigmaSpace=50)
    G = - Dg + Rd

    return np.stack((R, G, B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
ldgeb = forward_LDgEb(img)
rec = inverse_LDgEb(ldgeb)
compare_images(img,rec)