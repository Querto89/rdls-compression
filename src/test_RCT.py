import numpy as np
import imageio.v3 as iio
from tester import compare_images

def forward_RCT(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)
    
    Cu = B - G
    Cv = R - G
    t = np.floor( (Cu + Cv) >> 2 )
    Y = G + t
    
    return np.stack((Y, Cu, Cv), axis=2).astype(np.int32)

def inverse_RCT(rct_img):
    Y  = rct_img[:,:,0].astype(np.int32)
    Cu = rct_img[:,:,1].astype(np.int32)
    Cv = rct_img[:,:,2].astype(np.int32)

    t = np.floor( ( Cu + Cv ) >> 2 )
    G = Y - t
    R = Cv + G
    B = Cu + G

    return np.stack((R, G, B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
rct = forward_RCT(img)
rec = inverse_RCT(rct)
compare_images(img,rec)