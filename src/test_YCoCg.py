import numpy as np
import imageio.v3 as iio
from tester import compare_images

def forward_YCoCgR(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)
    
    Y  = (R + 2*G + B) >> 2
    Co = (R - B) >> 1
    Cg = (-R + 2*G - B) >> 2
    
    return np.stack((Y, Co, Cg), axis=2).astype(np.int32)

def inverse_YCoCgR(rct_img):
    Y  = rct_img[:,:,0].astype(np.int32)
    Co = rct_img[:,:,1].astype(np.int32)
    Cg = rct_img[:,:,2].astype(np.int32)
    
    t = Y - Cg
    R = t + Co
    G = Y + Cg
    B = t - Co
    
    return np.stack((R, G, B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
ycocg = forward_YCoCgR(img)
rec = inverse_YCoCgR(ycocg)
compare_images(img,rec)