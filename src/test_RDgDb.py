import numpy as np
import imageio.v3 as iio
from tester import compare_images

def forward_RDgDb(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)
   
    Dg = R - G
    Db = G - B
    
    return np.stack((R, Dg, Db), axis=2).astype(np.int32)

def inverse_RDgDb(rdgd_img):
    R = rdgd_img[:,:,0].astype(np.int32)
    Dg = rdgd_img[:,:,1].astype(np.int32)
    Db = rdgd_img[:,:,2].astype(np.int32)
    
    G = R - Dg
    B = G - Db
    return np.stack((R, G, B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
rdgdb = forward_RDgDb(img)
rec = inverse_RDgDb(rdgdb)
compare_images(img,rec)