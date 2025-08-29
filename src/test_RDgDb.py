import numpy as np
import cv2

def forward_RDgDb(img):
    R = img[:,:,2].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,0].astype(np.int32)
   
    Dg = R - G
    Db = G - B
    
    return np.stack((R, Dg, Db), axis=2).astype(np.int32)

def inverse_RDgDb(rdgd_img):
    R = rdgd_img[:,:,0].astype(np.int32)
    Dg = rdgd_img[:,:,1].astype(np.int32)
    Db = rdgd_img[:,:,2].astype(np.int32)
    
    G = R - Dg
    B = G - Db
    return np.stack((B, G, R), axis=2).astype(np.uint8)

# Test odwrotności
img = cv2.imread('data/natural/kodim04.png')
rdgd = forward_RDgDb(img)
rec = inverse_RDgDb(rdgd)

diff = np.abs(img.astype(np.int32) - rec.astype(np.int32))
print("Maksymalna różnica:", diff.max())