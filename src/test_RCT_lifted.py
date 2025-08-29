import numpy as np
import cv2

def forward_RCT(img):
    R = img[:,:,2].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,0].astype(np.int32)
    
    Cu = B - G
    Cv = R - G
    Y = G + np.floor((Cu+Cv)>>2)
    
    return np.stack((Y, Cu, Cv), axis=2).astype(np.int32)

def inverse_RCT(rct_img):
    Y  = rct_img[:,:,0].astype(np.int32)
    Cu = rct_img[:,:,1].astype(np.int32)
    Cv = rct_img[:,:,2].astype(np.int32)
    
    G = Y - np.floor((Cu+Cv)>>2)
    R = Cv + G
    B = Cu + G
    
    return np.stack((B, G, R), axis=2).astype(np.uint8)

# Test odwrotności
img = cv2.imread('data/natural/kodim01.png')  # przykładowy obraz
rct = forward_RCT(img)
rec = inverse_RCT(rct)

diff = np.abs(img.astype(np.int32) - rec.astype(np.int32))
print("Maksymalna różnica:", diff.max())