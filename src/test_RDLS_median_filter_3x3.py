import numpy as np
import cv2

def median_filter_3x3(channel):
    """Zwraca kanał po medianie 3x3"""
    return cv2.medianBlur(channel.astype(np.uint8), 3).astype(np.int32) #Non linear filtering

def forward_RCT_RDLS(img):
    R = img[:,:,2].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,0].astype(np.int32)

    # Klasyczny lifting RCT
    Co = R - B
    t = B + (Co >> 1)
    Cg = G - t
    Y = t + (Cg >> 1)

    # Tutaj wpleciesz medianę (RDLS)
    # Zmieniamy tylko Cg jako przykład:
    Cg_filtered = median_filter_3x3(Cg)

    return np.stack((Y, Co, Cg_filtered), axis=2)

def inverse_RCT_RDLS(rct_img):
    Y  = rct_img[:,:,0].astype(np.int32)
    Co = rct_img[:,:,1].astype(np.int32)
    Cg = rct_img[:,:,2].astype(np.int32)

    t = Y - (Cg >> 1)
    G = Cg + t
    B = t - (Co >> 1)
    R = B + Co

    return np.stack((B, G, R), axis=2).astype(np.uint8)

img = cv2.imread('data/natural/kodim01.png')
rct_rdls = forward_RCT_RDLS(img)
rec = inverse_RCT_RDLS(rct_rdls)

diff = np.abs(img.astype(np.int32) - rec.astype(np.int32))
print("Maksymalna różnica:", diff.max())
