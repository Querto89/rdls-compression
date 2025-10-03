#color_space_transform.py
import numpy as np
from noise_filter import apply_filter

def forward_RCT(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    
    Cv = R - G
    Cu = B - G
    Y = G + np.floor((Cv + Cu)/4)
    
    return np.stack((Y, Cu, Cv), axis=2)

def inverse_RCT(rct_img):
    Y  = rct_img[:,:,0]
    Cu = rct_img[:,:,1]
    Cv = rct_img[:,:,2]

    G = Y - np.floor((Cu + Cv)/4)
    B = Cu + G
    R = Cv + G

    return np.stack((R, G, B), axis=2)

def forward_LDgEb(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    
    Dg = - G + R
    L = R - np.floor(Dg/2)
    Eb = B - L
    
    return np.stack((L, Dg, Eb), axis=2)

def inverse_LDgEb(ldgeb_img):
    L = ldgeb_img[:,:,0]
    Dg = ldgeb_img[:,:,1]
    Eb = ldgeb_img[:,:,2]

    B = Eb + L
    R = L + np.floor(Dg/2)
    G = - Dg + R

    return np.stack((R, G, B), axis=2)

def forward_YCoCgR(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    
    Co = R - B
    Cg = - B - np.floor(Co/2) + G
    Y = B + np.floor(Co/2) + np.floor(Cg/2)
    
    return np.stack((Y, Co, Cg), axis=2)

def inverse_YCoCgR(ycocgr_img):
    Y  = ycocgr_img[:,:,0]
    Co = ycocgr_img[:,:,1]
    Cg = ycocgr_img[:,:,2]
    
    G = Cg + Y - np.floor(Cg/2)
    B = Y - np.floor(Cg/2) - np.floor(Co/2)
    R = B + Co
    
    return np.stack((R, G, B), axis=2)

def forward_RDgDb(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
   
    Dg = R - G
    Db = G - B
    
    return np.stack((R, Dg, Db), axis=2)

def inverse_RDgDb(rdgd_img):
    R = rdgd_img[:,:,0]
    Dg = rdgd_img[:,:,1]
    Db = rdgd_img[:,:,2]
    
    G = R - Dg
    B = G - Db

    return np.stack((R, G, B), axis=2)

def forward_RCT_RDLS(img, method, params: dict = None):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    Gd = apply_filter(G,method,params)
    Cv = R - Gd
    Cu = B - Gd
    Cvd = apply_filter(Cv,method,params)
    Cud = apply_filter(Cu,method,params)
    Y = G + np.floor((Cvd + Cud)/4)

    return np.stack((Y, Cu, Cv), axis=2)

def inverse_RCT_RDLS(rct_img, method, params: dict = None):
    Y  = rct_img[:,:,0]
    Cu = rct_img[:,:,1]
    Cv = rct_img[:,:,2]

    Cvd = apply_filter(Cv,method,params)
    Cud = apply_filter(Cu,method,params)
    G = Y - np.floor((Cvd + Cud)/4)
    Gd = apply_filter(G,method,params)
    B = Cu + Gd
    R = Cv + Gd

    return np.stack((R,G,B), axis=2)

def forward_LDgEb_RDLS(img, method, params: dict = None):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    
    Rd = apply_filter(R,method,params)
    Dg = - G + Rd
    Dgd = apply_filter(Dg,method,params)
    L = R - np.floor(Dgd/2)
    Ld = apply_filter(L,method,params)
    Eb = B - Ld
    
    return np.stack((L, Dg, Eb), axis=2)

def inverse_LDgEb_RDLS(ldgeb_img, method, params: dict = None):
    L = ldgeb_img[:,:,0]
    Dg = ldgeb_img[:,:,1]
    Eb = ldgeb_img[:,:,2]
    
    Ld = apply_filter(L,method,params)
    B = Eb + Ld
    Dgd = apply_filter(Dg,method,params)
    R = L + np.floor(Dgd/2)
    Rd = apply_filter(R,method,params)
    G = - Dg + Rd

    return np.stack((R, G, B), axis=2)

def forward_YCoCgR_RDLS(img, method, params: dict = None):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    
    Bd = apply_filter(B,method,params)
    Co = R - Bd
    Cod = apply_filter(Co,method,params)
    Gd = apply_filter(G,method,params)
    Cg = - B - np.floor(Cod/2) + Gd
    Cgd = apply_filter(Cg,method,params)
    Y = G - np.ceil(Cgd/2)
    
    return np.stack((Y, Co, Cg), axis=2)

def inverse_YCoCgR_RDLS(ycocgr_img, method, params: dict = None):
    Y  = ycocgr_img[:,:,0]
    Co = ycocgr_img[:,:,1]
    Cg = ycocgr_img[:,:,2]
    
    Cgd = apply_filter(Cg,method,params)
    G = Y + np.ceil(Cgd/2)
    Cod = apply_filter(Co,method,params)
    Gd = apply_filter(G,method,params)
    B = - Cg - np.floor(Cod/2) + Gd
    Bd = apply_filter(B,method,params)
    R = Co + Bd

    return np.stack((R, G, B), axis=2)

def forward_RDgDb_RDLS(img, method, params: dict = None):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    Gd = apply_filter(G,method,params)
    Db = - B + Gd
    Rd = apply_filter(R,method,params)
    Dg = - G + Rd

    return np.stack((R, Dg, Db), axis=2)

def inverse_RDgDb_RDLS(rdgd_img, method, params: dict = None):
    R = rdgd_img[:,:,0]
    Dg = rdgd_img[:,:,1]
    Db = rdgd_img[:,:,2]
    
    Rd = apply_filter(R,method,params)
    G = - Dg + Rd
    Gd = apply_filter(G,method,params)
    B = - Db + Gd

    return np.stack((R, G, B), axis=2)

def color_space_transform(img: np.ndarray, color_space: str = "RCT", filter_method: str = "bilateral", filter_params: dict = None )-> np.ndarray:
    img_frwd = None
    img_rec = None
    match(color_space):
        case "RCT":
            img_frwd = forward_RCT(img)
            img_rec = inverse_RCT(img_frwd)
        case "LDgEb":
            img_frwd = forward_LDgEb(img)
            img_rec = inverse_LDgEb(img_frwd)
        case "YCoCg-R":
            img_frwd = forward_YCoCgR(img)
            img_rec = inverse_YCoCgR(img_frwd)
        case "RDgDb":
            img_frwd = forward_RDgDb(img)
            img_rec = inverse_RDgDb(img_frwd)
        case "RDLS-RCT":
            img_frwd = forward_RCT_RDLS(img,filter_method,filter_params)
            img_rec = inverse_RCT_RDLS(img_frwd,filter_method,filter_params)
        case "RDLS-LDgEb":
            img_frwd = forward_LDgEb_RDLS(img,filter_method,filter_params)
            img_rec = inverse_LDgEb_RDLS(img_frwd,filter_method,filter_params)
        case "RDLS-YCoCg-R":
            img_frwd = forward_YCoCgR_RDLS(img,filter_method,filter_params)
            img_rec = inverse_YCoCgR_RDLS(img_frwd,filter_method,filter_params)
        case "RDLS-RDgDb":
            img_frwd = forward_RDgDb_RDLS(img,filter_method,filter_params)
            img_rec = inverse_RDgDb_RDLS(img_frwd,filter_method,filter_params)
        case _:
            raise ValueError(f"Nieznany typ przestrzeni barw: {color_space}")

    return img_rec