#img_compares.py
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as PSNR
import matplotlib.gridspec as gridspec

def compare_images(img_org: np.ndarray, img_rec: np.ndarray):
    # --- Test PSNR
    data_range = img_org.max() - img_org.min()
    psnr = PSNR(img_org, img_rec, data_range=data_range)

    # --- Test zgodności bitowej
    bit_acc = (img_org.shape == img_rec.shape)
    
    # --- Bit-perfect check
    bit_perfect = np.array_equal(img_org,img_rec)

    # --- Hash check
    h1 = hashlib.sha256(img_org.tobytes()).hexdigest()
    h2 = hashlib.sha256(img_rec.tobytes()).hexdigest()
    hash_check = (h1==h2)

    # --- Różnice pixeli
    diff = img_rec.astype(np.int32) - img_org.astype(np.int32)
    max_diff = np.abs(diff).max()
    nonzero = np.count_nonzero(diff)

    return {
        "psnr": psnr,
        "same_shape": bit_acc,
        "bit_perfect": bit_perfect,
        "hash_check": hash_check,
        "max_diff": max_diff,
        "nonzero_diff": nonzero,
    }