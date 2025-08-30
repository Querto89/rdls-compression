import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

def forward_RCT(img):
    R = img[:,:,0].astype(np.int32)
    G = img[:,:,1].astype(np.int32)
    B = img[:,:,2].astype(np.int32)
    
    Y  = (R + 2*G + B) >> 2
    Co = (R - B) >> 1
    Cg = (-R + 2*G - B) >> 2
    
    return np.stack((Y, Co, Cg), axis=2).astype(np.int32)

def inverse_RCT(rct_img):
    Y  = rct_img[:,:,0].astype(np.int32)
    Co = rct_img[:,:,1].astype(np.int32)
    Cg = rct_img[:,:,2].astype(np.int32)
    
    t = Y - Cg
    R = t + Co
    G = Y + Cg
    B = t - Co
    
    return np.stack((R, G, B), axis=2).astype(np.uint8)

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
rct = forward_RCT(img)
rec = inverse_RCT(rct)

# Obliczenie PSNR
psnr_value = peak_signal_noise_ratio(img, rec, data_range=255)
print("PSNR:", psnr_value)
diff = np.abs(img.astype(np.int32) - rec.astype(np.int32))
print("\nMaksymalna różnica:", diff.max())

# Tworzenie figury z dwoma obrazami
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img)
axes[0].set_title("Oryginał")
axes[0].axis("off")

axes[1].imshow(rec)
axes[1].set_title("Rekonstrukcja")
axes[1].axis("off")

# Dodanie PSNR jako tekstu globalnego pod obrazami
fig.text(0.5, 0.1, f"PSNR: {psnr_value:.2f} dB", ha='center', fontsize=12)
fig.text(0.5, 0.05, f"Maksymalna różnica: {diff.max():.2f}", ha='center', fontsize=12)

# Dopasowanie layoutu i zapis PNG
plt.tight_layout(pad=0.5)
fig.savefig("porownanie_psnr.png", bbox_inches='tight', pad_inches=0.1, dpi=150)
plt.show()