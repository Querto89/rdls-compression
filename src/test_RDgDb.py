import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

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

img = iio.imread('data/natural/kodim01.png')  # przykładowy obraz
rdgdb = forward_RDgDb(img)
rec = inverse_RDgDb(rdgdb)

# Obliczenie PSNR
psnr_value = peak_signal_noise_ratio(img, rec, data_range=255)
print("PSNR:", psnr_value)
diff = np.abs(img.astype(np.int32) - rec.astype(np.int32))
print("Maksymalna różnica:", diff.max())

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