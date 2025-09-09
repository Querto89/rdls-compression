import numpy as np
import hashlib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as PSNR
import matplotlib.gridspec as gridspec

def compare_images(img_orig: np.ndarray, img_rec: np.ndarray, title=""):
    # --- Test PSNR
    data_range = img_orig.max() - img_orig.min()
    psnr_value = PSNR(img_orig, img_rec, data_range=data_range)
    psnr_result = f"PSNR = {psnr_value:.2f} dB"

    # --- Test zgodności bitowej
    bit_acc_result = ""
    if(img_orig.shape == img_rec.shape):
        bit_acc_result = "Bit-acc: Obrazy mają takie same rozmiary"
    else:
        bit_acc_result = "Bit-acc: Obrazy mają różne rozmiary"
    
    # --- Bit-perfect check
    bit_equal = np.array_equal(img_orig,img_rec)
    bit_equal_result = f"Bit-perfect: {bit_equal}"

    # --- Hash check
    h1 = hashlib.sha256(img_orig.tobytes()).hexdigest()
    h2 = hashlib.sha256(img_rec.tobytes()).hexdigest()
    hash_test = (h1==h2)
    hash_result = f"Hash-equal: {hash_test}"

    # --- Różnice pixeli
    diff = img_rec.astype(np.int32) - img_orig.astype(np.int32)
    max_diff = np.abs(diff).max()
    nonzero = np.count_nonzero(diff)
    max_diff_result = f"Maksymalna różnica: {max_diff}"
    nonzero_result = f"Liczba różniących się pikseli: {nonzero}"

    fig = plt.figure(figsize=(12,10))
    fig.subplots_adjust(bottom=0.18, hspace=0.3)

    gs = gridspec.GridSpec(2, 2, figure=fig)

    # --- górny rząd: obrazy
    ax0 = fig.add_subplot(gs[0,0])
    ax0.imshow(img_orig)
    ax0.set_title("Oryginalny")
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0,1])
    ax1.imshow(img_rec)
    ax1.set_title("Zrekonstruowany")
    ax1.axis('off')

    # --- dolny rząd: histogram + mapa błędu
    ax2 = fig.add_subplot(gs[1,0])
    ax2.hist(diff.ravel(), bins=513, range=(-256, 256), color="blue")
    ax2.set_title("Histogram różnic")
    ax2.set_xlabel("Różnica (rec - orig)")
    ax2.set_ylabel("Liczba pikseli")
    ax2.set_aspect('auto')  # wymusza równy układ w hist

    ax3 = fig.add_subplot(gs[1,1])
    im = ax3.imshow(np.abs(diff).sum(axis=2), cmap="hot")
    ax3.set_title("Mapa błędu")
    ax3.set_aspect('auto')  # wymusza proporcje
    plt.colorbar(im, ax=ax3)

    # Dodanie wyników testów pod obrazami
    results = [bit_acc_result, bit_equal_result, max_diff_result, nonzero_result, psnr_result]
    for i, text in enumerate(results):
        fig.text(0.5, 0.01 + i*0.03, text, ha='center', fontsize=12)

    plt.show()
    return 0