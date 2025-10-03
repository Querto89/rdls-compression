#utils.py
import numpy as np

def to_int32(img):
    """Konwertuje obraz uint8 (BGR) do int32 kanałów B,G,R"""
    return img.astype(np.int32)

def merge_channels(B, G, R):
    """Składa kanały B,G,R do uint8 obrazu"""
    return np.stack((B, G, R), axis=2).astype(np.uint8)

def _to_float01(img):
    """Konwertuje wejście do float32 w zakresie [0,1] oraz zwraca metadane do odtworzenia dtype/range."""
    dtype = img.dtype
    if np.issubdtype(dtype, np.floating):
        # zakładamy, że floaty już w [0,1] (jeśli nie, użytkownik powinien skalować)
        return img.astype(np.float32), dtype, 1.0
    # dla typów całkowitych (np. uint8, uint16)
    info_max = np.iinfo(dtype).max
    return img.astype(np.float32) / info_max, dtype, info_max

def _from_float01(img_float, orig_dtype, orig_max, clip=True):
    """Konwertuje float [0,1] z powrotem do orig_dtype."""
    if clip:
        img_float = np.clip(img_float, 0.0, 1.0)
    if np.issubdtype(orig_dtype, np.floating):
        return img_float.astype(orig_dtype)
    # integer
    out = (img_float * orig_max).round()
    out = np.clip(out, 0, orig_max)
    return out.astype(orig_dtype)
