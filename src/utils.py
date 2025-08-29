import numpy as np

def to_int32(img):
    """Konwertuje obraz uint8 (BGR) do int32 kanałów B,G,R"""
    return img.astype(np.int32)

def merge_channels(B, G, R):
    """Składa kanały B,G,R do uint8 obrazu"""
    return np.stack((B, G, R), axis=2).astype(np.uint8)
