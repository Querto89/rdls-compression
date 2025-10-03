# entropy.py
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import math


def entropy_H0(data: np.ndarray) -> float:
    """
    Entropia H0 (zerowego rzędu) = -sum(p(x) * log2(p(x))).
    data: numpy array (dowolny kształt, rzutowany do 1D).
    """
    flat = data.flatten()
    counts = Counter(flat)
    total = len(flat)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def entropy_H1(data: np.ndarray) -> float:
    """
    Entropia H1 (pierwszego rzędu) = -sum(p(x,y) * log2(p(y|x))).
    Liczona dla par kolejnych wartości w 1D (skan liniowy obrazu).
    """
    flat = data.flatten()
    if len(flat) < 2:
        return 0.0

    # policz pary (x, y)
    pairs = list(zip(flat[:-1], flat[1:]))
    pair_counts = Counter(pairs)
    total_pairs = len(pairs)

    # rozkład p(x,y)
    p_xy = {pair: c / total_pairs for pair, c in pair_counts.items()}

    # rozkład p(x)
    x_counts = Counter(flat[:-1])
    p_x = {x: c / total_pairs for x, c in x_counts.items()}

    # entropia warunkowa
    h1 = 0.0
    for (x, y), pxy in p_xy.items():
        px = p_x[x]
        p_y_given_x = pxy / px if px > 0 else 0
        if p_y_given_x > 0:
            h1 -= pxy * math.log2(p_y_given_x)
    return h1


def get_entropy(data: np.ndarray, method_name: str, image_name: str):
    """
    Liczy H0 i H1 dla obrazu i zapisuje do CSV.
    Jeśli plik CSV istnieje, dopisuje nowe wiersze.
    """
    h0 = entropy_H0(data)
    h1 = entropy_H1(data)

    return h0,h1

def get_channel_entropy(img: np.ndarray):
    """
    img: obraz 3-kanałowy uint8
    zwraca: entropię H0 dla każdego kanału
    """
    R_H0 = entropy_H0(img[:,:,0])
    G_H0 = entropy_H0(img[:,:,1])
    B_H0 = entropy_H0(img[:,:,2])

    return {
        "R_H0": R_H0,
        "G_H0": G_H0,
        "B_H0": B_H0
    }