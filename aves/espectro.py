"""
Transformada de Fourier de tiempo corto (STFT) y espectrograma.

    X[m, k] = Σ_{n=0}^{N-1} x[n + mH] · w[n] · e^{-j2πkn/N}

m: marco (t_m = mH/fs), k: bin de frecuencia (f_k = k·fs/N), H: salto, w: ventana.
Cada fila es la FFT de un pedazo de N muestras: O(N log N) en lugar de O(N²).
"""

import numpy as np
from scipy.signal import get_window

NOMBRES_SCIPY = {"rectangular": "boxcar"}


def ventana(p):
    """Ventana periódica de N muestras (la adecuada para análisis espectral)."""
    return get_window(NOMBRES_SCIPY.get(p.ventana, p.ventana), p.n_ventana)


def stft(x, p):
    """Devuelve X (marcos × bins, complejo), los tiempos de cada marco (s) y las
    frecuencias de cada bin (Hz)."""
    n = p.n_ventana
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    marcos = np.lib.stride_tricks.sliding_window_view(x, n)[::p.salto]
    X = np.fft.rfft(marcos * ventana(p), axis=1)
    t = (np.arange(len(marcos)) * p.salto + n / 2) / p.fs
    f = np.fft.rfftfreq(n, 1 / p.fs)
    return X, t, f


def a_db(X):
    """Magnitud en decibeles: 20·log10|X|."""
    return 20 * np.log10(np.abs(X) + 1e-10)


def banda(f, p):
    """Máscara de los bins dentro de la banda del filtro."""
    return (f >= p.f_min) & (f <= p.f_max)


def energia_banda_db(X, f, p):
    """Energía de cada marco dentro de la banda, en dB (por Parseval, la energía se
    puede medir sumando |X|² en frecuencia)."""
    return 10 * np.log10(np.sum(np.abs(X[:, banda(f, p)]) ** 2, axis=1) + 1e-20)


def marcos_activos(energia_db, piso_db, p):
    """Marcos con canto: los que superan el piso de ruido en p.umbral_actividad_db."""
    return energia_db > piso_db + p.umbral_actividad_db
