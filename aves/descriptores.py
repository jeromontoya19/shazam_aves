"""
Motor B: coeficientes cepstrales en escala mel (MFCC) y un clasificador.

    MFCC = DCT( log( banco_mel · |X|² ) )

El banco mel agrupa los bins de la FFT en bandas que imitan la escala del oído; la
DCT (pariente de la transformada de Fourier) resume la forma del espectro en pocos
números. Se descarta c0: un cambio de volumen g suma log(g²) a todos los log-mel, y la
DCT de una constante solo afecta a c0, así que el resto no depende del volumen.

Cada fragmento se resume con la media y la desviación de sus MFCC en los marcos con
canto (2·n_mfcc números). El clasificador es una regresión logística multinomial:
p_k = e^{z_k} / Σ_j e^{z_j}, que ya da probabilidades.
"""

from functools import lru_cache

import librosa
import numpy as np
from scipy.fft import dct
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@lru_cache(maxsize=8)
def banco_mel(p):
    return librosa.filters.mel(sr=p.fs, n_fft=p.n_ventana, n_mels=p.n_mel,
                               fmin=p.f_min, fmax=p.f_max)


def mfcc(X, p):
    """MFCC de cada marco: arreglo (marcos × n_mfcc)."""
    mel = (np.abs(X) ** 2) @ banco_mel(p).T
    return dct(np.log(mel + 1e-12), type=2, norm="ortho", axis=1)[:, 1:p.n_mfcc + 1]


def vector(X, activos, p):
    """Descriptor de un fragmento: media y desviación de los MFCC de los marcos con canto."""
    c = mfcc(X, p)
    if activos.sum() >= 5:
        c = c[activos]
    return np.concatenate([c.mean(axis=0), c.std(axis=0)])


def nuevo_clasificador():
    """Estandariza cada descriptor (media 0, desviación 1) y aplica la regresión logística."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced"))


def probabilidades(clasificador, v, especies):
    """Probabilidades del clasificador en el orden de `especies`."""
    if clasificador is None:
        return None
    proba = dict(zip(clasificador.classes_, clasificador.predict_proba([v])[0]))
    return np.array([proba.get(e, 0.0) for e in especies])
