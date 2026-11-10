import numpy as np
from audio import FS


def stft(x, N):
    # se parte la señal en ventanas de N muestras (50% de traslape)
    # y a cada una se le aplica Hann y la FFT
    hann = np.hanning(N)
    salto = N // 2
    tramas = []
    for i in range(0, len(x) - N, salto):
        X = np.fft.rfft(x[i:i + N] * hann)
        tramas.append(np.abs(X))
    return np.array(tramas)     # filas = tiempo, columnas = frecuencia


def frecuencias(N):
    # frecuencia en Hz de cada columna de la STFT: f = k * FS / N
    return np.fft.rfftfreq(N, 1 / FS)