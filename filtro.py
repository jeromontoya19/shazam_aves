from scipy.signal import butter, sosfiltfilt
from audio import FS

F_MIN = 300      # Hz. Por debajo: viento, motores, golpes al micrófono
F_MAX = 10000    # Hz. Por encima: casi solo ruido
ORDEN = 4

# coeficientes del filtro Butterworth pasabanda (se calculan una sola vez)
sos = butter(ORDEN, [F_MIN, F_MAX], btype="bandpass", fs=FS, output="sos")


def pasabanda(x):
    # filtra la señal hacia adelante y hacia atrás para que no quede desfasada
    return sosfiltfilt(sos, x)