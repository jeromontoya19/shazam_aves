"""
La señal antes de Fourier: cargarla, grabarla y preprocesarla.

Preprocesado = quitar la componente DC  ->  filtro pasa-banda Butterworth  ->  normalizar.
"""

import numpy as np
import librosa
from scipy import signal


def cargar(ruta, p):
    """Lee cualquier audio (wav, mp3, ...) en mono y a p.fs muestras por segundo."""
    x, _ = librosa.load(ruta, sr=p.fs, mono=True)
    return x.astype(np.float64)


def grabar(segundos, p, cada_segundo=None):
    """Graba del micrófono. Se graba a la frecuencia propia del dispositivo y luego se
    remuestrea a p.fs, porque no todos los micrófonos aceptan 22 050 Hz."""
    import sounddevice as sd        # solo hace falta para grabar en vivo

    fs_mic = int(sd.query_devices(kind="input")["default_samplerate"])
    datos = sd.rec(int(segundos * fs_mic), samplerate=fs_mic, channels=1, dtype="float32")
    for restante in range(int(segundos), 0, -1):
        if cada_segundo:
            cada_segundo(restante)
        sd.sleep(1000)
    sd.wait()
    x = datos[:, 0].astype(np.float64)
    return librosa.resample(x, orig_sr=fs_mic, target_sr=p.fs) if fs_mic != p.fs else x


# --------------------------------- Filtro ---------------------------------
def prototipo_analogico(p):
    """Polos del Butterworth pasa-banda en el plano s (Laplace).

    El pasa-bajos de orden n tiene |H(jω)|² = 1 / (1 + (ω/ωc)^(2n)) y sus polos
    están repartidos en una semicircunferencia del semiplano izquierdo; la
    transformación s -> (s² + ω0²) / (B·s) lo convierte en pasa-banda.
    """
    w = 2 * np.pi * np.array([p.f_min, p.f_max])
    return signal.butter(p.orden_filtro, w, btype="bandpass", analog=True, output="zpk")


def disenar_filtro(p):
    """El mismo filtro, ya digital, en secciones de segundo orden. scipy lo diseña en el
    plano s y lo pasa al plano z con la transformada bilineal s = (2/T)(z − 1)/(z + 1),
    con predistorsión para que las frecuencias de corte queden exactas."""
    return signal.butter(p.orden_filtro, [p.f_min, p.f_max], btype="bandpass",
                         fs=p.fs, output="sos")


def es_estable(sos):
    """Un filtro digital es estable (BIBO) si todos sus polos están dentro del
    círculo unitario |z| < 1, igual que uno analógico lo es si están en Re(s) < 0."""
    _, polos, _ = signal.sos2zpk(sos)
    return bool(np.all(np.abs(polos) < 1))


# ------------------------------- Preprocesado -------------------------------
def normalizar(x):
    """Escala la señal para que su pico valga 1 (volumen independiente de la grabación)."""
    pico = np.max(np.abs(x)) if len(x) else 0.0
    return x / pico if pico > 0 else x


def preprocesar(x, p):
    x = x - np.mean(x)                                  # componente DC: la frecuencia 0
    x = signal.sosfiltfilt(disenar_filtro(p), x)        # ida y vuelta: no desfasa la señal
    return normalizar(x)


def agregar_ruido(x, snr_db, rng, ruido=None):
    """Suma ruido con una relación señal/ruido dada: SNR = 10·log10(P_señal / P_ruido).
    Si no se da un ruido grabado, se usa ruido blanco gaussiano."""
    if ruido is None or len(ruido) < len(x):
        ruido = rng.standard_normal(len(x))
    else:
        inicio = rng.integers(0, len(ruido) - len(x) + 1)
        ruido = ruido[inicio:inicio + len(x)]
    p_senal, p_ruido = np.mean(x ** 2), np.mean(ruido ** 2)
    if p_senal == 0 or p_ruido == 0:
        return x
    return x + ruido * np.sqrt(p_senal / (p_ruido * 10 ** (snr_db / 10)))
