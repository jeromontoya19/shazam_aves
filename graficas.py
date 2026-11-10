import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfreqz
from audio import cargar, FS
from filtro import sos, pasabanda, F_MIN, F_MAX
from frecuencia import stft, frecuencias
from reconocimiento import CARPETA


def espectrograma(eje, x, N, titulo):
    S = stft(x, N)
    t = np.arange(len(S)) * (N // 2) / FS
    S_db = 20 * np.log10(S.T + 1e-10)
    eje.pcolormesh(t, frecuencias(N) / 1000, S_db, shading="auto", cmap="magma", vmin=S_db.max() - 80)
    eje.set_title(titulo, fontsize=9)
    eje.set_xlabel("Tiempo (s)")
    eje.set_ylabel("Frecuencia (kHz)")


def primer_audio(especie):
    archivos = sorted(a for a in os.listdir(os.path.join(CARPETA, especie)) if a.lower().endswith((".mp3", ".wav")))
    return cargar(os.path.join(CARPETA, especie, archivos[0]))[:FS * 10]    # primeros 10 s


especies = sorted(e for e in os.listdir(CARPETA) if os.path.isdir(os.path.join(CARPETA, e)))

# Figura 1: un espectrograma por especie
filas = (len(especies) + 2) // 3
fig, ejes = plt.subplots(filas, 3, figsize=(15, 4 * filas))
for eje, especie in zip(ejes.flat, especies):
    espectrograma(eje, primer_audio(especie), 1024, especie.replace("_", " "))
plt.tight_layout()
plt.savefig("espectrogramas_especies.png", dpi=150)

# Figura 2: la misma grabación con tres tamaños de ventana
x = primer_audio(especies[0])
fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
for eje, N in zip(ejes, [256, 1024, 4096]):
    espectrograma(eje, x, N, f"N = {N}  ({1000 * N / FS:.0f} ms, {FS / N:.1f} Hz)")
plt.tight_layout()
plt.savefig("tamano_ventana.png", dpi=150)

# Figura 3: respuesta en frecuencia del filtro Butterworth
f, H = sosfreqz(sos, worN=8192, fs=FS)
plt.figure(figsize=(8, 4))
plt.semilogx(f[1:], 20 * np.log10(np.abs(H[1:]) ** 2 + 1e-12))   # al cuadrado: se aplica ida y vuelta
plt.axvline(F_MIN, color="gray", linestyle="--")
plt.axvline(F_MAX, color="gray", linestyle="--")
plt.ylim(-80, 5)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Ganancia (dB)")
plt.title("Filtro Butterworth pasabanda 300 Hz - 10 kHz")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("filtro.png", dpi=150)

# Figura 4: una grabación antes y después del filtro
fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
espectrograma(ejes[0], x, 1024, "Antes del filtro")
espectrograma(ejes[1], pasabanda(x), 1024, "Después del filtro")
plt.tight_layout()
plt.savefig("antes_despues_filtro.png", dpi=150)

plt.show()
