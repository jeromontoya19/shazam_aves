import os
import numpy as np
from audio import cargar
from filtro import pasabanda, F_MIN, F_MAX
from frecuencia import stft, frecuencias

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
N_PRUEBA = 2            # grabaciones de cada especie que se dejan para probar
N_RECONOCER = 4096      # ventana para reconocer un audio nuevo


def huella(x, N):
    x = pasabanda(x)                                # 1. filtro
    S = stft(x, N)                                  # 2. al dominio de la frecuencia
    f = frecuencias(N)
    S = S[:, (f >= F_MIN) & (f <= F_MAX)]           #    solo las frecuencias de la banda
    energia = S.sum(axis=1)
    S = S[energia > np.percentile(energia, 75)]     # 3. momentos donde suena el canto
    h = 20 * np.log10(S.mean(axis=0) + 1e-10)       # 4. espectro promedio en dB
    return (h - h.mean()) / h.std()                 #    normalizado


def parecido(h1, h2):
    return np.mean(h1 * h2)     # correlación entre dos huellas (de -1 a 1)


def ranking(h, base):
    # el mejor parecido de cada especie, de mayor a menor
    mejores = {}
    for especie, hb in base:
        p = parecido(h, hb)
        if p > mejores.get(especie, -2):
            mejores[especie] = p
    return sorted(mejores.items(), key=lambda par: par[1], reverse=True)


def leer_base():
    entrenamiento = []
    prueba = []
    for especie in sorted(os.listdir(CARPETA)):
        ruta_especie = os.path.join(CARPETA, especie)
        if not os.path.isdir(ruta_especie):
            continue
        archivos = sorted(a for a in os.listdir(ruta_especie) if a.lower().endswith((".mp3", ".wav")))
        print("Cargando", especie, "-", len(archivos), "grabaciones")
        for i, archivo in enumerate(archivos):
            x = cargar(os.path.join(ruta_especie, archivo))
            # si la especie tiene pocas grabaciones, todas van a la base
            if len(archivos) > N_PRUEBA and i >= len(archivos) - N_PRUEBA:
                prueba.append((especie, archivo, x))
            else:
                entrenamiento.append((especie, archivo, x))
    return entrenamiento, prueba


def evaluar(entrenamiento, prueba, N, mostrar=False):
    base = [(especie, huella(x, N)) for especie, archivo, x in entrenamiento]
    aciertos = 0
    for especie, archivo, x in prueba:
        resultado, p = ranking(huella(x, N), base)[0]
        if resultado == especie:
            aciertos += 1
        if mostrar:
            marca = "BIEN" if resultado == especie else "MAL "
            print(f"  {marca}  {archivo[:20]:20s}  real: {especie:26s} dijo: {resultado} ({p:.2f})")
    return aciertos / len(prueba)


def reconocer(x, entrenamiento, prueba):
    if np.sqrt(np.mean(x ** 2)) < 0.005:
        print("No se escuchó nada, intenta de nuevo más cerca")
        return
    base = [(especie, huella(xb, N_RECONOCER)) for especie, archivo, xb in entrenamiento + prueba]
    resultado = ranking(huella(x, N_RECONOCER), base)
    print(f"EL AVE ES: {resultado[0][0].replace('_', ' ')}  (parecido {resultado[0][1]:.0%})")
    print("Otras opciones:")
    for especie, p in resultado[1:3]:
            print(f"   {especie.replace('_', ' ')}  ({p:.0%})")