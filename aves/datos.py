"""
Lectura de data/metadata.csv (lo crea preparar_cantos.py) y fragmentos para evaluar.
"""

import csv
import sys

import numpy as np

from .audio import cargar, preprocesar
from .config import BASE, METADATA, RUIDO
from .espectro import energia_banda_db, marcos_activos, stft


def leer_metadata():
    if not METADATA.exists():
        sys.exit("No existe data/metadata.csv. Corre primero:  python preparar_cantos.py")
    with open(METADATA, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def grabaciones(filas, conjunto=None):
    """[(especie, ruta_wav)] del conjunto pedido ("entrenamiento", "prueba" o None = todas)."""
    return [(fila["especie"], BASE / fila["archivo_wav"]) for fila in filas
            if conjunto is None or fila["conjunto"] == conjunto]


def nombres_comunes(filas):
    return {fila["especie"]: fila.get("nombre_comun") or fila["especie"] for fila in filas}


def fragmentos(lista, p, segundos):
    """Corta cada grabación en fragmentos seguidos de `segundos` (señal cruda, como la
    grabaría el micrófono) y se queda con los que tienen canto. Devuelve [(especie, x)]."""
    largo = int(segundos * p.fs)
    resultado = []
    for especie, ruta in lista:
        x = cargar(ruta, p)
        X, _, f = stft(preprocesar(x, p), p)
        energia = energia_banda_db(X, f, p)
        activos = marcos_activos(energia, np.median(energia), p)
        for inicio in range(0, len(x) - largo + 1, largo):
            m0 = inicio // p.salto
            if activos[m0:m0 + largo // p.salto].mean() >= p.fraccion_activa_min:
                resultado.append((especie, x[inicio:inicio + largo]))
    return resultado


def ruido_grabado(p):
    """Une los audios de data/ruido/ (si existen) en una sola señal de ruido ambiente."""
    archivos = sorted(RUIDO.glob("*.*")) if RUIDO.exists() else []
    partes = [cargar(a, p) for a in archivos if a.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}]
    return np.concatenate(partes) if partes else None
