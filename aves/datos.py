"""
Las grabaciones de data/raw/<Genero_especie>/ y los fragmentos para evaluar.

El nombre de cada carpeta es la especie. Basta con poner ahí los audios: la partición en
entrenamiento y prueba se calcula aquí, por grabación completa, para que una grabación
nunca quede en los dos conjuntos.
"""

import random
import re
import sys

import numpy as np

from .audio import cargar, preprocesar
from .config import EXTENSIONES, FRACCION_ENTRENAMIENTO, MAX_SEGUNDOS, RAW, RUIDO, SEMILLA
from .espectro import energia_banda_db, marcos_activos, stft

NUMERO_XC = re.compile(r"(?<![a-z])XC\s*-?\s*(\d+)", re.IGNORECASE)


def _audios(carpeta):
    """Audios de una carpeta de especie. Un número XC repetido se omite: si la misma
    grabación cayera en entrenamiento y en prueba, el modelo ya conocería la respuesta."""
    audios, vistos = [], set()
    for archivo in sorted(carpeta.iterdir(), key=lambda a: a.name):
        if archivo.suffix.lower() not in EXTENSIONES:
            print(f"AVISO: se ignora {carpeta.name}/{archivo.name}: conviértelo a mp3 o wav.")
            continue
        xc = NUMERO_XC.search(archivo.name)
        if xc and xc[1] in vistos:
            print(f"AVISO: se ignora {carpeta.name}/{archivo.name}: grabación repetida.")
            continue
        if xc:
            vistos.add(xc[1])
        audios.append(archivo)
    return audios


def particion():
    """(entrenamiento, prueba), cada uno como lista de (especie, ruta). Con semilla fija:
    mientras no cambien los audios, la partición es siempre la misma. Cada especie baraja
    por separado para que agregar una especie no cambie la partición de las demás."""
    if not RAW.exists():
        sys.exit("No existe data/raw/. Crea data/raw/<Genero_especie>/ y pon ahí los audios.")
    entrenamiento, prueba = [], []
    for carpeta in sorted((c for c in RAW.iterdir() if c.is_dir()), key=lambda c: c.name):
        especie = carpeta.name.replace("_", " ")
        audios = _audios(carpeta)
        barajados = random.Random(f"{SEMILLA}-{especie}").sample(audios, len(audios))
        n = max(1, round(len(audios) * FRACCION_ENTRENAMIENTO))
        entrenamiento += [(especie, a) for a in barajados[:n]]
        prueba += [(especie, a) for a in barajados[n:]]
    return entrenamiento, prueba


def fragmentos(lista, p, segundos):
    """Corta cada grabación en fragmentos seguidos de `segundos` (señal cruda, como la
    grabaría el micrófono) y se queda con los que tienen canto. Devuelve [(especie, x)]."""
    largo = int(segundos * p.fs)
    resultado = []
    for especie, ruta in lista:
        x = cargar(ruta, p, MAX_SEGUNDOS)
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
    partes = [cargar(a, p) for a in archivos if a.suffix.lower() in EXTENSIONES]
    return np.concatenate(partes) if partes else None
