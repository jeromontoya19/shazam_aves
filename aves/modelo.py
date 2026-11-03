"""
Entrenar y reconocer con los dos motores y combinar sus probabilidades:

    p = w·p_A + (1 − w)·p_B        (w = PESO_HUELLAS)

La incertidumbre se mide con la entropía normalizada H = −Σ p_k log p_k / log K:
vale 0 % cuando una especie tiene toda la probabilidad y 100 % cuando todas empatan.
"""

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import descriptores, huellas
from .audio import cargar, preprocesar
from .config import MODELO, PESO_HUELLAS, VOTOS_MINIMOS
from .espectro import energia_banda_db, marcos_activos, stft


@dataclass
class Modelo:
    p: object                   # Parametros con los que se entrenó
    especies: list
    nombres: dict               # especie -> nombre común
    base: huellas.BaseHuellas   # motor A
    clasificador: object        # motor B (None si hay menos de dos especies)
    conjunto: str               # "entrenamiento" o "todo"
    fragmentos: int

    def nombre(self, especie):
        comun = self.nombres.get(especie, "")
        return f"{especie} ({comun})" if comun and comun != especie else especie


@dataclass
class Resultado:
    especies: list
    p: object                   # probabilidades combinadas (None si no hay evidencia)
    p_a: object
    p_b: object
    votos: dict                 # especie -> (votos alineados, grabación, desfase)
    hay_canto: bool
    X: np.ndarray
    t: np.ndarray
    f: np.ndarray
    picos: np.ndarray
    coincidencias: np.ndarray

    @property
    def ganadora(self):
        return None if self.p is None else self.especies[int(np.argmax(self.p))]

    @property
    def confianza(self):
        return 0.0 if self.p is None else float(np.max(self.p))

    @property
    def incertidumbre(self):
        if self.p is None or len(self.p) < 2:
            return 1.0
        q = self.p[self.p > 0]
        return float(-np.sum(q * np.log(q)) / np.log(len(self.p)))

    def ranking(self, p=None, n=3):
        p = self.p if p is None else p
        if p is None:
            return []
        return [(self.especies[i], float(p[i])) for i in np.argsort(p)[::-1][:n]]


def combinar(p_a, p_b, peso=PESO_HUELLAS):
    """Si uno de los motores no tiene evidencia, manda el otro."""
    if p_a is None:
        return p_b
    if p_b is None:
        return p_a
    return peso * p_a + (1 - peso) * p_b


def entrenar(grabaciones, p, nombres=None, conjunto="entrenamiento", informar=print):
    """grabaciones: lista de (especie, ruta). Construye la base de huellas con las
    grabaciones completas y entrena el clasificador con fragmentos de p.segmento_s."""
    base = huellas.BaseHuellas()
    vectores, etiquetas = [], []
    largo, paso = p.marcos(1000 * p.segmento_s), p.marcos(1000 * p.salto_segmento_s)

    for especie, ruta in grabaciones:
        X, _, f = stft(preprocesar(cargar(ruta, p), p), p)
        picos = huellas.constelacion(X, f, p)
        lista = huellas.huellas(picos, p)
        base.agregar(especie, lista)

        energia = energia_banda_db(X, f, p)
        piso = np.median(energia)            # piso de ruido de toda la grabación
        n = 0
        for inicio in range(0, max(1, len(X) - largo + 1), paso):
            activos = marcos_activos(energia[inicio:inicio + largo], piso, p)
            if activos.mean() >= p.fraccion_activa_min:
                vectores.append(descriptores.vector(X[inicio:inicio + largo], activos, p))
                etiquetas.append(especie)
                n += 1
        informar(f"  {Path(ruta).stem:12s} {especie:30s} {len(picos):6d} picos "
                 f"{len(lista):8d} huellas {n:4d} fragmentos")

    clasificador = None
    if len(set(etiquetas)) >= 2:
        clasificador = descriptores.nuevo_clasificador().fit(np.array(vectores), etiquetas)
    return Modelo(p=p, especies=sorted(set(base.especie_de)), nombres=nombres or {},
                  base=base, clasificador=clasificador, conjunto=conjunto,
                  fragmentos=len(vectores))


def reconocer(modelo, x):
    """x: señal cruda a modelo.p.fs. Preprocesa, consulta los dos motores y combina."""
    p = modelo.p
    X, t, f = stft(preprocesar(x, p), p)
    energia = energia_banda_db(X, f, p)
    activos = marcos_activos(energia, np.median(energia), p)

    picos = huellas.constelacion(X, f, p)
    lista = huellas.huellas(picos, p)
    votos = modelo.base.buscar(lista, p)
    p_a = huellas.probabilidades(votos, modelo.especies, VOTOS_MINIMOS)
    p_b = descriptores.probabilidades(modelo.clasificador,
                                      descriptores.vector(X, activos, p), modelo.especies)

    res = Resultado(especies=modelo.especies, p=combinar(p_a, p_b), p_a=p_a, p_b=p_b,
                    votos=votos, hay_canto=activos.mean() >= p.fraccion_activa_min,
                    X=X, t=t, f=f, picos=picos, coincidencias=np.empty((0, 2), dtype=int))
    if res.ganadora in votos:
        _, g, desfase = votos[res.ganadora]
        res.coincidencias = modelo.base.coincidencias(lista, g, desfase, p)
    return res


def guardar(modelo, ruta=MODELO):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)


def cargar_modelo(ruta=MODELO):
    if not ruta.exists():
        sys.exit("No hay modelo entrenado. Corre primero:  python entrenar.py")
    with open(ruta, "rb") as f:
        return pickle.load(f)
