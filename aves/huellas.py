"""
Motor A: huellas tipo Shazam (Wang, 2003, "An Industrial-Strength Audio Search Algorithm").

1. Constelación: los máximos locales del espectrograma que sobresalen del piso de ruido.
2. Huellas: cada pico "ancla" (t1, f1) se empareja con los picos siguientes de su zona
   objetivo (t2, f2). La huella (f1, f2, Δt = t2 − t1) no cambia con el volumen ni con
   el momento en que empezó la grabación.
3. Búsqueda: cada huella de la consulta que también está en la base vota por la pareja
   (grabación, desfase = t_base − t_consulta). Si la consulta y una grabación de la base
   tienen el mismo canto, sus votos se amontonan en un solo desfase; las coincidencias
   al azar se reparten entre muchos.
"""

from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import maximum_filter

from .espectro import a_db, banda


def constelacion(X, f, p):
    """Picos del espectrograma como arreglo de filas (marco, bin), ordenado por marco."""
    S = a_db(X)
    en_banda = banda(f, p)
    S_banda = np.where(en_banda[None, :], S, -np.inf)           # fuera de la banda no hay picos
    piso = np.median(S[:, en_banda])
    vecindad = (p.marcos(p.vecindad_ms), p.bins(p.vecindad_hz))
    es_maximo = maximum_filter(S_banda, size=vecindad, mode="constant", cval=-np.inf) == S_banda
    m, k = np.nonzero(es_maximo & (S_banda > piso + p.umbral_db))

    duracion_s = X.shape[0] * p.salto / p.fs
    tope = max(1, int(p.picos_por_segundo * duracion_s))
    if len(m) > tope:                                           # se quedan los más fuertes
        fuertes = np.argsort(S[m, k])[::-1][:tope]
        m, k = m[fuertes], k[fuertes]
    orden = np.lexsort((k, m))
    return np.column_stack([m[orden], k[orden]])


def codificar(f1, f2, dt, p):
    """Empaqueta (f1, f2, Δt), ya cuantizados, en un solo entero de 36 bits."""
    qf, qt = p.bins(p.cuantizacion_hz), p.marcos(p.cuantizacion_ms)
    return (int(f1 // qf) << 24) | (int(f2 // qf) << 12) | int(dt // qt)


def huellas(picos, p):
    """Lista de (código, marco del ancla, bin del ancla)."""
    dt_max, df_max = p.marcos(p.dt_max_ms), p.bins(p.df_max_hz)
    resultado = []
    for i, (t1, f1) in enumerate(picos):
        parejas = 0
        for t2, f2 in picos[i + 1:]:
            dt = t2 - t1
            if dt > dt_max:
                break
            if dt == 0 or abs(f2 - f1) > df_max:
                continue
            resultado.append((codificar(f1, f2, dt, p), int(t1), int(f1)))
            parejas += 1
            if parejas == p.fan_out:
                break
    return resultado


class BaseHuellas:
    """Tabla hash: código -> [(grabación, marco)], más la especie de cada grabación."""

    def __init__(self):
        self.tabla = defaultdict(list)
        self.especie_de = []

    def agregar(self, especie, lista):
        g = len(self.especie_de)
        self.especie_de.append(especie)
        for codigo, t, _ in lista:
            self.tabla[codigo].append((g, t))

    def __len__(self):
        return sum(len(v) for v in self.tabla.values())

    def buscar(self, lista, p):
        """Votos alineados por especie: {especie: (votos, grabación, desfase)}.
        Para cada grabación cuenta el desfase con más votos, y cada especie se
        queda con su mejor grabación."""
        qt = p.marcos(p.cuantizacion_ms)
        votos = Counter()
        for codigo, tq, _ in lista:
            for g, tb in self.tabla.get(codigo, ()):
                votos[(g, (tb - tq) // qt)] += 1
        mejor = {}
        for (g, desfase), n in votos.items():
            especie = self.especie_de[g]
            if n > mejor.get(especie, (0,))[0]:
                mejor[especie] = (n, g, desfase)
        return mejor

    def coincidencias(self, lista, g, desfase, p):
        """Picos ancla de la consulta cuyas huellas votaron por (g, desfase): son los
        que se marcan en la gráfica."""
        qt = p.marcos(p.cuantizacion_ms)
        anclas = set()
        for codigo, tq, fq in lista:
            if any(gb == g and (tb - tq) // qt == desfase for gb, tb in self.tabla.get(codigo, ())):
                anclas.add((tq, fq))
        return np.array(sorted(anclas), dtype=int).reshape(-1, 2)


def probabilidades(mejor, especies, minimo):
    """Convierte los votos alineados en probabilidades. A cada especie se le resta el
    nivel de coincidencias al azar, estimado con la mediana de los votos de todas, y
    se normaliza: p_k = v'_k / Σ v'_j. Devuelve None si ninguna llega a `minimo` votos
    o si ninguna sobresale."""
    v = np.array([mejor.get(e, (0,))[0] for e in especies], dtype=float)
    if v.max(initial=0) < minimo:
        return None
    v = np.clip(v - np.median(v), 0, None)
    return v / v.sum() if v.sum() > 0 else None
