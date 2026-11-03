"""
Exactitud de los motores sobre fragmentos que el modelo no ha visto.
"""

import numpy as np

from .audio import agregar_ruido
from .modelo import combinar, reconocer

MOTORES = ("Huellas (A)", "MFCC (B)", "Combinado")


def _prediccion(especies, p):
    return None if p is None else especies[int(np.argmax(p))]


def evaluar(modelo, clips, snr_db=None, ruido=None, semilla=0):
    """Reconoce cada fragmento (con ruido si se pide) y devuelve las predicciones:
    [{"real", "p_a", "p_b", <motor>: especie predicha}]."""
    rng = np.random.default_rng(semilla)
    predicciones = []
    for especie, x in clips:
        if snr_db is not None:
            x = agregar_ruido(x, snr_db, rng, ruido)
        res = reconocer(modelo, x)
        predicciones.append({
            "real": especie, "p_a": res.p_a, "p_b": res.p_b,
            MOTORES[0]: _prediccion(res.especies, res.p_a),
            MOTORES[1]: _prediccion(res.especies, res.p_b),
            MOTORES[2]: _prediccion(res.especies, res.p),
        })
    return predicciones


def exactitud(predicciones, motor):
    """Fracción de fragmentos bien reconocidos (un motor sin respuesta cuenta como error)."""
    return float(np.mean([f[motor] == f["real"] for f in predicciones])) if predicciones else 0.0


def exactitudes(predicciones):
    return {m: exactitud(predicciones, m) for m in MOTORES}


def exactitud_por_peso(predicciones, especies, pesos):
    """Exactitud del motor combinado para varios pesos w (p = w·p_A + (1 − w)·p_B)."""
    resultado = []
    for w in pesos:
        aciertos = [_prediccion(especies, combinar(f["p_a"], f["p_b"], w)) == f["real"]
                    for f in predicciones]
        resultado.append(float(np.mean(aciertos)) if aciertos else 0.0)
    return resultado


def matriz_confusion(predicciones, especies, motor=MOTORES[2]):
    """matriz[i, j] = fragmentos de la especie i que se reconocieron como la j."""
    indice = {e: i for i, e in enumerate(especies)}
    matriz = np.zeros((len(especies), len(especies)), dtype=int)
    for f in predicciones:
        if f[motor] is not None and f["real"] in indice:
            matriz[indice[f["real"]], indice[f[motor]]] += 1
    return matriz
