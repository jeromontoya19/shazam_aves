"""
experimentos.py
Experimentos para la sección de Resultados del informe. Cada uno guarda su gráfica
en resultados/ e imprime la tabla con los números.

Uso:
    python experimentos.py filtro     -> polos (planos s y z) y respuesta del Butterworth
    python experimentos.py fuga       -> fuga espectral de un tono puro con cada ventana
    python experimentos.py ventana    -> exactitud contra el tamaño de ventana N (reentrena)
    python experimentos.py ventanas   -> exactitud con ventana rectangular, Hamming y Hann (reentrena)
    python experimentos.py ruido      -> exactitud contra la relación señal/ruido
    python experimentos.py duracion   -> exactitud contra la duración del fragmento
    python experimentos.py todos

'filtro' y 'fuga' no necesitan datos. Los demás entrenan con el conjunto de
entrenamiento y evalúan con el de prueba. Si todavía no hay conjunto de prueba,
agrega --con-entrenamiento para probar que el código corre (los números saldrán
inflados, porque el modelo ya conoce esos audios).

Para el experimento de ruido con ruido real, pon grabaciones de ruido ambiente
(tráfico, viento, lluvia...) en data/ruido/; si no hay, se usa ruido blanco.
"""

import sys
from dataclasses import replace

import numpy as np

from aves import graficas
from aves.audio import disenar_filtro, es_estable, prototipo_analogico
from aves.config import RESULTADOS, SEGUNDOS_GRABACION, Parametros
from aves.datos import fragmentos, grabaciones, leer_metadata, nombres_comunes, ruido_grabado
from aves.espectro import ventana
from aves.evaluacion import MOTORES, evaluar, exactitudes
from aves.modelo import entrenar

TAMANOS_N = (256, 512, 1024, 2048, 4096)
VENTANAS = ("rectangular", "hamming", "hann")
SNR_DB = (None, 30, 20, 10, 5, 0, -5)
DURACIONES_S = (1, 2, 3, 5, 8, 10)


# ------------------------------- Utilidades -------------------------------
def datos(con_entrenamiento):
    """Grabaciones para entrenar y para evaluar."""
    filas = leer_metadata()
    entrenamiento = grabaciones(filas, "entrenamiento")
    prueba = entrenamiento if con_entrenamiento else grabaciones(filas, "prueba")
    if not prueba:
        sys.exit("No hay grabaciones de prueba todavía. Agrega más grabaciones por especie y "
                 "corre preparar_cantos.py, o usa --con-entrenamiento para probar el código.")
    return entrenamiento, prueba, nombres_comunes(filas)


def medir(entrenamiento, prueba, nombres, p, segundos=SEGUNDOS_GRABACION, **ruido):
    modelo = entrenar(entrenamiento, p, nombres, informar=lambda *_: None)
    clips = fragmentos(prueba, p, segundos)
    return exactitudes(evaluar(modelo, clips, **ruido)), len(clips)


def tabla(encabezado, filas):
    print(f"\n{encabezado:>14s}  " + "  ".join(f"{m:>11s}" for m in MOTORES) + "  fragmentos")
    for etiqueta, valores, n in filas:
        print(f"{etiqueta:>14s}  " + "  ".join(f"{100 * valores[m]:10.1f}%" for m in MOTORES)
              + f"  {n:10d}")


def series(filas):
    return {m: [valores[m] for _, valores, _ in filas] for m in MOTORES}


# ------------------------------- Experimentos -------------------------------
def exp_filtro(_):
    p = Parametros()
    sos = disenar_filtro(p)
    _, polos_s, _ = prototipo_analogico(p)
    print(f"Butterworth pasa-banda {p.f_min:.0f}-{p.f_max:.0f} Hz, orden {p.orden_filtro} "
          f"({len(polos_s)} polos)")
    print(f"  Mayor Re(s) de los polos analógicos: {polos_s.real.max():.1f} rad/s (< 0: estable)")
    print(f"  Polos digitales dentro del círculo unitario: {'sí' if es_estable(sos) else 'NO'}")
    graficas.filtro(p, RESULTADOS / "filtro_polos_respuesta.png")
    print("Guardado: resultados/filtro_polos_respuesta.png")


def exp_fuga(_):
    """Un tono que no cae justo en un bin reparte su energía en los vecinos (fuga);
    la ventana decide cuánto: lóbulos laterales de ~-13 dB (rectangular), ~-31 dB
    (Hann) y ~-43 dB (Hamming)."""
    p = Parametros()
    f0 = 3000 + 0.4 * p.resolucion_hz
    n = np.arange(p.n_ventana)
    tono = np.sin(2 * np.pi * f0 * n / p.fs)
    relleno = 16 * p.n_ventana                    # relleno con ceros: solo suaviza la curva
    f = np.fft.rfftfreq(relleno, 1 / p.fs)
    cerca = np.abs(f - f0) < 800
    espectros = {}
    print(f"Tono de {f0:.1f} Hz, N = {p.n_ventana}")
    for nombre in VENTANAS:
        X = np.abs(np.fft.rfft(tono * ventana(replace(p, ventana=nombre)), relleno))
        db = 20 * np.log10(X / X.max() + 1e-12)
        espectros[nombre.capitalize()] = db[cerca]
        print(f"  {nombre:12s} lóbulo lateral más alto: {lobulo_lateral(db):6.1f} dB")
    graficas.fuga(f[cerca], espectros, f"Fuga espectral de un tono de {f0:.0f} Hz",
                  RESULTADOS / "fuga_espectral.png")
    print("Guardado: resultados/fuga_espectral.png")


def lobulo_lateral(db):
    """Nivel del lóbulo lateral más alto: el máximo después del primer mínimo."""
    i = int(np.argmax(db))
    while i + 1 < len(db) and db[i + 1] < db[i]:
        i += 1
    return float(db[i:].max())


def exp_ventana(con_entrenamiento):
    entrenamiento, prueba, nombres = datos(con_entrenamiento)
    filas = []
    for n in TAMANOS_N:
        p = replace(Parametros(), n_ventana=n, salto=n // 4)
        print(f"  N = {n:5d}: {1000 * n / p.fs:6.1f} ms por ventana, "
              f"resolución {p.resolucion_hz:6.1f} Hz ...", flush=True)
        valores, clips = medir(entrenamiento, prueba, nombres, p)
        filas.append((str(n), valores, clips))
    tabla("N", filas)
    graficas.lineas(list(TAMANOS_N), series(filas), "Tamaño de ventana N (muestras)",
                    "Exactitud contra el tamaño de ventana", RESULTADOS / "exactitud_ventana_N.png",
                    escala_log=True)
    print("Guardado: resultados/exactitud_ventana_N.png")


def exp_ventanas(con_entrenamiento):
    entrenamiento, prueba, nombres = datos(con_entrenamiento)
    filas = []
    for nombre in VENTANAS:
        print(f"  ventana {nombre} ...", flush=True)
        valores, clips = medir(entrenamiento, prueba, nombres, replace(Parametros(), ventana=nombre))
        filas.append((nombre, valores, clips))
    tabla("ventana", filas)
    graficas.lineas(list(range(len(VENTANAS))), series(filas), "Ventana",
                    "Exactitud según la ventana de la STFT", RESULTADOS / "exactitud_ventanas.png",
                    marcas_x=[v.capitalize() for v in VENTANAS])
    print("Guardado: resultados/exactitud_ventanas.png")


def exp_ruido(con_entrenamiento):
    entrenamiento, prueba, nombres = datos(con_entrenamiento)
    p = Parametros()
    ruido = ruido_grabado(p)
    print(f"  Ruido: {'grabado (data/ruido/)' if ruido is not None else 'blanco gaussiano'}")
    modelo = entrenar(entrenamiento, p, nombres, informar=lambda *_: None)
    clips = fragmentos(prueba, p, SEGUNDOS_GRABACION)
    filas = []
    for snr in SNR_DB:
        etiqueta = "sin ruido" if snr is None else f"{snr} dB"
        print(f"  SNR {etiqueta} ...", flush=True)
        filas.append((etiqueta, exactitudes(evaluar(modelo, clips, snr_db=snr, ruido=ruido)),
                      len(clips)))
    tabla("SNR", filas)
    graficas.lineas(list(range(len(SNR_DB))), series(filas), "Relación señal/ruido",
                    "Exactitud contra el ruido", RESULTADOS / "exactitud_ruido.png",
                    marcas_x=[e for e, _, _ in filas])
    print("Guardado: resultados/exactitud_ruido.png")


def exp_duracion(con_entrenamiento):
    entrenamiento, prueba, nombres = datos(con_entrenamiento)
    p = Parametros()
    modelo = entrenar(entrenamiento, p, nombres, informar=lambda *_: None)
    filas = []
    for s in DURACIONES_S:
        print(f"  fragmentos de {s} s ...", flush=True)
        clips = fragmentos(prueba, p, s)
        filas.append((f"{s} s", exactitudes(evaluar(modelo, clips)), len(clips)))
    tabla("duración", filas)
    graficas.lineas(list(DURACIONES_S), series(filas), "Duración del fragmento (s)",
                    "Exactitud contra la duración de la grabación",
                    RESULTADOS / "exactitud_duracion.png")
    print("Guardado: resultados/exactitud_duracion.png")


EXPERIMENTOS = {"filtro": exp_filtro, "fuga": exp_fuga, "ventana": exp_ventana,
                "ventanas": exp_ventanas, "ruido": exp_ruido, "duracion": exp_duracion}


def main():
    argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
    con_entrenamiento = "--con-entrenamiento" in sys.argv
    if not argumentos or argumentos[0] not in {*EXPERIMENTOS, "todos"}:
        sys.exit(__doc__)
    nombres = list(EXPERIMENTOS) if argumentos[0] == "todos" else [argumentos[0]]
    if con_entrenamiento:
        print("AVISO: se evalúa con el conjunto de entrenamiento; los números salen inflados.")
    for nombre in nombres:
        print(f"\n=== {nombre} ===")
        EXPERIMENTOS[nombre](con_entrenamiento)


if __name__ == "__main__":
    main()
