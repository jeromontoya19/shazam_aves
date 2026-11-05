"""
entrenar.py
Construye el modelo a partir de los audios de data/raw/<Genero_especie>/:
  - Motor A: la base de huellas (constelación + hash) de cada grabación completa.
  - Motor B: el clasificador con los MFCC de fragmentos de 5 s con canto.
Guarda todo en modelos/modelo.pkl.

Para agregar datos: pon los audios en la carpeta de su especie (o crea una carpeta
nueva con el nombre Genero_especie) y vuelve a correr este programa.

Uso:
    python entrenar.py          -> solo con el conjunto de entrenamiento (para poder evaluar)
    python entrenar.py todo     -> con todas las grabaciones (para la demo final)
Luego:
    python reconocer.py     o     python evaluar.py

Dependencias: pip install -r requirements.txt
"""

import sys
import time

from aves.audio import disenar_filtro, es_estable
from aves.config import MODELO, Parametros
from aves.datos import particion
from aves.modelo import entrenar, guardar


def main():
    conjunto = "todo" if len(sys.argv) > 1 and sys.argv[1] == "todo" else "entrenamiento"
    de_entrenamiento, de_prueba = particion()
    lista = de_entrenamiento + de_prueba if conjunto == "todo" else de_entrenamiento
    if not lista:
        sys.exit("No encontré audios. Ponlos en data/raw/<Genero_especie>/.")

    p = Parametros()
    print(f"Filtro Butterworth {p.f_min:.0f}-{p.f_max:.0f} Hz, orden {p.orden_filtro}: "
          f"{'estable' if es_estable(disenar_filtro(p)) else 'INESTABLE'}")
    print(f"STFT: N = {p.n_ventana} ({1000 * p.n_ventana / p.fs:.0f} ms), ventana {p.ventana}, "
          f"resolución {p.resolucion_hz:.1f} Hz y {p.resolucion_ms:.1f} ms")
    print(f"\nEntrenando con {len(lista)} grabaciones ({conjunto}):\n")

    inicio = time.time()
    modelo = entrenar(lista, p, conjunto)
    guardar(modelo)

    print(f"\nListo en {time.time() - inicio:.0f} s: {len(modelo.especies)} especies, "
          f"{len(modelo.base):,} huellas, {modelo.fragmentos} fragmentos para el clasificador.")
    if modelo.clasificador is None:
        print("AVISO: el motor B necesita al menos dos especies con canto; solo se usará el A.")
    if conjunto != "todo":
        print(f"Quedan {len(de_prueba)} grabaciones para probar.")
        sin_prueba = sorted({e for e, _ in de_entrenamiento} - {e for e, _ in de_prueba})
        if sin_prueba:
            print(f"AVISO: sin audio de prueba (necesitan 2 grabaciones o más): {', '.join(sin_prueba)}")
    print(f"Modelo guardado en {MODELO.relative_to(MODELO.parent.parent).as_posix()}")
    print("Siguiente:  python reconocer.py   o   python evaluar.py")


if __name__ == "__main__":
    main()
