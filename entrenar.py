"""
entrenar.py
Construye el modelo a partir de las grabaciones de data/metadata.csv:
  - Motor A: la base de huellas (constelación + hash) de cada grabación completa.
  - Motor B: el clasificador con los MFCC de fragmentos de 5 s con canto.
Guarda todo en modelos/modelo.pkl.

Para agregar datos: pon los audios en data/raw/<Genero_especie>/ y corre
    python preparar_cantos.py
    python entrenar.py

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
from aves.datos import grabaciones, leer_metadata, nombres_comunes
from aves.modelo import entrenar, guardar


def main():
    conjunto = "todo" if len(sys.argv) > 1 and sys.argv[1] == "todo" else "entrenamiento"
    filas = leer_metadata()
    lista = grabaciones(filas, None if conjunto == "todo" else "entrenamiento")
    if not lista:
        sys.exit("No hay grabaciones de entrenamiento en data/metadata.csv.")

    p = Parametros()
    print(f"Filtro Butterworth {p.f_min:.0f}-{p.f_max:.0f} Hz, orden {p.orden_filtro}: "
          f"{'estable' if es_estable(disenar_filtro(p)) else 'INESTABLE'}")
    print(f"STFT: N = {p.n_ventana} ({1000 * p.n_ventana / p.fs:.0f} ms), ventana {p.ventana}, "
          f"resolución {p.resolucion_hz:.1f} Hz y {p.resolucion_ms:.1f} ms")
    print(f"\nEntrenando con {len(lista)} grabaciones ({conjunto}):\n")

    inicio = time.time()
    modelo = entrenar(lista, p, nombres_comunes(filas), conjunto)
    guardar(modelo)

    print(f"\nListo en {time.time() - inicio:.0f} s: {len(modelo.especies)} especies, "
          f"{len(modelo.base):,} huellas, {modelo.fragmentos} fragmentos para el clasificador.")
    if modelo.clasificador is None:
        print("AVISO: el motor B necesita al menos dos especies con canto; solo se usará el A.")
    print(f"Modelo guardado en {MODELO.relative_to(MODELO.parent.parent).as_posix()}")
    print("Siguiente:  python reconocer.py   o   python evaluar.py")


if __name__ == "__main__":
    main()
