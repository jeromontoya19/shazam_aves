"""
evaluar.py
Mide qué tan bien reconoce el modelo: corta las grabaciones de PRUEBA en fragmentos
de 8 s (como una grabación en vivo), los reconoce y compara con la especie real.

Guarda en resultados/:
  matriz_confusion.png    qué especies se confunden con cuáles (motor combinado)
  predicciones.csv        una fila por fragmento

Uso:
    python evaluar.py                  -> con el conjunto de prueba
    python evaluar.py entrenamiento    -> con el de entrenamiento: sale casi perfecto porque
                                          el modelo ya conoce esos audios; sirve solo para
                                          revisar que todo funciona cuando aún no hay datos de prueba
Antes:  python entrenar.py
"""

import csv
import sys
from collections import defaultdict

from aves import graficas
from aves.config import PESO_HUELLAS, RESULTADOS, SEGUNDOS_GRABACION
from aves.datos import fragmentos, particion
from aves.evaluacion import (MOTORES, exactitud, exactitud_por_peso, exactitudes,
                             matriz_confusion, evaluar)
from aves.modelo import cargar_modelo

PESOS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    conjunto = "entrenamiento" if len(sys.argv) > 1 and sys.argv[1] == "entrenamiento" else "prueba"
    modelo = cargar_modelo()
    if conjunto == "prueba" and modelo.conjunto == "todo":
        print("OJO: el modelo se entrenó con TODAS las grabaciones (entrenar.py todo), así que\n"
              "también conoce las de prueba y la exactitud saldrá inflada. Para evaluar de\n"
              "verdad, corre antes:  python entrenar.py")

    de_entrenamiento, de_prueba = particion()
    # Si los audios cambiaron, cambió la partición: el modelo podría conocer los de "prueba"
    conocidos = de_entrenamiento + de_prueba if modelo.conjunto == "todo" else de_entrenamiento
    if modelo.archivos != [ruta.name for _, ruta in conocidos]:
        sys.exit("Los audios de data/raw/ cambiaron desde el último entrenamiento. "
                 "Corre primero:  python entrenar.py")

    lista = de_entrenamiento if conjunto == "entrenamiento" else de_prueba
    clips = fragmentos(lista, modelo.p, SEGUNDOS_GRABACION)
    if not clips:
        sys.exit(f"No hay fragmentos de {SEGUNDOS_GRABACION} s con canto en el conjunto de {conjunto}."
                 + ("\nCada especie necesita al menos 2 grabaciones para tener audio de prueba. "
                    "Mientras tanto:  python evaluar.py entrenamiento" if conjunto == "prueba" else ""))

    print(f"Evaluando {len(clips)} fragmentos de {SEGUNDOS_GRABACION} s "
          f"({len(lista)} grabaciones de {conjunto})...")
    predicciones = evaluar(modelo, clips)

    print("\nExactitud:")
    for motor, valor in exactitudes(predicciones).items():
        print(f"  {motor:12s} {100 * valor:5.1f} %")

    print(f"\nExactitud del combinado según el peso w de las huellas (ahora w = {PESO_HUELLAS}):")
    for w, valor in zip(PESOS, exactitud_por_peso(predicciones, modelo.especies, PESOS)):
        print(f"  w = {w:4.2f}   {100 * valor:5.1f} %")

    por_especie = defaultdict(list)
    for f in predicciones:
        por_especie[f["real"]].append(f)
    print(f"\n{'Especie':48s} {'fragmentos':>10s}  " + "  ".join(f"{m:>11s}" for m in MOTORES))
    for especie in modelo.especies:
        filas = por_especie.get(especie, [])
        valores = "  ".join(f"{100 * exactitud(filas, m):10.0f}%" if filas else f"{'-':>11s}"
                            for m in MOTORES)
        print(f"{modelo.nombre(especie)[:48]:48s} {len(filas):10d}  {valores}")

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    nombres = [modelo.nombres.get(e) or e for e in modelo.especies]
    graficas.confusion(matriz_confusion(predicciones, modelo.especies), nombres,
                       f"Matriz de confusión, motor combinado ({conjunto})",
                       RESULTADOS / "matriz_confusion.png")
    with open(RESULTADOS / "predicciones.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["real", *MOTORES])
        for fila in predicciones:
            w.writerow([fila["real"], *(fila[m] or "" for m in MOTORES)])
    print("\nGuardado en resultados/: matriz_confusion.png y predicciones.csv")


if __name__ == "__main__":
    main()
