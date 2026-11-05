"""
reconocer.py
Reconoce el canto de un ave desde la terminal y muestra su espectrograma.

Uso:
    python reconocer.py                    -> en vivo: Enter para grabar, q para salir
    python reconocer.py ruta/al/audio.mp3  -> analiza un archivo (wav, mp3, flac, ...)
    python reconocer.py -s 12              -> graba 12 s en lugar de 8
    python reconocer.py --sin-grafica      -> no abre la ventana de la gráfica

Cada grabación en vivo se guarda en grabaciones/ (sirve para revisarla o para
agregarla después a data/raw/ como dato nuevo).

Antes:  python entrenar.py
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf

from aves import graficas
from aves.audio import cargar, grabar
from aves.config import CONFIANZA_MINIMA, GRABACIONES, RESULTADOS, SEGUNDOS_GRABACION
from aves.modelo import cargar_modelo, reconocer


def porcentaje(v):
    return f"{100 * v:.0f} %"


def imprimir(res, modelo):
    if res.p is None:
        print("\nNo encontré ninguna especie conocida en el audio.")
        return
    nombre = modelo.nombre(res.ganadora)
    if res.p_a is None and not res.hay_canto:
        print("\nNo detecté el canto de ninguna especie conocida.")
        print(f"  (Lo más parecido según los MFCC sería {nombre}, con {porcentaje(res.confianza)}.)")
        return
    if res.confianza >= CONFIANZA_MINIMA:
        print(f"\nLa especie es {nombre} en un {porcentaje(res.confianza)}.")
    else:
        print(f"\nNo estoy seguro. La más probable es {nombre}, con {porcentaje(res.confianza)}.")
    if not res.hay_canto:
        print("  (Casi no se oye canto en la grabación: el resultado puede no ser confiable.)")

    for titulo, p in (("Huellas (A)", res.p_a), ("MFCC (B)   ", res.p_b)):
        if p is None:
            print(f"  {titulo}:  sin coincidencias")
            continue
        especie, valor = res.ranking(p, n=1)[0]
        extra = ""
        if p is res.p_a and especie in res.votos:
            extra = f"  ({res.votos[especie][0]} huellas alineadas)"
        print(f"  {titulo}:  {modelo.nombres.get(especie) or especie} {porcentaje(valor)}{extra}")
    otras = ", ".join(f"{modelo.nombres.get(e) or e} {porcentaje(v)}" for e, v in res.ranking(n=4)[1:])
    if otras:
        print(f"  Otras opciones: {otras}")
    print(f"  Incertidumbre (entropía normalizada): {porcentaje(res.incertidumbre)}")


def analizar(x, modelo, titulo, mostrar):
    print("Analizando...")
    res = reconocer(modelo, x)
    imprimir(res, modelo)
    ruta = RESULTADOS / "ultimo_reconocimiento.png"
    if mostrar:
        print("  (Cierra la ventana de la gráfica para continuar.)")
    graficas.reconocimiento(res, modelo, titulo, ruta=ruta, mostrar=mostrar)


def en_vivo(modelo, segundos, mostrar):
    while True:
        try:
            orden = input(f"\nPulsa Enter para grabar {segundos} s (q + Enter para salir): ")
        except (EOFError, KeyboardInterrupt):
            break
        if orden.strip().lstrip("﻿").lower() in {"q", "salir"}:
            break
        print("Grabando...", end=" ", flush=True)
        try:
            x = grabar(segundos, modelo.p, cada_segundo=lambda r: print(r, end=" ", flush=True))
        except Exception as e:
            print(f"\nNo pude grabar del micrófono ({e}).")
            continue
        print("listo")
        GRABACIONES.mkdir(parents=True, exist_ok=True)
        ruta = GRABACIONES / f"grabacion_{datetime.now():%Y-%m-%d_%H-%M-%S}.wav"
        sf.write(ruta, x, modelo.p.fs, subtype="PCM_16")
        analizar(x, modelo, f"Grabación en vivo ({ruta.name})", mostrar)


def main():
    parser = argparse.ArgumentParser(description="Reconoce el canto de un ave.")
    parser.add_argument("archivo", nargs="?", help="audio a analizar; sin él, graba en vivo")
    parser.add_argument("-s", "--segundos", type=int, default=SEGUNDOS_GRABACION)
    parser.add_argument("--sin-grafica", action="store_true")
    args = parser.parse_args()

    modelo = cargar_modelo()
    nombres = ", ".join(modelo.nombres.get(e) or e for e in modelo.especies)
    print(f"Reconocedor de cantos: {len(modelo.especies)} especies ({nombres}).")
    if modelo.conjunto != "todo":
        print("(Modelo entrenado solo con el conjunto de entrenamiento.)")

    if args.archivo:
        ruta = Path(args.archivo)
        if not ruta.exists():
            sys.exit(f"No existe {ruta}")
        analizar(cargar(ruta, modelo.p), modelo, ruta.name, not args.sin_grafica)
    else:
        en_vivo(modelo, args.segundos, not args.sin_grafica)


if __name__ == "__main__":
    main()
