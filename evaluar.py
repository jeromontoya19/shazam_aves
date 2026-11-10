from audio import FS
from reconocimiento import leer_base, evaluar

VENTANAS = [256, 512, 1024, 2048, 4096]

entrenamiento, prueba = leer_base()
print(f"\n{len(entrenamiento)} grabaciones de entrenamiento, {len(prueba)} de prueba\n")

# porcentaje de aciertos para cada tamaño de ventana
resultados = {}
for N in VENTANAS:
    resultados[N] = evaluar(entrenamiento, prueba, N)
    print(f"N = {N:4d}  ({1000 * N / FS:5.1f} ms, resolución {FS / N:5.1f} Hz):  "
          f"{resultados[N]:.0%} de aciertos")

mejor = max(resultados, key=resultados.get)
print(f"\nDetalle con la mejor ventana (N = {mejor}):")
evaluar(entrenamiento, prueba, mejor, mostrar=True)