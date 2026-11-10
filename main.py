from audio import grabar
from reconocimiento import leer_base, reconocer

entrenamiento, prueba = leer_base()

while True:
    input("\nPresiona Enter para grabar...")
    reconocer(grabar(), entrenamiento, prueba)
    if input("\n¿Grabar otro canto? (s/n): ").strip().lower() != "s":
        break