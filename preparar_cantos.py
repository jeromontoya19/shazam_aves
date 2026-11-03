"""
preparar_cantos.py
Ruta MANUAL: organiza los cantos que descargaste a mano desde xeno-canto.org
y los deja en el mismo formato que usa el resto del proyecto.

Antes de correrlo debes tener:
  data/raw/<Genero_especie>/   <- los MP3 descargados (no hace falta renombrarlos)
  data/anotaciones.csv         <- los datos de cada grabación (recomendado)

Al correrlo se crean:
  data/wav/<Genero_especie>/XC######.wav   (mono, 44.1 kHz, 16 bits)
  data/metadata.csv                         (una fila por grabación; no editar a mano)

Uso:           python preparar_cantos.py
Dependencias:  pip install librosa soundfile scipy numpy
"""

import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

import librosa
import soundfile as sf
from scipy.io import wavfile

SR = 44100                       # todos los WAV quedan a la misma frecuencia de muestreo
MAX_SEGUNDOS_WAV = 60            # recorta grabaciones muy largas
FRACCION_ENTRENAMIENTO = 0.6     # de 8 grabaciones: 5 para la base, 3 para probar
SEMILLA = 42                     # para que la partición sea siempre la misma
EXTENSIONES = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".aif", ".aiff"}   # las que lee libsndfile
NO_SOPORTADAS = {".m4a", ".aac", ".mp4", ".3gp", ".amr", ".wma"}           # hay que convertirlas antes

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RAW = DATA / "raw"
WAV = DATA / "wav"
ANOTACIONES = DATA / "anotaciones.csv"
METADATA = DATA / "metadata.csv"


def numero_xc(texto):
    """Saca el número de catálogo de un nombre de archivo o de la columna id."""
    m = re.search(r"XC\s*-?\s*(\d+)", texto, re.IGNORECASE) or re.search(r"(\d{4,})", texto)
    return m.group(1) if m else None


def identificador(archivo):
    """XC###### para las grabaciones de xeno-canto; para las propias (sin número XC en el
    nombre), el nombre del archivo."""
    m = re.search(r"(?<![a-z])XC\s*-?\s*(\d+)", archivo.name, re.IGNORECASE)
    return f"XC{m.group(1)}" if m else re.sub(r"[^\w-]+", "_", archivo.stem)


def leer_anotaciones():
    """Lee data/anotaciones.csv. Acepta ',' o ';' y archivos guardados desde Excel."""
    if not ANOTACIONES.exists():
        print("No hay data/anotaciones.csv: se usarán solo las carpetas.")
        return {}
    for codificacion in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            texto = ANOTACIONES.read_text(encoding=codificacion)
            break
        except UnicodeDecodeError:
            continue
    lineas = [l for l in texto.splitlines() if l.strip()]
    if not lineas:
        return {}
    separador = ";" if lineas[0].count(";") > lineas[0].count(",") else ","

    anotaciones = {}
    for fila in csv.DictReader(lineas, delimiter=separador):
        fila = {k.strip().lower(): (v or "").strip() for k, v in fila.items() if isinstance(k, str)}
        n = numero_xc(fila.get("id", ""))
        if n:
            anotaciones[f"XC{n}"] = fila
    print(f"Anotaciones leídas: {len(anotaciones)} grabaciones.")
    return anotaciones


def main():
    if not RAW.exists():
        sys.exit("No existe data/raw/. Crea data/raw/<Genero_especie>/ y pon ahí los MP3.")

    anotaciones = leer_anotaciones()
    rng = random.Random(SEMILLA)
    filas, vistos = [], set()

    for carpeta in sorted(p for p in RAW.iterdir() if p.is_dir()):
        especie = carpeta.name.replace("_", " ")
        print(f"\n{especie}")
        archivos = []
        for archivo in sorted(carpeta.iterdir()):
            if archivo.suffix.lower() in NO_SOPORTADAS:
                print(f"  {archivo.name}: formato no soportado; conviértelo a mp3 o wav")
            if archivo.suffix.lower() not in EXTENSIONES:
                continue
            xc = identificador(archivo)
            # Un número XC es único en todo xeno-canto; un nombre propio, solo dentro de su carpeta
            clave = xc if re.fullmatch(r"XC\d+", xc) else f"{carpeta.name}/{xc}"
            if clave in vistos:
                print(f"  {archivo.name}: repetido; se omite")
            else:
                vistos.add(clave)
                archivos.append((archivo, xc))
        if not archivos:
            continue

        # Partición por grabación completa (una grabación nunca queda en los dos conjuntos)
        orden = list(range(len(archivos)))
        rng.shuffle(orden)
        entrenamiento = set(orden[:max(1, round(len(archivos) * FRACCION_ENTRENAMIENTO))])
        (WAV / carpeta.name).mkdir(parents=True, exist_ok=True)

        for i, (archivo, xc) in enumerate(archivos):
            ruta_wav = WAV / carpeta.name / f"{xc}.wav"
            try:
                if not ruta_wav.exists():
                    y, _ = librosa.load(archivo, sr=SR, mono=True, duration=MAX_SEGUNDOS_WAV)
                    sf.write(ruta_wav, y, SR, subtype="PCM_16")
                rate, x = wavfile.read(ruta_wav)    # prueba: así lo leerá el análisis
            except Exception as e:
                print(f"  {xc}: no se pudo convertir ({e})")
                continue

            info = anotaciones.get(xc, {})
            aviso = "" if info or not re.fullmatch(r"XC\d+", xc) else "   (falta en anotaciones.csv)"
            print(f"  {xc}: listo, {len(x) / rate:.1f} s a {rate} Hz{aviso}")
            if info.get("especie") and info["especie"].lower() != especie.lower():
                print(f"    OJO: anotaciones.csv dice '{info['especie']}', "
                      f"pero el audio está en la carpeta {carpeta.name}")

            fila = {
                "id": xc,
                "especie": especie,
                "nombre_comun": info.get("nombre_comun") or especie,
                "conjunto": "entrenamiento" if i in entrenamiento else "prueba",
                "archivo_wav": ruta_wav.relative_to(BASE).as_posix(),
                "archivo_original": archivo.relative_to(BASE).as_posix(),
                "duracion_wav_s": round(len(x) / rate, 1),
            }
            for clave, valor in info.items():
                fila.setdefault(clave, valor)
            fila["url"] = f"https://xeno-canto.org/{xc[2:]}" if re.fullmatch(r"XC\d+", xc) else ""
            filas.append(fila)

    if not filas:
        sys.exit("No encontré ningún audio dentro de data/raw/<Genero_especie>/.")

    campos = [k for k in dict.fromkeys(k for f in filas for k in f) if k != "url"] + ["url"]
    with open(METADATA, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos, restval="")
        w.writeheader()
        w.writerows(filas)

    print(f"\nListo: {len(filas)} grabaciones en data/metadata.csv\n")
    resumen = Counter((f["especie"], f["conjunto"]) for f in filas)
    for especie in dict.fromkeys(f["especie"] for f in filas):
        print(f"  {especie:28s} entrenamiento={resumen[(especie, 'entrenamiento')]}  "
              f"prueba={resumen[(especie, 'prueba')]}")

    sin_audio = [xc for xc in anotaciones if xc not in vistos]
    if sin_audio:
        print(f"\nEstán en anotaciones.csv pero no encontré su audio: {', '.join(sin_audio)}")


if __name__ == "__main__":
    main()