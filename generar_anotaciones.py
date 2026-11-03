"""
generar_anotaciones.py
Crea data/anotaciones.csv a partir de los audios que hay en data/raw/<Genero_especie>/.

Toma los datos de cada grabación de una de dos fuentes:
  - Con clave de xeno-canto (variable XC_API_KEY): consulta la API por el número XC
    y llena TODO (recordista, localidad, coordenadas, calidad, tipo, licencia...).
  - Sin clave: lee las etiquetas que xeno-canto guarda dentro del MP3
    (nombre en inglés, recordista, copyright) y deja vacías las demás columnas
    para completarlas a mano mirando la página.

Si anotaciones.csv ya existe, se conserva lo que hayas escrito en 'nombre_comun'
y en cualquier celda que la fuente deje vacía.

Uso:           python generar_anotaciones.py
Luego:         python preparar_cantos.py
Dependencias:  pip install mutagen requests
"""

import csv
import os
import re
import time
from pathlib import Path

import requests
from mutagen import File as AudioFile

API_URL = "https://xeno-canto.org/api/3/recordings"
API_KEY = os.environ.get("XC_API_KEY", "").strip()
HEADERS = {"User-Agent": "UCO-MatematicasEspeciales-CantosAves/1.0"}
EXTENSIONES = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
ANOTACIONES = BASE / "data" / "anotaciones.csv"

COLUMNAS = ["id", "especie", "nombre_comun", "nombre_en", "recordista", "localidad", "pais",
            "lat", "lon", "elevacion_m", "fecha", "hora", "tipo", "calidad", "duracion_s",
            "otras_especies", "licencia", "fuente"]

# Nombres comunes que se llenan solos; los demás escríbelos tú en el CSV.
NOMBRES_COMUNES = {
    "Zonotrichia capensis": "Copetón",
    "Thraupis episcopus": "Azulejo",
    "Tyrannus melancholicus": "Sirirí",
    "Momotus aequatorialis": "Barranquero andino",
    "Myiothlypis nigrocristata": "Reinita crestinegra",
    "Pheugopedius mystacalis": "Cucarachero bigotudo",
    "Grallaria nuchalis ruficeps": "Tororoí nuquicastaño",
    "Drymophila striaticeps": "Tiluchí colilargo",
    "Synallaxis albescens": "Pijuí pechiblanco",
}


def numero_xc(texto):
    m = re.search(r"XC\s*-?\s*(\d+)", texto, re.IGNORECASE) or re.search(r"(\d{4,})", texto)
    return m.group(1) if m else None


def duracion_texto_a_s(texto):
    try:
        s = 0.0
        for parte in str(texto).split(":"):
            s = s * 60 + float(parte)
        return round(s)
    except ValueError:
        return ""


def leer_existentes():
    """Lee el anotaciones.csv actual (si existe) para no perder lo escrito a mano."""
    if not ANOTACIONES.exists():
        return {}
    for cod in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            texto = ANOTACIONES.read_text(encoding=cod)
            break
        except UnicodeDecodeError:
            continue
    lineas = [l for l in texto.splitlines() if l.strip()]
    if not lineas:
        return {}
    sep = ";" if lineas[0].count(";") > lineas[0].count(",") else ","
    existentes = {}
    for fila in csv.DictReader(lineas, delimiter=sep):
        fila = {k.strip().lower(): (v or "").strip() for k, v in fila.items() if isinstance(k, str)}
        n = numero_xc(fila.get("id", ""))
        if n:
            existentes[f"XC{n}"] = fila
    return existentes


def desde_api(n):
    """Consulta una grabación por su número de catálogo (etiqueta nr:)."""
    r = requests.get(API_URL, params={"query": f"nr:{n}", "key": API_KEY},
                     headers=HEADERS, timeout=60)
    datos = r.json()
    if r.status_code != 200 or "error" in datos:
        raise RuntimeError(f"API ({r.status_code}): {datos}")
    recs = datos.get("recordings", [])
    if not recs:
        return None
    g = recs[0]
    also = g.get("also") or []
    if isinstance(also, str):
        also = [also]
    lic = str(g.get("lic", ""))
    return {
        "especie_api": f"{g.get('gen', '')} {g.get('sp', '')}".strip(),
        "nombre_en": g.get("en", ""),
        "recordista": g.get("rec", ""),
        "localidad": g.get("loc", ""),
        "pais": g.get("cnt", ""),
        "lat": g.get("lat", ""),
        "lon": g.get("lon", g.get("lng", "")),
        "elevacion_m": g.get("alt", ""),
        "fecha": g.get("date", ""),
        "hora": g.get("time", ""),
        "tipo": g.get("type", ""),
        "calidad": g.get("q", ""),
        "duracion_s": duracion_texto_a_s(g.get("length", "")),
        "otras_especies": "; ".join(s for s in also if s.strip()),
        "licencia": ("https:" + lic) if lic.startswith("//") else lic,
        "fuente": "API xeno-canto",
    }


def desde_etiquetas(archivo):
    """Lee las etiquetas que xeno-canto deja dentro del MP3."""
    audio = AudioFile(archivo)
    info = {"fuente": "etiquetas MP3"}
    if audio is None:
        return info
    if getattr(audio, "info", None) and getattr(audio.info, "length", None):
        info["duracion_s"] = round(audio.info.length)
    tags = audio.tags or {}

    def texto(clave):
        frame = tags.get(clave) if hasattr(tags, "get") else None
        return str(frame.text[0]).strip() if frame is not None and getattr(frame, "text", None) else ""

    titulo = texto("TIT2")                      # p. ej. "Rufous-collared Sparrow (Zonotrichia capensis)"
    info["nombre_en"] = titulo.split(" (")[0] if titulo else ""
    info["recordista"] = texto("TPE1")
    info["licencia"] = texto("TCOP")
    info["fecha"] = texto("TDRC")
    return info


def main():
    if not RAW.exists():
        raise SystemExit("No existe data/raw/. Corre este script desde la carpeta proyecto-aves.")
    existentes = leer_existentes()
    print("Fuente:", "API de xeno-canto" if API_KEY else
          "etiquetas del MP3 (sin XC_API_KEY: localidad, calidad y tipo quedan para llenar a mano)")

    filas = []
    for carpeta in sorted(p for p in RAW.iterdir() if p.is_dir()):
        especie = carpeta.name.replace("_", " ")
        for archivo in sorted(f for f in carpeta.iterdir() if f.suffix.lower() in EXTENSIONES):
            n = numero_xc(archivo.name)
            if not n:
                print(f"  {archivo.name}: sin número XC en el nombre; se omite")
                continue
            xc = f"XC{n}"
            try:
                info = desde_api(n) if API_KEY else desde_etiquetas(archivo)
                if API_KEY:
                    time.sleep(1)
            except Exception as e:
                print(f"  {xc}: no se pudieron leer los datos ({e}); se usan las etiquetas del MP3")
                info = desde_etiquetas(archivo)
            if info is None:
                print(f"  {xc}: la API no encontró esa grabación; se usan las etiquetas del MP3")
                info = desde_etiquetas(archivo)

            especie_api = info.pop("especie_api", "")
            if especie_api and especie_api.lower() not in especie.lower():
                print(f"  OJO {xc}: xeno-canto dice '{especie_api}', pero está en la carpeta {carpeta.name}")

            previa = existentes.get(xc, {})
            fila = {c: "" for c in COLUMNAS}
            fila.update({"id": xc, "especie": especie})
            fila.update({k: v for k, v in info.items() if k in fila})
            # Lo que escribiste a mano gana sobre lo vacío; nombre_comun siempre se conserva
            for c in COLUMNAS:
                if previa.get(c) and (c == "nombre_comun" or fila[c] in ("", None)):
                    fila[c] = previa[c]
            if not fila["nombre_comun"]:
                fila["nombre_comun"] = NOMBRES_COMUNES.get(especie, "")
            filas.append(fila)
            print(f"  {xc:>10}  {especie:28s} {fila['recordista']}")

    with open(ANOTACIONES, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)

    vacios = [f["id"] for f in filas if not f["nombre_comun"]]
    print(f"\nListo: {len(filas)} grabaciones en data/anotaciones.csv")
    if vacios:
        print("Falta el nombre común de:", ", ".join(vacios))
    print("Siguiente paso: python preparar_cantos.py")


if __name__ == "__main__":
    main()