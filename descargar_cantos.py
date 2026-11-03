"""
descargar_cantos.py  (versión 2)
Completa las carpetas que ya tienes en data/raw/<Genero_especie>/ con grabaciones
de xeno-canto, hasta tener POR_ESPECIE en cada una. No toca las que ya descargaste
a mano ni descarga repetidas. NECESITA la clave de la API de xeno-canto.

Clave (https://xeno-canto.org/account), en la terminal de VS Code:
    $env:XC_API_KEY="tu_clave"

Uso:
    python descargar_cantos.py            -> completa todas las carpetas de data/raw
    python descargar_cantos.py explorar   -> especies con más grabaciones en el Oriente Antioqueño

Después, como siempre:
    python generar_anotaciones.py
    python preparar_cantos.py

Dependencias: pip install requests
"""

import csv
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

# ----------------------------- Configuración -----------------------------
API_URL = "https://xeno-canto.org/api/3/recordings"
API_KEY = os.environ.get("XC_API_KEY", "").strip()      # nunca escribas la clave en el código
HEADERS = {"User-Agent": "UCO-MatematicasEspeciales-CantosAves/1.0"}

POR_ESPECIE = 6                      # grabaciones que debe tener cada carpeta
CALIDADES = {"A", "B"}               # A = mejor ... E = peor
TIPOS = ("song", "call")             # se prefieren los cantos; los llamados solo si faltan
DURACION_MIN, DURACION_MAX = 5, 120  # segundos
BOX = (5.70, -75.55, 6.45, -74.90)   # Oriente Antioqueño: lat_min, lon_min, lat_max, lon_max
RIONEGRO = (6.155, -75.374)          # se eligen primero las grabaciones más cercanas
CERCA_KM = 150                       # fuera del Oriente, se prefiere lo que esté a menos de esto
EXTENSIONES = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
REGION = BASE / "data" / "region.csv"


# ------------------------------- Utilidades -------------------------------
def consultar(query):
    """Devuelve todas las grabaciones de una consulta, recorriendo todas las páginas."""
    if not API_KEY:
        sys.exit("Falta la clave. En la terminal escribe:  $env:XC_API_KEY=\"tu_clave\"")
    grabaciones, pagina = [], 1
    while True:
        r = requests.get(API_URL, params={"query": query, "key": API_KEY, "page": pagina},
                         headers=HEADERS, timeout=60)
        try:
            datos = r.json()
        except ValueError:
            sys.exit(f"Respuesta inesperada ({r.status_code}):\n{r.text[:500]}")
        if r.status_code != 200 or "error" in datos:
            sys.exit(f"Error de la API ({r.status_code}): {datos}")
        grabaciones.extend(datos.get("recordings", []))
        paginas = int(datos.get("numPages") or 1)
        if pagina >= paginas:
            break
        pagina += 1
        time.sleep(1)                # no saturar el servidor
    return grabaciones


def numero_xc(texto):
    m = re.search(r"XC\s*-?\s*(\d+)", texto, re.IGNORECASE) or re.search(r"(\d{4,})", texto)
    return m.group(1) if m else None


def duracion_s(texto):
    try:
        s = 0.0
        for parte in str(texto).split(":"):
            s = s * 60 + float(parte)
        return s
    except ValueError:
        return 0.0


def distancia_km(g):
    """Distancia (fórmula de haversine) entre el sitio de grabación y Rionegro."""
    try:
        lat, lon = float(g.get("lat")), float(g.get("lon", g.get("lng")))
    except (TypeError, ValueError):
        return float("inf")
    f1, l1, f2, l2 = map(math.radians, (*RIONEGRO, lat, lon))
    a = math.sin((f2 - f1) / 2) ** 2 + math.cos(f1) * math.cos(f2) * math.sin((l2 - l1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def zona(g):
    """0 = Oriente Antioqueño, 1 = a menos de CERCA_KM de Rionegro, 2 = resto de Colombia."""
    try:
        lat, lon = float(g.get("lat")), float(g.get("lon", g.get("lng")))
    except (TypeError, ValueError):
        return 2
    if BOX[0] <= lat <= BOX[2] and BOX[1] <= lon <= BOX[3]:
        return 0
    return 1 if distancia_km(g) <= CERCA_KM else 2


def otras_especies(g):
    also = g.get("also") or []
    if isinstance(also, str):
        also = [also]
    return [s for s in also if s.strip()]


def es_canto(g):
    return "song" in str(g.get("type", "")).lower()


def sirve(g):
    tipo = str(g.get("type", "")).lower()
    return (g.get("q") in CALIDADES
            and any(t in tipo for t in TIPOS)
            and DURACION_MIN <= duracion_s(g.get("length")) <= DURACION_MAX)


# --------------------------------- Modos ----------------------------------
def completar():
    carpetas = sorted(p for p in RAW.iterdir() if p.is_dir()) if RAW.exists() else []
    if not carpetas:
        sys.exit("No hay carpetas en data/raw/. Crea una por especie, p. ej. data/raw/Zonotrichia_capensis")

    nuevas = 0
    for carpeta in carpetas:
        partes = carpeta.name.split("_")
        if len(partes) < 2:
            print(f"\n{carpeta.name}: el nombre debe ser Genero_especie; se omite")
            continue
        genero, epiteto = partes[0], partes[1]
        actuales = {numero_xc(f.name) for f in carpeta.iterdir()
                    if f.suffix.lower() in EXTENSIONES} - {None}
        faltan = POR_ESPECIE - len(actuales)
        print(f"\n{genero} {epiteto}: tiene {len(actuales)}", end="")
        if faltan <= 0:
            extra = f" (sobran {-faltan}; bórralas a mano si quieres)" if faltan < 0 else ""
            print(f", completa{extra}")
            continue
        print(f", faltan {faltan}")

        candidatas = [g for g in consultar(f"gen:{genero} sp:{epiteto} cnt:colombia")
                      if sirve(g) and str(g.get("id")) not in actuales]
        # Orden: primero el Oriente, luego lo cercano y al final el resto de Colombia; dentro de
        # cada zona, cantos antes que llamados, sin aves de fondo antes que con, y las más cercanas
        candidatas.sort(key=lambda g: (zona(g), not es_canto(g), len(otras_especies(g)) > 0,
                                       distancia_km(g)))

        bajadas = 0
        for g in candidatas:
            if bajadas == faltan:
                break
            ext = Path(str(g.get("file-name", ""))).suffix.lower() or ".mp3"
            destino = carpeta / f"XC{g['id']}{ext}"
            url = str(g.get("file", ""))
            if url.startswith("//"):
                url = "https:" + url
            try:
                r = requests.get(url, headers=HEADERS, timeout=120)
                r.raise_for_status()
                destino.write_bytes(r.content)
            except Exception as e:
                print(f"  XC{g['id']}: no se pudo descargar ({e}); pruebo con otra")
                continue
            bajadas += 1
            nuevas += 1
            d = distancia_km(g)
            lugar = f"{d:6.1f} km" if not math.isinf(d) else "   ? km"
            print(f"  XC{g['id']:>7}  {g.get('q')}  {str(g.get('type', ''))[:12]:12s} "
                  f"{str(g.get('length', '')):>5}  {lugar}  {str(g.get('loc', ''))[:40]}")
            time.sleep(1)
        if bajadas < faltan:
            print(f"  AVISO: solo encontré {bajadas} de {faltan}. Prueba bajando la exigencia "
                  f"(agrega 'C' a CALIDADES) o búscalas a mano.")

    print(f"\nListo: {nuevas} grabaciones nuevas.")
    print("Siguiente:  python generar_anotaciones.py   y luego   python preparar_cantos.py")


def explorar():
    print("Consultando grabaciones de aves del Oriente Antioqueño...")
    grabaciones = consultar("grp:birds box:{},{},{},{}".format(*BOX))
    REGION.parent.mkdir(parents=True, exist_ok=True)
    with open(REGION, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "especie", "nombre_en", "calidad", "tipo", "duracion", "localidad", "url"])
        for g in grabaciones:
            w.writerow([f"XC{g.get('id')}", f"{g.get('gen', '')} {g.get('sp', '')}", g.get("en", ""),
                        g.get("q", ""), g.get("type", ""), g.get("length", ""), g.get("loc", ""),
                        f"https://xeno-canto.org/{g.get('id')}"])
    conteo = Counter(f"{g.get('gen', '')} {g.get('sp', '')}" for g in grabaciones if sirve(g))
    print(f"\n{len(grabaciones)} grabaciones en la región (detalle en data/region.csv).")
    print("Especies con más grabaciones útiles:\n")
    for especie, n in conteo.most_common(25):
        print(f"  {n:3d}  {especie}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "explorar":
        explorar()
    else:
        completar()