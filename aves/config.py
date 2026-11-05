"""
Parámetros del reconocedor. Todo lo ajustable está aquí, en unidades físicas
(Hz, ms, dB), para poder citarlo en el informe y para que cambiar el tamaño de
la ventana no cambie el significado de los demás parámetros.
"""

from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RAW = DATA / "raw"                     # los audios, en data/raw/<Genero_especie>/
RUIDO = DATA / "ruido"                 # opcional: grabaciones de ruido ambiente para los experimentos
MODELO = BASE / "modelos" / "modelo.pkl"
GRABACIONES = BASE / "grabaciones"     # aquí se guarda cada grabación en vivo
RESULTADOS = BASE / "resultados"       # gráficas y tablas de evaluar.py y experimentos.py

# Audios de data/raw
EXTENSIONES = {".mp3", ".mpeg", ".wav", ".flac", ".ogg", ".opus", ".aif", ".aiff"}
MAX_SEGUNDOS = 60                 # de cada grabación solo se usa este tramo inicial
FRACCION_ENTRENAMIENTO = 0.6      # de 8 grabaciones: 5 para la base, 3 para probar
SEMILLA = 42                      # para que la partición sea siempre la misma

# Una especie que no esté aquí se muestra con su nombre científico (el de su carpeta)
NOMBRES_COMUNES = {
    "Drymophila striaticeps": "Tiluchí colilargo",
    "Grallaria nuchalis ruficeps": "Tororoí nuquicastaño",
    "Momotus aequatorialis": "Barranquero",
    "Myiothlypis nigrocristata": "Reinita crestinegra",
    "Pheugopedius mystacalis": "Cucarachero bigotudo",
    "Synallaxis albescens": "Pijuí pechiblanco",
    "Thraupis episcopus": "Azulejo",
    "Tyrannus melancholicus": "Sirirí",
    "Zonotrichia capensis": "Copetón",
}

# Reconocimiento (no cambian el modelo entrenado)
SEGUNDOS_GRABACION = 8
PESO_HUELLAS = 0.5          # p = w·p_A + (1 − w)·p_B
CONFIANZA_MINIMA = 0.40     # por debajo, la respuesta es "no estoy seguro"
VOTOS_MINIMOS = 10          # huellas alineadas para que el motor A cuente; al azar salen hasta ~9


@dataclass(frozen=True)
class Parametros:
    # Muestreo: con fs = 22 050 Hz, Nyquist = 11 025 Hz, por encima de casi todos los cantos
    fs: int = 22050

    # Filtro pasa-banda Butterworth (diseñado en el plano s)
    f_min: float = 250.0            # Hz; el barranquero canta grave (~400 Hz)
    f_max: float = 10000.0          # Hz
    orden_filtro: int = 4

    # STFT: N = 1024 -> 46 ms por ventana, Δf = fs/N ≈ 21.5 Hz; salto N/4 -> Δt ≈ 11.6 ms
    n_ventana: int = 1024
    salto: int = 256
    ventana: str = "hann"           # "hann", "hamming" o "rectangular"

    # Constelación (motor A)
    vecindad_ms: float = 100.0      # un pico debe ser el máximo en ±50 ms ...
    vecindad_hz: float = 200.0      # ... y en ±100 Hz
    umbral_db: float = 10.0         # dB por encima de la mediana del espectrograma (piso de ruido)
    picos_por_segundo: int = 25     # se conservan los más fuertes

    # Huellas (motor A)
    fan_out: int = 5                # parejas por cada pico ancla
    dt_max_ms: float = 700.0        # zona objetivo: hasta 0.7 s después del ancla ...
    df_max_hz: float = 2000.0       # ... y a menos de 2 kHz de distancia
    cuantizacion_hz: float = 65.0   # frecuencias agrupadas: tolera diferencias entre individuos
    cuantizacion_ms: float = 23.0   # tiempos agrupados

    # MFCC (motor B)
    n_mel: int = 40
    n_mfcc: int = 20                # sin contar c0
    segmento_s: float = 5.0         # el clasificador se entrena con fragmentos de 5 s ...
    salto_segmento_s: float = 2.5   # ... que se solapan a la mitad

    # Detección de canto
    umbral_actividad_db: float = 6.0    # energía de la banda por encima de la mediana
    fraccion_activa_min: float = 0.10   # un fragmento sirve si al menos el 10 % tiene canto

    @property
    def resolucion_hz(self):
        return self.fs / self.n_ventana

    @property
    def resolucion_ms(self):
        return 1000 * self.salto / self.fs

    def marcos(self, ms):
        """Convierte milisegundos a marcos de la STFT."""
        return max(1, round(ms / self.resolucion_ms))

    def bins(self, hz):
        """Convierte hercios a bins de frecuencia de la FFT."""
        return max(1, round(hz / self.resolucion_hz))
