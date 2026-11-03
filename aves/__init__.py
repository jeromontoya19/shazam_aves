"""
Reconocedor de cantos de aves con análisis de Fourier.

  config       parámetros (unidades físicas) y rutas
  audio        carga, grabación y preprocesado (DC, filtro Butterworth, normalización)
  espectro     STFT, espectrograma y detección de canto
  huellas      motor A: constelación, huellas hash y votación (tipo Shazam)
  descriptores motor B: MFCC y clasificador
  modelo       entrenar, guardar, cargar y reconocer con los dos motores
  datos        lectura de data/metadata.csv y fragmentos para evaluar
  graficas     figuras del reconocimiento, la evaluación y los experimentos
"""
