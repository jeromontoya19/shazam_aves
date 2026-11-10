import time
import librosa
import sounddevice as sd

FS = 22050       # frecuencia de muestreo con la que trabaja todo el proyecto (Hz)
SEGUNDOS = 10    # duración de la grabación con el micrófono


def cargar(ruta):
    # abre un mp3 o wav, lo pasa a mono y a la frecuencia FS
    x, fs = librosa.load(ruta, sr=FS, mono=True, duration=60)
    return x


def grabar():
    fs_mic = 44100
    print(f"\nGRABANDO... acerca el micrófono al pájaro ({SEGUNDOS} s)")
    x = sd.rec(SEGUNDOS * fs_mic, samplerate=fs_mic, channels=1, dtype="float32")
    for s in range(SEGUNDOS, 0, -1):
        print(f"  {s}...", end=" ", flush=True)
        time.sleep(1)
    sd.wait()
    print("\nGrabación terminada\n")
    return librosa.resample(x.flatten(), orig_sr=fs_mic, target_sr=FS)