"""
Figuras del reconocimiento, la evaluación y los experimentos (matplotlib).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal

from .audio import disenar_filtro, prototipo_analogico
from .espectro import a_db, banda

SUPERFICIE, TINTA, TINTA_2, TENUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
REJILLA, EJE = "#e1e0d9", "#c3c2b7"
COLORES = {"Huellas (A)": "#2a78d6", "MFCC (B)": "#eb6834", "Combinado": "#1baf7a"}
NARANJA = "#eb6834"
AZULES = LinearSegmentedColormap.from_list(
    "azules", [SUPERFICIE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": REJILLA, "grid.linewidth": 0.8, "xtick.color": TENUE, "ytick.color": TENUE,
    "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2, "text.color": TINTA,
    "legend.frameon": False, "font.size": 10, "lines.linewidth": 2, "lines.markersize": 6,
})


def _terminar(fig, ruta, mostrar):
    fig.tight_layout()
    if ruta:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150)
    if mostrar:
        plt.show()
    plt.close(fig)


def reconocimiento(res, modelo, titulo, ruta=None, mostrar=True):
    """Espectrograma con la constelación, las huellas que coincidieron y las probabilidades."""
    p = modelo.p
    fig, (ax, ax_p) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1.2]})

    S = a_db(res.X)
    en_banda = S[:, banda(res.f, p)]
    piso, alto = np.median(en_banda), np.percentile(en_banda, 99.5)   # el piso de ruido queda blanco
    ax.imshow(S.T, origin="lower", aspect="auto", cmap=AZULES, vmin=piso, vmax=alto,
              extent=[res.t[0], res.t[-1], res.f[0] / 1000, res.f[-1] / 1000])
    ax.grid(False)
    if len(res.picos):
        ax.scatter(res.t[res.picos[:, 0]], res.f[res.picos[:, 1]] / 1000, s=14,
                   facecolors="none", edgecolors=NARANJA, linewidths=0.8, label="Constelación")
    if len(res.coincidencias):
        ax.scatter(res.t[res.coincidencias[:, 0]], res.f[res.coincidencias[:, 1]] / 1000, s=36,
                   color=NARANJA, edgecolors=SUPERFICIE, linewidths=1.5,
                   label="Huellas que coinciden")
    for borde in (p.f_min, p.f_max):
        ax.axhline(borde / 1000, color=TINTA_2, linewidth=1, linestyle="--")
    ax.set_ylim(0, min(res.f[-1], p.f_max * 1.1) / 1000)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia (kHz)")
    ax.set_title(titulo, loc="left")
    if len(res.picos):
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, labelcolor=TINTA,
                  borderaxespad=0.2)

    ranking = [r for r in res.ranking(n=5) if r[1] >= 0.01] or res.ranking(n=1)
    ax_p.set_axisbelow(True)
    if ranking:
        nombres = [modelo.nombres.get(e) or e for e, _ in ranking][::-1]
        valores = [100 * v for _, v in ranking][::-1]
        barras = ax_p.barh(nombres, valores, color=COLORES["Huellas (A)"], height=0.6)
        for barra, v in zip(barras, valores):
            ax_p.text(barra.get_width() + 1.5, barra.get_y() + barra.get_height() / 2,
                      f"{v:.0f} %", va="center", color=TINTA_2)
    ax_p.set_ylim(len(ranking) - 5.5, len(ranking) - 0.5)     # siempre espacio para 5 barras
    ax_p.set_xlim(0, 115)
    ax_p.set_xlabel("Probabilidad (%)")
    ax_p.set_title("Especies más probables", loc="left")
    ax_p.grid(axis="y", visible=False)
    _terminar(fig, ruta, mostrar)


def lineas(x, series, etiqueta_x, titulo, ruta, escala_log=False, mostrar=False, marcas_x=None):
    """Exactitud (%) de cada motor contra un parámetro."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for nombre, y in series.items():
        ax.plot(x, 100 * np.asarray(y), marker="o", color=COLORES.get(nombre), label=nombre)
        ax.annotate(nombre, (x[-1], 100 * y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=TINTA_2)
    if escala_log:
        ax.set_xscale("log", base=2)
    ax.set_xticks(x, marcas_x or [str(v) for v in x])
    ax.set_ylim(0, 105)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel("Exactitud (%)")
    ax.set_title(titulo, loc="left")
    ax.legend(loc="lower right", labelcolor=TINTA)
    _terminar(fig, ruta, mostrar)


def confusion(matriz, nombres, titulo, ruta, mostrar=False):
    """Matriz de confusión normalizada por fila (qué fracción de cada especie real
    se predijo como cada especie)."""
    filas = matriz.sum(axis=1, keepdims=True)
    normal = np.divide(matriz, filas, out=np.zeros_like(matriz, dtype=float), where=filas > 0)
    fig, ax = plt.subplots(figsize=(1.1 * len(nombres) + 3, 0.8 * len(nombres) + 2.5))
    ax.imshow(normal, cmap=AZULES, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            if matriz[i, j]:
                ax.text(j, i, int(matriz[i, j]), ha="center", va="center",
                        color="#ffffff" if normal[i, j] > 0.55 else TINTA)
    ax.set_xticks(range(len(nombres)), nombres, rotation=40, ha="right")
    ax.set_yticks(range(len(nombres)), nombres)
    ax.set_xlabel("Especie predicha")
    ax.set_ylabel("Especie real")
    ax.set_title(titulo, loc="left")
    _terminar(fig, ruta, mostrar)


def filtro(p, ruta, mostrar=False):
    """Polos y ceros en el plano s (Laplace) y en el plano z, y respuesta en frecuencia."""
    z_s, p_s, _ = prototipo_analogico(p)
    sos = disenar_filtro(p)
    z_z, p_z, _ = signal.sos2zpk(sos)
    fig, (ax_s, ax_z, ax_h) = plt.subplots(1, 3, figsize=(15, 4.6))

    ax_s.scatter(p_s.real / 1000, p_s.imag / 1000, marker="x", s=60, color=COLORES["Huellas (A)"],
                 label="Polos")
    ax_s.scatter(z_s.real / 1000, z_s.imag / 1000, s=60, facecolors="none",
                 edgecolors=NARANJA, label="Ceros")
    ax_s.axvline(0, color=TINTA_2, linewidth=1)
    ax_s.set_xlabel("Re(s) (krad/s)")
    ax_s.set_ylabel("Im(s) (krad/s)")
    ax_s.set_title("Plano s: polos en Re(s) < 0 ⇒ estable", loc="left")
    ax_s.legend(labelcolor=TINTA)

    theta = np.linspace(0, 2 * np.pi, 400)
    ax_z.plot(np.cos(theta), np.sin(theta), color=TINTA_2, linewidth=1)
    ax_z.scatter(p_z.real, p_z.imag, marker="x", s=60, color=COLORES["Huellas (A)"], label="Polos")
    ax_z.scatter(z_z.real, z_z.imag, s=60, facecolors="none", edgecolors=NARANJA, label="Ceros")
    ax_z.set_aspect("equal")
    ax_z.set_xlabel("Re(z)")
    ax_z.set_ylabel("Im(z)")
    ax_z.set_title("Plano z: polos en |z| < 1 ⇒ estable", loc="left")
    ax_z.legend(labelcolor=TINTA)

    f, h = signal.sosfreqz(sos, worN=4096, fs=p.fs)
    ax_h.plot(f / 1000, 20 * np.log10(np.abs(h) + 1e-12), color=COLORES["Huellas (A)"])
    for borde in (p.f_min, p.f_max):
        ax_h.axvline(borde / 1000, color=TINTA_2, linewidth=1, linestyle="--")
    ax_h.set_ylim(-80, 5)
    ax_h.set_xlabel("Frecuencia (kHz)")
    ax_h.set_ylabel("|H(f)| (dB)")
    ax_h.set_title(f"Respuesta en frecuencia, orden {p.orden_filtro}", loc="left")
    _terminar(fig, ruta, mostrar)


def fuga(f, espectros, titulo, ruta, mostrar=False):
    """Espectro de un tono puro con distintas ventanas (fuga espectral)."""
    colores = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (nombre, db), color in zip(espectros.items(), colores):
        ax.plot(f, db, color=color, label=nombre)
    ax.set_ylim(-120, 5)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud (dB, 0 = pico)")
    ax.set_title(titulo, loc="left")
    ax.legend(labelcolor=TINTA)
    _terminar(fig, ruta, mostrar)
