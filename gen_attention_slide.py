#!/usr/bin/env python3
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "slides"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 180

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

COL = {
    "input":     "#cfe2f3",
    "frontend":  "#b6d7a8",
    "temporal":  "#f9cb9c",
    "spectral":  "#f6b26b",
    "fusion":    "#b4a7d6",
    "embed":     "#9fc5e8",
    "tree":      "#f4a582",
    "future":    "#e8e8e8",
    "delta":     "#ead1dc",
    "text":      "#1a1a1a",
    "muted":     "#7a7a7a",
    "accent":    "#1155cc",
    "arc":       "#b22222",
    "bg_box":    "#f4f4f4",
    "highlight": "#fff2cc",
}

def _darken(hexc, f=0.62):
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    return (r / 255 * f, g / 255 * f, b / 255 * f)

def _round(ax, x, y, w, h, fc, ec=None, lw=1.2, rounding=0.015, ls="-", alpha=1.0, z=2):
    if ec is None:
        ec = _darken(fc) if fc != "white" and fc != "#ffffff" else "#cccccc"
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
        alpha=alpha, mutation_aspect=1.0, zorder=z,
    )
    ax.add_patch(p)
    return p

def _txt(ax, x, y, s, size=10, weight="normal", color=None, ha="center", va="center", style="normal", z=4):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
            style=style, color=color or COL["text"], zorder=z)

def _arrow(ax, p0, p1, color=None, lw=1.6, style="-|>", rad=0.0, z=3, ls="-", mut=12):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=mut,
        linewidth=lw, color=color or COL["muted"],
        connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=z,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    return a

def slide_general():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    
    _txt(ax, 0.5, 0.94, "Le Mécanisme d'Attention en Deep Learning", size=22, weight="bold", color=COL["text"])
    _txt(ax, 0.5, 0.88, "Principe Fondamental : Requête (Query), Clé (Key), Valeur (Value)", size=14, color=COL["muted"], style="italic")
    ax.plot([0.05, 0.95], [0.82, 0.82], color="#e0e0e0", lw=1.5)
    
    # Left side: Explanation
    _txt(ax, 0.04, 0.76, "L'Analogie de la Recherche", size=14, weight="bold", color=COL["accent"], ha="left")
    
    _round(ax, 0.04, 0.43, 0.44, 0.30, fc=COL["bg_box"])
    _txt(ax, 0.26, 0.69, "L'attention agit comme un filtre intelligent", size=10, weight="bold")
    _txt(ax, 0.26, 0.65, "qui interroge ses propres connaissances :", size=10, weight="bold")
    
    _txt(ax, 0.26, 0.58, "1. Query (Q) : La Question", size=11, color=COL["accent"], weight="bold")
    _txt(ax, 0.26, 0.54, "Quelles sont les caractéristiques les plus importantes ?", size=9)
    
    _txt(ax, 0.26, 0.48, "2. Key (K) : L'Espace de Connaissance", size=11, color=COL["accent"], weight="bold")
    _txt(ax, 0.26, 0.45, "Où se trouve l'information pertinente dans le signal.", size=9)

    _round(ax, 0.04, 0.16, 0.44, 0.23, fc="white", ec=COL["accent"])
    _txt(ax, 0.26, 0.35, "Le Fonctionnement", size=11, weight="bold", color=COL["accent"])
    _txt(ax, 0.26, 0.30, "Le modèle croise la Question (Q) avec la Clé (K) pour", size=9.5)
    _txt(ax, 0.26, 0.26, "calculer des poids d'attention. Ces poids servent", size=9.5)
    _txt(ax, 0.26, 0.22, "à attribuer de l'attention exclusivement aux", size=9.5)
    _txt(ax, 0.26, 0.18, "Valeurs (V) cruciales, en ignorant le bruit.", size=9.5, weight="bold")

    # Right side: Visual Diagram of Attention
    _txt(ax, 0.52, 0.75, "Fonctionnement Graphique (Q, K, V)", size=14, weight="bold", color=COL["accent"], ha="left")
    
    # Q, K, V blocks
    _round(ax, 0.55, 0.60, 0.12, 0.08, fc="#f4a582")
    _txt(ax, 0.61, 0.64, "Query (Q)", size=11, weight="bold")
    
    _round(ax, 0.70, 0.60, 0.12, 0.08, fc="#f6b26b")
    _txt(ax, 0.76, 0.64, "Key (K)", size=11, weight="bold")
    
    _round(ax, 0.85, 0.60, 0.12, 0.08, fc="#b6d7a8")
    _txt(ax, 0.91, 0.64, "Value (V)", size=11, weight="bold")
    
    # Arrows to MatMul
    _arrow(ax, (0.61, 0.59), (0.65, 0.49))
    _arrow(ax, (0.76, 0.59), (0.70, 0.49))
    
    # MatMul / Compatibilité
    _round(ax, 0.58, 0.41, 0.18, 0.07, fc="white", ec=COL["muted"])
    _txt(ax, 0.67, 0.445, "Comparaison (Q × K)", size=10, weight="bold")
    
    # Arrow to Softmax
    _arrow(ax, (0.67, 0.40), (0.67, 0.32))
    
    # Softmax / Poids
    _round(ax, 0.58, 0.24, 0.18, 0.07, fc=COL["accent"])
    _txt(ax, 0.67, 0.275, "Poids d'Attention\n(Softmax)", size=10, weight="bold", color="white")
    
    # Arrow from Softmax and Value to final Mult
    _arrow(ax, (0.67, 0.23), (0.74, 0.14))
    _arrow(ax, (0.91, 0.59), (0.80, 0.14), rad=-0.2)
    
    # Final multiplication
    _round(ax, 0.71, 0.06, 0.18, 0.07, fc=COL["highlight"], ec=COL["arc"], lw=2)
    _txt(ax, 0.80, 0.095, "Signal Filtré\n(Poids × V)", size=10, weight="bold", color=COL["arc"])
    
    fig.savefig(OUT / "slide_05_attention_general.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Slide general generated.")

def slide_arcfaultnet():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    
    _txt(ax, 0.5, 0.94, "L'Attention dans Arc-FaultNet V2", size=22, weight="bold", color=COL["text"])
    _txt(ax, 0.5, 0.88, "Deux mécanismes complémentaires pour fiabiliser la détection", size=14, color=COL["muted"], style="italic")
    ax.plot([0.05, 0.95], [0.82, 0.82], color="#e0e0e0", lw=1.5)
    
    # ==========================================
    # LEFT: Channel Attention
    # ==========================================
    _txt(ax, 0.25, 0.75, "1. Channel Attention (Filtrage)", size=13, weight="bold", color=COL["accent"])
    _txt(ax, 0.25, 0.68, "Le réseau extrait de multiples caractéristiques (canaux).", size=9)
    _txt(ax, 0.25, 0.64, "L'attention canal évalue la pertinence de chacun.", size=9)
    
    _round(ax, 0.06, 0.44, 0.10, 0.14, fc="#e0e0e0")
    _txt(ax, 0.11, 0.51, "Canal 1\n(Bruit)", size=9)
    
    _round(ax, 0.20, 0.44, 0.10, 0.14, fc="#f4a582")
    _txt(ax, 0.25, 0.51, "Canal 2\n(Signal Arc)", size=9, weight="bold")
    
    _round(ax, 0.34, 0.44, 0.10, 0.14, fc="#e0e0e0")
    _txt(ax, 0.39, 0.51, "Canal 3\n(Moteur)", size=9)
    
    _arrow(ax, (0.25, 0.42), (0.25, 0.34))
    
    _round(ax, 0.15, 0.24, 0.20, 0.08, fc="white", ec=COL["accent"])
    _txt(ax, 0.25, 0.28, "Évaluation (Score Beta)", size=10, weight="bold", color=COL["accent"])
    
    _arrow(ax, (0.25, 0.22), (0.25, 0.14))
    
    _round(ax, 0.06, 0.04, 0.10, 0.08, fc="#e0e0e0")
    _txt(ax, 0.11, 0.08, "Beta ≈ 0\n(Ignoré)", size=8)
    
    _round(ax, 0.20, 0.04, 0.10, 0.08, fc=COL["highlight"], ec=COL["arc"], lw=2)
    _txt(ax, 0.25, 0.08, "Beta ≈ 1\n(Amplifié)", size=8, weight="bold")
    
    _round(ax, 0.34, 0.04, 0.10, 0.08, fc="#e0e0e0")
    _txt(ax, 0.39, 0.08, "Beta ≈ 0\n(Ignoré)", size=8)

    # ==========================================
    # RIGHT: Cross Attention
    # ==========================================
    _txt(ax, 0.75, 0.75, "2. Cross-Attention (Coopération)", size=13, weight="bold", color=COL["accent"])
    _txt(ax, 0.75, 0.68, "Les branches Temporelle et Spectrale s'échangent", size=9)
    _txt(ax, 0.75, 0.64, "des informations pour s'auto-corriger.", size=9)
    
    _round(ax, 0.58, 0.46, 0.14, 0.10, fc=COL["temporal"])
    _txt(ax, 0.65, 0.51, "Domaine\nTemporel", size=10, weight="bold")
    
    _round(ax, 0.78, 0.46, 0.14, 0.10, fc=COL["spectral"])
    _txt(ax, 0.85, 0.51, "Domaine\nSpectral", size=10, weight="bold")
    
    # Arrows crossing
    _arrow(ax, (0.65, 0.44), (0.83, 0.33), rad=-0.2, color=COL["accent"])
    _arrow(ax, (0.85, 0.44), (0.67, 0.33), rad=-0.2, color=COL["accent"])
    
    _round(ax, 0.58, 0.22, 0.14, 0.09, fc="white", ec=COL["temporal"])
    _txt(ax, 0.65, 0.265, "Temporel\nFiltré", size=9, weight="bold")
    
    _round(ax, 0.78, 0.22, 0.14, 0.09, fc="white", ec=COL["spectral"])
    _txt(ax, 0.85, 0.265, "Spectral\nFiltré", size=9, weight="bold")
    
    _round(ax, 0.54, 0.03, 0.42, 0.15, fc=COL["bg_box"])
    _txt(ax, 0.75, 0.145, "L'Intuition :", size=9, weight="bold")
    _txt(ax, 0.75, 0.105, "Si le spectre fréquentiel est parasité par un appareil,", size=8.5)
    _txt(ax, 0.75, 0.075, "la forme temporelle va aider le modèle à ignorer", size=8.5)
    _txt(ax, 0.75, 0.045, "ce bruit spectral (et inversement).", size=8.5)

    fig.savefig(OUT / "slide_06_attention_arcfaultnet.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Slide arcfaultnet generated.")

if __name__ == "__main__":
    slide_general()
    slide_arcfaultnet()
