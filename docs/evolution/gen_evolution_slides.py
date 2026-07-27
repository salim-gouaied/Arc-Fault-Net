#!/usr/bin/env python3
"""
Arc-FaultNet — chronological evolution slides + V1 vs V2 comparison.

Generates PNG slides (16:9) for presentations:

  slides/00_title.png
  slides/01_timeline.png
  slides/02_v1_initial.png
  slides/03_enhancements_se.png
  slides/04_ablation.png
  slides/05_data_2048.png
  slides/06_v2_architecture.png
  slides/07_channel_experiments.png
  slides/08_architecture_comparison.png

Run:
    python docs/evolution/gen_evolution_slides.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "slides"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 150
SLIDE = (13.33, 7.5)  # 16:9

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

C = {
    "v1": "#cfe2f3",
    "v2": "#d5e8d4",
    "accent": "#1155cc",
    "muted": "#6a6a6a",
    "text": "#1a1a1a",
    "orange": "#f9cb9c",
    "purple": "#b4a7d6",
    "green": "#6aa84f",
    "red": "#b22222",
    "grey": "#e8e8e8",
    "timeline": "#f0f4fa",
}


def _darken(hexc: str, f: float = 0.55):
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    return (r / 255 * f, g / 255 * f, b / 255 * f)


def _ax():
    fig, ax = plt.subplots(figsize=SLIDE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _round(ax, x, y, w, h, fc, ec=None, lw=1.4, alpha=1.0, ls="-", z=2):
    ec = ec or _darken(fc)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
        alpha=alpha, zorder=z,
    ))


def _txt(ax, x, y, s, size=10, weight="normal", color=None, ha="center",
         va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
            style=style, color=color or C["text"])


def _save(fig, name: str):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


def _slide_header(ax, title: str, subtitle: str = "", step: str = ""):
    if step:
        _round(ax, 0.03, 0.88, 0.08, 0.08, C["accent"], ec=C["accent"], z=1)
        _txt(ax, 0.07, 0.92, step, size=14, weight="bold", color="white")
    _txt(ax, 0.5, 0.93, title, size=22, weight="bold", color=C["accent"])
    if subtitle:
        _txt(ax, 0.5, 0.87, subtitle, size=11, style="italic", color=C["muted"])
    ax.plot([0.05, 0.95], [0.84, 0.84], color="#cccccc", lw=0.8, zorder=1)


def _bullets(ax, items, x=0.08, y0=0.76, dy=0.072, size=11):
    y = y0
    for item in items:
        _txt(ax, x, y, "•", size=13, weight="bold", color=C["accent"], ha="left")
        _txt(ax, x + 0.025, y, item, size=size, ha="left", va="center")
        y -= dy


def slide_title():
    fig, ax = _ax()
    _round(ax, 0.08, 0.28, 0.84, 0.52, C["timeline"], ec=C["accent"], lw=2.0)
    _txt(ax, 0.5, 0.62, "Arc-FaultNet", size=36, weight="bold", color=C["accent"])
    _txt(ax, 0.5, 0.52, "Évolution chronologique de l'architecture", size=18)
    _txt(ax, 0.5, 0.42,
         "V1 (Gabor + Joint Attention)  →  ablations  →  V2 (front-end physique + IA embarquée)",
         size=12, color=C["muted"])
    _txt(ax, 0.5, 0.18, "Slides générées depuis l'historique git du dépôt",
         size=10, style="italic", color=C["muted"])
    _save(fig, "00_title.png")


MILESTONES = [
    ("7878c1c", "V1 initiale", "Dual-branch · Gabor · Joint Attention"),
    ("9590f47", "HP search", "Optimisation hyperparamètres"),
    ("1d0d0c3", "Anti-overfit", "Réduction des paramètres"),
    ("8e07ba9", "SE + amplitude", "Blocs SE · Gabor amplifiable"),
    ("0ac2070", "Ablation", "6 variantes · validation empirique"),
    ("576526e", "Dataset 2048", "102,4 kHz · M = 2048"),
    ("eac9c34", "V2", "Nouvelle architecture single-cycle"),
    ("92088d8", "Eval V2", "Métriques & généralisation"),
    ("6abef3d", "Dowalla", "Résidu inter-cycle ΔI_k"),
    ("966b620", "HEAD", "Retour |ΔI| intra-cycle"),
]


def slide_timeline():
    fig, ax = _ax()
    _slide_header(ax, "Chronologie des jalons", "Commits git — ordre temporel")
    n = len(MILESTONES)
    x0, w, gap = 0.04, 0.088, 0.004
    y_node, y_sub = 0.58, 0.44
    for i, (commit, title, sub) in enumerate(MILESTONES):
        x = x0 + i * (w + gap)
        col = C["v2"] if i >= 6 else C["v1"]
        if i >= 8:
            col = C["orange"]
        _round(ax, x, y_node - 0.06, w, 0.12, col, ec=_darken(col), lw=1.2)
        _txt(ax, x + w / 2, y_node + 0.02, title, size=7.8, weight="bold")
        _txt(ax, x + w / 2, y_sub, sub, size=6.2, color=C["muted"])
        _txt(ax, x + w / 2, y_node - 0.10, commit[:7], size=6.0,
             color=C["accent"], style="italic")
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap * 0.3, y_node),
                        xytext=(x + w, y_node),
                        arrowprops=dict(arrowstyle="-|>", color=C["muted"], lw=1.2))
    _round(ax, 0.05, 0.12, 0.42, 0.22, "white", ec=C["muted"])
    _txt(ax, 0.26, 0.28, "Phase V1", size=10, weight="bold", color=C["accent"])
    _txt(ax, 0.26, 0.20, "Gabor · SE · ablation sur combined_dataset (20k)",
         size=8.5, color=C["muted"])
    _round(ax, 0.53, 0.12, 0.42, 0.22, C["v2"], ec=C["green"], alpha=0.5)
    _txt(ax, 0.74, 0.28, "Phase V2", size=10, weight="bold", color=C["green"])
    _txt(ax, 0.74, 0.20, "Front-end I(t) · FrequencyGate · XGBoost head",
         size=8.5, color=C["muted"])
    _save(fig, "01_timeline.png")


def slide_v1_initial():
    fig, ax = _ax()
    _slide_header(ax, "Étape 1 — Arc-FaultNet V1 (initial)",
                  "commit 7878c1c · dual-branch CNN + Joint Attention", "01")
    _bullets(ax, [
        "Entrée : 2 canaux bruts [V_ligne, I] — 20 000 échantillons / cycle @ 1 MHz",
        "Branche 1D : ParametricConv1d — filtres Gabor (f₀, σ apprenables)",
        "Branche 2D : STFT → Conv2d sur spectrogramme (tranche HF fixe)",
        "Fusion : Joint Attention — CAM (canaux) + SAM (temps) croisés",
        "Classifieur : tête FC → P(arc)",
        "Inspiration MC-VSAttn ; contribution : dual-branch + attention inter-branches",
    ])
    # mini diagram
    _round(ax, 0.55, 0.48, 0.12, 0.14, C["v1"])
    _txt(ax, 0.61, 0.58, "[V, I]", size=9, weight="bold")
    _round(ax, 0.72, 0.55, 0.11, 0.10, C["orange"])
    _txt(ax, 0.775, 0.60, "Gabor 1D", size=8, weight="bold")
    _round(ax, 0.72, 0.38, 0.11, 0.10, C["orange"])
    _txt(ax, 0.775, 0.43, "STFT 2D", size=8, weight="bold")
    _round(ax, 0.86, 0.46, 0.10, 0.18, C["purple"])
    _txt(ax, 0.91, 0.58, "Joint", size=8, weight="bold")
    _txt(ax, 0.91, 0.50, "Attention", size=8, weight="bold")
    for p0, p1 in [((0.67, 0.55), (0.72, 0.60)), ((0.67, 0.45), (0.72, 0.43)),
                   ((0.83, 0.60), (0.86, 0.55)), ((0.83, 0.43), (0.86, 0.50))]:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                     color=C["muted"], lw=1.4))
    _save(fig, "02_v1_initial.png")


def slide_se():
    fig, ax = _ax()
    _slide_header(ax, "Étape 2 — Enrichissements V1",
                  "commits 9590f47 · 1d0d0c3 · 8e07ba9", "02")
    _bullets(ax, [
        "SEBlock : Squeeze-and-Excitation après chaque couche conv (1D et 2D)",
        "Amplitude Gabor apprenable (use_amplitude) — gain par filtre",
        "Flags : --use-se · --use-amplitude · --deep-clf",
        "Réduction des paramètres (1d0d0c3) pour limiter l'overfitting",
        "Pipeline étendu : K-fold, LOCO, hyperparamètres (lr, weight-decay…)",
    ])
    _round(ax, 0.55, 0.35, 0.38, 0.38, C["grey"], ec=C["muted"], alpha=0.4)
    _txt(ax, 0.74, 0.66, "Bloc conv typique V1 enrichi", size=10, weight="bold")
    blocks = ["Conv / Gabor", "BatchNorm", "ReLU", "SEBlock", "MaxPool"]
    y = 0.58
    for b in blocks:
        _round(ax, 0.60, y, 0.28, 0.07, "white", ec=C["accent"] if b == "SEBlock" else C["muted"])
        _txt(ax, 0.74, y + 0.035, b, size=9, weight="bold" if b == "SEBlock" else "normal")
        y -= 0.09
    _save(fig, "03_enhancements_se.png")


def slide_ablation():
    fig, ax = _ax()
    _slide_header(ax, "Étape 3 — Étude d'ablation",
                  "commit 0ac2070 · combined_dataset · seed=3", "03")
    rows = [
        ("standard_conv (sans Gabor)", "96,68 %", "+0,55 % vs Gabor"),
        ("arcfaultnet (référence)", "96,08 %", "Gabor + Joint Attn"),
        ("no_attention", "94,76 %", "concat simple"),
        ("1d_only (sans STFT)", "65,88 %", "branche 2D indispensable"),
        ("baseline_cnn", "89,82 %", "sans dual-branch"),
    ]
    _txt(ax, 0.12, 0.72, "Variante", size=9, weight="bold", ha="left", color=C["accent"])
    _txt(ax, 0.52, 0.72, "F1", size=9, weight="bold", color=C["accent"])
    _txt(ax, 0.68, 0.72, "Insight", size=9, weight="bold", ha="left", color=C["accent"])
    ax.plot([0.08, 0.94], [0.70, 0.70], color="#ccc", lw=0.8)
    y = 0.64
    for name, f1, insight in rows:
        hl = "standard_conv" in name
        _round(ax, 0.08, y - 0.04, 0.86, 0.055,
               "#fff8e6" if hl else "white", ec="#e6a23c" if hl else C["muted"], lw=1.0)
        _txt(ax, 0.12, y, name, size=9, ha="left")
        _txt(ax, 0.52, y, f1, size=9, weight="bold")
        _txt(ax, 0.68, y, insight, size=8.5, ha="left", color=C["muted"])
        y -= 0.065
    _round(ax, 0.08, 0.12, 0.86, 0.10, C["v2"], ec=C["green"], alpha=0.35)
    _txt(ax, 0.5, 0.18,
         "→ Conclusion : les Gabor n'aident pas ; la dualité temporel/spectral + fusion oui → motivation V2",
         size=10, weight="bold", color=_darken(C["green"], 0.35))
    _save(fig, "04_ablation.png")


def slide_2048():
    fig, ax = _ax()
    _slide_header(ax, "Étape 4 — Format données 2048",
                  "commit 576526e · combined_dataset_2048", "04")
    _bullets(ax, [
        "Décimation : 20 000 → 2 048 échantillons / cycle (102,4 kHz effectif)",
        "fs et n_fft propagés dans le modèle (init Gabor V1 cohérente)",
        "QA décimation : spectres et formes d'onde validés",
        "Entraînement cross-fold sur le nouveau format",
        "Prépare le déploiement AFDD embarqué (fenêtre plus courte)",
    ])
    _round(ax, 0.55, 0.22, 0.38, 0.48, C["grey"], ec=C["muted"], alpha=0.3)
    _txt(ax, 0.74, 0.64, "Avant", size=10, weight="bold")
    _txt(ax, 0.60, 0.56, "M = 20 000", size=11, ha="left")
    _txt(ax, 0.60, 0.50, "fs = 1 MHz", size=11, ha="left")
    _txt(ax, 0.74, 0.40, "Après", size=10, weight="bold", color=C["green"])
    _txt(ax, 0.60, 0.32, "M = 2 048", size=11, ha="left", weight="bold")
    _txt(ax, 0.60, 0.26, "fs = 102,4 kHz", size=11, ha="left", weight="bold")
    ax.annotate("", xy=(0.82, 0.38), xytext=(0.82, 0.52),
                arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=2))
    _save(fig, "05_data_2048.png")


def slide_v2():
    fig, ax = _ax()
    _slide_header(ax, "Étape 5 — Arc-FaultNet V2",
                  "commit eac9c34 · refonte single-cycle", "05")
    _bullets(ax, [
        "Front-end : 4 canaux dérivés de I(t) seul — I_norm, |ΔI|, TKEO, RMS_slide",
        "V(t) retiré de l'entrée modèle (segmentation / labels uniquement)",
        "TemporalBranchV2 : Conv1d + GELU (Gabor supprimé — arc impulsif, non périodique)",
        "SpectralBranchV2 : FrequencyGate + pooling asymétrique",
        "RevisedCrossAttention : deux gates CAM conditionnés → embedding z (128-d)",
        "Décision en 2 phases : FC (train) puis XGBoost sur z (deploy)",
        "~0,35 M paramètres · docs/architecture + train_xgb_head.py",
    ])
    _save(fig, "06_v2_architecture.png")


def slide_channels():
    fig, ax = _ax()
    _slide_header(ax, "Étape 6 — Expériences canal 1",
                  "commits 6abef3d → 966b620 (HEAD actuel)", "06")
    _round(ax, 0.08, 0.38, 0.38, 0.38, C["v1"], ec=C["muted"])
    _txt(ax, 0.27, 0.68, "V2 initial (eac9c34)", size=10, weight="bold")
    _txt(ax, 0.27, 0.58, "Canal 1 = |ΔI|", size=11, weight="bold")
    _txt(ax, 0.27, 0.50, "dérivée intra-cycle", size=9, color=C["muted"])
    _txt(ax, 0.27, 0.42, "|I[n] − I[n−1]|", size=9, style="italic", color=C["muted"])

    _round(ax, 0.54, 0.38, 0.38, 0.38, C["orange"], ec="#e69138")
    _txt(ax, 0.73, 0.68, "Expérience Dowalla (6abef3d)", size=10, weight="bold")
    _txt(ax, 0.73, 0.58, "Canal 1 = ΔI_k", size=11, weight="bold")
    _txt(ax, 0.73, 0.50, "résidu inter-cycle", size=9, color=C["muted"])
    _txt(ax, 0.73, 0.42, "I_k − I_(k−1) + metadata.csv", size=9, style="italic",
         color=C["muted"])

    _round(ax, 0.08, 0.12, 0.84, 0.18, "#fff4e5", ec="#e6a23c")
    _txt(ax, 0.5, 0.24,
         "HEAD actuel (966b620) : retour à |ΔI| intra-cycle",
         size=12, weight="bold", color="#b8740f")
    _txt(ax, 0.5, 0.16,
         "V2 inchangée côté réseau — seul le front-end canal 1 a été testé puis restauré",
         size=9, color=C["muted"])
    _save(fig, "07_channel_experiments.png")


def _arch_column(ax, x, title, color, items, diagram_rows):
    w = 0.42
    _round(ax, x, 0.11, w, 0.66, color, ec=_darken(color), alpha=0.25, lw=1.6)
    _txt(ax, x + w / 2, 0.78, title, size=14, weight="bold", color=_darken(color, 0.4))
    y = 0.72
    for label, value in items:
        _txt(ax, x + 0.03, y, label, size=8.0, weight="bold", ha="left", color=C["accent"])
        _txt(ax, x + 0.20, y, value, size=7.8, ha="left")
        y -= 0.044
    # flow diagram (compact, above footer)
    dy, bh = 0.078, 0.048
    y = 0.24
    for i, (lbl, fc) in enumerate(diagram_rows):
        _round(ax, x + 0.06, y, w - 0.12, bh, fc, ec=_darken(fc))
        _txt(ax, x + w / 2, y + bh / 2, lbl, size=7.8, weight="bold")
        if i < len(diagram_rows) - 1:
            ax.annotate("", xy=(x + w / 2, y - 0.01), xytext=(x + w / 2, y + bh),
                        arrowprops=dict(arrowstyle="-|>", color=C["muted"], lw=1.2))
        y -= dy


def slide_comparison():
    fig, ax = _ax()
    _slide_header(ax, "Comparaison d'architecture — V1 vs V2", "Vue côte à côte")

    v1_items = [
        ("Entrée", "[V_ligne, I] bruts"),
        ("Longueur", "20 000 @ 1 MHz (ou 2048)"),
        ("Branche 1D", "ParametricConv1d (Gabor)"),
        ("Branche 2D", "STFT 2 canaux · slice HF fixe"),
        ("SE blocks", "Optionnels (--use-se)"),
        ("Fusion", "Joint Attention CAM+SAM"),
        ("Tête", "FC uniquement"),
        ("Params", "~344 k"),
    ]
    v1_flow = [
        ("[V, I]", C["v1"]),
        ("Gabor 1D ∥ STFT 2D", C["orange"]),
        ("Joint Attention", C["purple"]),
        ("FC → P(arc)", C["accent"]),
    ]

    v2_items = [
        ("Entrée", "I(t) seul — 4 canaux dérivés"),
        ("Longueur", "2 048 @ 102,4 kHz"),
        ("Branche 1D", "Conv1d + GELU (sans Gabor)"),
        ("Branche 2D", "STFT(I) · FrequencyGate"),
        ("SE blocks", "Non (design allégé)"),
        ("Fusion", "RevisedCrossAttention"),
        ("Tête", "FC (train) + XGBoost (deploy)"),
        ("Params", "~350 k"),
    ]
    v2_flow = [
        ("4 dérivées I + STFT(I)", C["v2"]),
        ("Encodeur 1D ∥ Encodeur 2D", C["orange"]),
        ("Cross-Attention → z", C["purple"]),
        ("FC / XGBoost → P(arc)", C["accent"]),
    ]

    _arch_column(ax, 0.05, "Arc-FaultNet V1", C["v1"], v1_items, v1_flow)
    _arch_column(ax, 0.53, "Arc-FaultNet V2", C["v2"], v2_items, v2_flow)

    # centre arrow
    _txt(ax, 0.5, 0.50, "→", size=28, weight="bold", color=C["accent"])
    _txt(ax, 0.5, 0.42, "évolution", size=9, style="italic", color=C["muted"])

    # legend removed keys
    _round(ax, 0.05, 0.02, 0.90, 0.075, "#f5f9f0", ec=C["green"], alpha=0.5, lw=1.0)
    changes = [
        ("Gabor → Conv1d", "arc impulsif, non oscillatoire"),
        ("[V,I] → 4×I", "capteur unique, load-invariant"),
        ("Joint Attn → Cross-Attn", "fusion sans ambiguïté canal"),
        ("FC → FC+XGB", "calibration déploiement"),
    ]
    y = 0.078
    for a, b in changes:
        _txt(ax, 0.07, y, f"▸ {a}", size=7.2, ha="left", weight="bold", color=C["green"])
        _txt(ax, 0.30, y, b, size=7.2, ha="left", color=C["muted"])
        y -= 0.016

    _save(fig, "08_architecture_comparison.png")


def main():
    print("Arc-FaultNet evolution slides →", OUT)
    slide_title()
    slide_timeline()
    slide_v1_initial()
    slide_se()
    slide_ablation()
    slide_2048()
    slide_v2()
    slide_channels()
    slide_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
