#!/usr/bin/env python3
"""
Arc-FaultNet V2 — Presentation slides generator (clean minimal style).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

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

# Minimal palette — black dominant
BLK = "#1a1a1a"
GRY = "#6a6a6a"
LGRY = "#d0d0d0"
VLGRY = "#f2f2f2"
ACCENT = "#1155cc"
RED = "#c0392b"
GREEN = "#27ae60"


def _darken(hexc, f=0.55):
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i+2], 16) for i in (0, 2, 4))
    return (r/255*f, g/255*f, b/255*f)


def _rr(ax, x, y, w, h, fc, ec=None, lw=0.8, rounding=0.008, alpha=1.0, z=2):
    if ec is None:
        ec = LGRY
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle="-",
        alpha=alpha, mutation_aspect=1.0, zorder=z)
    ax.add_patch(p)


def _table(ax, headers, rows, top, col_xs, col_ws, row_h=0.045,
           highlight_col=None, highlight_fn=None):
    """Draw a clean grid table."""
    total_w = sum(col_ws)
    start_x = col_xs[0]
    total_h = row_h * (len(rows) + 1)
    bottom = top - len(rows) * row_h
    
    # Outer border and background
    _rr(ax, start_x, bottom, total_w, total_h, "white", ec=BLK, lw=1.2, rounding=0)
    
    # Header background
    _rr(ax, start_x, top, total_w, row_h, BLK, ec=BLK, lw=1.2, rounding=0)
    
    # Vertical lines
    for i in range(1, len(col_xs)):
        ax.plot([col_xs[i], col_xs[i]], [bottom, top + row_h], color=BLK, lw=1.2)
        
    # Horizontal lines
    for i in range(len(rows)):
        y = top - i * row_h
        ax.plot([start_x, start_x + total_w], [y, y], color=LGRY, lw=0.8)
    
    # Header text
    for j, (hdr, cx) in enumerate(zip(headers, col_xs)):
        ax.text(cx + col_ws[j]/2, top + row_h/2, hdr,
                fontsize=9.5, fontweight="bold", ha="center", va="center", color="white")
    
    # Rows text
    for i, row_vals in enumerate(rows):
        y = top - (i+1) * row_h
        for j, (v, cx) in enumerate(zip(row_vals, col_xs)):
            color = BLK
            fw = "normal"
            if highlight_col is not None and j == highlight_col and highlight_fn:
                color, fw = highlight_fn(v)
            ax.text(cx + col_ws[j]/2, y + row_h/2, str(v),
                    fontsize=9, fontweight=fw, ha="center", va="center", color=color)


# =====================================================================
# SLIDE 1 — Threshold Analysis
# =====================================================================
def slide_threshold():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Title
    ax.text(0.50, 0.96, "Analyse du Seuil de Decision", fontsize=20,
            fontweight="bold", ha="center", va="top", color=BLK)
    ax.text(0.50, 0.91, "Arc-FaultNet V2  |  seed = 42  |  combined_dataset_2048",
            fontsize=10, ha="center", va="top", color=GRY)

    # Section 1 label
    ax.text(0.06, 0.845, "Effet du seuil sur les metriques", fontsize=12,
            fontweight="bold", ha="left", va="center", color=BLK)
    ax.plot([0.06, 0.94], [0.825, 0.825], color=LGRY, lw=0.8)

    # Table
    headers = ["Seuil", "TP", "FP", "FN", "TN", "Precision", "Rappel", "F1"]
    rows = [
        ["0.3", "734", "5",  "23", "868", "99.32%", "96.96%", "98.13%"],
        ["0.4", "727", "4",  "30", "869", "99.45%", "96.04%", "97.72%"],
        ["0.5", "721", "3",  "36", "870", "99.59%", "95.24%", "97.37%"],
        ["0.6", "716", "3",  "41", "870", "99.58%", "94.58%", "97.02%"],
        ["0.7", "710", "3",  "47", "870", "99.58%", "93.79%", "96.60%"],
        ["0.8", "700", "2",  "57", "871", "99.72%", "92.47%", "95.96%"],
    ]
    col_xs = [0.06, 0.17, 0.28, 0.38, 0.48, 0.59, 0.72, 0.85]
    col_ws = [0.09, 0.09, 0.08, 0.08, 0.09, 0.11, 0.11, 0.09]

    def fp_highlight(v):
        try:
            n = int(v)
        except ValueError:
            return BLK, "normal"
        if n <= 2: return GREEN, "bold"
        if n >= 5: return RED, "bold"
        return BLK, "bold"

    _table(ax, headers, rows, 0.78, col_xs, col_ws, row_h=0.042,
           highlight_col=2, highlight_fn=fp_highlight)

    # Section 2 label
    ax.text(0.06, 0.44, "Compromis", fontsize=12,
            fontweight="bold", ha="left", va="center", color=BLK)
    ax.plot([0.06, 0.94], [0.42, 0.42], color=LGRY, lw=0.8)

    # Left box
    _rr(ax, 0.06, 0.22, 0.38, 0.17, VLGRY, ec=LGRY, rounding=0.012)
    ax.text(0.25, 0.365, "Seuil BAS (< 0.5)", fontsize=11,
            fontweight="bold", ha="center", va="center", color=BLK)
    ax.text(0.25, 0.315, "Rappel eleve  /  Plus de FP", fontsize=9,
            ha="center", va="center", color=GRY)
    ax.text(0.25, 0.265, "Risque : declenchements\nintempestifs du disjoncteur",
            fontsize=8.5, ha="center", va="center", color=RED, style="italic", linespacing=1.5)

    # Arrow
    ax.annotate("", xy=(0.56, 0.30), xytext=(0.44, 0.30),
                arrowprops=dict(arrowstyle="<->", lw=1.8, color=BLK))

    # Right box
    _rr(ax, 0.56, 0.22, 0.38, 0.17, VLGRY, ec=LGRY, rounding=0.012)
    ax.text(0.75, 0.365, "Seuil HAUT (> 0.5)", fontsize=11,
            fontweight="bold", ha="center", va="center", color=BLK)
    ax.text(0.75, 0.315, "Moins de FP  /  Plus de FN", fontsize=9,
            ha="center", va="center", color=GRY)
    ax.text(0.75, 0.265, "Risque : arc non detecte\n= danger d'incendie",
            fontsize=8.5, ha="center", va="center", color=RED, style="italic", linespacing=1.5)

    # Bottom note
    ax.text(0.50, 0.12, "Le seuil est un parametre post-entrainement.\n"
            "Il ne modifie pas le modele, seulement la decision finale.",
            fontsize=9, ha="center", va="center", color=GRY, style="italic", linespacing=1.6)

    path = OUT / "slide_01_threshold_analysis.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
# SLIDE 2 — Single-Mode Comparison
# =====================================================================
def slide_single_comparison():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.50, 0.95, "Comparaison des Modeles Single", fontsize=20,
            fontweight="bold", ha="center", va="top", color=BLK)
    ax.text(0.50, 0.88, "Arc-FaultNet V2  |  combined_dataset_2048  |  fs = 102.4 kHz",
            fontsize=10, ha="center", va="top", color=GRY)

    headers = ["Modele", "Accuracy", "FP", "F1-Score", "Precision", "Rappel"]
    rows = [
        ["Seed 42",       "97.61%", "3",  "97.37%", "99.59%", "95.24%"],
        ["Seed 3",        "96.99%", "7",  "96.77%", "99.05%", "94.58%"],
        ["Seed 42 (bis)", "97.48%", "5",  "97.25%", "98.77%", "95.77%"],
        ["Seed 4",        "97.24%", "22", "96.82%", "96.89%", "96.75%"],
    ]
    col_xs = [0.06, 0.22, 0.38, 0.50, 0.66, 0.82]
    col_ws = [0.14, 0.14, 0.10, 0.14, 0.14, 0.12]

    def fp_hl(v):
        try: n = int(v)
        except: return BLK, "normal"
        if n <= 3: return GREEN, "bold"
        if n >= 10: return RED, "bold"
        return BLK, "bold"

    _table(ax, headers, rows, 0.78, col_xs, col_ws, row_h=0.055,
           highlight_col=2, highlight_fn=fp_hl)

    # Summary
    ax.plot([0.06, 0.94], [0.48, 0.48], color=LGRY, lw=0.8)
    ax.text(0.50, 0.42, "Moyenne :  Accuracy = 97.33% (+/- 0.26%)   |   "
            "F1 = 97.05% (+/- 0.27%)   |   FP moyen = 9.25",
            fontsize=10, ha="center", va="center", color=BLK, fontweight="bold")
    ax.text(0.50, 0.34, "Le modele Seed 42 obtient le meilleur compromis : FP = 3 et F1 = 97.37%",
            fontsize=10, ha="center", va="center", color=BLK)
    ax.text(0.50, 0.22, "Architecture stable : la performance varie de moins de 0.3%\n"
            "entre les differents seeds d'initialisation.",
            fontsize=9, ha="center", va="center", color=GRY, style="italic", linespacing=1.5)

    path = OUT / "slide_02_single_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
# SLIDE 3 — Confusion Matrices
# =====================================================================
def slide_confusion_matrices():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle("Matrices de Confusion  —  Arc-FaultNet V2 (4 modeles Single)",
                 fontsize=16, fontweight="bold", color=BLK, y=1.04)

    cms = [
        ("Seed 42", np.array([[870, 3], [36, 721]])),
        ("Seed 3",  np.array([[848, 7], [42, 733]])),
        ("Seed 42 (bis)", np.array([[865, 8], [33, 724]])),
        ("Seed 4",  np.array([[900, 22], [23, 685]])),
    ]

    for idx, (title, cm) in enumerate(cms):
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Greys", aspect="auto", vmin=0, vmax=920)

        labels_ax = ["Normal", "Arc"]
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels_ax, fontsize=9, color=BLK)
        ax.set_yticklabels(labels_ax, fontsize=9, color=BLK)
        ax.set_xlabel("Predit", fontsize=9, fontweight="bold", color=BLK)
        if idx == 0:
            ax.set_ylabel("Vrai label", fontsize=9, fontweight="bold", color=BLK)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                if (i == 0 and j == 1):
                    lbl = f"FP={val}"
                    color = RED if val > 5 else BLK
                elif (i == 1 and j == 0):
                    lbl = f"FN={val}"
                    color = RED if val > 30 else BLK
                elif (i == 0 and j == 0):
                    lbl = f"TN={val}"
                    color = "white" if val > 500 else BLK
                else:
                    lbl = f"TP={val}"
                    color = "white" if val > 500 else BLK
                ax.text(j, i, lbl, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color=BLK)

        for spine in ax.spines.values():
            spine.set_edgecolor(LGRY)
            spine.set_linewidth(1)

    plt.tight_layout()
    path = OUT / "slide_03_confusion_matrices.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
# SLIDE 4 — GroupKFold Comparison
# =====================================================================
def slide_groupkfold():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.50, 0.97, "Comparaison GroupKFold Recording", fontsize=20,
            fontweight="bold", ha="center", va="top", color=BLK)
    ax.text(0.50, 0.925, "Arc-FaultNet V2  |  Validation croisee par enregistrements (K=5)",
            fontsize=10, ha="center", va="top", color=GRY)

    # Table
    headers = ["Metrique", "Modele A (08/06)", "Modele B (10/06)", "Delta"]
    rows = [
        ["Accuracy",    "87.9% +/- 11.3%", "92.3% +/- 11.5%", "+4.4%"],
        ["F1-Score",    "84.0% +/- 16.3%", "92.1% +/- 10.9%", "+8.0%"],
        ["Precision",   "93.8% +/- 7.5%",  "90.8% +/- 13.5%", "-3.0%"],
        ["Rappel",      "79.9% +/- 23.4%", "93.8% +/- 8.2%",  "+13.9%"],
        ["Specificite", "95.0% +/- 6.0%",  "91.1% +/- 14.8%", "-3.9%"],
    ]
    col_xs = [0.06, 0.26, 0.50, 0.78]
    col_ws = [0.18, 0.22, 0.26, 0.16]

    def delta_hl(v):
        if v.startswith("+"): return GREEN, "bold"
        if v.startswith("-"): return RED, "bold"
        return BLK, "normal"

    _table(ax, headers, rows, 0.82, col_xs, col_ws, row_h=0.045,
           highlight_col=3, highlight_fn=delta_hl)

    # Bar chart
    bar_ax = fig.add_axes([0.08, 0.06, 0.84, 0.32])
    fold_A = [77.43, 91.49, 71.89, 99.58, 99.08]
    fold_B = [98.70, 100.0, 100.0, 92.67, 70.00]

    x = np.arange(5)
    w = 0.32
    bars_a = bar_ax.bar(x - w/2, fold_A, w, label="Modele A (08/06)",
                        color="#aaaaaa", edgecolor="#666666", lw=0.6)
    bars_b = bar_ax.bar(x + w/2, fold_B, w, label="Modele B (10/06)",
                        color="#444444", edgecolor="#222222", lw=0.6)

    bar_ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold", color=BLK)
    bar_ax.set_xlabel("Fold", fontsize=10, fontweight="bold", color=BLK)
    bar_ax.set_title("Accuracy par Fold", fontsize=12, fontweight="bold", color=BLK, pad=8)
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(["Fold 1\n(IJL)", "Fold 2\n(IJL)", "Fold 3\n(IJL)",
                            "Fold 4\n(multi)", "Fold 5\n(multi)"], fontsize=8, color=BLK)
    bar_ax.set_ylim(60, 107)
    bar_ax.axhline(y=90, color=LGRY, linestyle="--", lw=0.8)
    bar_ax.legend(fontsize=9, loc="lower left")
    bar_ax.grid(axis="y", linestyle="--", alpha=0.2)
    bar_ax.tick_params(colors=BLK)

    for bar in bars_a:
        h = bar.get_height()
        bar_ax.text(bar.get_x()+bar.get_width()/2, h+0.8, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=7.5, color=GRY)
    for bar in bars_b:
        h = bar.get_height()
        bar_ax.text(bar.get_x()+bar.get_width()/2, h+0.8, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=7.5, color=BLK)

    # Conclusion
    ax.text(0.50, 0.44, "Le Modele B ameliore le F1 moyen de +8% et le Rappel de +14%.\n"
            "La variance elevee (std > 10%) traduit le domain gap entre les sources de donnees.",
            fontsize=9.5, ha="center", va="center", color=BLK, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=VLGRY, edgecolor=LGRY, lw=0.8))

    path = OUT / "slide_04_groupkfold_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  Saved -> {path}")


if __name__ == "__main__":
    print("Generating slides...")
    slide_threshold()
    slide_single_comparison()
    slide_confusion_matrices()
    slide_groupkfold()
    print(f"\nDone -> {OUT}/")
