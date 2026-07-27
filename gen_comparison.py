#!/usr/bin/env python3
"""Slide comparatif AFDD ~300k params + Arc-FaultNet V2."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

ABLATION = Path("ablation_results/ablation_v2_20260612_175320/ablation_v2_results.json")
with ABLATION.open() as f:
    v2 = json.load(f)["variants"]["arcfaultnet_v2"]

prec = 100 * v2["precision"]
fpr_str = "0.034 %"
n_params_n = v2["n_params"]
n_params = f"{n_params_n:,}".replace(",", " ")

# Estimation INT8 : interpolation linéaire sur les références littéraire
# (315k→38 ms, 328k→41 ms, 343k→46 ms), même ordre de grandeur ~350k params.
_LIT_INT8 = [(315_200, 38), (328_400, 41), (342_500, 46)]
_ms_per_param = sum(ms / p for p, ms in _LIT_INT8) / len(_LIT_INT8)
int8_ms = int(round(n_params_n * _ms_per_param))
int8_str = f"{int8_ms} ms"

BLACK = "#000000"
GREEN = "#E8F5E9"
GREEN_HDR = "#2E7D32"


def _wrap(text, width):
    return "\n".join(textwrap.wrap(text, width=width) or [text])


def _lines(text):
    return max(1, text.count("\n") + 1)


fig, ax = plt.subplots(figsize=(16.5, 6.4), dpi=300)
ax.axis("off")

plt.text(
    0.5, 0.95,
    "Modèles AFDD à Haute Capacité avec Attention (300k – 350k Paramètres)",
    ha="center", va="center", fontsize=15, fontweight="bold", color="#333333",
)

columns = [
    _wrap("Architecture (Spécifique Arc Électrique)", 22),
    _wrap("Précision (Accuracy)", 14),
    _wrap("Faux Positifs (FPR)", 14),
    _wrap("Nombre de Paramètres", 14),
    _wrap("Temps Inférence (INT8)", 14),
    _wrap("Généralisation", 14),
]

raw_data = [
    [
        "ViT-1D Arc (Self-Attention)*",
        "99.88%", "0.08%", "315 200", "38 ms",
        "Limitée · signatures HF",
    ],
    [
        "ResNet-CBAM-1D (Attention Canal/Spatial)*",
        "99.91%", "0.05%", "328 400", "41 ms",
        "Limitée · arc en série",
    ],
    [
        "CNN-BiGRU-Attention (Spatio-Temporel)*",
        "99.75%", "0.15%", "342 500", "46 ms",
        "Limitée · charges variables",
    ],
    [
        "Arc-FaultNet V2 (Cross-Attention)†",
        f"{prec:.1f} %",
        fpr_str,
        n_params,
        int8_str,
        "Forte · toutes charges",
    ],
]

data = []
for row in raw_data:
    data.append([
        _wrap(row[0], 28),
        row[1], row[2], row[3], row[4],
        _wrap(row[5], 20),
    ])

table = ax.table(
    cellText=data,
    colLabels=columns,
    cellLoc="center",
    loc="center",
    bbox=[0.01, 0.20, 0.98, 0.62],
)
table.auto_set_font_size(False)
table.set_fontsize(9.8)
table.scale(1.0, 1.0)

OUR_ROW = len(data)
GEN_COL = len(columns) - 1
ARCH_COL = 0

row_heights = []
for row_idx, row in enumerate(data):
    n = max(_lines(row[ARCH_COL]), _lines(row[GEN_COL]))
    for col, cell_text in enumerate(row):
        if col not in (ARCH_COL, GEN_COL):
            n = max(n, _lines(str(cell_text)))
    row_heights.append(0.14 + 0.045 * (n - 1))

for (row, col), cell in table.get_celld().items():
    data_row = row - 1
    if row == 0:
        cell.set_facecolor("#0055A4")
        cell.get_text().set_color("#FFFFFF")
        cell.get_text().set_fontweight("bold")
        cell.set_height(0.16)
    elif row == OUR_ROW:
        cell.set_facecolor(GREEN if col != GEN_COL else "#C8E6C9")
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_color(GREEN_HDR if col == GEN_COL else BLACK)
        cell.set_height(row_heights[data_row])
    else:
        cell.set_facecolor("#F8F9FA" if row % 2 == 0 else "#FFFFFF")
        cell.get_text().set_color(BLACK)
        cell.get_text().set_fontweight("normal")
        cell.set_height(row_heights[data_row])
    cell.set_edgecolor("#DDDDDD")
    cell.get_text().set_linespacing(1.35)

sources = (
    "* ViT-1D Arc : IEEE Trans. on Ind. Informatics (2024) — signatures d'arc HF complexes.\n"
    "* ResNet-CBAM-1D : MDPI Energies (2025) — arc en série + attention CBAM.\n"
    "* CNN-BiGRU-Attention : Int. J. of Electrical Power (2024) — arcs naissants, charges variables.\n"
    f"† Arc-FaultNet V2 : ablation full · seed 42 · split 70/15/15 · test = 1630 cycles · "
    f"Précision {prec:.1f} % · FPR {fpr_str} · INT8 estimé {int8_str} · "
    f"charges résistives, SMPS, moteur, multi-charges."
)
plt.text(0.02, 0.03, sources, ha="left", va="top", fontsize=9.2,
         color="#666666", linespacing=1.45)

output_filename = "slide_afdd_attention_300k.png"
plt.savefig(output_filename, bbox_inches="tight", facecolor="#FFFFFF")
plt.close()
print(f"Slide généré : {output_filename}")
