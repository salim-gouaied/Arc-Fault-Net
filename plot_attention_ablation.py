#!/usr/bin/env python3
"""
plot_attention_ablation.py — publication figures for the attention-ablation section.

Data sources (read-only):
    runs/arcfaultnet_v2_single_20260724_154038/eval/metrics.json  -> "all mechanisms"
    ablation_attention_results/results.json                       -> ablated variants

Figures produced in ablation_attention_results/figures/ (no titles, no captions —
add them via \\caption{} in LaTeX):
    fig1_radar.png        radar: all mechanisms / – cross-attention / no attention
    fig2_metrics.png      metric bars: all mechanisms vs no attention
    fig3_confusion.png    confusion matrices: all mechanisms vs no attention

Usage:
    ./venv/bin/python plot_attention_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ABLATION = ROOT / "ablation_attention_results" / "results.json"
REFERENCE = ROOT / "runs/arcfaultnet_v2_single_20260724_154038/eval/metrics.json"
OUT_DIR = ROOT / "ablation_attention_results" / "figures"

# ── Palette ────────────────────────────────────────────────────────────────
C_ATTN = "#1f4e9c"    # all mechanisms (reference run)
C_XATTN = "#c07a10"   # – sequential cross-attention
C_NONE = "#b3261e"    # no attention
GRID = "#d5d9e0"
INK = "#1a1d23"

L_ATTN = "All mechanisms"
L_XATTN = "– Sequential cross-attention"
L_NONE = "No attention"

METRICS = ["accuracy", "precision", "recall", "specificity", "f1"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score"]


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#8a9099",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load() -> tuple[dict, dict, dict]:
    variants = {r["variant"]: r for r in json.loads(ABLATION.read_text())}

    attn = json.loads(REFERENCE.read_text())
    # eval/metrics.json carries no specificity — derive it from the counts.
    attn["specificity"] = attn["tn"] / (attn["tn"] + attn["fp"])
    attn["n_params"] = 309_833

    return attn, variants["no_xattn"], variants["none"]


# ══════════════════════════════════════════════════════════════════════════
#  Figure 1 — radar
# ══════════════════════════════════════════════════════════════════════════

def fig_radar(series: list[tuple[dict, str, str]], path: Path) -> None:
    n = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.0, 6.2), subplot_kw={"polar": True})

    ax.set_ylim(90.0, 100.0)
    ax.set_yticks([92, 94, 96, 98, 100])
    ax.set_yticklabels(["92", "94", "96", "98", "100 %"], fontsize=8.5,
                       color="#6b7280")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, fontsize=11)
    ax.tick_params(axis="x", pad=14)
    ax.grid(color=GRID, linewidth=0.7)
    ax.spines["polar"].set_color(GRID)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for m, label, colour in series:
        vals = [100 * m[k] for k in METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, "-", color=colour, linewidth=2.3, label=label,
                marker="o", markersize=4.5, zorder=3)
        ax.fill(angles, vals, color=colour, alpha=0.09, zorder=2)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=1,
              frameon=False, fontsize=10.5, handlelength=1.8)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════
#  Figure 2 — metric-by-metric bars
# ══════════════════════════════════════════════════════════════════════════

def fig_metrics(attn: dict, none: dict, path: Path) -> None:
    x = np.arange(len(METRICS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    a = np.array([100 * attn[m] for m in METRICS])
    b = np.array([100 * none[m] for m in METRICS])

    ax.bar(x - width / 2, a, width, color=C_ATTN, label=L_ATTN, zorder=3)
    ax.bar(x + width / 2, b, width, color=C_NONE, label=L_NONE, zorder=3)

    for xi, v in zip(x - width / 2, a):
        ax.text(xi, v + 0.18, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C_ATTN)
    for xi, v in zip(x + width / 2, b):
        ax.text(xi, v + 0.18, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C_NONE)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=11)
    ax.set_ylim(90, 101)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_yticks([90, 92, 94, 96, 98, 100])
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.legend(frameon=False, fontsize=10.5, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════
#  Figure 3 — confusion matrices
# ══════════════════════════════════════════════════════════════════════════

def _draw_cm(ax, m: dict, title: str, colour: str) -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mono", ["#ffffff", colour])
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)

    for i in range(2):
        for j in range(2):
            strong = pct[i, j] > 55
            ax.text(j, i - 0.10, f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=19, fontweight="bold",
                    color="white" if strong else INK)
            ax.text(j, i + 0.22, f"{pct[i, j]:.2f} %", ha="center", va="center",
                    fontsize=9.5, color="#e8eaed" if strong else "#6b7280")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Arc"], fontsize=10.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Arc"], fontsize=10.5)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10, color=colour)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_color("#c3c8d0")


def fig_confusion(attn: dict, none: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    _draw_cm(axes[0], attn, L_ATTN, C_ATTN)
    _draw_cm(axes[1], none, L_NONE, C_NONE)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Drop figures from earlier naming schemes so the directory stays unambiguous.
    for stale in OUT_DIR.glob("*_vs_none.png"):
        stale.unlink()
    for stale in OUT_DIR.glob("fig1_radar_variants.png"):
        stale.unlink()

    attn, no_xattn, none = load()

    print("Figures written to", OUT_DIR.relative_to(ROOT))
    fig_radar([(attn, L_ATTN, C_ATTN),
               (no_xattn, L_XATTN, C_XATTN),
               (none, L_NONE, C_NONE)],
              OUT_DIR / "fig1_radar.png")
    fig_metrics(attn, none, OUT_DIR / "fig2_metrics.png")
    fig_confusion(attn, none, OUT_DIR / "fig3_confusion.png")

    print(f"\n  {'model':30s} {'acc':>6} {'prec':>6} {'rec':>6} "
          f"{'spec':>6} {'F1':>6}  {'FP':>3} {'FN':>3}   params")
    for m, label in ((attn, L_ATTN), (no_xattn, L_XATTN), (none, L_NONE)):
        print(f"  {label:30s} "
              f"{100*m['accuracy']:6.2f} {100*m['precision']:6.2f} "
              f"{100*m['recall']:6.2f} {100*m['specificity']:6.2f} "
              f"{100*m['f1']:6.2f}  {m['fp']:3d} {m['fn']:3d}   "
              f"{m['n_params']:,}")


if __name__ == "__main__":
    main()
