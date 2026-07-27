#!/usr/bin/env python3
"""
Comparaison V1 vs V2 — Accuracy, Recall, Spécificité (moyennes ± écart-type).
4 seeds par architecture, runs single sur combined_dataset_2048 (V2) ou V1 final.

Usage:
    ./venv/bin/python plot_v1_v2_comparison.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "presentation" / "diagrams" / "14_comparaison_metrics_v1_v2.png"

# 4 seeds × V1 (arcfaultnet) — architecture Gabor + Joint Attention
V1_RUNS = [
    ROOT / "runs/arcfaultnet_single_20260526_120146/results.json",      # seed 2
    ROOT / "runs/arcfaultnet_single_20260526_120829/results.json",      # seed 3
    ROOT / "runs/arcfaultnet_single_20260603_115307/results.json",      # seed 4 · combined_dataset_2048
    ROOT / "runs/arcfaultnet_single_20260522_114209/results.json",      # seed 42
]

# 4 seeds × V2 (arcfaultnet_v2) — combined_dataset_2048 · batch 09–10/06
V2_RUNS = [
    ROOT / "runs/arcfaultnet_v2_single_20260610_124020/results.json",   # seed 2
    ROOT / "runs/arcfaultnet_v2_single_20260609_120534/results.json",    # seed 3
    ROOT / "runs/arcfaultnet_v2_single_20260610_130056/results.json",   # seed 4
    ROOT / "runs/arcfaultnet_v2_single_20260610_124344/results.json",   # seed 42
]

METRICS = ("accuracy", "recall", "specificity")
LABELS = ("Accuracy", "Recall", "Spécificité")
V1_COLOR, V2_COLOR = "#94a3b8", "#4f46e5"


def load_group(paths: list[Path]) -> tuple[dict[str, list[float]], list[int]]:
    values = {k: [] for k in METRICS}
    seeds = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        seeds.append(int(data["seed"]))
        for k in METRICS:
            values[k].append(float(data[f"test_{k}"]) * 100)
    return values, seeds


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def main():
    v1_vals, v1_seeds = load_group(V1_RUNS)
    v2_vals, v2_seeds = load_group(V2_RUNS)

    v1_mean, v1_std, v2_mean, v2_std = {}, {}, {}, {}
    for k in METRICS:
        v1_mean[k], v1_std[k] = mean_std(v1_vals[k])
        v2_mean[k], v2_std[k] = mean_std(v2_vals[k])

    x = np.arange(len(METRICS))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, [v1_mean[k] for k in METRICS], width,
           yerr=[v1_std[k] for k in METRICS], label="V1 — Gabor + Joint Attention",
           capsize=5, color=V1_COLOR, edgecolor="#64748b", linewidth=0.8)
    ax.bar(x + width / 2, [v2_mean[k] for k in METRICS], width,
           yerr=[v2_std[k] for k in METRICS], label="V2 — front-end physique + Cross-Attention",
           capsize=5, color=V2_COLOR, edgecolor="#3730a3", linewidth=0.8)

    ax.set_ylabel("Pourcentage (%)", fontsize=12)
    ax.set_title("V1 vs V2 — Accuracy, Recall et Spécificité (4 seeds)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(82, 101)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for i, k in enumerate(METRICS):
        ax.text(x[i] - width / 2, v1_mean[k] + v1_std[k] + 0.25,
                f"{v1_mean[k]:.1f}%", ha="center", va="bottom", fontsize=9, color="#475569")
        ax.text(x[i] + width / 2, v2_mean[k] + v2_std[k] + 0.25,
                f"{v2_mean[k]:.1f}%", ha="center", va="bottom", fontsize=9, color="#312e81")

    fig.text(
        0.5, 0.02,
        f"V1 seeds {v1_seeds}  ·  V2 seeds {v2_seeds}  ·  "
        "V2 sur combined_dataset_2048  ·  barres = moyenne ± σ (n=4)",
        ha="center", fontsize=9, style="italic", color="#64748b",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {OUT}")
    print("\nV1:")
    for k in METRICS:
        print(f"  {k:14s}  {v1_mean[k]:6.2f}% ± {v1_std[k]:.2f}")
    print("\nV2:")
    for k in METRICS:
        print(f"  {k:14s}  {v2_mean[k]:6.2f}% ± {v2_std[k]:.2f}")


if __name__ == "__main__":
    main()
