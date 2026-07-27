#!/usr/bin/env python3
"""
Ablation V2 chart — combined ablation study.
Data source (only):
    ablation_results/ablation_v2_20260612_175320/ablation_v2_results.json

Usage:
    ./venv/bin/python plot_ablation_contributions.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ABLATION = ROOT / "ablation_results/ablation_v2_20260612_175320/ablation_v2_results.json"
OUT_DIR = ROOT / "docs/presentation/diagrams"

FULL = "#4f46e5"
TEMPORAL = "#ea580c"
NO_ATTN = "#94a3b8"

METRICS = ("accuracy", "recall", "specificity")
LABELS = ("Accuracy", "Recall", "Specificity")


def load_variants() -> dict:
    with open(ABLATION) as f:
        data = json.load(f)
    out = {}
    for name, v in data["variants"].items():
        out[name] = {
            "accuracy": v["accuracy"] * 100,
            "recall": v["recall"] * 100,
            "specificity": v["specificity"] * 100,
        }
    return out


def main():
    if not ABLATION.is_file():
        raise FileNotFoundError(f"Missing ablation results: {ABLATION}")

    models = [
        ("arcfaultnet_v2", "ArcFaultNet V2 (full)", FULL),
        ("v2_temporal_only", "Temporal branch only (no STFT)", TEMPORAL),
        ("v2_baseline_cnn", "Without attention mechanisms", NO_ATTN),
    ]

    variants = load_variants()
    x = np.arange(len(LABELS))
    width = 0.24
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (key, label, color) in enumerate(models):
        vals = [variants[key][m] for m in METRICS]
        pos = x + offsets[i] * width
        bars = ax.bar(
            pos, vals, width, label=label, color=color,
            edgecolor="#475569" if color == NO_ATTN else "#312e81",
            linewidth=0.7,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.35,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8.5,
                color="#1e293b",
            )

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Ablation Study", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.55)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "15_ablation_study.png"
    plt.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
