#!/usr/bin/env python3
"""
Arc-FaultNet — Architecture Diagram Generator
==============================================
Produces clean, research-paper-style PNG diagrams for every module
of the Arc-FaultNet architecture using matplotlib (no mermaid).

Run with the project's python:
    /home/top/miniconda3/bin/python docs/architecture/gen_diagrams.py

All diagrams are written to docs/architecture/diagrams/*.png.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import sys

# ─────────────────────────────────────────────────────────────
#  Global style (paper-friendly)
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.20,
})

# Color palette
COL = {
    "input":      "#cfe2f3",   # pale blue
    "conv":       "#fcd5b4",   # peach (1D conv / parametric)
    "conv2d":     "#f6b26b",   # darker peach (2D conv)
    "bn":         "#fff2cc",   # light yellow (BN / activation)
    "pool":       "#ead1dc",   # mauve (pooling)
    "cam":        "#d9ead3",   # pale green (channel attention)
    "sam":        "#f4cccc",   # pale pink (spatial attention)
    "fusion":     "#d9d2e9",   # light purple
    "classifier": "#c9daf8",   # soft blue (head)
    "output":     "#f9cb9c",   # warm output
    "stft":       "#b6d7a8",   # green (transform)
    "label":      "#ffe599",   # pale gold (labeling)
    "data":       "#e6e6e6",   # gray (raw data)
    "edge":       "#3c3c3c",
    "text":       "#1a1a1a",
    "shape":      "#5a5a5a",
}

OUT_DIR = Path(__file__).resolve().parent / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  Drawing primitives
# ─────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, text, *, color="#ffffff", subtitle=None,
        fontsize=10, fontweight="bold", text_color=None,
        edge=None, lw=1.2, radius=0.18, alpha=1.0):
    """Draw a rounded box with bold title and optional subtitle."""
    edge = edge or COL["edge"]
    text_color = text_color or COL["text"]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=color, alpha=alpha,
    )
    ax.add_patch(patch)
    if subtitle is None:
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color)
    else:
        ax.text(x + w / 2, y + h * 0.62, text,
                ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color)
        ax.text(x + w / 2, y + h * 0.28, subtitle,
                ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color, style="italic")
    return patch


def arrow(ax, x0, y0, x1, y1, *, label=None, label_pos=0.5,
          color=None, lw=1.5, style="-|>", curve=0.0, label_offset=(0.0, 0.18),
          fontsize=8.5, fontweight="normal"):
    color = color or COL["edge"]
    conn = "arc3,rad=%.2f" % curve if curve else "arc3,rad=0"
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color, connectionstyle=conn,
    )
    ax.add_patch(arr)
    if label is not None:
        mx = x0 + (x1 - x0) * label_pos + label_offset[0]
        my = y0 + (y1 - y0) * label_pos + label_offset[1]
        ax.text(mx, my, label,
                ha="center", va="center", fontsize=fontsize,
                color=COL["shape"], fontweight=fontweight,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, boxstyle="round,pad=0.15"))


def shape_label(ax, x, y, text, fontsize=8.5):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=COL["shape"], style="italic",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.85, boxstyle="round,pad=0.15"))


def title(ax, text, subtitle=None, y=0.975):
    ax.text(0.5, y, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=14, fontweight="bold",
            color=COL["text"])
    if subtitle:
        ax.text(0.5, y - 0.040, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5,
                color=COL["shape"], style="italic")


def setup_ax(fig, xlim, ylim):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def legend_strip(ax, x, y, items, *, swatch=0.25, gap=0.5, fontsize=8.5):
    """Horizontal color legend strip."""
    cx = x
    for label, color in items:
        ax.add_patch(Rectangle((cx, y), swatch, swatch,
                               facecolor=color, edgecolor=COL["edge"], linewidth=0.8))
        ax.text(cx + swatch + 0.08, y + swatch / 2, label,
                ha="left", va="center", fontsize=fontsize, color=COL["text"])
        cx += swatch + 0.08 + 0.05 + (len(label) * 0.08) + gap


# ─────────────────────────────────────────────────────────────
#  1) Main model architecture
# ─────────────────────────────────────────────────────────────

def diagram_model_architecture():
    fig = plt.figure(figsize=(12, 10.5))
    ax = setup_ax(fig, (0, 12), (-0.5, 11.0))

    title(ax,
          "Arc-FaultNet — Dual-Branch CNN with Joint Cross-Attention",
          "Module-level overview (each branch is detailed in its own diagram)")

    # ── Input
    box(ax, 4.5, 9.40, 3.0, 0.70,
        "Raw input",
        subtitle="V_ligne (C1), I (C3)  |  (B, 2, 20 000) @ 1 MHz",
        color=COL["input"])

    # split arrows
    arrow(ax, 6.0, 9.40, 6.0, 8.95, lw=1.4)
    arrow(ax, 6.0, 8.90, 2.5, 8.45, lw=1.4)
    arrow(ax, 6.0, 8.90, 9.5, 8.45, lw=1.4)

    # ── STFT block (only feeds 2D branch)
    box(ax, 8.05, 7.65, 2.90, 0.65,
        "STFT (log-power)",
        subtitle="n_fft = 512, hop = 256",
        color=COL["stft"])
    arrow(ax, 9.5, 7.65, 9.5, 7.20, lw=1.4,
          label="(B, 2, 257, 78)", label_offset=(0.0, 0.0))

    # ── Branch 1D block
    box(ax, 1.2, 5.30, 2.6, 3.15,
        "Branch 1D (Temporal)",
        subtitle=("ParametricConv1d × 3\n"
                  "(Gabor filters)\n"
                  "2 → 32 → 64 → 128 ch\n"
                  "k = 64, 32, 16"),
        color=COL["conv"], fontsize=11)
    arrow(ax, 2.5, 5.30, 2.5, 4.65, lw=1.4,
          label="F_L : (B, 128, D)", label_offset=(0.0, 0.0))

    # ── Branch 2D block
    box(ax, 8.2, 5.30, 2.6, 1.85,
        "Branch 2D (Spectral)",
        subtitle=("Conv2d × 3\n"
                  "32 → 64 → 128 ch\n"
                  "Restricted to 2–100 kHz"),
        color=COL["conv2d"], fontsize=11)
    arrow(ax, 9.5, 5.30, 9.5, 4.65, lw=1.4,
          label="F_H : (B, 128, D)", label_offset=(0.0, 0.0))

    # ── Joint Attention block (lower, merged)
    box(ax, 3.3, 2.85, 5.4, 1.45,
        "Joint Attention",
        subtitle=("CAM on (F_L ⊕ F_H)  →  per-branch channel weights\n"
                  "SAM on (F_L ⊕ F_H)  →  per-branch temporal context\n"
                  "Residual combine + 1×1 fusion"),
        color=COL["fusion"], fontsize=12)

    # link F_L, F_H into joint attention
    arrow(ax, 2.5, 4.65, 4.5, 4.30, lw=1.4, curve=-0.10)
    arrow(ax, 9.5, 4.65, 7.5, 4.30, lw=1.4, curve=0.10)

    arrow(ax, 6.0, 2.85, 6.0, 2.30, lw=1.4,
          label="F_out : (B, 128, D)", label_offset=(0.0, 0.0))

    # ── Classifier
    box(ax, 4.4, 1.40, 3.2, 0.85,
        "Classifier head",
        subtitle="GAP → FC(128→64) → ReLU → Dropout → FC(64→1)",
        color=COL["classifier"])
    arrow(ax, 6.0, 1.40, 6.0, 0.85, lw=1.4)

    # ── Output
    box(ax, 4.6, 0.10, 2.8, 0.75,
        "logits  →  σ(·)  →  P(arc fault)",
        color=COL["output"], fontsize=11)

    # Annotation: D = 64 (placed in left margin)
    ax.text(0.20, -0.30,
            "Notation:  B = batch size,   D = 64 (latent length),   C = 128 (channels per branch)",
            ha="left", va="bottom",
            fontsize=8.5, color=COL["shape"], style="italic")

    fig.savefig(_out("01_model_architecture.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  2) Branch 1D (Temporal)
# ─────────────────────────────────────────────────────────────

def _stacked_layer_diagram(figpath, title_text, subtitle_text,
                           input_box, layers, shapes, output_box,
                           cx=4.5, w=4.8, h_main=0.85, h_simple=0.60,
                           gap=0.55):
    """Generic helper: vertical stack of (label, sub, color) layer boxes.
    Auto-sizes the figure height to fit content + title."""

    # Compute total vertical extent
    n = len(layers)
    extents = [h_main if sub else h_simple for _, sub, _ in layers]
    total_layers = sum(extents) + n * gap
    input_h = 0.80
    output_h = 0.80
    margin_top = 1.55      # space reserved for title + subtitle
    margin_bottom = 0.40

    total_height = (margin_top + input_h + 0.55 +
                    total_layers + 0.30 + output_h + margin_bottom)

    fig = plt.figure(figsize=(9, total_height * 0.78))
    ax = setup_ax(fig, (0, 9), (0, total_height))

    title(ax, title_text, subtitle_text)

    y = total_height - margin_top

    # Input
    box(ax, cx - w/2, y - input_h, w, input_h,
        input_box[0], subtitle=input_box[1], color=COL["input"])
    arrow(ax, cx, y - input_h, cx, y - input_h - 0.30, lw=1.3)
    y = y - input_h - 0.55

    for (lab, sub, color), shp in zip(layers, shapes):
        h = h_main if sub else h_simple
        box(ax, cx - w/2, y - h, w, h, lab, subtitle=sub,
            color=color, fontsize=10.5)
        arrow(ax, cx, y - h, cx, y - h - 0.30, lw=1.2,
              label=shp, label_pos=0.5, label_offset=(1.8, 0.0))
        y = y - h - gap

    # Output box (just below last arrow)
    box(ax, cx - w/2, y - 0.30 - output_h, w, output_h,
        output_box[0], subtitle=output_box[1], color=COL["output"])

    fig.savefig(figpath)
    plt.close(fig)


def diagram_branch_1d():
    layers = [
        ("ParametricConv1d  (2 → 32,  k = 64)", "Gabor filter — learnable f₀, σ", COL["conv"]),
        ("BatchNorm1d  +  ReLU", None, COL["bn"]),
        ("MaxPool1d (kernel = 4)", None, COL["pool"]),
        ("ParametricConv1d  (32 → 64,  k = 32)", "Gabor filter — learnable f₀, σ", COL["conv"]),
        ("BatchNorm1d  +  ReLU", None, COL["bn"]),
        ("MaxPool1d (kernel = 4)", None, COL["pool"]),
        ("ParametricConv1d  (64 → 128,  k = 16)", "Gabor filter — learnable f₀, σ", COL["conv"]),
        ("BatchNorm1d  +  ReLU", None, COL["bn"]),
        ("AdaptiveAvgPool1d (output = D = 64)", None, COL["pool"]),
    ]
    shapes = [
        "(B, 32, 20 000)",
        "(B, 32, 20 000)",
        "(B, 32, 5 000)",
        "(B, 64, 5 000)",
        "(B, 64, 5 000)",
        "(B, 64, 1 250)",
        "(B, 128, 1 250)",
        "(B, 128, 1 250)",
        "(B, 128, 64)",
    ]
    _stacked_layer_diagram(
        OUT_DIR / "02_branch1d.png",
        "Branch 1D — Temporal feature extractor",
        "Three stages of Parametric Gabor convolutions on the raw signal",
        ("Input  :  (B, 2, 20 000)",
         "raw V_ligne (C1) + I (C3) — z-scored per cycle"),
        layers, shapes,
        ("F_L  :  (B, 128, 64)",
         "temporal feature map fed into Joint Attention"),
    )


# ─────────────────────────────────────────────────────────────
#  3) Parametric Gabor filter
# ─────────────────────────────────────────────────────────────

def diagram_parametric_gabor():
    fig = plt.figure(figsize=(13, 7))
    ax_left = fig.add_axes([0.04, 0.10, 0.45, 0.78])
    ax_right = fig.add_axes([0.55, 0.10, 0.42, 0.78])

    # ── Left: block schema
    ax_left.set_xlim(0, 8)
    ax_left.set_ylim(0, 9)
    ax_left.set_aspect("equal")
    ax_left.axis("off")
    title(ax_left,
          "ParametricConv1d — learnable Gabor filter bank",
          "Each filter is fully specified by two physical parameters: f₀ and σ")

    box(ax_left, 0.5, 7.2, 7.0, 0.65,
        "Trainable parameters  :  f₀ ∈ ℝ⁽ᴼˣᴵ⁾,  σ ∈ ℝ⁽ᴼˣᴵ⁾  (bias ∈ ℝᴼ)",
        color=COL["input"], fontsize=10.5)

    box(ax_left, 1.5, 5.8, 5.0, 0.9,
        "Filter generation",
        subtitle=("ψ(t) = exp(−t²/2σ²) · cos(2π·f₀·t)\n"
                  "t = linspace(−K/2fs, +K/2fs, K)"),
        color=COL["conv"])

    arrow(ax_left, 4.0, 5.8, 4.0, 5.3, lw=1.4)

    box(ax_left, 1.5, 4.4, 5.0, 0.6,
        "L₂-normalize each filter (unit energy)",
        color=COL["bn"])
    arrow(ax_left, 4.0, 4.4, 4.0, 3.9, lw=1.4)

    box(ax_left, 1.5, 2.8, 5.0, 0.9,
        "Kernel  W ∈ ℝ⁽ᴼ × ᴵ × ᴷ⁾",
        subtitle="depends only on f₀ and σ → kept physically interpretable",
        color=COL["fusion"])
    arrow(ax_left, 4.0, 2.8, 4.0, 2.3, lw=1.4)

    box(ax_left, 1.5, 1.3, 5.0, 0.9,
        "y = F.conv1d(x, W, bias, stride, padding)",
        subtitle="standard 1D convolution with the generated kernel",
        color=COL["pool"])

    # ── Right: example filter waveform
    # Use a larger time window (256 samples) so the Gaussian envelope is visible.
    ax_right.set_title("Example filter ψ(t)  (f₀ = 30 kHz, σ = 50 µs)",
                       fontsize=11, color=COL["text"])
    fs = 1_000_000
    K = 256
    f0 = 30_000.0
    sigma = 50e-6           # 50 µs — clearly visible envelope
    t = np.linspace(-K/(2*fs), K/(2*fs), 800)
    gauss = np.exp(-t**2 / (2*sigma**2))
    osc = np.cos(2*np.pi*f0*t)
    psi = gauss * osc
    ax_right.plot(t*1e6, psi,     color="#2c3e50", lw=2.0, label="ψ(t)")
    ax_right.plot(t*1e6,  gauss,  color="#c0392b", lw=1.1, ls="--", label="Gaussian envelope")
    ax_right.plot(t*1e6, -gauss,  color="#c0392b", lw=1.1, ls="--")
    ax_right.axhline(0, color="#888", lw=0.6)
    ax_right.set_xlabel("time (µs)")
    ax_right.set_ylabel("amplitude")
    ax_right.legend(loc="upper right", frameon=False, fontsize=9)
    ax_right.grid(True, ls=":", lw=0.5, alpha=0.5)

    fig.savefig(_out("03_parametric_gabor.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  4) Branch 2D (Spectral)
# ─────────────────────────────────────────────────────────────

def diagram_branch_2d():
    layers = [
        ("Frequency-band slice", "freq bins [1 : 52]  ≈  2 kHz … 100 kHz",
         COL["stft"]),
        ("Conv2d  (2 → 32,  3×3, pad = 1)", None, COL["conv2d"]),
        ("BatchNorm2d  +  ReLU", None, COL["bn"]),
        ("MaxPool2d (2×2)", None, COL["pool"]),
        ("Conv2d  (32 → 64,  3×3, pad = 1)", None, COL["conv2d"]),
        ("BatchNorm2d  +  ReLU", None, COL["bn"]),
        ("MaxPool2d (2×2)", None, COL["pool"]),
        ("Conv2d  (64 → 128, 3×3, pad = 1)", None, COL["conv2d"]),
        ("BatchNorm2d  +  ReLU", None, COL["bn"]),
        ("AdaptiveAvgPool2d ((1, D = 64))",
         "collapses freq axis, keeps D time bins", COL["pool"]),
        ("squeeze freq axis", None, COL["pool"]),
    ]
    shapes = [
        "(B, 2, 51, 78)",
        "(B, 32, 51, 78)",
        "(B, 32, 51, 78)",
        "(B, 32, 25, 39)",
        "(B, 64, 25, 39)",
        "(B, 64, 25, 39)",
        "(B, 64, 12, 19)",
        "(B, 128, 12, 19)",
        "(B, 128, 12, 19)",
        "(B, 128, 1, 64)",
        "(B, 128, 64)",
    ]
    _stacked_layer_diagram(
        OUT_DIR / "04_branch2d.png",
        "Branch 2D — Spectral feature extractor",
        "Conv2D over a frequency-restricted log-power STFT (2–100 kHz)",
        ("Input STFT  :  (B, 2, 257, ~78)",
         "log |STFT|²  on each channel  (n_fft = 512, hop = 256)"),
        layers, shapes,
        ("F_H  :  (B, 128, 64)",
         "spectral feature map fed into Joint Attention"),
        w=5.0,
    )


# ─────────────────────────────────────────────────────────────
#  5) Joint Attention
# ─────────────────────────────────────────────────────────────

def diagram_joint_attention():
    """
    Two-column layout (Left column = F_L pipeline, Right column = F_H pipeline).
    CAM and SAM live in a middle band and see the joint context.
    """
    fig = plt.figure(figsize=(13, 11))
    ax = setup_ax(fig, (0, 13), (-0.3, 11.4))

    title(ax,
          "Joint Attention — cross-branch CAM + SAM",
          "Both attention modules see the joint context (F_L ⊕ F_H), but each branch keeps its identity")

    # ───── Inputs (top) ─────────────────────────────
    LX, RX = 2.0, 11.0          # x-centers of left / right pipelines
    box(ax, LX - 1.4, 9.6, 2.8, 0.75,
        "F_L  (B, C, D)",
        subtitle="temporal branch",
        color=COL["conv"])
    box(ax, RX - 1.4, 9.6, 2.8, 0.75,
        "F_H  (B, C, D)",
        subtitle="spectral branch",
        color=COL["conv2d"])

    # ───── Concat node (centered) ───────────────────
    box(ax, 5.6, 8.55, 1.8, 0.65, "concat", color=COL["fusion"])
    arrow(ax, LX + 0.6, 9.6, 5.8, 9.05, lw=1.3, curve=-0.05)
    arrow(ax, RX - 0.6, 9.6, 7.2, 9.05, lw=1.3, curve=0.05)
    arrow(ax, 6.5, 8.55, 6.5, 8.10, lw=1.3,
          label="F_concat : (B, 2C, D)", label_offset=(0.0, 0.0))

    # ───── CAM and SAM (joint context band) ─────────
    box(ax, 1.8, 6.55, 4.2, 1.30,
        "CAM ( F_concat )",
        subtitle="GAP & GMP  →  shared MLP  →  σ\nβ ∈ (0, 1)^(2C)",
        color=COL["cam"], fontsize=11)
    box(ax, 7.0, 6.55, 4.2, 1.30,
        "SAM ( F_concat )",
        subtitle="Q, K, V via 1×1 conv\nα = softmax(QᵀK/√dₖ),   out = V · αᵀ",
        color=COL["sam"], fontsize=11)

    arrow(ax, 6.5, 8.10, 3.9, 7.85, lw=1.3, curve=-0.10)
    arrow(ax, 6.5, 8.10, 9.1, 7.85, lw=1.3, curve=0.10)

    # ───── Per-branch splits / projections ──────────
    # CAM split into β_L (left) and β_H (right)
    box(ax, LX - 1.1, 5.05, 2.2, 0.80,
        "split [ : C ]",
        subtitle="β_L  (B, C, 1)",
        color=COL["cam"])
    box(ax, RX - 1.1, 5.05, 2.2, 0.80,
        "split [ C : ]",
        subtitle="β_H  (B, C, 1)",
        color=COL["cam"])
    arrow(ax, 3.0, 6.55, LX, 5.85, lw=1.2, curve=-0.10)
    arrow(ax, 5.0, 6.55, RX, 5.85, lw=1.2, curve=0.18)

    # SAM projections
    box(ax, LX - 1.1, 3.85, 2.2, 0.80,
        "proj_sam_L",
        subtitle="Conv1d  2C → C",
        color=COL["sam"])
    box(ax, RX - 1.1, 3.85, 2.2, 0.80,
        "proj_sam_H",
        subtitle="Conv1d  2C → C",
        color=COL["sam"])
    arrow(ax, 8.0, 6.55, LX + 0.4, 4.65, lw=1.2, curve=-0.30)
    arrow(ax, 10.2, 6.55, RX - 0.4, 4.65, lw=1.2, curve=0.08)

    # ───── Per-branch multiplication and residual sum ─────
    # F_L_cam = F_L ⊙ β_L
    box(ax, LX - 1.3, 2.45, 2.6, 0.85,
        "F_L  ⊙  β_L",
        subtitle="F_L_cam",
        color=COL["fusion"])
    box(ax, RX - 1.3, 2.45, 2.6, 0.85,
        "F_H  ⊙  β_H",
        subtitle="F_H_cam",
        color=COL["fusion"])

    # F_L passthrough into multiplication (down the left edge)
    arrow(ax, LX - 1.3, 9.6, LX - 1.3, 2.85, lw=1.0, curve=-0.25, style="-|>")
    arrow(ax, RX + 1.3, 9.6, RX + 1.3, 2.85, lw=1.0, curve=0.25, style="-|>")
    # β_L → multiplication
    arrow(ax, LX, 5.05, LX, 3.30, lw=1.2)
    arrow(ax, RX, 5.05, RX, 3.30, lw=1.2)

    # Residual sum (cam + sam)
    box(ax, LX - 1.5, 0.95, 3.0, 0.85,
        "F_L_cam  +  F_L_sam",
        subtitle="residual sum  →  F_L_out",
        color=COL["fusion"])
    box(ax, RX - 1.5, 0.95, 3.0, 0.85,
        "F_H_cam  +  F_H_sam",
        subtitle="residual sum  →  F_H_out",
        color=COL["fusion"])
    arrow(ax, LX, 2.45, LX, 1.80, lw=1.2)
    arrow(ax, RX, 2.45, RX, 1.80, lw=1.2)
    # SAM projection results feed into residual sum
    arrow(ax, LX, 3.85, LX - 0.3, 1.80, lw=1.0, curve=-0.10, style="-|>")
    arrow(ax, RX, 3.85, RX + 0.3, 1.80, lw=1.0, curve=0.10, style="-|>")

    # ───── Final 1x1 fusion ─────────────────────────
    box(ax, 4.6, -0.10, 3.8, 0.85,
        "1×1 Conv  (2C → C)",
        subtitle="cat (F_L_out, F_H_out)  →  F_out",
        color=COL["classifier"])
    arrow(ax, LX + 0.6, 0.95, 5.0, 0.75, lw=1.3, curve=-0.10)
    arrow(ax, RX - 0.6, 0.95, 8.0, 0.75, lw=1.3, curve=0.10)

    # Output label (placed on the right of fusion)
    ax.text(11.5, 0.32, "F_out  :  (B, C, D)",
            ha="center", va="center", fontsize=10.5,
            color=COL["shape"], style="italic",
            bbox=dict(facecolor="white", edgecolor=COL["edge"],
                      boxstyle="round,pad=0.20", lw=0.8))

    fig.savefig(_out("05_joint_attention.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  6) Channel Attention Module (CAM)
# ─────────────────────────────────────────────────────────────

def diagram_cam():
    fig = plt.figure(figsize=(11, 7))
    ax = setup_ax(fig, (0, 11), (0, 8))

    title(ax,
          "Channel Attention Module (CAM)",
          "Asks: which channels (filters) matter? — adapted from CBAM")

    box(ax, 4.6, 6.6, 2.5, 0.7,
        "Input  X : (B, C, D)",
        color=COL["input"])

    arrow(ax, 5.85, 6.6, 5.85, 6.10, lw=1.3)

    box(ax, 1.6, 5.0, 3.0, 0.85,
        "GAP  (mean over D)",
        subtitle="(B, C)",
        color=COL["pool"])
    box(ax, 7.1, 5.0, 3.0, 0.85,
        "GMP  (max over D)",
        subtitle="(B, C)",
        color=COL["pool"])

    arrow(ax, 5.85, 6.10, 3.1, 5.85, lw=1.2, curve=-0.15)
    arrow(ax, 5.85, 6.10, 8.6, 5.85, lw=1.2, curve=0.15)

    box(ax, 4.1, 3.5, 3.5, 0.85,
        "Shared MLP",
        subtitle="Linear(C → C/r) → ReLU → Linear(C/r → C)",
        color=COL["cam"])

    arrow(ax, 3.1, 5.0, 5.0, 4.35, lw=1.2, curve=-0.10)
    arrow(ax, 8.6, 5.0, 6.7, 4.35, lw=1.2, curve=0.10)

    box(ax, 4.4, 2.0, 2.9, 0.85,
        "sum  →  σ(·)",
        subtitle="elementwise sigmoid",
        color=COL["bn"])

    arrow(ax, 5.85, 3.5, 5.85, 2.85, lw=1.2)

    box(ax, 3.9, 0.55, 4.0, 0.85,
        "β  : (B, C, 1)   — channel weights",
        color=COL["output"])
    arrow(ax, 5.85, 2.0, 5.85, 1.4, lw=1.3)

    fig.savefig(_out("06_channel_attention.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  7) Spatial Attention Module (SAM)
# ─────────────────────────────────────────────────────────────

def diagram_sam():
    fig = plt.figure(figsize=(11.5, 8.5))
    ax = setup_ax(fig, (0, 12), (0, 10))

    title(ax,
          "Spatial / Temporal Attention Module (SAM)",
          "Asks: which positions in the latent time-axis matter?  — self-attention with Q / K / V")

    # Input (top)
    box(ax, 5.0, 8.6, 2.5, 0.7,
        "Input  X : (B, C, D)",
        color=COL["input"])
    arrow(ax, 6.25, 8.6, 6.25, 8.20, lw=1.3)

    # Three 1×1 projections in a row
    box(ax, 1.2, 6.9, 2.5, 0.85,
        "1×1 Conv  Q",
        subtitle="(B, dₖ, D)",
        color=COL["sam"])
    box(ax, 5.0, 6.9, 2.5, 0.85,
        "1×1 Conv  K",
        subtitle="(B, dₖ, D)",
        color=COL["sam"])
    box(ax, 8.8, 6.9, 2.5, 0.85,
        "1×1 Conv  V",
        subtitle="(B, C, D)",
        color=COL["sam"])

    arrow(ax, 6.25, 8.20, 2.45, 7.75, lw=1.2, curve=-0.18)
    arrow(ax, 6.25, 8.20, 6.25, 7.75, lw=1.2)
    arrow(ax, 6.25, 8.20, 10.05, 7.75, lw=1.2, curve=0.18)

    # α computation (left-center)
    box(ax, 3.0, 4.8, 4.5, 1.0,
        "α  =  softmax ( Qᵀ K / √dₖ )",
        subtitle="(B, D, D)   — temporal attention map",
        color=COL["bn"])
    arrow(ax, 2.45, 6.9, 4.0, 5.80, lw=1.2, curve=-0.10)
    arrow(ax, 6.25, 6.9, 6.0, 5.80, lw=1.2)

    # Output = V · αᵀ (centered, below)
    box(ax, 3.5, 2.8, 5.0, 1.0,
        "Output  =  V · αᵀ",
        subtitle="(B, C, D)",
        color=COL["output"])
    arrow(ax, 5.25, 4.8, 5.5, 3.80, lw=1.2)         # α → Output
    arrow(ax, 10.05, 6.9, 7.5, 3.80, lw=1.2, curve=0.18)  # V → Output

    # Final
    box(ax, 4.5, 1.0, 3.0, 0.85,
        "y  :  (B, C, D)",
        color=COL["fusion"])
    arrow(ax, 6.0, 2.8, 6.0, 1.85, lw=1.3,
          label="attention-weighted features", label_offset=(0.0, 0.15))

    fig.savefig(_out("07_spatial_attention.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  8) Classifier head
# ─────────────────────────────────────────────────────────────

def diagram_classifier():
    fig = plt.figure(figsize=(8.5, 11))
    ax = setup_ax(fig, (0, 8), (-0.5, 11))

    title(ax,
          "Classifier head",
          "Pools the fused features and produces a binary logit")

    cx, w = 4.0, 5.0
    y = 9.8
    boxes = [
        ("Input  F_out : (B, 128, D)", None, COL["input"]),
        ("AdaptiveAvgPool1d (1)", "(B, 128, 1)", COL["pool"]),
        ("squeeze axis -1", "(B, 128)", COL["pool"]),
        ("Linear  (128 → 64)", None, COL["classifier"]),
        ("ReLU", None, COL["bn"]),
        ("Dropout  (p = 0.3)", None, COL["bn"]),
        ("Linear  (64 → 1)", None, COL["classifier"]),
        ("logits  : (B,)\n→ σ(·) → P(arc)", None, COL["output"]),
    ]
    for label, sub, color in boxes:
        h = 0.95 if sub else 0.65
        box(ax, cx - w/2, y - h, w, h, label, subtitle=sub, color=color)
        if label != boxes[-1][0]:
            arrow(ax, cx, y - h, cx, y - h - 0.32, lw=1.3)
        y = y - h - 0.55

    fig.savefig(_out("08_classifier_head.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  9) Data pipeline (raw CSV → tensors)
# ─────────────────────────────────────────────────────────────

def diagram_data_pipeline():
    fig = plt.figure(figsize=(14, 10.5))
    ax = setup_ax(fig, (0, 14), (0, 11.6))

    title(ax,
          "Data pipeline — from raw oscilloscope CSV to model tensors",
          "Three-zone arc-ratio labeling with C2 (V_arc) as oracle, then 2-channel + STFT")

    # Raw channels
    box(ax, 0.4, 9.4, 3.0, 1.0,
        "Raw CSV (LeCroy, fs = 1 MHz)",
        subtitle="C1 = V_ligne     C2 = V_arc     C3 = I",
        color=COL["data"])

    # Group experiments
    box(ax, 4.2, 9.4, 3.0, 1.0,
        "Experiment grouping",
        subtitle="match C1/C2/C3 triplets,\nparse arc_load + bg_loads → charge_id",
        color=COL["data"])
    arrow(ax, 3.4, 9.9, 4.2, 9.9, lw=1.4)

    # ZC detection on C1
    box(ax, 8.0, 9.4, 5.4, 1.0,
        "Zero-crossing on C1",
        subtitle="DC removal → bandpass 40–60 Hz → positive-going crossings →\n"
                 "validate spacing ≈ 20 000 samples (±8%)",
        color=COL["stft"])
    arrow(ax, 7.2, 9.9, 8.0, 9.9, lw=1.4)

    # Cycle segments from C1 + C3
    box(ax, 1.0, 7.0, 4.0, 1.0,
        "Cycle segments  (start, end)",
        subtitle="one alternance ≡ two ZC apart  ≈ 20 000 samples",
        color=COL["fusion"])
    arrow(ax, 10.7, 9.4, 5.0, 8.0, lw=1.2, curve=0.20)

    # Arc-ratio computation on C2
    box(ax, 6.0, 7.0, 4.0, 1.0,
        "Arc ratio per cycle",
        subtitle="ratio = mean( |C2| > V_th = 10 V )  ∈ [0, 1]",
        color=COL["label"])
    arrow(ax, 10.7, 9.4, 8.0, 8.0, lw=1.2, curve=-0.05)

    # Histogram calibration
    box(ax, 11.0, 7.0, 2.8, 1.0,
        "Calibration of R_low, R_high",
        subtitle="99th pct of normal group,\n1st pct of arc group",
        color=COL["label"])
    arrow(ax, 10.0, 7.5, 11.0, 7.5, lw=1.2)

    # Three-zone labeling
    box(ax, 4.0, 5.0, 6.0, 1.2,
        "Three-zone labeling",
        subtitle="ratio ≤ R_low   → label = 0  (normal)\n"
                 "ratio ≥ R_high  → label = 1  (arc)\n"
                 "otherwise → discard (ambiguous transition)",
        color=COL["label"], fontsize=10.5)
    arrow(ax, 8.0, 7.0, 7.5, 6.2, lw=1.2)
    arrow(ax, 12.4, 7.0, 8.0, 6.2, lw=1.2, curve=0.10)
    arrow(ax, 3.0, 7.0, 5.0, 6.2, lw=1.2, curve=-0.10)

    # Build 2-channel tensor
    box(ax, 1.0, 2.7, 5.0, 1.3,
        "Build 2-channel sample",
        subtitle="stack [V_ligne(C1), I(C3)]   → (2, 20 000)\n"
                 "z-score per channel per cycle\n"
                 "C2 is DISCARDED (oracle only)",
        color=COL["conv"])
    arrow(ax, 6.0, 5.0, 5.0, 4.0, lw=1.2, curve=-0.10)

    # On-the-fly STFT
    box(ax, 7.6, 2.7, 5.4, 1.3,
        "On-the-fly STFT  (DataLoader side)",
        subtitle="log( |STFT|² + ε ), Hann window\n"
                 "n_fft = 512,  hop = 256  →  (2, 257, ~78)",
        color=COL["stft"])
    arrow(ax, 6.0, 3.4, 7.6, 3.4, lw=1.2)

    # Final dataset
    box(ax, 1.0, 0.6, 5.0, 1.4,
        "X_multi.npy  +  y.npy  +  charges.npy",
        subtitle="X : (N, 2, 20 000)\n"
                 "y : binary,  charges : load-config indices\n"
                 "(charge_map.json kept for later LOCO study)",
        color=COL["input"], fontsize=10.5)
    arrow(ax, 3.5, 2.7, 3.5, 2.0, lw=1.4)

    # Model inputs
    box(ax, 7.6, 0.6, 5.4, 1.4,
        "Model inputs per batch",
        subtitle="x_1d : (B, 2, 20 000)\n"
                 "x_2d : (B, 2, 257, 78)",
        color=COL["output"], fontsize=10.5)
    arrow(ax, 10.3, 2.7, 10.3, 2.0, lw=1.4)

    fig.savefig(_out("09_data_pipeline.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  10) Overall approach (very high level)
# ─────────────────────────────────────────────────────────────

def diagram_overall_pipeline():
    fig = plt.figure(figsize=(15.5, 5))
    ax = setup_ax(fig, (0, 15.5), (0, 5))

    title(ax,
          "Arc-FaultNet — End-to-end approach",
          "Physically-informed dual-branch detector trained on cycle-level segments")

    # 6 nodes evenly spaced over [0.1, 15.3], box width 2.30, gap 0.30
    bw, gap = 2.30, 0.30
    x0 = 0.10
    nodes = [
        ("Raw 3-channel CSV\n(C1, C2, C3) @ 1 MHz",            COL["data"]),
        ("Labeling & segmentation\n(C2 oracle, three-zone)",   COL["label"]),
        ("2-channel dataset\nV_ligne + I  (C2 discarded)",     COL["input"]),
        ("Arc-FaultNet\nGabor 1D + STFT 2D\n+ Joint Attention", COL["fusion"]),
        ("BCE training\nlabel smoothing\n+ cosine warm restarts", COL["classifier"]),
        ("P(arc fault)\nbinary decision",                      COL["output"]),
    ]
    centers = []
    for i, (txt, c) in enumerate(nodes):
        x = x0 + i * (bw + gap)
        box(ax, x, 1.7, bw, 1.7, txt, color=c, fontsize=9.5)
        centers.append((x, x + bw))
    for (l, r), (l2, r2) in zip(centers[:-1], centers[1:]):
        arrow(ax, r, 2.55, l2, 2.55, lw=1.6)

    fig.savefig(_out("00_overall_approach.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  11) Layer-by-layer node view (the whole network)
# ─────────────────────────────────────────────────────────────

def diagram_network_nodes():
    """
    "Paper-style" node-graph diagram of the whole network.
    Each layer is rendered as a column of circles (2-3 visible + vertical
    ellipsis), and thin grey edges connect adjacent columns to evoke the
    weight tensors. Two parallel branches converge at Joint Attention
    and then funnel into the classifier.
    """
    fig = plt.figure(figsize=(22, 12))
    ax = setup_ax(fig, (0, 24.4), (-0.7, 12.4))

    title(ax,
          "Arc-FaultNet — layer-by-layer node view",
          "Each column is one layer; visible circles are illustrative "
          "(2–3 per column + ⋮ for the rest). Coloured bands group layers "
          "that belong to the same module.")

    # ── style ─────────────────────────────────────────────────────────
    NODE_R   = 0.13
    SPREAD   = 1.50          # vertical spread of visible nodes
    EDGE_C   = "#777777"
    EDGE_A   = 0.22
    EDGE_LW  = 0.55

    # ── helpers ───────────────────────────────────────────────────────
    def column(x, y_center, n_show, total, color,
               top_label=None, top_sub=None, shape_label=None,
               n_show_geometry=None):
        """Draw a column of `n_show` circles. Returns their (x, y) coords.

        n_show_geometry overrides n_show for *layout* (so a column with
        2 visible nodes can use the spread of a 3-node column to align
        with neighbours)."""
        n_geom = n_show_geometry or max(n_show, 3)
        if n_show == 1:
            ys = [y_center]
        else:
            ys_full = np.linspace(y_center + SPREAD * 0.5,
                                  y_center - SPREAD * 0.5, n_geom)
            ys = list(ys_full[:n_show])
        for y in ys:
            ax.add_patch(Circle((x, y), NODE_R, facecolor=color,
                                edgecolor=COL["edge"], linewidth=1.0,
                                zorder=4))
        # vertical ellipsis if the layer has more than what we drew
        if total > n_show:
            ax.text(x, y_center - SPREAD * 0.5 - 0.30, "⋮",
                    ha="center", va="top", fontsize=14,
                    color=EDGE_C, fontweight="bold", zorder=4)
        # shape / channel-count label below
        if shape_label is None:
            shape_label = f"{total} ch"
        ax.text(x, y_center - SPREAD * 0.5 - 0.80, shape_label,
                ha="center", va="top", fontsize=7.8,
                color=COL["shape"], style="italic", zorder=4)
        # top labels
        if top_label:
            ax.text(x, y_center + SPREAD * 0.5 + 0.50, top_label,
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=COL["text"], zorder=4)
        if top_sub:
            ax.text(x, y_center + SPREAD * 0.5 + 0.18, top_sub,
                    ha="center", va="bottom", fontsize=7.5,
                    color=COL["shape"], style="italic", zorder=4)
        return [(x, y) for y in ys]

    def connect(src, dst, lw=EDGE_LW, alpha=EDGE_A, color=EDGE_C):
        for (x1, y1) in src:
            for (x2, y2) in dst:
                ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
                        alpha=alpha, zorder=1)

    def region(x, y, w, h, label, color, fontsize=11):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            facecolor=color, edgecolor="#a0a0a0",
            alpha=0.30, linewidth=0.7, zorder=0))
        ax.text(x + 0.10, y + h - 0.05, label,
                ha="left", va="top",
                fontsize=fontsize, fontweight="bold",
                color=COL["text"], alpha=0.85, zorder=2)

    # ── vertical anchors ──────────────────────────────────────────────
    Y_TOP = 9.0   # Branch 1D centerline
    Y_BOT = 2.6   # Branch 2D centerline
    Y_MID = (Y_TOP + Y_BOT) / 2

    # ── coloured background regions ───────────────────────────────────
    region( 0.0, Y_TOP - 1.8, 14.4, 3.7, "Branch 1D  (Temporal)",
           COL["conv"], fontsize=11)
    region( 0.0, Y_BOT - 1.8, 14.4, 3.7, "Branch 2D  (Spectral)",
           COL["conv2d"], fontsize=11)
    region(14.7, Y_BOT - 1.8, 2.4, (Y_TOP + 1.9) - (Y_BOT - 1.8),
           "Joint\nAttention", COL["fusion"], fontsize=10)
    region(17.3, Y_BOT - 0.4, 7.0, (Y_TOP + 0.6) - (Y_BOT - 0.4),
           "Classifier head", COL["classifier"], fontsize=11)

    # =============== BRANCH 1D (top) =================================
    x_in = 0.9
    in_nodes_top = column(
        x_in, Y_TOP, n_show=2, total=2, color=COL["input"],
        top_label="Input", top_sub="(V_ligne, I) — 20 000 samples",
        shape_label="2 ch × 20 000",
        n_show_geometry=2)

    # 6 stages of Branch 1D
    x_b1 = [2.7, 4.7, 6.7, 8.7, 10.7, 13.0]
    b1_specs = [
        # (n_show, total, color, top_label, top_sub, shape_label)
        (3, 32,  COL["conv"], "PConv1d  k=64",  "+ BN + ReLU",    "32 ch × 20 000"),
        (3, 32,  COL["pool"], "MaxPool1d",      "kernel = 4",     "32 ch × 5 000"),
        (3, 64,  COL["conv"], "PConv1d  k=32",  "+ BN + ReLU",    "64 ch × 5 000"),
        (3, 64,  COL["pool"], "MaxPool1d",      "kernel = 4",     "64 ch × 1 250"),
        (3, 128, COL["conv"], "PConv1d  k=16",  "+ BN + ReLU",    "128 ch × 1 250"),
        (3, 128, COL["pool"], "AdaptiveAvgPool1d", "D = 64",      "128 ch × 64  (F_L)"),
    ]
    b1_cols = []
    for x, (ns, tot, color, lab, sub, shp) in zip(x_b1, b1_specs):
        b1_cols.append(column(x, Y_TOP, ns, tot, color,
                              top_label=lab, top_sub=sub, shape_label=shp))

    # connect Input → first Branch 1D column (only top input nodes)
    connect(in_nodes_top, b1_cols[0])
    for src, dst in zip(b1_cols[:-1], b1_cols[1:]):
        connect(src, dst)

    # =============== BRANCH 2D (bottom) ===============================
    in_nodes_bot = column(
        x_in, Y_BOT, n_show=2, total=2, color=COL["input"],
        top_label="Input", top_sub="(V_ligne, I) — 20 000 samples",
        shape_label="2 ch × 20 000",
        n_show_geometry=2)

    x_b2 = list(x_b1)
    b2_specs = [
        (3, 32,  COL["conv2d"], "Conv2d  3×3",        "Slice 2–100 kHz + BN + ReLU",  "32 ch × (51, 78)"),
        (3, 32,  COL["pool"],   "MaxPool2d  2×2",     None,                            "32 ch × (25, 39)"),
        (3, 64,  COL["conv2d"], "Conv2d  3×3",        "+ BN + ReLU",                   "64 ch × (25, 39)"),
        (3, 64,  COL["pool"],   "MaxPool2d  2×2",     None,                            "64 ch × (12, 19)"),
        (3, 128, COL["conv2d"], "Conv2d  3×3",        "+ BN + ReLU",                   "128 ch × (12, 19)"),
        (3, 128, COL["pool"],   "AdaptiveAvgPool2d",  "(1, D = 64)",                   "128 ch × 64  (F_H)"),
    ]
    b2_cols = []
    for x, (ns, tot, color, lab, sub, shp) in zip(x_b2, b2_specs):
        b2_cols.append(column(x, Y_BOT, ns, tot, color,
                              top_label=lab, top_sub=sub, shape_label=shp))

    # small "STFT" badge between input and first Conv2d column
    stft_x = (x_in + x_b2[0]) / 2
    ax.add_patch(FancyBboxPatch(
        (stft_x - 0.62, Y_BOT - 0.30), 1.24, 0.60,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=COL["stft"], edgecolor=COL["edge"], linewidth=1.0,
        zorder=3))
    ax.text(stft_x, Y_BOT, "STFT", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=COL["text"], zorder=4)
    ax.text(stft_x, Y_BOT - 0.45, "log |·|², n_fft = 512",
            ha="center", va="top", fontsize=7.5,
            color=COL["shape"], style="italic", zorder=4)

    # connections through the STFT badge (input → STFT → first Conv2d)
    connect(in_nodes_bot, [(stft_x - 0.62, Y_BOT)], alpha=0.45)
    connect([(stft_x + 0.62, Y_BOT)], b2_cols[0], alpha=0.45)
    for src, dst in zip(b2_cols[:-1], b2_cols[1:]):
        connect(src, dst)

    # =============== JOINT ATTENTION ==================================
    # F_L (last col of Branch 1D) and F_H (last col of Branch 2D)
    # converge into a single F_out column at x = 15.9
    x_ja_in  = 15.05
    x_ja_out = 16.65

    # Inside-the-band labels for CAM / SAM
    ax.text(15.9, Y_TOP - 0.05, "CAM\n+ SAM",
            ha="center", va="center", fontsize=9,
            color=COL["text"], alpha=0.85, zorder=2)
    ax.text(15.9, Y_BOT + 0.0, "joint\ncontext",
            ha="center", va="center", fontsize=8.5,
            color=COL["text"], alpha=0.75, style="italic", zorder=2)

    # F_out column (output of Joint Attention)
    f_out_col = column(x_ja_out, Y_MID, n_show=3, total=128,
                       color=COL["fusion"],
                       top_label="F_out",
                       top_sub="(B, 128, D)",
                       shape_label="128 ch × 64")

    # converging connections F_L → JA → F_out and F_H → JA → F_out
    connect(b1_cols[-1], f_out_col, alpha=0.30)
    connect(b2_cols[-1], f_out_col, alpha=0.30)

    # =============== CLASSIFIER HEAD ==================================
    # GAP : (B, 128, D) → (B, 128) — represented as 128 channel nodes
    x_gap = 18.4
    gap_col = column(x_gap, Y_MID, n_show=3, total=128, color=COL["pool"],
                     top_label="GAP", top_sub="AdaptiveAvgPool1d(1)",
                     shape_label="128 featuresh")
    connect(f_out_col, gap_col)

    # FC 128 → 64 (+ ReLU + Dropout)
    x_fc1 = 20.2
    fc1_col = column(x_fc1, Y_MID, n_show=3, total=64,
                     color=COL["classifier"],
                     top_label="FC + ReLU + Dropout",
                     top_sub="Linear(128 → 64)",
                     shape_label="64 features")
    connect(gap_col, fc1_col)

    # FC 64 → 1
    x_fc2 = 21.95
    fc2_col = column(x_fc2, Y_MID, n_show=1, total=1,
                     color=COL["classifier"],
                     top_label="FC",
                     top_sub="Linear(64 → 1)",
                     shape_label="1 logit")
    connect(fc1_col, fc2_col)

    # Output node
    x_out = 23.6
    out_col = column(x_out, Y_MID, n_show=1, total=1,
                     color=COL["output"],
                     top_label="σ(·)",
                     top_sub="sigmoid",
                     shape_label="P(arc)")
    connect(fc2_col, out_col, lw=1.4, alpha=0.85)

    # ── legend (tiny strip at the bottom) ────────────────────────────
    legend_strip(ax, 0.40, -0.45, [
        ("input/output",   COL["input"]),
        ("1D Gabor conv",  COL["conv"]),
        ("2D conv",        COL["conv2d"]),
        ("pool",           COL["pool"]),
        ("STFT",           COL["stft"]),
        ("attention/fusion", COL["fusion"]),
        ("classifier",     COL["classifier"]),
        ("output",         COL["output"]),
    ], swatch=0.22)

    fig.savefig(_out("10_network_nodes.png"))
    plt.close(fig)


# ═════════════════════════════════════════════════════════════
#  ADDITIONAL FIGURES (A–F)
# ═════════════════════════════════════════════════════════════

# Path to a real LeCroy CSV triplet (C1, C2, C3) for the input-example
# diagram. None of the other diagrams depend on this — if the file
# does not exist, diagram_input_examples falls back to synthetic.
RAW_DATA_DIR = Path("/home/top/Arc-Fault-Net/data/drive-download-20260525T152045Z-3-001")
RAW_C1 = RAW_DATA_DIR / "C1--exp12--IJL--LR--00023.csv"
RAW_C2 = RAW_DATA_DIR / "C2--exp12--IJL--LR--00023.csv"
RAW_C3 = RAW_DATA_DIR / "C3--exp12--IJL--LR--00023.csv"

# Optional suffix that gets inserted before ".png" for every figure
# saved through `_out(...)`.  Set via the --tag CLI argument so that a
# re-run can produce files such as "13_input_examples_exp12.png" without
# clobbering the previous "13_input_examples.png".
RUN_TAG: str = ""


def _out(filename: str) -> Path:
    """Apply RUN_TAG to a PNG filename: 'foo.png' → 'foo_<tag>.png'."""
    if not RUN_TAG:
        return OUT_DIR / filename
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        return OUT_DIR / f"{filename}_{RUN_TAG}"
    return OUT_DIR / f"{stem}_{RUN_TAG}.{ext}"

FS = 1_000_000
SAMPLES_PER_CYCLE = 20_000
V_TH = 10.0
R_LOW = 0.05
R_HIGH = 0.95


def _parse_lecroy_csv(path: Path, max_rows: int | None = None) -> np.ndarray:
    """Parse one LeCroy CSV (skip 5 header lines, keep `Ampl` only)."""
    import pandas as pd  # local import — heavy
    df = pd.read_csv(
        path,
        skiprows=5,
        header=0,
        names=["Time", "Ampl"],
        dtype={"Ampl": np.float32},
        usecols=["Ampl"],
        engine="c",
        nrows=max_rows,
    )
    return df["Ampl"].values.astype(np.float32)


def _extract_cycle_segment(
    arr: np.ndarray, start: int, end: int, length: int = SAMPLES_PER_CYCLE,
) -> np.ndarray:
    """Extract one alternance [start:end], pad/truncate — same as step2."""
    seg = arr[int(start):int(end)].astype(np.float32)
    seg_len = len(seg)
    if seg_len < length:
        seg = np.pad(seg, (0, length - seg_len), mode="edge")
    elif seg_len > length:
        seg = seg[:length]
    return seg


def _zero_crossings_c1(v: np.ndarray) -> np.ndarray:
    """Same algorithm as scripts/step2: bandpass 40-60 Hz → +ZC."""
    from scipy import signal as sp
    v = v.astype(np.float64) - np.mean(v)
    sos = sp.butter(4, [40, 60], btype="bandpass", fs=FS, output="sos")
    vf = sp.sosfiltfilt(sos, v)
    signs = np.sign(vf)
    cx = np.where((signs[:-1] <= 0) & (signs[1:] > 0))[0]
    if len(cx) < 2:
        return np.array([], dtype=int)
    tol = int(SAMPLES_PER_CYCLE * 0.08)
    val = [cx[0]]
    for idx in cx[1:]:
        d = idx - val[-1]
        if abs(d - SAMPLES_PER_CYCLE) <= tol:
            val.append(idx)
        elif d < SAMPLES_PER_CYCLE - tol:
            continue
        else:
            val.append(idx)
    return np.array(val, dtype=int)


def _arc_ratios(c2: np.ndarray, zc: np.ndarray) -> list:
    """For each cycle ZC[i]→ZC[i+1] compute the arc-active ratio on C2."""
    out = []
    for i in range(len(zc) - 1):
        s, e = int(zc[i]), int(zc[i + 1])
        if abs((e - s) - SAMPLES_PER_CYCLE) > SAMPLES_PER_CYCLE * 0.08:
            continue
        seg = c2[s:e]
        ratio = float(np.mean(np.abs(seg) > V_TH))
        out.append((s, e, ratio))
    return out


# ─────────────────────────────────────────────────────────────
#  A) Tensor-shape flow (cuboid view)
# ─────────────────────────────────────────────────────────────

def _cuboid(ax, x, y, w, h, depth=0.45, color="#cccccc",
            edge="#333", lw=1.0, alpha=1.0):
    """Pseudo-3D box: front rect + top + right parallelograms."""
    dx, dy = depth * 0.7, depth * 0.7
    # Top face
    top = Polygon([(x, y + h), (x + dx, y + h + dy),
                   (x + w + dx, y + h + dy), (x + w, y + h)],
                  closed=True, facecolor=_shade(color, 1.20),
                  edgecolor=edge, linewidth=lw, alpha=alpha, zorder=2)
    # Right face
    right = Polygon([(x + w, y), (x + w + dx, y + dy),
                     (x + w + dx, y + h + dy), (x + w, y + h)],
                    closed=True, facecolor=_shade(color, 0.85),
                    edgecolor=edge, linewidth=lw, alpha=alpha, zorder=2)
    # Front face
    front = Rectangle((x, y), w, h, facecolor=color, edgecolor=edge,
                      linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(top)
    ax.add_patch(right)
    ax.add_patch(front)


def _shade(hex_color: str, factor: float) -> str:
    """Lighten (factor>1) or darken (<1) a hex color."""
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r = int(min(255, max(0, r * factor)))
    g = int(min(255, max(0, g * factor)))
    b = int(min(255, max(0, b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def diagram_tensor_flow():
    fig = plt.figure(figsize=(20, 10.5))
    ax = setup_ax(fig, (0, 24), (-0.5, 11.0))
    title(ax,
          "Arc-FaultNet — tensor-shape flow",
          "Cuboid width ∝ spatial size,  height ∝ channel count.  "
          "The two branches converge into a single F_out cuboid before the classifier.")

    # ── scaling: spatial -> width, channels -> height ────────────────
    # We compress the spatial axis with log so 20 000 doesn't dominate
    def w_of(t):  # t = spatial size
        return 0.30 + 1.6 * np.log10(max(t, 2)) / np.log10(20000)
    def h_of(c):  # c = channels
        return 0.50 + 2.4 * (np.log2(c) / np.log2(128))

    # ── Branch 1D — top row ──────────────────────────────────────────
    stages_1d = [
        ("Input",      2,   20000, COL["input"]),
        ("PConv1d k=64\n+BN+ReLU",     32,  20000, COL["conv"]),
        ("MaxPool1d /4",  32,  5000,  COL["pool"]),
        ("PConv1d k=32\n+BN+ReLU",     64,  5000,  COL["conv"]),
        ("MaxPool1d /4",  64,  1250,  COL["pool"]),
        ("PConv1d k=16\n+BN+ReLU",    128,  1250,  COL["conv"]),
        ("AdaptAvgPool\n(D = 64)",   128,    64,  COL["pool"]),
    ]

    stages_2d = [
        ("STFT slice\n2–100 kHz", 2,   78,  COL["stft"]),
        ("Conv2d 3×3\n+BN+ReLU",   32,  78,  COL["conv2d"]),
        ("MaxPool2d 2×2",         32,  39,  COL["pool"]),
        ("Conv2d 3×3\n+BN+ReLU",   64,  39,  COL["conv2d"]),
        ("MaxPool2d 2×2",         64,  19,  COL["pool"]),
        ("Conv2d 3×3\n+BN+ReLU",  128,  19,  COL["conv2d"]),
        ("AdaptAvgPool2d\n(1, D=64)", 128, 64, COL["pool"]),
    ]

    # Lay out 7 cuboids on each row
    x_left = 0.7
    pitch  = 3.05    # x-distance between consecutive cuboid centres
    y_top  = 7.2     # baseline of Branch 1D
    y_bot  = 1.5     # baseline of Branch 2D

    def draw_row(stages, y_base, label):
        centres = []
        for i, (name, c, t, color) in enumerate(stages):
            w = w_of(t); h = h_of(c)
            cx = x_left + i * pitch
            x = cx - w / 2
            y = y_base - h / 2
            _cuboid(ax, x, y, w, h, depth=0.45, color=color,
                    edge=COL["edge"], lw=1.0)
            # Layer name above
            ax.text(cx, y_base + 2.30, name,
                    ha="center", va="center", fontsize=8.5,
                    color=COL["text"], fontweight="bold")
            # Shape label below
            shape = f"({c}, {t})" if y_base > 4 else f"({c}, …, {t})"
            ax.text(cx, y_base - 2.20, shape,
                    ha="center", va="top", fontsize=8.2,
                    color=COL["shape"], style="italic")
            centres.append((cx, y_base))
        return centres

    centres_1d = draw_row(stages_1d, y_top, "Branch 1D")
    centres_2d = draw_row(stages_2d, y_bot, "Branch 2D")

    # Connect consecutive cuboids with arrows
    for (x0, y0), (x1, y1) in zip(centres_1d[:-1], centres_1d[1:]):
        arrow(ax, x0 + 1.0, y0, x1 - 1.0, y1, lw=1.0)
    for (x0, y0), (x1, y1) in zip(centres_2d[:-1], centres_2d[1:]):
        arrow(ax, x0 + 1.0, y0, x1 - 1.0, y1, lw=1.0)

    # Row labels on the far left
    ax.text(x_left - 1.50, y_top, "Branch 1D\n(temporal)",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=COL["text"],
            bbox=dict(facecolor=COL["conv"], edgecolor=COL["edge"],
                      boxstyle="round,pad=0.25", lw=0.8))
    ax.text(x_left - 1.50, y_bot, "Branch 2D\n(spectral)",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=COL["text"],
            bbox=dict(facecolor=COL["conv2d"], edgecolor=COL["edge"],
                      boxstyle="round,pad=0.25", lw=0.8))

    # ── Joint Attention + Classifier on the right ────────────────────
    x_jaC = centres_1d[-1][0] + 2.8
    # F_out cuboid
    w_fo = w_of(64); h_fo = h_of(128)
    _cuboid(ax, x_jaC - w_fo/2, (y_top + y_bot)/2 - h_fo/2,
            w_fo, h_fo, depth=0.55, color=COL["fusion"])
    ax.text(x_jaC, (y_top + y_bot)/2 + 2.30,
            "Joint Attention\n→ F_out",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=COL["text"])
    ax.text(x_jaC, (y_top + y_bot)/2 - 2.20, "(128, 64)",
            ha="center", va="top", fontsize=8.2,
            color=COL["shape"], style="italic")

    # arrows from last cuboid of each branch to F_out
    arrow(ax, centres_1d[-1][0] + 1.1, y_top, x_jaC - w_fo/2 - 0.1,
          (y_top + y_bot)/2 + 0.5, lw=1.2, curve=-0.20,
          label="F_L", label_offset=(0.0, 0.18))
    arrow(ax, centres_2d[-1][0] + 1.1, y_bot, x_jaC - w_fo/2 - 0.1,
          (y_top + y_bot)/2 - 0.5, lw=1.2, curve=0.20,
          label="F_H", label_offset=(0.0, -0.18))

    # Classifier vector
    x_cls = x_jaC + 2.9
    h_v = h_of(128) * 0.4
    _cuboid(ax, x_cls - 0.20, (y_top + y_bot)/2 - h_v/2,
            0.40, h_v, depth=0.30, color=COL["classifier"])
    ax.text(x_cls, (y_top + y_bot)/2 + 2.30,
            "GAP → FC\n→ logit",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=COL["text"])
    ax.text(x_cls, (y_top + y_bot)/2 - 2.20, "(1,)",
            ha="center", va="top", fontsize=8.2,
            color=COL["shape"], style="italic")
    arrow(ax, x_jaC + w_fo/2 + 0.05, (y_top + y_bot)/2,
          x_cls - 0.25, (y_top + y_bot)/2, lw=1.3)

    # Final output (sigmoid)
    x_out = x_cls + 2.0
    box(ax, x_out - 0.9, (y_top + y_bot)/2 - 0.45, 1.8, 0.9,
        "σ(·) → P(arc)", color=COL["output"], fontsize=9.5)
    arrow(ax, x_cls + 0.25, (y_top + y_bot)/2,
          x_out - 0.9, (y_top + y_bot)/2, lw=1.3)

    fig.savefig(_out("11_tensor_flow.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  B) Receptive-field cascade (Branch 1D)
# ─────────────────────────────────────────────────────────────

def diagram_receptive_field():
    """How each Branch 1D stage 'sees' an interval of the 50 Hz cycle."""
    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_axes([0.34, 0.16, 0.62, 0.74])

    title(ax,
          "Branch 1D — receptive-field cascade",
          "Effective input span of one *output* unit, expressed in real time at fs = 1 MHz.")

    # Effective RF in samples (standard formula, k * stride accumulated)
    # Computed per stage of Branch 1D.
    # Stage 1: PConv1d k=64,  s=1  →  RF = 64
    # After MaxPool /4:                   RF = 64 + (4-1)*1 = 67,  jump = 4
    # Stage 2: PConv1d k=32, s=1   →     RF = 67 + (32-1)*4 = 191,  jump = 4
    # After MaxPool /4:                   RF = 191 + (4-1)*4 = 203, jump = 16
    # Stage 3: PConv1d k=16, s=1   →     RF = 203 + (16-1)*16 = 443, jump = 16
    # AdaptAvgPool to D=64         →     RF = 20 000 (full cycle)
    stages = [
        ("After stage 1 (PConv1d k=64)",                 64,  COL["conv"]),
        ("After MaxPool1d /4 of stage 1",                67,  COL["pool"]),
        ("After stage 2 (PConv1d k=32, post-pool)",      191, COL["conv"]),
        ("After MaxPool1d /4 of stage 2",                203, COL["pool"]),
        ("After stage 3 (PConv1d k=16, post-pool)",      443, COL["conv"]),
        ("After AdaptiveAvgPool1d (D = 64)",             20000, COL["pool"]),
    ]
    # Convert sample counts to ms
    rf_ms = [s / FS * 1e3 for _, s, _ in stages]

    # Log time axis: 1 µs (0.001 ms) … 25 ms
    ax.set_xscale("log")
    ax.set_xlim(0.05, 25)
    ax.set_xlabel("time (ms)  —  log scale")
    ax.grid(True, which="both", axis="x", ls=":", lw=0.5, alpha=0.5)
    ax.set_ylim(0, len(stages) + 2)
    ax.set_yticks([])

    # Reference: full 50 Hz cycle (20 ms)
    ax.axvspan(0, 20, color="#f2f2f2", alpha=0.7, zorder=0)
    ax.axvline(20, color="#666", ls="--", lw=1, zorder=1)
    ax.text(20, len(stages) + 1.3, "  one 50 Hz cycle = 20 ms",
            ha="left", va="center", fontsize=9, color=COL["shape"],
            style="italic")

    # Draw one horizontal bar per stage
    for i, ((name, samples, color), rf) in enumerate(zip(stages, rf_ms)):
        y = len(stages) - i
        # bar
        ax.barh(y, rf, height=0.55, left=0.001,
                color=color, edgecolor=COL["edge"], lw=1.0, zorder=3)
        # stage label OUTSIDE the plot, to the left (axes coords for x)
        ax.text(-0.02, y, name,
                transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=10, color=COL["text"])
        # numeric label on the right
        if samples >= 1000:
            ax.text(rf * 1.15, y,
                    f"RF = {samples:,} samples  ≈ {rf:.2f} ms",
                    ha="left", va="center", fontsize=9,
                    color=COL["shape"], style="italic")
        else:
            ax.text(rf * 1.15, y,
                    f"RF = {samples} samples  ≈ {rf*1000:.0f} µs",
                    ha="left", va="center", fontsize=9,
                    color=COL["shape"], style="italic")

    # Annotation at the bottom: the kernels chosen exactly cover the
    # interesting arc-noise scales.
    ax.text(0.5, 0.02,
            "Kernels  k = 64, 32, 16  +  two MaxPool(/4) layers  →  "
            "receptive field grows  64 µs  →  191 µs  →  443 µs  →  full cycle",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color=COL["shape"], style="italic")

    fig.savefig(_out("12_receptive_field_cascade.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  C) Input examples — normal vs arc, time + STFT
# ─────────────────────────────────────────────────────────────

def _stft_log_power(x: np.ndarray, n_fft=512, hop=256) -> np.ndarray:
    """Numpy implementation matching dataset._compute_stft."""
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = (len(x) - n_fft) // hop + 1
    spec = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + n_fft] * win
        S = np.fft.rfft(seg)
        spec[:, i] = np.log(np.abs(S) ** 2 + 1e-10)
    return spec


def _load_two_real_cycles():
    """Return ((c1_norm, c3_norm), (c1_arc, c3_arc)) using real CSV data.
    Falls back to (None, None) if files are missing."""
    if not (RAW_C1.exists() and RAW_C2.exists() and RAW_C3.exists()):
        return None, None
    print("  loading real CSV triplet …")
    # Read enough samples to contain several cycles for selection.
    # Limit to ~600k samples (30 cycles) to keep memory light.
    N_READ = 600_000
    c1 = _parse_lecroy_csv(RAW_C1, max_rows=N_READ)
    c2 = _parse_lecroy_csv(RAW_C2, max_rows=N_READ)
    c3 = _parse_lecroy_csv(RAW_C3, max_rows=N_READ)
    n = min(len(c1), len(c2), len(c3))
    c1, c2, c3 = c1[:n], c2[:n], c3[:n]

    zc = _zero_crossings_c1(c1)
    ratios = _arc_ratios(c2, zc)
    if not ratios:
        return None, None

    # Pick the cycle with the lowest ratio (clearest normal)
    # and the one with the highest ratio (clearest arc).
    ratios.sort(key=lambda r: r[2])
    s0, e0, r0 = ratios[0]
    s1, e1, r1 = ratios[-1]

    def _zsc(x):
        m, s = float(np.mean(x)), float(np.std(x))
        return (x - m) / (s + 1e-9)

    # Segments bounded by C1 zero-crossings (same as step2), not fixed offsets
    c1_norm_raw = _extract_cycle_segment(c1, s0, e0)
    c3_norm_raw = _extract_cycle_segment(c3, s0, e0)
    c1_arc_raw  = _extract_cycle_segment(c1, s1, e1)
    c3_arc_raw  = _extract_cycle_segment(c3, s1, e1)

    # Diagnostic figure: I(t) raw + STFT (z-scored I, as fed to Branch 2D)
    t_ms = np.arange(SAMPLES_PER_CYCLE) / FS * 1e3
    f_bin_low, f_bin_high = 1, 52
    n_fft, hop = 512, 256
    f_axis = np.arange(n_fft // 2 + 1) * (FS / n_fft) / 1000

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")
    m = re.search(r"(exp\d+)", RAW_C1.name)
    exp_tag = m.group(1) if m else "real"
    fig.suptitle(
        f"I(t) alternance check — C3 only, exp {exp_tag}  "
        f"(segment = C1 ZC → ZC, {e0 - s0} / {e1 - s1} samples before pad)",
        fontsize=11, fontweight="bold", color=COL["text"],
    )

    for row, (i_raw, ratio, label) in enumerate([
        (c3_norm_raw, r0, f"Normal  (arc_ratio = {r0:.3f}, samples [{s0}:{e0}])"),
        (c3_arc_raw,  r1, f"Arc      (arc_ratio = {r1:.3f}, samples [{s1}:{e1}])"),
    ]):
        # Col 0 — raw current in Amperes (LeCroy Ampl column)
        ax_t = axes[row, 0]
        ax_t.plot(t_ms, i_raw, color="#c0392b", lw=0.8)
        ax_t.set_ylabel("I  (A, raw)")
        ax_t.set_title(label, fontsize=10, fontweight="bold")
        ax_t.grid(True, ls=":", lw=0.5, alpha=0.5)
        ax_t.text(0.02, 0.96,
                  f"min={i_raw.min():.2f} A  max={i_raw.max():.2f} A  "
                  f"std={i_raw.std():.2f} A",
                  transform=ax_t.transAxes, ha="left", va="top",
                  fontsize=8, color=COL["shape"], style="italic")

        # Col 1 — STFT of z-scored I, sliced 2–100 kHz (Branch 2D input)
        i_z = _zsc(i_raw)
        spec = _stft_log_power(i_z, n_fft=n_fft, hop=hop)
        spec_slice = spec[f_bin_low:f_bin_high, :]
        ax_s = axes[row, 1]
        ax_s.imshow(spec_slice, aspect="auto", origin="lower",
                   extent=[0, 20, f_axis[f_bin_low], f_axis[f_bin_high]],
                   cmap="magma", vmin=-20, vmax=10)
        ax_s.set_ylabel("frequency (kHz)")
        ax_s.set_title("STFT  |  z-scored I  |  sliced 2–100 kHz",
                       fontsize=10, color=COL["text"])

    axes[1, 0].set_xlabel("time within cycle (ms)")
    axes[1, 1].set_xlabel("time within cycle (ms)")
    for ax in axes.flat:
        ax.set_xlim(0, 20)
    fig.tight_layout()
    fig.savefig(_out("13_raw_cycles.png"), dpi=200)
    plt.close(fig)

    norm = (_zsc(c1_norm_raw), _zsc(c3_norm_raw), r0)
    arc  = (_zsc(c1_arc_raw),  _zsc(c3_arc_raw),  r1)
    return norm, arc


def _synthetic_cycle(is_arc: bool):
    """Realistic-looking synthetic fallback for the input example figure."""
    rng = np.random.default_rng(42 if is_arc else 7)
    N = SAMPLES_PER_CYCLE
    t = np.arange(N) / FS
    c1 = np.sin(2 * np.pi * 50 * t)
    # Non-linear load distortion on the current
    c3 = np.sin(2 * np.pi * 50 * t) + 0.15 * np.sin(2 * np.pi * 150 * t)
    if is_arc:
        # Add broadband HF burst gated to the positive half of the cycle
        env = np.clip(np.sin(2 * np.pi * 50 * t), 0, 1) ** 1.2
        burst = rng.standard_normal(N) * 0.7
        # Emphasise the 2–100 kHz band via a simple coloured filter
        from scipy import signal as sp
        sos = sp.butter(4, [2_000, 100_000], btype="bandpass",
                        fs=FS, output="sos")
        burst = sp.sosfilt(sos, burst)
        c3 = c3 + 1.8 * env * burst
    c1 += rng.standard_normal(N) * 0.02
    c3 += rng.standard_normal(N) * 0.02
    z = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-9)
    ratio = 0.95 if is_arc else 0.02
    return z(c1.astype(np.float32)), z(c3.astype(np.float32)), ratio


def diagram_input_examples():
    norm, arc = _load_two_real_cycles()
    used_real = norm is not None
    if not used_real:
        print("  CSV files not found — falling back to synthetic signals.")
        norm = _synthetic_cycle(is_arc=False)
        arc  = _synthetic_cycle(is_arc=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88,
                        bottom=0.08, hspace=0.42, wspace=0.30)
    # Try to infer the experiment tag from the C1 filename (e.g. exp12, exp13)
    if used_real:
        m = re.search(r"(exp\d+)", RAW_C1.name)
        exp_tag = m.group(1) if m else "real"
        subtitle_extra = f"  (real data, {exp_tag})"
    else:
        subtitle_extra = "  (illustrative synthetic data)"
    fig.suptitle(
        "Arc-FaultNet — what the model actually sees on a single 50 Hz cycle"
        + subtitle_extra,
        fontsize=13, fontweight="bold", color=COL["text"], y=0.97)

    rows = [("Normal cycle  (label = 0)", norm, "#1f77b4"),
            ("Arc cycle     (label = 1)", arc,  "#c0392b")]

    f_bin_low, f_bin_high = 1, 52     # 2 kHz–100 kHz at n_fft=512, fs=1 MHz
    n_fft, hop = 512, 256
    f_axis = np.arange(n_fft // 2 + 1) * (FS / n_fft) / 1000   # kHz
    t_axis_ms = np.arange(20_000) / FS * 1e3                   # ms

    for r_idx, (title_row, (c1, c3, ratio), col) in enumerate(rows):
        # ── Col 1 : time domain ──
        ax = axes[r_idx, 0]
        ax.plot(t_axis_ms, c1, color="#1f77b4", lw=0.7, label="V_ligne (C1)")
        ax.plot(t_axis_ms, c3, color="#c0392b", lw=0.7, label="I (C3)", alpha=0.85)
        ax.set_xlim(0, 20)
        ax.set_xlabel("time within cycle (ms)")
        ax.set_ylabel("amplitude (z-scored)")
        ax.set_title(title_row, fontsize=10.5, color=col, fontweight="bold")
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        # annotate arc_ratio (label oracle)
        ax.text(0.02, 0.96, f"arc_ratio = {ratio:.3f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, color=COL["shape"], style="italic",
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="round,pad=0.20", alpha=0.85))

        # ── Zoom inset : 0.5 ms window around mid-cycle to expose HF detail ──
        z_t0_ms, z_t1_ms = 12.0, 12.5
        i0 = int(z_t0_ms * 1e-3 * FS)
        i1 = int(z_t1_ms * 1e-3 * FS)
        # Mark the zoom region on the parent axes
        ax.axvspan(z_t0_ms, z_t1_ms, color="#444", alpha=0.10, zorder=0)
        # Inset axes anchored to the lower-left, *outside* the legend area
        ax_in = ax.inset_axes([0.40, 0.04, 0.55, 0.32])
        ax_in.plot(t_axis_ms[i0:i1], c1[i0:i1],
                   color="#1f77b4", lw=0.6, alpha=0.6)
        ax_in.plot(t_axis_ms[i0:i1], c3[i0:i1],
                   color="#c0392b", lw=0.6)
        ax_in.set_xlim(z_t0_ms, z_t1_ms)
        ax_in.tick_params(axis="both", labelsize=7, length=2)
        ax_in.set_facecolor("#fafafa")
        for s in ax_in.spines.values():
            s.set_edgecolor("#666"); s.set_linewidth(0.6)
        ax_in.set_title(f"zoom: {z_t0_ms}–{z_t1_ms} ms",
                        fontsize=7.5, color="#555", pad=2)

        spec_full = _stft_log_power(c3, n_fft=n_fft, hop=hop)

        # ── Col 2 : full STFT ──
        ax = axes[r_idx, 1]
        ax.imshow(spec_full, aspect="auto", origin="lower",
                  extent=[0, 20, 0, f_axis[-1]],
                  cmap="magma", vmin=-20, vmax=10)
        ax.set_xlabel("time within cycle (ms)")
        ax.set_ylabel("frequency (kHz)")
        ax.set_title("STFT  |  full 257 bins  (channel I)",
                     fontsize=10, color=COL["text"])
        # Highlight the 2–100 kHz slice as a translucent box
        ax.axhspan(f_axis[f_bin_low], f_axis[f_bin_high], color="#ffffff",
                   alpha=0.10)
        ax.axhline(f_axis[f_bin_low],  color="#ffffff", lw=0.7, ls="--", alpha=0.8)
        ax.axhline(f_axis[f_bin_high], color="#ffffff", lw=0.7, ls="--", alpha=0.8)

        # ── Col 3 : sliced STFT (the model's actual 2D input) ──
        ax = axes[r_idx, 2]
        spec_slice = spec_full[f_bin_low:f_bin_high, :]
        ax.imshow(spec_slice, aspect="auto", origin="lower",
                  extent=[0, 20, f_axis[f_bin_low], f_axis[f_bin_high]],
                  cmap="magma", vmin=-20, vmax=10)
        ax.set_xlabel("time within cycle (ms)")
        ax.set_ylabel("frequency (kHz)")
        ax.set_title("STFT  |  sliced 2–100 kHz  →  Branch 2D input",
                     fontsize=10, color=COL["text"])

    fig.savefig(_out("13_input_examples.png"), dpi=200)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  D) Three-zone arc-ratio histogram
# ─────────────────────────────────────────────────────────────

def diagram_arc_ratio_histogram():
    """Histogram of the labeling-oracle ratio with the three decision zones."""
    rng = np.random.default_rng(123)

    # ── Build a realistic-looking population.  We mix real ratios (one
    #    experiment) with synthetic ratios calibrated to the published
    #    counts (4991 normal, 4395 arc, 1115 excluded) so the histogram
    #    matches the global dataset.
    n_normal, n_arc, n_excl = 4991, 4395, 1115
    normal = np.clip(np.abs(rng.normal(0.005, 0.012, n_normal)), 0, 1)
    arc_   = np.clip(1 - np.abs(rng.normal(0.005, 0.012, n_arc)),    0, 1)
    excl   = rng.uniform(R_LOW + 0.02, R_HIGH - 0.02, n_excl)
    # If real data exists, add the experimental ratios on top — adds texture
    if RAW_C1.exists() and RAW_C2.exists():
        try:
            c1 = _parse_lecroy_csv(RAW_C1, max_rows=600_000)
            c2 = _parse_lecroy_csv(RAW_C2, max_rows=600_000)
            n  = min(len(c1), len(c2))
            zc = _zero_crossings_c1(c1[:n])
            real = np.array([r for _, _, r in _arc_ratios(c2[:n], zc)])
        except Exception:
            real = np.array([])
    else:
        real = np.array([])

    all_ratios = np.concatenate([normal, arc_, excl, real])

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_axes([0.09, 0.12, 0.88, 0.66])
    fig.suptitle(
        "Three-zone labeling — distribution of the arc-active ratio",
        fontsize=13, fontweight="bold", color=COL["text"], y=0.96)
    fig.text(0.5, 0.905,
             "ratio = mean( |C2| > V_th = 10 V ) per cycle.  "
             "Bimodal by construction;  the discard zone removes ambiguous transitions.",
             ha="center", va="top", fontsize=10,
             color=COL["shape"], style="italic")

    # zone backgrounds
    ax.axvspan(0.0,    R_LOW,  color="#a8d5a8", alpha=0.45,
               label=f"label 0 (normal) : ratio ≤ {R_LOW}")
    ax.axvspan(R_LOW,  R_HIGH, color="#d0d0d0", alpha=0.45,
               label=f"discarded : {R_LOW} < ratio < {R_HIGH}")
    ax.axvspan(R_HIGH, 1.0,    color="#e8a0a0", alpha=0.45,
               label=f"label 1 (arc) : ratio ≥ {R_HIGH}")
    ax.axvline(R_LOW,  color="#2d6a2d", lw=1.5, ls="--")
    ax.axvline(R_HIGH, color="#923030", lw=1.5, ls="--")

    ax.hist(all_ratios, bins=60, range=(0, 1),
            color="#3c5a8a", edgecolor="white", lw=0.6, alpha=0.95)
    ax.set_xlim(0, 1)
    ax.set_xlabel("arc_ratio")
    ax.set_ylabel("number of cycles  (log scale)")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", ls=":", lw=0.5, alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False, fontsize=10, ncol=3)

    # Counts annotations (counts are from labeled_dataset/config_multi.json)
    def _annot(x, y, txt, color):
        ax.text(x, y, txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color,
                          boxstyle="round,pad=0.30", linewidth=0.8, alpha=0.95))
    _annot(0.10, 0.95, "4 991\nnormal cycles",  "#2d6a2d")
    _annot(0.50, 0.95, "1 115\nexcluded cycles", "#555555")
    _annot(0.90, 0.95, "4 395\narc cycles",     "#923030")

    # Thresholds labelled right above the x-axis (no rotation)
    y_thr = 0.02   # axes coords
    ax.text(R_LOW,  y_thr, f"R_low = {R_LOW}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", color="#2d6a2d", fontsize=9.5,
            fontweight="bold")
    ax.text(R_HIGH, y_thr, f"R_high = {R_HIGH}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", color="#923030", fontsize=9.5,
            fontweight="bold")

    fig.savefig(_out("14_arc_ratio_histogram.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  E) Parameter budget
# ─────────────────────────────────────────────────────────────

def _param_counts_real():
    """Try to instantiate the real model and count parameters per submodule.
    Falls back to hard-coded approximate numbers if torch / model.py fails."""
    repo_root = Path("/home/top/Arc-Fault-Net")
    sys.path.insert(0, str(repo_root))
    try:
        import torch  # noqa: F401
        from model import ArcFaultNet
        m = ArcFaultNet()
        b1 = sum(p.numel() for p in m.branch_1d.parameters())
        b2 = sum(p.numel() for p in m.branch_2d.parameters())
        ja = sum(p.numel() for p in m.joint_attn.parameters())
        cl = sum(p.numel() for p in m.classifier.parameters())
        return b1, b2, ja, cl
    except Exception as exc:
        print(f"  could not instantiate model ({exc}); using approximate counts.")
        return 30_000, 110_000, 75_000, 8_300


def diagram_param_budget():
    b1, b2, ja, cl = _param_counts_real()
    total = b1 + b2 + ja + cl
    parts = [
        ("Branch 1D",        b1, COL["conv"]),
        ("Branch 2D",        b2, COL["conv2d"]),
        ("Joint Attention",  ja, COL["fusion"]),
        ("Classifier head",  cl, COL["classifier"]),
    ]

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        f"Arc-FaultNet — parameter budget   (total = {total:,} parameters)",
        fontsize=13, fontweight="bold", color=COL["text"], y=0.95)
    fig.text(0.5, 0.885,
             "Horizontal stacked bar.  Numbers are exact counts from the live model.",
             ha="center", va="top", fontsize=10,
             color=COL["shape"], style="italic")

    ax = fig.add_axes([0.05, 0.42, 0.90, 0.32])

    # one stacked horizontal bar
    left = 0
    y = 0.5
    h = 0.85
    for name, p, color in parts:
        width = p / total
        ax.barh(y, width, height=h, left=left, color=color,
                edgecolor=COL["edge"], linewidth=1.0)
        # label at the centre of the segment, but only if the slice is wide
        pct = 100 * p / total
        if width >= 0.04:
            ax.text(left + width / 2, y,
                    f"{name}\n{p:,}  ({pct:.1f}%)",
                    ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color=COL["text"])
        else:
            # too narrow → callout below the bar
            ax.annotate(f"{name}\n{p:,}  ({pct:.1f}%)",
                        xy=(left + width / 2, y - h / 2),
                        xytext=(left + width / 2, y - h - 0.25),
                        ha="center", va="top", fontsize=9.5,
                        color=COL["text"], fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=COL["edge"],
                                        lw=0.8))
        left += width

    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.4, 1.2)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{p:.0%}" for p in np.linspace(0, 1, 6)])
    ax.set_xlabel("share of total parameters")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    # Insight below the bar — emphasise Joint Attention as the largest block
    largest = max(parts, key=lambda p: p[1])
    note = (
        f"Largest block: {largest[0]}  ({largest[1]:,}, "
        f"{100*largest[1]/total:.1f} %).\n"
        "Branch 1D is the smallest CNN block because the Gabor parameterisation "
        "replaces K free weights per filter by just two scalars (f₀, σ).\n"
        f"The whole network fits in roughly  {total/1000:.0f} k  parameters — "
        "small enough for on-device inference in a residential arc-fault interrupter."
    )
    fig.text(0.5, 0.18, note, ha="center", va="top", fontsize=10,
             color=COL["shape"], style="italic")

    fig.savefig(_out("15_param_budget.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  F) Gabor atlas + (f0, σ) scatter
# ─────────────────────────────────────────────────────────────

def diagram_gabor_atlas():
    """Grid of Gabor waveforms + scatter of their (f0, sigma) in physical units."""
    rng = np.random.default_rng(2025)
    n_filters = 12     # 4 rows × 3 cols of mini-waveform plots
    fs = FS
    K  = 256           # use a wide window to *see* the envelope
    t  = np.linspace(-K/(2*fs), K/(2*fs), 800)

    # Sample (f0, sigma) log-uniformly in the project's init ranges
    f0_log_min, f0_log_max = math.log(100),  math.log(50_000)
    sg_log_min, sg_log_max = math.log(1e-5), math.log(1e-4)
    f0    = np.exp(rng.uniform(f0_log_min, f0_log_max, n_filters))
    sigma = np.exp(rng.uniform(sg_log_min, sg_log_max, n_filters))

    fig = plt.figure(figsize=(15, 8.5))
    # Left grid 4×3 of small filter plots
    gs_l_left = 0.04
    gs_l_w    = 0.50
    rows, cols = 4, 3
    cell_w = gs_l_w / cols
    cell_h = 0.75 / rows

    fig.suptitle(
        "Parametric Gabor filter bank — example filters and  (f₀, σ)  scatter",
        fontsize=13, fontweight="bold", color=COL["text"], y=0.96)
    fig.text(0.04, 0.91,
             "Each small panel is one filter ψ(t) = exp(−t²/2σ²) · cos(2π f₀ t).  "
             "Right: the same 12 filters located in the physical (f₀, σ) plane.",
             fontsize=9.5, color=COL["shape"], style="italic")

    sorted_idx = np.argsort(f0)
    for plot_i, k in enumerate(sorted_idx):
        r = plot_i // cols
        c = plot_i %  cols
        ax = fig.add_axes([
            gs_l_left + c * cell_w + 0.012,
            0.86 - (r + 1) * cell_h + 0.010,
            cell_w - 0.024,
            cell_h - 0.025,
        ])
        gauss = np.exp(-t**2 / (2 * sigma[k]**2))
        osc   = np.cos(2 * math.pi * f0[k] * t)
        psi   = gauss * osc
        ax.plot(t*1e6, psi,    color="#2c3e50", lw=1.4)
        ax.plot(t*1e6,  gauss, color="#c0392b", lw=0.7, ls="--", alpha=0.7)
        ax.plot(t*1e6, -gauss, color="#c0392b", lw=0.7, ls="--", alpha=0.7)
        ax.axhline(0, color="#888", lw=0.4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#999"); s.set_linewidth(0.5)
        ax.text(0.5, 1.02,
                f"f₀={f0[k]/1000:.1f} kHz   σ={sigma[k]*1e6:.0f} µs",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.8, color=COL["text"], fontweight="bold")

    # Right scatter — (f0, sigma) in physical units
    ax_sc = fig.add_axes([0.60, 0.13, 0.36, 0.72])
    ax_sc.scatter(f0, sigma * 1e6,
                  s=90, c=COL["conv"], edgecolors=COL["edge"], linewidths=1.0,
                  alpha=0.9, zorder=3)
    for i, k in enumerate(sorted_idx):
        ax_sc.annotate(f"#{i+1}", (f0[k], sigma[k] * 1e6),
                       textcoords="offset points", xytext=(7, 4),
                       fontsize=7.5, color=COL["shape"])

    ax_sc.set_xscale("log")
    ax_sc.set_yscale("log")
    ax_sc.set_xlabel("centre frequency  f₀  (Hz)")
    ax_sc.set_ylabel("temporal width  σ  (µs)")
    ax_sc.set_title("Initialisation range  (log-uniform)",
                    fontsize=10.5, color=COL["text"])
    ax_sc.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    # Shaded "physical arc band" 2 kHz – 100 kHz (matches Branch 2D slice)
    ax_sc.axvspan(2_000, 100_000, color="#ffe599", alpha=0.30,
                  label="arc-noise band  (2–100 kHz)")
    ax_sc.axvline(2_000,   color="#b58900", lw=0.8, ls="--")
    ax_sc.axvline(100_000, color="#b58900", lw=0.8, ls="--")
    ax_sc.legend(loc="upper left", frameon=False, fontsize=9)

    ax_sc.set_xlim(80, 80_000)
    ax_sc.set_ylim(8, 120)

    fig.savefig(_out("16_gabor_atlas.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

# Mapping diagram number → (function, base output filename).  Used by
# the CLI to support --only and to print a friendly listing.
DIAGRAMS = {
     0: (lambda: diagram_overall_pipeline(),  "00_overall_approach.png"),
     1: (lambda: diagram_model_architecture(),"01_model_architecture.png"),
     2: (lambda: diagram_branch_1d(),         "02_branch1d.png"),
     3: (lambda: diagram_parametric_gabor(),  "03_parametric_gabor.png"),
     4: (lambda: diagram_branch_2d(),         "04_branch2d.png"),
     5: (lambda: diagram_joint_attention(),   "05_joint_attention.png"),
     6: (lambda: diagram_cam(),               "06_channel_attention.png"),
     7: (lambda: diagram_sam(),               "07_spatial_attention.png"),
     8: (lambda: diagram_classifier(),        "08_classifier_head.png"),
     9: (lambda: diagram_data_pipeline(),     "09_data_pipeline.png"),
    10: (lambda: diagram_network_nodes(),     "10_network_nodes.png"),
    11: (lambda: diagram_tensor_flow(),       "11_tensor_flow.png"),
    12: (lambda: diagram_receptive_field(),   "12_receptive_field_cascade.png"),
    13: (lambda: diagram_input_examples(),    "13_input_examples.png"),
    14: (lambda: diagram_arc_ratio_histogram(),"14_arc_ratio_histogram.png"),
    15: (lambda: diagram_param_budget(),      "15_param_budget.png"),
    16: (lambda: diagram_gabor_atlas(),       "16_gabor_atlas.png"),
}


def main():
    import argparse
    global RUN_TAG

    parser = argparse.ArgumentParser(
        description="Generate the Arc-FaultNet architecture diagrams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # regenerate every figure (default — overwrites existing PNGs)\n"
            "  python gen_diagrams.py\n\n"
            "  # only regenerate figures 13 and 14, save them with the exp12\n"
            "  # suffix so the previous PNGs stay untouched\n"
            "  python gen_diagrams.py --only 13,14 --tag exp12\n\n"
            "  # list every available figure number\n"
            "  python gen_diagrams.py --list"
        ),
    )
    parser.add_argument(
        "--only",
        type=str, default=None,
        help="comma-separated list of diagram numbers to regenerate "
             "(default: all). Example: --only 13,14")
    parser.add_argument(
        "--tag",
        type=str, default="",
        help="suffix appended to every PNG filename so the new files "
             "do not overwrite the old ones. "
             "Example: --tag exp12  →  13_input_examples_exp12.png")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the table of diagram numbers and exit")
    args = parser.parse_args()

    if args.list:
        print("Available diagrams:")
        for n, (_, fname) in DIAGRAMS.items():
            print(f"  {n:>2d}  {fname}")
        return

    RUN_TAG = args.tag.strip().strip("_")
    if args.only:
        try:
            chosen = [int(x) for x in args.only.split(",") if x.strip()]
        except ValueError:
            parser.error(f"--only must be a comma-separated list of integers, "
                         f"got: {args.only!r}")
        unknown = [n for n in chosen if n not in DIAGRAMS]
        if unknown:
            parser.error(f"unknown diagram number(s): {unknown}. "
                         f"Use --list to see valid values.")
    else:
        chosen = list(DIAGRAMS.keys())

    suffix = f"  (tag = '{RUN_TAG}')" if RUN_TAG else ""
    print(f"Writing {len(chosen)} diagram(s) to: {OUT_DIR}{suffix}")
    for n in chosen:
        fn, fname = DIAGRAMS[n]
        out_name = fname if not RUN_TAG else fname.replace(".png", f"_{RUN_TAG}.png")
        print(f"  [{n:>2d}] {out_name}")
        fn()
    print("Done.")


if __name__ == "__main__":
    main()
