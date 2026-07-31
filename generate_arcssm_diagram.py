#!/usr/bin/env python3
"""
generate_arcssm_diagram.py — Publication-quality architecture diagram for ArcSSM.

Generates a detailed, ML-style architecture diagram matching the exact
configuration from the best single run (arcssm_single_20260728_143957):
  - Backbone: S4D (LTI, bidirectional)
  - d_model=128, d_state=64, n_layers=4
  - Classifier: shallow (Linear→GELU→Dropout→Linear)
  - 359K parameters

Colour palette: 3-colour scheme (blue / orange / grey) for readability.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  MINIMAL 3-COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════
BLUE       = '#3B7DD8'
BLUE_LIGHT = '#D6E4F0'
ORANGE     = '#E07B39'
ORANGE_LT  = '#FAEADB'
GREY       = '#6B7B8D'
GREY_LIGHT = '#E8ECF0'
TEXT       = '#1E2A38'
DIM        = '#8899AA'
BG         = '#FAFBFD'
WHITE      = '#FFFFFF'
GREEN_RES  = '#3DA06B'


def rbox(ax, x, y, w, h, label, sub=None, color=BLUE, fill=BLUE_LIGHT,
         fs=9, sub_fs=7, radius=0.012):
    """Draw a rounded rectangle with centred label and optional sub-label."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=color, linewidth=1.4, zorder=3
    )
    ax.add_patch(box)
    cy = y + h / 2
    if sub:
        cy = y + h * 0.62
        ax.text(x + w / 2, y + h * 0.32, sub,
                ha='center', va='center', fontsize=sub_fs,
                color=DIM, family='monospace', zorder=4)
    ax.text(x + w / 2, cy, label,
            ha='center', va='center', fontsize=fs,
            color=TEXT, weight='bold', zorder=4)


def small_box(ax, x, y, w, h, label, color=GREY, fill=GREY_LIGHT,
              fs=7.5, bold=False):
    """Small inner box (for block internals)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.006",
        facecolor=fill, edgecolor=color, linewidth=1.0, zorder=3
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center', fontsize=fs,
            color=TEXT, weight='bold' if bold else 'normal', zorder=4)


def varrow(ax, x, y1, y2, color=GREY, lw=1.3):
    """Vertical arrow from y1 down to y2."""
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=11), zorder=2)


def dim(ax, x, y, text, ha='left', fs=6.5):
    """Dimensional annotation in monospace."""
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs,
            color=DIM, family='monospace', style='italic', zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(10, 16.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Layout constants
    CX = 0.48                # centre-x of main column
    BW = 0.34                # main box width
    BH = 0.026               # standard box height
    GAP = 0.013              # vertical gap
    SGAP = 0.007             # small gap

    # ══════════════════════════════════════════════════════
    # TITLE BAR
    # ══════════════════════════════════════════════════════
    title_bg = FancyBboxPatch(
        (0.06, 0.958), 0.88, 0.032,
        boxstyle="round,pad=0,rounding_size=0.007",
        facecolor=TEXT, edgecolor='none', zorder=3)
    ax.add_patch(title_bg)
    ax.text(0.50, 0.9745, 'ArcSSM  —  S4D Bidirectional Architecture',
            ha='center', va='center', fontsize=13, color=WHITE,
            weight='bold', zorder=4)
    ax.text(0.50, 0.963,
            '359K params  ·  d_model = 128  ·  d_state = 64  ·  4 layers  ·  bidirectional',
            ha='center', va='center', fontsize=7, color='#AABBCC',
            family='monospace', zorder=4)

    y = 0.940

    # ══════════════════════════════════════════════════════
    # INPUT CHANNELS  (4 boxes side-by-side)
    # ══════════════════════════════════════════════════════
    y -= 0.038
    ch_w, ch_h = 0.15, 0.030
    ch_gap = 0.012
    total = 4 * ch_w + 3 * ch_gap
    x0 = CX - total / 2

    channels = ['I(t)', '|ΔI(t)|', 'TKEO(I)', 'RMS_slide(I)']
    for i, ch in enumerate(channels):
        xi = x0 + i * (ch_w + ch_gap)
        rbox(ax, xi, y, ch_w, ch_h, ch, color=BLUE, fill=BLUE_LIGHT,
             fs=8.5, radius=0.007)

    input_bot = y
    dim(ax, CX, y - 0.011, '(B, 4, 2048)', ha='center')

    # Merge lines → single arrow
    y_merge = y - 0.018
    for i in range(4):
        xi = x0 + i * (ch_w + ch_gap) + ch_w / 2
        ax.plot([xi, CX], [input_bot, y_merge], color=GREY, lw=0.7, zorder=2)
    varrow(ax, CX, y_merge, y_merge - 0.012)

    # ══════════════════════════════════════════════════════
    # CONV1D CHANNEL EMBEDDING
    # ══════════════════════════════════════════════════════
    y = y_merge - 0.012 - BH - SGAP
    rbox(ax, CX - BW/2, y, BW, BH + 0.006,
         'Conv1d  Embedding', sub='4 → 128 ch  |  k=7  |  stride=1',
         color=BLUE, fill=BLUE_LIGHT, fs=10, sub_fs=7)
    dim(ax, CX + BW/2 + 0.02, y + (BH + 0.006) / 2, '(B, 128, 2048)')
    conv_bot = y

    # GELU
    y -= (SGAP + BH * 0.65)
    varrow(ax, CX, conv_bot, y + BH * 0.65)
    small_box(ax, CX - 0.065, y, 0.13, BH * 0.65,
              'GELU', color=GREY, fill=GREY_LIGHT, fs=8, bold=True)

    # Transpose note
    y -= (SGAP + BH * 0.55)
    varrow(ax, CX, y + BH * 0.55 + SGAP, y + BH * 0.55)
    small_box(ax, CX - 0.10, y, 0.20, BH * 0.55,
              'transpose → (B, L, H)', color=GREY, fill=GREY_LIGHT, fs=6.5)
    dim(ax, CX + 0.10 + 0.02, y + BH * 0.275, '(B, 2048, 128)')

    # ══════════════════════════════════════════════════════
    # S4 BLOCK  (detailed, with ×4 bracket)
    # ══════════════════════════════════════════════════════
    s4_entry = y - GAP * 0.5

    # Block outer frame
    blk_h = 0.230
    blk_w = BW + 0.08
    blk_x = CX - blk_w / 2
    blk_top = s4_entry
    blk_bot = blk_top - blk_h

    block_bg = FancyBboxPatch(
        (blk_x, blk_bot), blk_w, blk_h,
        boxstyle="round,pad=0,rounding_size=0.010",
        facecolor=ORANGE_LT, edgecolor=ORANGE, linewidth=2.0, zorder=1)
    ax.add_patch(block_bg)

    varrow(ax, CX, s4_entry + GAP * 0.5, blk_top)

    # Block title
    ax.text(CX, blk_top - 0.014,
            'S4Block  (Pre-Norm Residual)',
            ha='center', va='center', fontsize=10, color=ORANGE,
            weight='bold', zorder=4)

    # ── Inner layers ──
    iw = 0.28           # inner box width
    ih = 0.020          # inner box height
    ig = 0.006          # inner gap

    iy = blk_top - 0.034

    # LayerNorm
    small_box(ax, CX - iw/2, iy, iw, ih,
              'LayerNorm(128)', color=GREY, fill=GREY_LIGHT, fs=8)

    # ── S4D Bidirectional sub-frame ──
    iy -= (ig * 2 + ih * 0.5)
    varrow(ax, CX, iy + ih * 0.5 + ig * 2, iy + ih * 0.5)

    s4d_h = 0.095
    s4d_w = iw + 0.05
    s4d_x = CX - s4d_w / 2
    s4d_top = iy + ih * 0.5
    s4d_bot = s4d_top - s4d_h

    s4d_bg = FancyBboxPatch(
        (s4d_x, s4d_bot), s4d_w, s4d_h,
        boxstyle="round,pad=0,rounding_size=0.007",
        facecolor=WHITE, edgecolor=ORANGE,
        linewidth=1.0, linestyle='--', zorder=2)
    ax.add_patch(s4d_bg)

    ax.text(CX, s4d_top - 0.010,
            'S4D  Bidirectional  (LTI)',
            ha='center', va='center', fontsize=8,
            color=ORANGE, weight='bold', zorder=4)
    ax.text(CX, s4d_top - 0.020,
            'H=128 channels  ·  N=64 complex states',
            ha='center', va='center', fontsize=5.5,
            color=DIM, family='monospace', zorder=4)

    # Two kernel boxes
    kw = 0.105
    kh = 0.020
    kg = 0.030
    ky = s4d_top - 0.042

    fwd_x = CX - kg / 2 - kw
    bwd_x = CX + kg / 2

    small_box(ax, fwd_x, ky, kw, kh,
              'Kernel  →', color=BLUE, fill=BLUE_LIGHT, fs=6.5, bold=True)
    small_box(ax, bwd_x, ky, kw, kh,
              '←  Kernel', color=BLUE, fill=BLUE_LIGHT, fs=6.5, bold=True)

    dim(ax, fwd_x + kw/2, ky - 0.009, 'FFT conv', ha='center', fs=5)
    dim(ax, bwd_x + kw/2, ky - 0.009, 'FFT conv (flip)', ha='center', fs=5)

    # Sum circle
    sum_y = ky - 0.022
    ax.plot([fwd_x + kw/2, CX], [ky, sum_y + 0.004],
            color=GREY, lw=0.8, zorder=2)
    ax.plot([bwd_x + kw/2, CX], [ky, sum_y + 0.004],
            color=GREY, lw=0.8, zorder=2)

    circle = plt.Circle((CX, sum_y), 0.005, facecolor=WHITE,
                         edgecolor=ORANGE, linewidth=1.2, zorder=3)
    ax.add_patch(circle)
    ax.text(CX, sum_y, '+', ha='center', va='center', fontsize=9,
            color=ORANGE, weight='bold', zorder=4)
    ax.text(CX + 0.025, sum_y, '+ D·u', ha='left', va='center',
            fontsize=5.5, color=DIM, weight='bold', zorder=4)

    s4d_out_y = s4d_bot

    # GELU (inside block)
    iy = s4d_out_y - ig
    varrow(ax, CX, s4d_out_y, iy)
    iy -= ih * 0.6
    small_box(ax, CX - 0.055, iy, 0.11, ih * 0.6,
              'GELU', color=GREY, fill=GREY_LIGHT, fs=7.5, bold=True)

    # Linear channel mixing
    iy -= (ig + ih * 0.6)
    varrow(ax, CX, iy + ih * 0.6 + ig, iy + ih * 0.6)
    small_box(ax, CX - iw/2, iy, iw, ih * 0.6,
              'Linear(128→128)  channel mix', color=GREY, fill=GREY_LIGHT, fs=7)

    # Dropout
    iy -= (ig * 0.5 + ih * 0.5)
    small_box(ax, CX - 0.06, iy, 0.12, ih * 0.5,
              'Dropout(0.1)', color=GREY, fill=GREY_LIGHT, fs=6)

    # Residual arrow (right side, curved)
    res_x = blk_x + blk_w - 0.012
    ax.annotate('', xy=(res_x, iy + ih * 0.25),
                xytext=(res_x, blk_top - 0.030),
                arrowprops=dict(arrowstyle='->', color=GREEN_RES, lw=1.4,
                                connectionstyle='arc3,rad=-0.30',
                                linestyle='--', mutation_scale=10), zorder=2)
    mid_res_y = (blk_top - 0.030 + iy + ih * 0.25) / 2
    ax.text(res_x + 0.014, mid_res_y, '+ x',
            ha='left', va='center', fontsize=7,
            color=GREEN_RES, weight='bold', zorder=4)

    # ×4 bracket (left side)
    bkt_x = blk_x - 0.020
    ax.plot([bkt_x, bkt_x], [blk_bot, blk_top], color=ORANGE, lw=2.2, zorder=2)
    ax.plot([bkt_x, bkt_x + 0.010], [blk_top, blk_top], color=ORANGE, lw=2.2, zorder=2)
    ax.plot([bkt_x, bkt_x + 0.010], [blk_bot, blk_bot], color=ORANGE, lw=2.2, zorder=2)
    ax.text(bkt_x - 0.008, (blk_top + blk_bot) / 2, '× 4',
            ha='right', va='center', fontsize=9, color=ORANGE,
            weight='bold', rotation=90, zorder=5)

    # Repetition dots
    y = blk_bot - GAP * 0.4
    for i in range(3):
        ax.plot(CX, y - i * 0.007, 'o', color=ORANGE, markersize=2.5, zorder=3)

    y = y - 3 * 0.007 - GAP

    # ══════════════════════════════════════════════════════
    # POST-BACKBONE  (more vertical breathing room)
    # ══════════════════════════════════════════════════════
    BG2 = GAP * 2.0   # bigger gap for this section

    # LayerNorm (final)
    varrow(ax, CX, y + GAP, y + BH)
    rbox(ax, CX - BW/2, y, BW, BH,
         'LayerNorm(128)', color=GREY, fill=GREY_LIGHT, fs=9)
    dim(ax, CX + BW/2 + 0.02, y + BH/2, '(B, 2048, 128)')

    # Global Average Pool
    y -= (BG2 + BH + 0.003)
    varrow(ax, CX, y + BH + BG2 + 0.003, y + BH)
    rbox(ax, CX - BW/2, y, BW, BH + 0.006,
         'Global Average Pool', sub='mean over time  (dim=1)',
         color=BLUE, fill=BLUE_LIGHT, fs=9.5, sub_fs=7)
    dim(ax, CX + BW/2 + 0.02, y + (BH + 0.006) / 2, '(B, 128)')

    # Linear → embedding
    y -= (BG2 + BH)
    varrow(ax, CX, y + BH + BG2, y + BH)
    rbox(ax, CX - BW/2, y, BW, BH,
         'Linear(128 → 128)', color=BLUE, fill=BLUE_LIGHT, fs=9)
    dim(ax, CX + BW/2 + 0.02, y + BH/2, '128-d embedding')

    # ══════════════════════════════════════════════════════
    # CLASSIFIER HEAD
    # ══════════════════════════════════════════════════════
    clf_top = y - BG2
    clf_h = 0.100
    clf_w = BW + 0.04
    clf_x = CX - clf_w / 2

    clf_bg = FancyBboxPatch(
        (clf_x, clf_top - clf_h), clf_w, clf_h,
        boxstyle="round,pad=0,rounding_size=0.009",
        facecolor=ORANGE_LT, edgecolor=ORANGE, linewidth=1.8, zorder=1)
    ax.add_patch(clf_bg)

    varrow(ax, CX, clf_top + BG2, clf_top)

    ax.text(CX, clf_top - 0.012, 'Classifier  (shallow)',
            ha='center', va='center', fontsize=9.5, color=ORANGE,
            weight='bold', zorder=4)

    # Inner layers
    iy = clf_top - 0.030
    small_box(ax, CX - iw/2, iy, iw, ih,
              'Linear(128 → 64)', color=ORANGE, fill=ORANGE_LT, fs=8)
    iy -= (ig + ih * 0.65)
    small_box(ax, CX - 0.055, iy, 0.11, ih * 0.65,
              'GELU', color=GREY, fill=GREY_LIGHT, fs=7.5, bold=True)
    iy -= (ig * 0.5 + ih * 0.55)
    small_box(ax, CX - 0.06, iy, 0.12, ih * 0.55,
              'Dropout(0.3)', color=GREY, fill=GREY_LIGHT, fs=6)
    iy -= (ig + ih * 0.65)
    small_box(ax, CX - iw/2, iy, iw, ih * 0.65,
              'Linear(64 → 1)', color=ORANGE, fill=ORANGE_LT, fs=8)

    # ── OUTPUT ──
    y = iy - GAP - BH * 0.7
    varrow(ax, CX, iy, y + BH * 0.7)
    rbox(ax, CX - 0.13, y, 0.26, BH * 0.7,
         'σ  →  Arc / Normal', color=ORANGE, fill=ORANGE_LT,
         fs=9, radius=0.007)

    # ══════════════════════════════════════════════════════
    # LEGEND (bottom)
    # ══════════════════════════════════════════════════════
    leg_y = 0.024
    eqs = [
        ('S4D-Lin init:',   'A = −½ + iπn   (learnable resonator bank)'),
        ('Discretisation:',  'ZOH:  Ā = exp(Δ·A),   B̄ = (Ā−1)/A'),
        ('Kernel:',          'K[h,l] = Σₙ Re(C·B̄·Āˡ)   via FFT   O(L log L)'),
        ('Bidirectional:',   'y = conv(u, K_fwd) + flip(conv(flip(u), K_bwd)) + D·u'),
    ]
    for i, (label, desc) in enumerate(eqs):
        yi = leg_y + (len(eqs) - 1 - i) * 0.013
        ax.text(0.08, yi, label, ha='left', va='center', fontsize=6.5,
                color=ORANGE, weight='bold', zorder=4)
        ax.text(0.24, yi, desc, ha='left', va='center', fontsize=6.5,
                color=DIM, family='monospace', zorder=4)

    # Run metrics badge
    ax.text(0.94, 0.018,
            'Best run: arcssm_single_20260728\n'
            'Acc 97.98%   F1 97.77%   Prec 99.86%\n'
            'Rec 95.77%   Spec 99.89%',
            ha='right', va='bottom', fontsize=5.5, color=DIM,
            family='monospace', zorder=4,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=GREY_LIGHT,
                      edgecolor=GREY, alpha=0.7))

    # ── Save ──
    out = 'arcssm_architecture_diagram.png'
    fig.savefig(out, dpi=200, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print(f"✓  {out}")

    out_pdf = out.replace('.png', '.pdf')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print(f"✓  {out_pdf}")
    plt.close(fig)


if __name__ == '__main__':
    main()
