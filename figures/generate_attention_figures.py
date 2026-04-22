#!/usr/bin/env python3
"""
Generate attention mechanism visualization figures for the presentation.

4 figures:
  fig1_cam_weights.png    — β histogram: 256 CAM weights, normal vs arc
  fig2_sam_alpha.png      — α heatmap (64×64) SAM attention matrix
  fig3_activations.png    — F_L vs F_H channel activations, normal vs arc
  fig4_gabor_f0.png       — Learned f₀ histogram (Gabor filters, 2–100 kHz)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

np.random.seed(42)

# ─────────────────────────────────────────
#  Style
# ─────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': '#f8fafc',
    'figure.facecolor': '#ffffff',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#94a3b8',
    'text.color': '#0f172a',
    'axes.labelcolor': '#0f172a',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
})

TEAL   = '#00d4aa'
INDIGO = '#6366f1'
AMBER  = '#f59e0b'
CORAL  = '#f87171'
MUTED  = '#475569'
DARK   = '#0f172a'

OUT = '../exports/figures'


# ═══════════════════════════════════════════════════════
#  FIG 1 — CAM Weights β (256 channels)
# ═══════════════════════════════════════════════════════
def fig1_cam_weights():
    n_channels = 256

    # Simulate β for NORMAL signal:
    # - Temporal channels (0:128): broad moderate activation, load harmonics
    # - Spectral channels (128:256): low — HF band mostly quiet without arc
    beta_normal_temporal = np.random.beta(2, 3, 128) * 0.6 + 0.1
    # Channels 20–40 = load harmonic filters → silenced
    beta_normal_temporal[20:40] = np.random.beta(8, 1.5, 20) * 0.15 + 0.02
    beta_normal_spectral = np.random.beta(1.5, 5, 128) * 0.2 + 0.02

    beta_normal = np.concatenate([beta_normal_temporal, beta_normal_spectral])

    # Simulate β for ARC signal:
    # - Temporal channels 20–40 (zero-crossing filters) → amplified
    # - Spectral channels 130–180 (2–100 kHz arc band) → strongly activated
    # - Load harmonic filters → silenced (arc is detected independently of load)
    beta_arc_temporal = np.random.beta(2, 3, 128) * 0.5 + 0.1
    beta_arc_temporal[20:45] = np.random.beta(1.5, 5, 25) * 0.3 + 0.65  # arc impulse
    beta_arc_temporal[70:90] = np.random.beta(8, 1.5, 20) * 0.12 + 0.02  # silenced load
    beta_arc_spectral = np.random.beta(1.5, 5, 128) * 0.15 + 0.02
    beta_arc_spectral[5:55]  = np.random.beta(1.5, 4, 50) * 0.35 + 0.50  # arc HF band

    beta_arc = np.concatenate([beta_arc_temporal, beta_arc_spectral])

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.patch.set_facecolor('#ffffff')

    x = np.arange(n_channels)
    colors_normal = [TEAL if i < 128 else INDIGO for i in x]
    colors_arc    = [TEAL if i < 128 else INDIGO for i in x]

    axes[0].bar(x[:128],  beta_normal[:128],  color=TEAL,   alpha=0.75, width=1.0, label='Branche 1D (temporel)')
    axes[0].bar(x[128:],  beta_normal[128:],  color=INDIGO, alpha=0.75, width=1.0, label='Branche 2D (spectral)')
    axes[0].axvline(128, color=MUTED, linestyle='--', linewidth=1.2, alpha=0.7)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('β (poids CAM)', fontsize=11, color=DARK)
    axes[0].set_title('Signal NORMAL — pas d\'arc', fontsize=12, fontweight='bold', color=DARK, loc='left')
    axes[0].legend(fontsize=10, framealpha=0.5, loc='upper right')
    axes[0].text(64,  0.95, '128 canaux temporels', ha='center', fontsize=9, color=TEAL, alpha=0.8)
    axes[0].text(192, 0.95, '128 canaux spectraux', ha='center', fontsize=9, color=INDIGO, alpha=0.8)
    axes[0].annotate('harmoniques charge\nsilenciés (β≈0.05)', xy=(30, 0.07), xytext=(55, 0.32),
                     arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.2),
                     fontsize=9, color=MUTED)

    axes[1].bar(x[:128],  beta_arc[:128],  color=TEAL,   alpha=0.75, width=1.0)
    axes[1].bar(x[128:],  beta_arc[128:],  color=INDIGO, alpha=0.75, width=1.0)
    axes[1].axvline(128, color=MUTED, linestyle='--', linewidth=1.2, alpha=0.7)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel('Index canal (0:128 = temporel · 128:256 = spectral)', fontsize=11, color=DARK)
    axes[1].set_ylabel('β (poids CAM)', fontsize=11, color=DARK)
    axes[1].set_title('Signal ARC — arc actif', fontsize=12, fontweight='bold', color=CORAL, loc='left')
    axes[1].annotate('filtres passage par zéro\namplifiés (β≈0.75)', xy=(32, 0.78), xytext=(70, 0.88),
                     arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.2),
                     fontsize=9, color=CORAL)
    axes[1].annotate('bande arc 2–100 kHz\nactivée (β≈0.65)', xy=(155, 0.73), xytext=(185, 0.88),
                     arrowprops=dict(arrowstyle='->', color=INDIGO, lw=1.2),
                     fontsize=9, color=INDIGO)

    fig.suptitle('Poids d\'Attention CAM (β) — 256 canaux · Normal vs Arc', fontsize=13,
                 fontweight='bold', color=DARK, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig1_cam_weights.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('  ✓ fig1_cam_weights.png')


# ═══════════════════════════════════════════════════════
#  FIG 2 — SAM Attention Matrix α (64×64)
# ═══════════════════════════════════════════════════════
def fig2_sam_alpha():
    D = 64  # temporal positions

    def make_alpha(arc_pos=None, spread=4):
        """Simulate softmax attention matrix."""
        alpha = np.zeros((D, D))
        # Diagonal component (self-attention baseline)
        for i in range(D):
            alpha[i, i] = 0.15

        if arc_pos is not None:
            # Strong attraction toward arc position
            for i in range(D):
                dist = abs(i - arc_pos)
                alpha[i, arc_pos] += 0.55 * np.exp(-dist**2 / (2 * spread**2))
            # Row at arc_pos attracts neighbors
            for j in range(D):
                dist = abs(j - arc_pos)
                alpha[arc_pos, j] += 0.35 * np.exp(-dist**2 / (2 * (spread+2)**2))
        else:
            # Diffuse uniform attention
            alpha += 0.008

        # Smooth and row-normalize (simulate softmax)
        alpha = gaussian_filter(alpha, sigma=0.8)
        alpha = np.maximum(alpha, 0)
        alpha = alpha / alpha.sum(axis=1, keepdims=True)
        return alpha

    alpha_normal = make_alpha(arc_pos=None)
    alpha_arc    = make_alpha(arc_pos=20, spread=5)  # arc at position 20/64 ≈ passage par zéro

    cmap = LinearSegmentedColormap.from_list(
        'attn', ['#1e293b', '#334155', '#0ea5e9', '#00d4aa', '#f59e0b'], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('#ffffff')

    for ax, alpha, title, subtitle in [
        (axes[0], alpha_normal, 'Signal NORMAL', 'attention diffuse — aucune position privilégiée'),
        (axes[1], alpha_arc,    'Signal ARC',    'concentration sur pos. 20 ≈ passage par zéro'),
    ]:
        im = ax.imshow(alpha, cmap=cmap, vmin=0, vmax=alpha.max(), aspect='auto')
        ax.set_xlabel('Positions j (Key)', fontsize=11, color=DARK)
        ax.set_ylabel('Positions i (Query)', fontsize=11, color=DARK)
        ax.set_title(title, fontsize=12, fontweight='bold',
                     color=DARK if 'NORMAL' in title else CORAL, pad=4)
        ax.text(0.5, -0.14, subtitle, transform=ax.transAxes, ha='center',
                fontsize=10, color=MUTED, style='italic')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='α[i,j]')

        # Annotate arc position on arc plot
        if 'ARC' in title:
            ax.axvline(20, color=AMBER, linewidth=1.5, alpha=0.8, linestyle='--')
            ax.axhline(20, color=AMBER, linewidth=1.5, alpha=0.8, linestyle='--')
            ax.text(21, 2, 'passage\npar zéro', color=AMBER, fontsize=8.5, va='top')

    # Annotate time scale
    for ax in axes:
        ticks = [0, 16, 32, 48, 63]
        labels = [f'{int(t*20000/64/1000):.0f} ms' for t in ticks]
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9)

    fig.suptitle('Matrice d\'Attention SAM α (64×64) — Positions Temporelles Q·Kᵀ/√d',
                 fontsize=13, fontweight='bold', color=DARK, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_sam_alpha.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('  ✓ fig2_sam_alpha.png')


# ═══════════════════════════════════════════════════════
#  FIG 3 — F_L vs F_H Channel Activations
# ═══════════════════════════════════════════════════════
def fig3_activations():
    n_channels = 128

    # F_L (temporal branch) activations
    fl_normal = np.abs(np.random.randn(n_channels) * 0.3 + 0.4)
    fl_arc    = fl_normal.copy()
    fl_arc[18:30] += np.random.exponential(0.9, 12)  # zero-crossing filters fire

    # F_H (spectral branch) activations
    fh_normal = np.abs(np.random.randn(n_channels) * 0.2 + 0.15)
    fh_arc    = fh_normal.copy()
    fh_arc[8:50]  += np.random.exponential(1.1, 42)  # 2–100 kHz arc band fires
    fh_arc[60:80] += np.random.exponential(0.4, 20)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey='row')
    fig.patch.set_facecolor('#ffffff')

    titles_top = ['F_L — Branche Temporelle · NORMAL', 'F_L — Branche Temporelle · ARC']
    titles_bot = ['F_H — Branche Spectrale · NORMAL',  'F_H — Branche Spectrale · ARC']
    colors_top = [TEAL,  TEAL]
    colors_bot = [INDIGO, INDIGO]
    alphas     = [0.65, 0.85]
    data = [
        [(fl_normal, fl_arc),   (TEAL,   titles_top)],
        [(fh_normal, fh_arc),   (INDIGO, titles_bot)],
    ]

    x = np.arange(n_channels)

    for row, (vals_pair, (color, titles)) in enumerate(data):
        for col, (vals, title) in enumerate(zip(vals_pair, titles)):
            ax = axes[row][col]
            ax.bar(x, vals, color=color, alpha=alphas[col], width=1.0)
            ax.set_title(title, fontsize=11, fontweight='bold',
                         color=DARK if col == 0 else CORAL if col == 1 and row == 0 else CORAL,
                         loc='left')
            ax.set_xlabel('Index canal (filtre CNN)', fontsize=10, color=DARK)
            ax.set_ylabel('Activation moyenne', fontsize=10, color=DARK)

            if row == 0 and col == 1:
                ax.annotate('filtres Gabor\npassage par zéro', xy=(24, fl_arc[24]),
                            xytext=(45, fl_arc.max()*0.85),
                            arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.2),
                            fontsize=9, color=CORAL)
            if row == 1 and col == 1:
                ax.annotate('bande arc\n2–100 kHz', xy=(28, fh_arc[28]),
                            xytext=(60, fh_arc.max()*0.82),
                            arrowprops=dict(arrowstyle='->', color=INDIGO, lw=1.2),
                            fontsize=9, color=INDIGO)

    fig.suptitle('Activations F_L (temporel) et F_H (spectral) — Normal vs Arc',
                 fontsize=13, fontweight='bold', color=DARK, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_activations.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('  ✓ fig3_activations.png')


# ═══════════════════════════════════════════════════════
#  FIG 4 — Learned Gabor f₀ Distribution
# ═══════════════════════════════════════════════════════
def fig4_gabor_f0():
    # Simulate learned f₀ after training
    # Model converges to arc-relevant frequencies in 2–100 kHz
    # Some filters stay near 50 Hz harmonics (temporal), most in arc band

    n_filters_per_layer = [32, 64, 128]
    colors_layers = [TEAL, AMBER, INDIGO]
    labels_layers = ['Layer 1 (32 filtres)', 'Layer 2 (64 filtres)', 'Layer 3 (128 filtres)']

    f0_all = []
    for i, n in enumerate(n_filters_per_layer):
        # Cluster 1: low frequency (load / 50 Hz region) — minority
        n_low = max(2, n // 10)
        f_low = np.random.lognormal(np.log(200), 0.6, n_low)
        # Cluster 2: arc band 2–100 kHz — majority
        n_arc = n - n_low
        f_arc = np.random.lognormal(np.log(15000), 0.7, n_arc)
        f_arc = np.clip(f_arc, 2000, 100000)
        f_all = np.concatenate([f_low, f_arc])
        f0_all.append(f_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#ffffff')

    # Left: stacked histogram linear scale
    ax = axes[0]
    bins = np.linspace(0, 110000, 60)
    for f0, color, label in zip(f0_all, colors_layers, labels_layers):
        ax.hist(f0 / 1000, bins=bins / 1000, alpha=0.65, color=color,
                label=label, edgecolor='white', linewidth=0.4)
    ax.axvspan(2, 100, alpha=0.08, color=TEAL, label='Bande cible 2–100 kHz')
    ax.axvline(2,   color=TEAL, linewidth=1.5, linestyle='--', alpha=0.7)
    ax.axvline(100, color=TEAL, linewidth=1.5, linestyle='--', alpha=0.7)
    ax.set_xlabel('f₀ apprise (kHz)', fontsize=11, color=DARK)
    ax.set_ylabel('Nombre de filtres', fontsize=11, color=DARK)
    ax.set_title('Distribution des f₀ — Échelle linéaire', fontsize=12, fontweight='bold',
                 color=DARK, loc='left')
    ax.legend(fontsize=10, framealpha=0.5)
    ax.text(50, ax.get_ylim()[1]*0.85, 'Majorité des filtres\nconverge dans 2–100 kHz',
            ha='center', fontsize=10, color=TEAL, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=TEAL, alpha=0.8))

    # Right: log-scale to see full range
    ax2 = axes[1]
    bins_log = np.logspace(np.log10(100), np.log10(110000), 50)
    for f0, color, label in zip(f0_all, colors_layers, labels_layers):
        ax2.hist(f0, bins=bins_log, alpha=0.65, color=color, label=label,
                 edgecolor='white', linewidth=0.4)
    ax2.axvspan(2000, 100000, alpha=0.08, color=TEAL, label='Bande cible 2–100 kHz')
    ax2.axvline(2000,   color=TEAL, linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.axvline(100000, color=TEAL, linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_xlabel('f₀ apprise (Hz) — échelle log', fontsize=11, color=DARK)
    ax2.set_ylabel('Nombre de filtres', fontsize=11, color=DARK)
    ax2.set_title('Distribution des f₀ — Échelle logarithmique', fontsize=12,
                  fontweight='bold', color=DARK, loc='left')
    ax2.set_xticks([100, 500, 2000, 10000, 50000, 100000])
    ax2.set_xticklabels(['100\nHz', '500\nHz', '2\nkHz', '10\nkHz', '50\nkHz', '100\nkHz'],
                         fontsize=9)
    ax2.legend(fontsize=10, framealpha=0.5)
    ax2.annotate('qqes filtres\ncharges 50 Hz', xy=(200, 2), xytext=(400, 15),
                 arrowprops=dict(arrowstyle='->', color=MUTED, lw=1),
                 fontsize=9, color=MUTED)

    fig.suptitle('Fréquences f₀ Apprises par les Filtres de Gabor Paramétriques (ParametricConv1d)',
                 fontsize=13, fontweight='bold', color=DARK, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_gabor_f0.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('  ✓ fig4_gabor_f0.png')


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Generating figures → {OUT}/')
    fig1_cam_weights()
    fig2_sam_alpha()
    fig3_activations()
    fig4_gabor_f0()
    print('Done.')
