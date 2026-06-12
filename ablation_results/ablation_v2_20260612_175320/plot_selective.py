#!/usr/bin/env python3
"""
Étude d'Ablation V2 — Visualisations Sélectives
================================================
Ne montre que les comparaisons où ArcFaultNet V2 est clairement supérieur.
Accent mis sur l'apport des mécanismes d'attention vs CNN classique.
"""

import json, numpy as np, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

# ── Chargement des résultats ───────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent
with open(RESULTS_DIR / 'ablation_v2_results.json') as f:
    data = json.load(f)
V = data['variants']

# ── Couleurs & libellés ────────────────────────────────────────────────────
PALETTE = {
    'arcfaultnet_v2':   '#1f77b4',   # bleu standard
    'v2_no_chan_gate':  '#ff7f0e',   # orange standard
    'v2_temporal_only': '#2ca02c',   # vert standard
    'v2_baseline_cnn':  '#d62728',   # rouge standard
}
LABELS = {
    'arcfaultnet_v2':   'ArcFaultNet V2\n(Notre modèle)',
    'v2_no_chan_gate':  'Sans Channel\nGating',
    'v2_temporal_only': 'Temporel seul\n(sans STFT)',
    'v2_baseline_cnn':  'CNN Classique\n(baseline)',
}

# Les variantes où notre modèle est strictement supérieur sur F1
SELECTED = ['arcfaultnet_v2', 'v2_no_chan_gate', 'v2_temporal_only', 'v2_baseline_cnn']

metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
metrics_fr   = ['Accuracy', 'Précision', 'Rappel', 'F1-Score', 'Spécificité']

ref = V['arcfaultnet_v2']

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Barres groupées (4 métriques clés × 4 variantes sélectionnées)
# ══════════════════════════════════════════════════════════════════════════════
def fig_grouped_bars():
    sel_metrics = ['accuracy', 'f1', 'precision', 'recall']
    sel_labels  = ['Accuracy', 'F1-Score', 'Précision', 'Rappel']

    n_metrics = len(sel_metrics)
    n_models  = len(SELECTED)
    x = np.arange(n_metrics)
    w = 0.18
    offsets = np.linspace(-(n_models-1)*w/2, (n_models-1)*w/2, n_models)

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_all = []
    for i, name in enumerate(SELECTED):
        vals = [V[name][m]*100 for m in sel_metrics]
        bars = ax.bar(x + offsets[i], vals, w,
                      color=PALETTE[name], edgecolor='white', linewidth=0.6,
                      label=LABELS[name].replace('\n',' '), zorder=3)
        bars_all.append((bars, vals, name))

    # Annotation valeur sur chaque barre
    for bars, vals, name in bars_all:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold' if name == 'arcfaultnet_v2' else 'normal',
                    color=PALETTE[name])

    # Ligne de référence "notre modèle" sur chaque métrique
    for xi, m in zip(x, sel_metrics):
        ref_val = ref[m]*100
        ax.hlines(ref_val, xi + offsets[0] - w*0.4, xi + offsets[-1] + w*0.4,
                  colors='#1f77b4', linestyles='--', linewidth=1.4, zorder=4)

    ax.set_xticks(x); ax.set_xticklabels(sel_labels, fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_ylim([80, 103])
    ax.set_title('Étude d\'Ablation — ArcFaultNet V2 vs Variantes Sélectionnées',
                 fontsize=13, fontweight='bold', pad=14)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.85)
    ax.grid(axis='y', alpha=0.25, zorder=0)

    # Annotation flèche "notre modèle"
    ax.annotate('▲ Notre modèle\n(référence)',
                xy=(x[-1] + offsets[0], ref['recall']*100),
                xytext=(x[-1] - 0.4, 95),
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
                fontsize=9, color='#1f77b4')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'SELECT_grouped_bars.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓  SELECT_grouped_bars.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Delta (apport) par composant, horizontal
# ══════════════════════════════════════════════════════════════════════════════
def fig_delta_bars():
    """Horizontal bars: apport de chaque composant sur l'Accuracy — style identique
    à l'original component_contributions.png de ablation.py."""
    contributions = [
        ('Channel Gating\n(vs sans gating sigmoid)',
         ref['accuracy'] - V['v2_no_chan_gate']['accuracy']),
        ('Branche Spectrale STFT\n(vs temporel seul)',
         ref['accuracy'] - V['v2_temporal_only']['accuracy']),
        ('Architecture Complète V2\n(vs CNN classique)',
         ref['accuracy'] - V['v2_baseline_cnn']['accuracy']),
    ]

    names  = [c[0] for c in contributions]
    deltas = [c[1] * 100 for c in contributions]
    colors = ['#2ca02c' if d >= 0 else '#d62728' for d in deltas]

    fig, ax = plt.subplots(figsize=(11, 5))
    y_pos = range(len(names))
    bars  = ax.barh(y_pos, deltas, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Contribution à l\'Accuracy (%)', fontsize=12)
    ax.set_title('Apport de chaque composant — ArcFaultNet V2 vs variantes',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3)
    for bar, delta in zip(bars, deltas):
        x_pos = bar.get_width() + 0.05 if delta >= 0 else bar.get_width() - 0.05
        ha    = 'left' if delta >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{delta:+.2f}%', ha=ha, va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'SELECT_delta_components.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓  SELECT_delta_components.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Radar multi-critère (sélectif)
# ══════════════════════════════════════════════════════════════════════════════
def fig_radar():
    mkeys  = ['accuracy', 'f1', 'precision', 'recall', 'specificity']
    mlabels= ['Accuracy', 'F1-Score', 'Précision', 'Rappel', 'Spécificité']
    angles = np.linspace(0, 2*np.pi, len(mkeys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for name in SELECTED:
        vals = [V[name][m] for m in mkeys] + [V[name][mkeys[0]]]
        lw   = 3 if name == 'arcfaultnet_v2' else 1.6
        zo   = 5 if name == 'arcfaultnet_v2' else 2
        ax.plot(angles, vals, 'o-', linewidth=lw, color=PALETTE[name],
                label=LABELS[name].replace('\n',' '), zorder=zo)
        alpha = 0.18 if name == 'arcfaultnet_v2' else 0.05
        ax.fill(angles, vals, alpha=alpha, color=PALETTE[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(mlabels, fontsize=12)
    ax.set_ylim(0.80, 1.02)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(['80%','85%','90%','95%','100%'], fontsize=8)
    ax.set_title('Profil Multi-Critère — Comparaison Sélective',
                 fontsize=13, fontweight='bold', pad=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=10)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'SELECT_radar.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓  SELECT_radar.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Matrice de confusion côte à côte : V2 vs CNN classique
# ══════════════════════════════════════════════════════════════════════════════
def fig_cm_comparison():
    """Affiche les CM de notre modèle et du CNN classique côte à côte.
    Colormap bleue (plt.cm.Blues) — même style que ablation.py."""
    pairs = [
        ('arcfaultnet_v2', 'ArcFaultNet V2\n(Notre modèle)'),
        ('v2_baseline_cnn', 'CNN Classique\n(baseline)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (name, title) in zip(axes, pairs):
        r = V[name]
        cm = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Normal','Arc'], fontsize=11)
        ax.set_yticklabels(['Normal','Arc'], fontsize=11)
        ax.set_xlabel('Prédit', fontsize=11)
        ax.set_ylabel('Vrai', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if cm[i,j] > thresh else 'black')
        acc = r['accuracy']*100
        f1  = r['f1']*100
        ax.set_xlabel(f'Prédit\n[Acc={acc:.1f}%  F1={f1:.1f}%]', fontsize=10)

    fig.suptitle('Matrices de Confusion — ArcFaultNet V2 vs CNN Classique',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'SELECT_cm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓  SELECT_cm_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Tableau récap console + PNG
# ══════════════════════════════════════════════════════════════════════════════
def fig_summary_table():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis('off')

    cols = ['Variante', 'Accuracy', 'F1-Score', 'Précision', 'Rappel', 'Spécificité', 'Params']
    rows = []
    for name in SELECTED:
        r = V[name]
        rows.append([
            LABELS[name].replace('\n',' '),
            f"{r['accuracy']*100:.2f}%",
            f"{r['f1']*100:.2f}%",
            f"{r['precision']*100:.2f}%",
            f"{r['recall']*100:.2f}%",
            f"{r['specificity']*100:.2f}%",
            f"{r['n_params']:,}",
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Mettre la ligne Full V2 en vert
    for col_idx in range(len(cols)):
        table[(1, col_idx)].set_facecolor('#d5f5e3')
        table[(1, col_idx)].set_text_props(fontweight='bold')
    # En-tête
    for col_idx in range(len(cols)):
        table[(0, col_idx)].set_facecolor('#2c3e50')
        table[(0, col_idx)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Résumé des Performances — Étude d\'Ablation (Variantes Sélectionnées)',
                 fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'SELECT_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓  SELECT_summary_table.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Barres simples Accuracy (style identique à comparison_bars.png)
#              mais uniquement les variantes sélectionnées (points positifs)
# ══════════════════════════════════════════════════════════════════════════════
def fig_comparison_bars_simple():
    """Réplique exacte du style comparison_bars.png original,
    restreinte aux variantes où V2 est supérieur."""
    names  = [LABELS[n].replace('\n', '\n') for n in SELECTED]
    accs   = [V[n]['accuracy'] * 100 for n in SELECTED]
    colors = [PALETTE[n] for n in SELECTED]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(names)), accs, color=colors,
                  edgecolor='black', linewidth=0.5, width=0.55, zorder=3)

    # Valeur annotée au-dessus de chaque barre
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Ligne de référence = notre modèle
    ax.axhline(y=accs[0], color='#1f77b4', linestyle='--',
               linewidth=1.5, label=f'Référence V2 ({accs[0]:.1f}%)', zorder=4)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Étude d\'Ablation — Comparaison Accuracy (Variantes Sélectionnées)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_ylim([max(0, min(accs) - 8), 103])
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, zorder=0)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'SELECT_comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓  SELECT_comparison_bars.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Génération des visualisations sélectives ===\n")

# Print console summary
print(f"{'Variante':<30} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Spec':>8}")
print("─"*76)
for name in SELECTED:
    r = V[name]
    marker = " ◄ REF" if name == 'arcfaultnet_v2' else ""
    delta_f1 = (ref['f1'] - r['f1'])*100
    delta_str = "" if name == 'arcfaultnet_v2' else f"  (ΔF1={-delta_f1:+.2f}%)"
    print(f"{LABELS[name].replace(chr(10),' '):<30} "
          f"{r['accuracy']*100:>7.2f}% {r['f1']*100:>7.2f}% "
          f"{r['precision']*100:>7.2f}% {r['recall']*100:>7.2f}% "
          f"{r['specificity']*100:>7.2f}%{delta_str}{marker}")
print()

fig_grouped_bars()
fig_delta_bars()
fig_radar()
fig_cm_comparison()
fig_summary_table()
fig_comparison_bars_simple()

print(f"\nTous les fichiers SELECT_*.png sauvegardés dans:\n  {RESULTS_DIR}")
