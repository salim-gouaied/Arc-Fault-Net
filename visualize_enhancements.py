#!/usr/bin/env python3
"""
Visualization: Impact of SE Blocks + Deep Classifier on Arc-FaultNet V2
========================================================================
Compares baseline (use_se=False, deep_classifier=False) vs enhanced
(use_se=True, deep_classifier=True) across multiple training seeds.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'text.color': '#222222',
    'axes.labelcolor': '#222222',
    'xtick.color': '#444444',
    'ytick.color': '#444444',
    'axes.edgecolor': '#cccccc',
    'grid.color': '#e6e6e6',
    'savefig.facecolor': '#ffffff',
    'savefig.edgecolor': '#ffffff',
})

RUNS_DIR = Path(__file__).parent / "runs"
OUT_DIR = Path(__file__).parent / "docs" / "enhancement_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Collect results ──────────────────────────────────────────────────
baseline_runs = []
enhanced_runs = []

for run_dir in sorted(RUNS_DIR.glob("arcfaultnet_v2_single_*")):
    rfile = run_dir / "results.json"
    if not rfile.exists():
        continue
    with open(rfile) as f:
        r = json.load(f)
    if r.get("test_accuracy") is None:
        continue

    entry = {
        "name": run_dir.name,
        "accuracy": r["test_accuracy"],
        "f1": r["test_f1"],
        "precision": r["test_precision"],
        "recall": r["test_recall"],
        "specificity": r.get("test_specificity", 0),
        "params": r.get("n_params", 0),
        "best_epoch": r.get("best_epoch", 0),
        "seed": r.get("seed", 0),
    }

    if r.get("use_se", False) and r.get("deep_classifier", False):
        enhanced_runs.append(entry)
    elif not r.get("use_se", False) and not r.get("deep_classifier", False):
        baseline_runs.append(entry)

print(f"Baseline runs: {len(baseline_runs)}, Enhanced runs: {len(enhanced_runs)}")

metrics = ["accuracy", "f1", "precision", "recall", "specificity"]
labels  = ["Accuracy", "F1-Score", "Precision", "Recall", "Specificity"]

base_vals = {m: [r[m] for r in baseline_runs] for m in metrics}
enh_vals  = {m: [r[m] for r in enhanced_runs] for m in metrics}

# Colors
C_BASE = "#C0392B"     
C_ENH  = "#1565C0"    

C_BASE_LIGHT = "#C0392B40"
C_ENH_LIGHT  = "#1565C040"

# ═════════════════════════════════════════════════════════════════════
# FIGURE 1 — Bar chart: Mean ± Std across seeds
# ═════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(metrics))
w = 0.32

base_means = [np.mean(base_vals[m]) for m in metrics]
base_stds  = [np.std(base_vals[m])  for m in metrics]
enh_means  = [np.mean(enh_vals[m])  for m in metrics]
enh_stds   = [np.std(enh_vals[m])   for m in metrics]

bars1 = ax.bar(x - w/2, base_means, w, yerr=base_stds, capsize=4,
               color=C_BASE, edgecolor='white', linewidth=0.5,
               error_kw=dict(ecolor='#444444', lw=1.2), label='Baseline (no SE, shallow head)', zorder=3)
bars2 = ax.bar(x + w/2, enh_means, w, yerr=enh_stds, capsize=4,
               color=C_ENH, edgecolor='white', linewidth=0.5,
               error_kw=dict(ecolor='#444444', lw=1.2), label='Enhanced (SE + Deep Classifier)', zorder=3)

# Value labels
for bar, val in zip(bars1, base_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, color=C_BASE, fontweight='bold')
for bar, val in zip(bars2, enh_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, color=C_ENH, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Score")
ax.set_ylim(0.90, 1.005)
ax.set_title("Arc-FaultNet V2 — Baseline vs Enhanced (Mean ± Std across seeds)",
             fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10, framealpha=0.3, edgecolor='#cccccc')
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_mean_comparison.png", dpi=200)
print(f"Saved fig1_mean_comparison.png")

# ═════════════════════════════════════════════════════════════════════
# FIGURE 2 — Box + Swarm: Distribution across seeds
# ═════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 6))

positions_b = np.arange(len(metrics)) * 2.5
positions_e = positions_b + 0.8

bp_base = ax.boxplot([base_vals[m] for m in metrics], positions=positions_b,
                      widths=0.55, patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor=C_BASE_LIGHT, edgecolor=C_BASE, lw=1.5),
                      whiskerprops=dict(color=C_BASE), capprops=dict(color=C_BASE),
                      medianprops=dict(color='#ffffff', lw=2))

bp_enh = ax.boxplot([enh_vals[m] for m in metrics], positions=positions_e,
                     widths=0.55, patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor=C_ENH_LIGHT, edgecolor=C_ENH, lw=1.5),
                     whiskerprops=dict(color=C_ENH), capprops=dict(color=C_ENH),
                     medianprops=dict(color='#ffffff', lw=2))

# Scatter individual points
for i, m in enumerate(metrics):
    jitter_b = np.random.normal(0, 0.06, len(base_vals[m]))
    jitter_e = np.random.normal(0, 0.06, len(enh_vals[m]))
    ax.scatter(positions_b[i] + jitter_b, base_vals[m], color=C_BASE, s=25, alpha=0.7, zorder=5, edgecolors='white', linewidths=0.3)
    ax.scatter(positions_e[i] + jitter_e, enh_vals[m], color=C_ENH, s=25, alpha=0.7, zorder=5, edgecolors='white', linewidths=0.3)

ax.set_xticks((positions_b + positions_e) / 2)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Score")
ax.set_ylim(0.83, 1.005)
ax.set_title("Metric Distribution — Baseline vs Enhanced (all seeds)",
             fontsize=15, fontweight='bold', pad=15)

legend_handles = [
    mpatches.Patch(facecolor=C_BASE_LIGHT, edgecolor=C_BASE, label=f'Baseline (n={len(baseline_runs)})'),
    mpatches.Patch(facecolor=C_ENH_LIGHT, edgecolor=C_ENH, label=f'Enhanced (n={len(enhanced_runs)})'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=10, framealpha=0.3, edgecolor='#cccccc')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_distribution_comparison.png", dpi=200)
print(f"Saved fig2_distribution_comparison.png")

# ═════════════════════════════════════════════════════════════════════
# FIGURE 3 — Stability: Coefficient of Variation
# ═════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))

cv_base = [np.std(base_vals[m]) / np.mean(base_vals[m]) * 100 for m in metrics]
cv_enh  = [np.std(enh_vals[m])  / np.mean(enh_vals[m])  * 100 for m in metrics]

x = np.arange(len(metrics))
bars1 = ax.bar(x - w/2, cv_base, w, color=C_BASE, edgecolor='white', label='Baseline', zorder=3)
bars2 = ax.bar(x + w/2, cv_enh, w, color=C_ENH, edgecolor='white', label='Enhanced', zorder=3)

for bar, val in zip(bars1, cv_base):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9, color=C_BASE, fontweight='bold')
for bar, val in zip(bars2, cv_enh):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9, color=C_ENH, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Coefficient of Variation (%)")
ax.set_title("Training Stability — Lower CV = More Consistent Across Seeds",
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=10, framealpha=0.3, edgecolor='#cccccc')
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_stability_cv.png", dpi=200)
print(f"Saved fig3_stability_cv.png")

# ═════════════════════════════════════════════════════════════════════
# FIGURE 4 — Radar Chart: Best model comparison
# ═════════════════════════════════════════════════════════════════════
best_base = max(baseline_runs, key=lambda r: r["f1"])
best_enh  = max(enhanced_runs, key=lambda r: r["f1"])

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

base_r = [best_base[m] for m in metrics] + [best_base[metrics[0]]]
enh_r  = [best_enh[m]  for m in metrics] + [best_enh[metrics[0]]]

ax.fill(angles, base_r, alpha=0.15, color=C_BASE)
ax.plot(angles, base_r, 'o-', color=C_BASE, lw=2, markersize=6, label='Best Baseline')
ax.fill(angles, enh_r, alpha=0.15, color=C_ENH)
ax.plot(angles, enh_r, 'o-', color=C_ENH, lw=2, markersize=6, label='Best Enhanced')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.95, 1.0)
ax.set_rticks([0.96, 0.97, 0.98, 0.99, 1.0])
ax.set_yticklabels(['0.96', '0.97', '0.98', '0.99', '1.00'], fontsize=8, color='#444444')
ax.set_title("Best Model Radar — Baseline vs Enhanced",
             fontsize=14, fontweight='bold', pad=25, color='#222222')
ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), fontsize=10, framealpha=0.3, edgecolor='#cccccc')
ax.set_facecolor('#f8f9fa')
ax.grid(color='#e6e6e6')

fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_radar_best.png", dpi=200)
print(f"Saved fig4_radar_best.png")

# ═════════════════════════════════════════════════════════════════════
# FIGURE 5 — Summary Table
# ═════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis('off')

header = ['Metric', 'Baseline\n(Mean ± Std)', 'Enhanced\n(Mean ± Std)', 'Δ (pp)', 'Improvement']
rows = []
for i, (m, l) in enumerate(zip(metrics, labels)):
    bm, bs = np.mean(base_vals[m]), np.std(base_vals[m])
    em, es = np.mean(enh_vals[m]), np.std(enh_vals[m])
    delta = (em - bm) * 100
    sign = "+" if delta >= 0 else ""
    imp = "✓" if delta > 0 else "—"
    rows.append([l, f'{bm:.4f} ± {bs:.4f}', f'{em:.4f} ± {es:.4f}', f'{sign}{delta:.2f}', imp])

# Add params row
rows.append(['Parameters', f'{baseline_runs[0]["params"]:,}', f'{enhanced_runs[0]["params"]:,}',
             f'+{enhanced_runs[0]["params"] - baseline_runs[0]["params"]:,}', f'+{(enhanced_runs[0]["params"]/baseline_runs[0]["params"]-1)*100:.1f}%'])

table = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

# Style
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if row == 0:
        cell.set_facecolor('#e6e6e6')
        cell.set_text_props(fontweight='bold', color='#222222')
    else:
        cell.set_facecolor('#ffffff')
        cell.set_text_props(color='#222222')
    if col == 3 and row > 0:
        val = rows[row-1][3]
        if val.startswith('+') and row <= len(metrics):
            cell.set_text_props(color='#2ea043', fontweight='bold')
    if col == 4 and row > 0:
        if rows[row-1][4] == '✓':
            cell.set_text_props(color='#2ea043', fontweight='bold')

ax.set_title("Enhancement Impact Summary — SE Blocks + Deep Classifier",
             fontsize=14, fontweight='bold', pad=15, color='#222222')

fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_summary_table.png", dpi=200, bbox_inches='tight')
print(f"Saved fig5_summary_table.png")

print(f"\n{'='*60}")
print(f"All figures saved to: {OUT_DIR}")
print(f"Best Baseline: {best_base['name']}  F1={best_base['f1']:.4f}")
print(f"Best Enhanced: {best_enh['name']}   F1={best_enh['f1']:.4f}")
print(f"{'='*60}")
