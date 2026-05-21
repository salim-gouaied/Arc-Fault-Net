#!/usr/bin/env python3
"""
ARC-FAULTNET — Evaluation Script (adapted for random-split training)
=====================================================================
Comprehensive evaluation with metrics, confusion matrix, and visualizations.

Features:
  - Overall metrics (accuracy, F1, precision, recall, AUC-ROC, etc.)
  - Confusion matrix
  - ROC curve and Precision-Recall curve
  - **False-negative analysis**: identifies arc samples that the model
    misclassified as normal, traces them back to their source experiment
    via metadata.csv, and produces signal plots + a CSV report.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import random
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

from dataset import ArcFaultDataset
from model import get_model


# ═══════════════════════════════════════════════════════
#  REPRODUCIBILITY  (must match train.py)
# ═══════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════
#  EVALUATION METRICS
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataset: ArcFaultDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions for a subset of the dataset.

    Returns:
        labels: Ground truth labels
        probs:  Predicted probabilities
    """
    model.eval()

    all_labels = []
    all_probs = []

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]

        x_1d_batch = []
        x_2d_batch = []
        labels_batch = []

        for idx in batch_indices:
            x_1d, x_2d, label, _ = dataset[idx]
            x_1d_batch.append(x_1d)
            x_2d_batch.append(x_2d)
            labels_batch.append(label)

        x_1d = torch.stack(x_1d_batch).to(device)
        x_2d = torch.stack(x_2d_batch).to(device)

        logits = model(x_1d, x_2d)
        probs = torch.sigmoid(logits)

        all_labels.extend([l.item() for l in labels_batch])
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute comprehensive metrics."""
    preds = (probs > threshold).astype(int)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    # AUC
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Average precision
    ap = average_precision_score(labels, probs)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'auc_roc': float(roc_auc),
        'average_precision': float(ap),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


# ═══════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot a detailed, publication-quality confusion matrix with:
      - Raw counts
      - Row-wise percentages (what % of actual class was predicted as X)
      - Cell-level semantic labels (TN, FP, FN, TP)
      - Color-coded: correct predictions in green tones, errors in red tones
    """
    cm = confusion_matrix(labels, preds)
    # Row-normalised version (percentages)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / (row_sums + 1e-8) * 100

    total = cm.sum()
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Semantic labels for each cell
    cell_labels = np.array([['TN', 'FP'], ['FN', 'TP']])
    cell_descriptions = np.array([
        ['Normal → Normal\n(Specificity)', 'Normal → Arc\n(False Alarm)'],
        ['Arc → Normal\n(Missed Arc)',    'Arc → Arc\n(Sensitivity)']
    ])

    # Custom diverging colormap: red for errors, green for correct
    # Build a 2x2 color array
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a custom color matrix: green for diagonal (correct), red for off-diagonal (errors)
    # Intensity proportional to count
    color_matrix = np.zeros((2, 2, 4))  # RGBA
    max_val = cm.max()
    for i in range(2):
        for j in range(2):
            intensity = 0.3 + 0.7 * (cm[i, j] / (max_val + 1e-8))
            if i == j:  # correct predictions — green shades
                color_matrix[i, j] = [0.18, 0.55 * intensity + 0.2, 0.22, 0.85]
            else:       # errors — red/orange shades
                color_matrix[i, j] = [0.85 * intensity + 0.15, 0.15, 0.15, 0.75]

    # Draw colored cells
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  facecolor=color_matrix[i, j],
                                  edgecolor='white', linewidth=3)
            ax.add_patch(rect)

    # Add rich text annotations
    for i in range(2):
        for j in range(2):
            # Main count
            ax.text(j, i - 0.18, f"{cm[i, j]}",
                    ha='center', va='center', fontsize=28,
                    fontweight='bold', color='white')
            # Percentage
            ax.text(j, i + 0.1, f"({cm_pct[i, j]:.1f}%)",
                    ha='center', va='center', fontsize=16,
                    color='white', alpha=0.95)
            # Semantic label
            ax.text(j, i + 0.32, f"{cell_labels[i, j]}",
                    ha='center', va='center', fontsize=13,
                    fontweight='bold', color='white', alpha=0.8,
                    fontstyle='italic')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Arc'], fontsize=14, fontweight='bold')
    ax.set_yticklabels(['Normal', 'Arc'], fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)

    # Add summary metrics below the matrix
    accuracy = (tp + tn) / total * 100
    precision_val = tp / (tp + fp + 1e-8) * 100
    recall_val = tp / (tp + fn + 1e-8) * 100
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-8)
    specificity_val = tn / (tn + fp + 1e-8) * 100

    summary_text = (
        f"Accuracy: {accuracy:.1f}%   |   "
        f"Precision: {precision_val:.1f}%   |   "
        f"Recall: {recall_val:.1f}%   |   "
        f"F1: {f1_val:.1f}%   |   "
        f"Specificity: {specificity_val:.1f}%"
    )
    ax.text(0.5, 2.05, summary_text,
            ha='center', va='center', fontsize=11,
            transform=ax.get_yaxis_transform(),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50',
                      edgecolor='#34495e', alpha=0.9),
            color='white', fontweight='bold')

    # Add cell descriptions on the sides
    ax.text(-0.85, 0, 'Actual\nNormal', ha='center', va='center',
            fontsize=10, color='#555', fontstyle='italic')
    ax.text(-0.85, 1, 'Actual\nArc', ha='center', va='center',
            fontsize=10, color='#555', fontstyle='italic')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve"
):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve"
):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {ap:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ═══════════════════════════════════════════════════════
#  TRAINING CURVES
# ═══════════════════════════════════════════════════════

def _make_single_curve(epochs, train_data, val_data, best_epoch,
                       title, ylabel, filename, output_dir,
                       log_scale=False, pct=False, fill=False):
    """Helper: one standalone metric plot with train+val+best epoch."""
    colors = {'train': '#2196F3', 'val': '#FF5722', 'best': '#4CAF50'}
    scale = 100 if pct else 1
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')

    if train_data is not None:
        td = [v * scale for v in train_data]
        ax.plot(epochs[:len(td)], td, color=colors['train'],
                linewidth=2, label=f'Train {ylabel}', marker='o', markersize=3)
        if fill:
            ax.fill_between(epochs[:len(td)], td, alpha=0.08, color=colors['train'])
    if val_data is not None:
        vd = [v * scale for v in val_data]
        ax.plot(epochs[:len(vd)], vd, color=colors['val'],
                linewidth=2, label=f'Val {ylabel}', marker='s', markersize=3)
        if fill:
            ax.fill_between(epochs[:len(vd)], vd, alpha=0.08, color=colors['val'])
        if best_epoch is not None and best_epoch < len(vd):
            best_val = vd[best_epoch]
            ax.axvline(x=best_epoch + 1, color=colors['best'],
                       linestyle='--', alpha=0.7, label=f'Best epoch {best_epoch+1}')
            label_txt = f'{best_val:.1f}%' if pct else f'{best_val:.4f}'
            ax.annotate(label_txt, xy=(best_epoch + 1, best_val),
                        xytext=(best_epoch + 3, best_val),
                        fontsize=11, fontweight='bold', color=colors['best'],
                        arrowprops=dict(arrowstyle='->', color=colors['best']))
    if log_scale:
        ax.set_yscale('log')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()


def plot_training_curves(history_path: Path, output_dir: Path):
    """
    Plot training & validation metrics over epochs.

    Generates individual PNGs per metric:
      - loss_curve.png, accuracy_curve.png, f1_curve.png,
        precision_curve.png, recall_curve.png, lr_schedule.png
    Plus the combined overview: training_curves.png
    """
    with open(history_path) as f:
        history = json.load(f)

    epochs = list(range(1, len(history['train_loss']) + 1))
    best_epoch = history.get('best_epoch', None)

    # ── Individual metric plots ──────────────────────────────────
    _make_single_curve(epochs, history['train_loss'], history['val_loss'],
                       best_epoch, 'Loss Over Epochs (log scale)', 'Loss',
                       'loss_curve.png', output_dir, log_scale=True, fill=True)
    print(f"  Saved → loss_curve.png")

    _make_single_curve(epochs, history.get('train_acc'), history.get('val_acc'),
                       best_epoch, 'Accuracy Over Epochs', 'Accuracy (%)',
                       'accuracy_curve.png', output_dir, pct=True, fill=True)
    print(f"  Saved → accuracy_curve.png")

    if 'val_f1' in history:
        _make_single_curve(epochs, None, history['val_f1'],
                           best_epoch, 'F1 Score Over Epochs', 'F1 (%)',
                           'f1_curve.png', output_dir, pct=True, fill=True)
        print(f"  Saved → f1_curve.png")

    if 'val_precision' in history:
        _make_single_curve(epochs, None, history['val_precision'],
                           best_epoch, 'Precision Over Epochs', 'Precision (%)',
                           'precision_curve.png', output_dir, pct=True, fill=True)
        print(f"  Saved → precision_curve.png")

    if 'val_recall' in history:
        _make_single_curve(epochs, None, history['val_recall'],
                           best_epoch, 'Recall Over Epochs', 'Recall (%)',
                           'recall_curve.png', output_dir, pct=True, fill=True)
        print(f"  Saved → recall_curve.png")

    if 'lr' in history:
        colors_lr = '#9C27B0'
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#fafafa')
        ax.plot(epochs[:len(history['lr'])], history['lr'], color=colors_lr,
                linewidth=2, label='Learning Rate', marker='D', markersize=4)
        ax.fill_between(epochs[:len(history['lr'])], history['lr'],
                        alpha=0.1, color=colors_lr)
        ax.set_yscale('log')
        ax.set_title('Learning Rate Schedule', fontsize=15, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / 'lr_schedule.png', dpi=200,
                    bbox_inches='tight', facecolor='#fafafa')
        plt.close()
        print(f"  Saved → lr_schedule.png")

    # ── Combined overview (kept for convenience) ─────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#fafafa')
    colors = {'train': '#2196F3', 'val': '#FF5722', 'best': '#4CAF50'}

    def style_ax(ax, title, ylabel):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=10, framealpha=0.9)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color=colors['train'], lw=2, label='Train', marker='o', ms=2)
    ax.plot(epochs, history['val_loss'], color=colors['val'], lw=2, label='Val', marker='s', ms=2)
    if best_epoch is not None:
        ax.axvline(x=best_epoch+1, color=colors['best'], ls='--', alpha=.7, label=f'Best ({best_epoch+1})')
    ax.set_yscale('log')
    style_ax(ax, 'Loss (log)', 'Loss')

    ax = axes[0, 1]
    ax.plot(epochs, [a*100 for a in history['train_acc']], color=colors['train'], lw=2, label='Train', marker='o', ms=2)
    ax.plot(epochs, [a*100 for a in history['val_acc']], color=colors['val'], lw=2, label='Val', marker='s', ms=2)
    if best_epoch is not None:
        ax.axvline(x=best_epoch+1, color=colors['best'], ls='--', alpha=.7)
    style_ax(ax, 'Accuracy', 'Accuracy (%)')

    ax = axes[1, 0]
    if 'val_f1' in history:
        ax.plot(epochs, [f*100 for f in history['val_f1']], color=colors['val'], lw=2, label='Val F1', marker='s', ms=2)
        if best_epoch is not None:
            ax.axvline(x=best_epoch+1, color=colors['best'], ls='--', alpha=.7)
    style_ax(ax, 'F1 Score', 'F1 (%)')

    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(epochs[:len(history['lr'])], history['lr'], color='#9C27B0', lw=2, label='LR', marker='D', ms=2)
        ax.set_yscale('log')
    style_ax(ax, 'Learning Rate', 'LR')

    fig.suptitle('Training History — Overview', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved → training_curves.png")


# ═══════════════════════════════════════════════════════
#  FALSE-NEGATIVE ANALYSIS
# ═══════════════════════════════════════════════════════

def analyse_false_negatives(
    dataset: ArcFaultDataset,
    test_indices: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    threshold: float = 0.5,
    n_plot: int = 12
):
    """
    Identify false negatives (arc samples predicted as normal) and
    trace them back to their source experiment via metadata.csv.

    Generates:
      - false_negatives.csv   : detailed table of every FN
      - fn_by_experiment.csv  : aggregated counts per experiment
      - fn_signals_*.png      : signal plots for up to n_plot FNs
    """
    preds = (probs > threshold).astype(int)

    # Mask for false negatives within the test set
    fn_mask = (labels == 1) & (preds == 0)
    fn_local = np.where(fn_mask)[0]          # positions inside test_indices
    fn_global = test_indices[fn_local]        # positions inside full dataset

    print(f"\n  False Negatives: {len(fn_global)} / {int(labels.sum())} arc samples "
          f"in test set ({100*len(fn_global)/(labels.sum()+1e-8):.1f}%)")

    if len(fn_global) == 0:
        print("  No false negatives — nothing to analyse.")
        return

    # ── Build detailed FN table ──────────────────────────────────
    rows = []
    for local_i, global_i in zip(fn_local, fn_global):
        meta_row = metadata.iloc[global_i]
        rows.append({
            'dataset_index': int(global_i),
            'prob': float(probs[local_i]),
            'source_dir': meta_row.get('source_dir', ''),
            'exp_id': meta_row.get('exp_id', ''),
            'file_num': meta_row.get('file_num', ''),
            'alt_index': meta_row.get('alt_index', ''),
            'arc_ratio': meta_row.get('arc_ratio', ''),
            'start_sample': meta_row.get('start_sample', ''),
            'end_sample': meta_row.get('end_sample', ''),
        })

    fn_df = pd.DataFrame(rows).sort_values('prob', ascending=True)
    fn_df.to_csv(output_dir / 'false_negatives.csv', index=False)
    print(f"  Saved → false_negatives.csv ({len(fn_df)} rows)")

    # ── Aggregate by experiment ──────────────────────────────────
    fn_by_exp = fn_df.groupby(['source_dir', 'exp_id']).agg(
        count=('dataset_index', 'size'),
        mean_prob=('prob', 'mean'),
        min_prob=('prob', 'min'),
        max_prob=('prob', 'max'),
    ).sort_values('count', ascending=False).reset_index()

    fn_by_exp.to_csv(output_dir / 'fn_by_experiment.csv', index=False)
    print(f"  Saved → fn_by_experiment.csv ({len(fn_by_exp)} experiments)")

    # Print top offenders
    print(f"\n  Top experiments contributing FN:")
    for _, row in fn_by_exp.head(10).iterrows():
        print(f"    {row['source_dir']}/{row['exp_id']}  "
              f"FN={int(row['count'])}  "
              f"mean_prob={row['mean_prob']:.3f}")

    # ── Signal plots for worst FNs ───────────────────────────────
    n_plot = min(n_plot, len(fn_df))
    worst_fn = fn_df.head(n_plot)     # lowest probability = most confident errors

    channel_names = ['V_ligne', 'I(t)']

    for page_start in range(0, n_plot, 4):
        page_end = min(page_start + 4, n_plot)
        n_this = page_end - page_start
        fig, axes = plt.subplots(n_this, 2, figsize=(14, 3.5 * n_this))
        if n_this == 1:
            axes = axes.reshape(1, -1)

        for row_i, (_, fn_row) in enumerate(worst_fn.iloc[page_start:page_end].iterrows()):
            idx = int(fn_row['dataset_index'])
            x_1d, _, label, _ = dataset[idx]

            for c in range(2):
                ax = axes[row_i, c]
                signal = x_1d[c].numpy()
                ax.plot(signal, linewidth=0.5, color='steelblue')
                ax.set_title(f"{channel_names[c]}", fontsize=10)
                ax.set_xlim([0, len(signal)])

                if c == 0:
                    ax.set_ylabel(
                        f"idx={idx}  p={fn_row['prob']:.3f}\n"
                        f"{fn_row['source_dir']}\n"
                        f"{fn_row['exp_id']} / file {fn_row['file_num']}",
                        fontsize=8, color='red'
                    )

        fig.suptitle(f"False Negatives (arc predicted as normal) — "
                     f"page {page_start//4 + 1}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'fn_signals_page{page_start//4 + 1}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved → fn_signals_page*.png ({n_plot} signals plotted)")


# ═══════════════════════════════════════════════════════
#  FALSE-POSITIVE ANALYSIS (bonus)
# ═══════════════════════════════════════════════════════

def analyse_false_positives(
    dataset: ArcFaultDataset,
    test_indices: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    threshold: float = 0.5,
    n_plot: int = 16
):
    """CSV report + signal plots for false positives (normal predicted as arc)."""
    preds = (probs > threshold).astype(int)
    fp_mask = (labels == 0) & (preds == 1)
    fp_local = np.where(fp_mask)[0]
    fp_global = test_indices[fp_local]

    print(f"  False Positives: {len(fp_global)} / {int((labels==0).sum())} normal samples "
          f"in test set ({100*len(fp_global)/((labels==0).sum()+1e-8):.1f}%)")

    if len(fp_global) == 0:
        return

    rows = []
    for local_i, global_i in zip(fp_local, fp_global):
        meta_row = metadata.iloc[global_i]
        rows.append({
            'dataset_index': int(global_i),
            'prob': float(probs[local_i]),
            'source_dir': meta_row.get('source_dir', ''),
            'exp_id': meta_row.get('exp_id', ''),
            'file_num': meta_row.get('file_num', ''),
            'alt_index': meta_row.get('alt_index', ''),
            'arc_ratio': meta_row.get('arc_ratio', ''),
        })

    fp_df = pd.DataFrame(rows).sort_values('prob', ascending=False)
    fp_df.to_csv(output_dir / 'false_positives.csv', index=False)
    print(f"  Saved → false_positives.csv ({len(fp_df)} rows)")

    # ── Signal plots for most-confident FPs ──────────────────────
    n_plot = min(n_plot, len(fp_df))
    worst_fp = fp_df.head(n_plot)   # highest prob = most confidently wrong

    channel_names = ['V_ligne', 'I(t)']

    for page_start in range(0, n_plot, 4):
        page_end = min(page_start + 4, n_plot)
        n_this = page_end - page_start
        fig, axes = plt.subplots(n_this, 2, figsize=(14, 3.5 * n_this))
        if n_this == 1:
            axes = axes.reshape(1, -1)

        for row_i, (_, fp_row) in enumerate(worst_fp.iloc[page_start:page_end].iterrows()):
            idx = int(fp_row['dataset_index'])
            x_1d, _, label, _ = dataset[idx]

            for c in range(2):
                ax = axes[row_i, c]
                signal = x_1d[c].numpy()
                ax.plot(signal, linewidth=0.5, color='darkorange')
                ax.set_title(f"{channel_names[c]}", fontsize=10)
                ax.set_xlim([0, len(signal)])

                if c == 0:
                    ax.set_ylabel(
                        f"idx={idx}  p={fp_row['prob']:.3f}\n"
                        f"{fp_row['source_dir']}\n"
                        f"{fp_row['exp_id']} / file {fp_row['file_num']}",
                        fontsize=8, color='darkorange'
                    )

        fig.suptitle(f"False Positives (normal predicted as arc) — "
                     f"page {page_start//4 + 1}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'fp_signals_page{page_start//4 + 1}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved → fp_signals_page*.png ({n_plot} signals plotted)")


# ═══════════════════════════════════════════════════════
#  TRUE-POSITIVE ANALYSIS
# ═══════════════════════════════════════════════════════

def analyse_true_positives(
    dataset: ArcFaultDataset,
    test_indices: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    threshold: float = 0.5,
    n_plot: int = 16
):
    """
    Identify true positives (arc samples correctly predicted as arc) and
    trace them back to their source experiment via metadata.csv.

    Generates:
      - true_positives.csv    : detailed table of every TP
      - tp_by_experiment.csv  : aggregated counts per experiment
      - tp_signals_*.png      : signal plots for a sample of TPs
    """
    preds = (probs > threshold).astype(int)

    # Mask for true positives within the test set
    tp_mask = (labels == 1) & (preds == 1)
    tp_local = np.where(tp_mask)[0]          # positions inside test_indices
    tp_global = test_indices[tp_local]        # positions inside full dataset

    print(f"\n  True Positives: {len(tp_global)} / {int(labels.sum())} arc samples "
          f"in test set ({100*len(tp_global)/(labels.sum()+1e-8):.1f}%)")

    if len(tp_global) == 0:
        print("  No true positives — nothing to analyse.")
        return

    # ── Build detailed TP table ──────────────────────────────────
    rows = []
    for local_i, global_i in zip(tp_local, tp_global):
        meta_row = metadata.iloc[global_i]
        rows.append({
            'dataset_index': int(global_i),
            'prob': float(probs[local_i]),
            'source_dir': meta_row.get('source_dir', ''),
            'exp_id': meta_row.get('exp_id', ''),
            'file_num': meta_row.get('file_num', ''),
            'alt_index': meta_row.get('alt_index', ''),
            'arc_ratio': meta_row.get('arc_ratio', ''),
            'start_sample': meta_row.get('start_sample', ''),
            'end_sample': meta_row.get('end_sample', ''),
        })

    tp_df = pd.DataFrame(rows).sort_values('prob', ascending=False)
    tp_df.to_csv(output_dir / 'true_positives.csv', index=False)
    print(f"  Saved → true_positives.csv ({len(tp_df)} rows)")

    # ── Aggregate by experiment ──────────────────────────────────
    tp_by_exp = tp_df.groupby(['source_dir', 'exp_id']).agg(
        count=('dataset_index', 'size'),
        mean_prob=('prob', 'mean'),
        min_prob=('prob', 'min'),
        max_prob=('prob', 'max'),
    ).sort_values('count', ascending=False).reset_index()

    tp_by_exp.to_csv(output_dir / 'tp_by_experiment.csv', index=False)
    print(f"  Saved → tp_by_experiment.csv ({len(tp_by_exp)} experiments)")

    # Print top contributors
    print(f"\n  Top experiments contributing TP:")
    for _, row in tp_by_exp.head(10).iterrows():
        print(f"    {row['source_dir']}/{row['exp_id']}  "
              f"TP={int(row['count'])}  "
              f"mean_prob={row['mean_prob']:.3f}")

    # ── Signal plots for most-confident TPs ──────────────────────
    n_plot = min(n_plot, len(tp_df))
    best_tp = tp_df.head(n_plot)     # highest probability = most confident correct predictions

    channel_names = ['V_ligne', 'I(t)']

    for page_start in range(0, n_plot, 4):
        page_end = min(page_start + 4, n_plot)
        n_this = page_end - page_start
        fig, axes = plt.subplots(n_this, 2, figsize=(14, 3.5 * n_this))
        if n_this == 1:
            axes = axes.reshape(1, -1)

        for row_i, (_, tp_row) in enumerate(best_tp.iloc[page_start:page_end].iterrows()):
            idx = int(tp_row['dataset_index'])
            x_1d, _, label, _ = dataset[idx]

            for c in range(2):
                ax = axes[row_i, c]
                signal = x_1d[c].numpy()
                ax.plot(signal, linewidth=0.5, color='forestgreen')
                ax.set_title(f"{channel_names[c]}", fontsize=10)
                ax.set_xlim([0, len(signal)])

                if c == 0:
                    ax.set_ylabel(
                        f"idx={idx}  p={tp_row['prob']:.3f}\n"
                        f"{tp_row['source_dir']}\n"
                        f"{tp_row['exp_id']} / file {tp_row['file_num']}",
                        fontsize=8, color='green'
                    )

        fig.suptitle(f"True Positives (arc correctly predicted as arc) — "
                     f"page {page_start//4 + 1}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'tp_signals_page{page_start//4 + 1}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved → tp_signals_page*.png ({n_plot} signals plotted)")


# ═══════════════════════════════════════════════════════
#  ADVANCED ANALYSIS PLOTS
# ═══════════════════════════════════════════════════════

def plot_score_distribution(labels, probs, output_dir, threshold=0.5):
    """Histogram of predicted probabilities split by true class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')

    normal_probs = probs[labels == 0]
    arc_probs = probs[labels == 1]

    ax.hist(normal_probs, bins=50, alpha=0.65, color='#2196F3',
            label=f'Normal (n={len(normal_probs)})', edgecolor='white', linewidth=0.5)
    ax.hist(arc_probs, bins=50, alpha=0.65, color='#FF5722',
            label=f'Arc (n={len(arc_probs)})', edgecolor='white', linewidth=0.5)
    ax.axvline(x=threshold, color='#333', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold}')

    ax.set_xlabel('Predicted Probability (arc)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Score Distribution by True Class', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distribution.png', dpi=200,
                bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved → score_distribution.png")


def plot_threshold_analysis(labels, probs, output_dir):
    """F1, Precision, Recall, Accuracy as a function of threshold."""
    thresholds = np.linspace(0.01, 0.99, 200)
    f1s, precisions, recalls, accuracies = [], [], [], []

    for t in thresholds:
        p = (probs > t).astype(int)
        tp = np.sum((p == 1) & (labels == 1))
        fp = np.sum((p == 1) & (labels == 0))
        fn = np.sum((p == 0) & (labels == 1))
        tn = np.sum((p == 0) & (labels == 0))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        acc = (tp + tn) / (tp + tn + fp + fn)
        f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)
        accuracies.append(acc)

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.plot(thresholds, [v*100 for v in f1s], color='#FF5722', lw=2.5, label='F1')
    ax.plot(thresholds, [v*100 for v in precisions], color='#FF9800', lw=2, label='Precision', ls='--')
    ax.plot(thresholds, [v*100 for v in recalls], color='#00BCD4', lw=2, label='Recall', ls='--')
    ax.plot(thresholds, [v*100 for v in accuracies], color='#9C27B0', lw=1.5, label='Accuracy', ls=':')
    ax.axvline(x=best_t, color='#4CAF50', ls='--', lw=2, alpha=0.8,
               label=f'Best F1 threshold = {best_t:.2f}')
    ax.scatter([best_t], [f1s[best_idx]*100], color='#4CAF50', s=100, zorder=5)
    ax.annotate(f'F1={f1s[best_idx]*100:.1f}%', xy=(best_t, f1s[best_idx]*100),
                xytext=(best_t + 0.08, f1s[best_idx]*100 - 3),
                fontsize=11, fontweight='bold', color='#4CAF50',
                arrowprops=dict(arrowstyle='->', color='#4CAF50'))

    ax.set_xlabel('Decision Threshold', fontsize=13)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Threshold Sensitivity Analysis', fontsize=15, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 101])
    ax.legend(fontsize=11, loc='lower center')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=200,
                bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved → threshold_analysis.png  (optimal threshold={best_t:.3f})")


def plot_calibration_curve(labels, probs, output_dir, n_bins=10):
    """Reliability diagram: predicted probability vs actual frequency."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            bin_accs.append(np.nan)
            bin_confs.append(bin_centers[i])
            bin_counts.append(0)
        else:
            bin_accs.append(labels[mask].mean())
            bin_confs.append(probs[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#fafafa')

    # Top: calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfectly calibrated', alpha=0.5)
    ax1.plot(bin_confs, bin_accs, 'o-', color='#FF5722', lw=2, ms=8,
             label='Model calibration')
    ax1.fill_between(bin_confs, bin_accs, [c for c in bin_confs],
                     alpha=0.15, color='#FF5722')
    ax1.set_ylabel('Actual Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curve (Reliability Diagram)', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Bottom: histogram of predictions per bin
    ax2.bar(bin_centers, bin_counts, width=1/n_bins * 0.85, color='#2196F3',
            edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=200,
                bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved → calibration_curve.png")


def plot_confidence_distribution(labels, probs, output_dir, threshold=0.5):
    """Box/violin plot of prediction confidence for TP, FP, TN, FN."""
    preds = (probs > threshold).astype(int)
    categories = []
    confidences = []

    for label, pred, prob in zip(labels, preds, probs):
        if label == 1 and pred == 1:
            categories.append('TP')
            confidences.append(prob)
        elif label == 0 and pred == 1:
            categories.append('FP')
            confidences.append(prob)
        elif label == 1 and pred == 0:
            categories.append('FN')
            confidences.append(prob)
        else:
            categories.append('TN')
            confidences.append(prob)

    cat_order = ['TN', 'FP', 'FN', 'TP']
    cat_colors = {'TN': '#4CAF50', 'FP': '#f44336', 'FN': '#FF9800', 'TP': '#2196F3'}

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#fafafa')

    data_by_cat = {c: [] for c in cat_order}
    for cat, conf in zip(categories, confidences):
        data_by_cat[cat].append(conf)

    positions = range(len(cat_order))
    bp = ax.boxplot([data_by_cat[c] for c in cat_order], positions=positions,
                    patch_artist=True, widths=0.5, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.3))

    for patch, cat in zip(bp['boxes'], cat_order):
        patch.set_facecolor(cat_colors[cat])
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    counts = [len(data_by_cat[c]) for c in cat_order]
    ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(cat_order, counts)],
                       fontsize=12, fontweight='bold')
    ax.axhline(y=threshold, color='#333', ls='--', lw=1.5, alpha=0.6,
               label=f'Threshold = {threshold}')
    ax.set_ylabel('Predicted Probability', fontsize=13)
    ax.set_title('Confidence Distribution by Prediction Category', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=200,
                bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved → confidence_distribution.png")


def plot_per_experiment_metrics(test_indices, labels, probs, metadata, output_dir, threshold=0.5):
    """Grouped bar chart: TP/FP/FN/TN counts and accuracy per experiment."""
    preds = (probs > threshold).astype(int)

    # Build per-experiment stats
    exp_stats = {}
    for local_i, global_i in enumerate(test_indices):
        meta_row = metadata.iloc[global_i]
        exp = f"{meta_row.get('source_dir', '?')}\n{meta_row.get('exp_id', '?')}"
        if exp not in exp_stats:
            exp_stats[exp] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        l, p = int(labels[local_i]), int(preds[local_i])
        if l == 1 and p == 1: exp_stats[exp]['tp'] += 1
        elif l == 0 and p == 1: exp_stats[exp]['fp'] += 1
        elif l == 1 and p == 0: exp_stats[exp]['fn'] += 1
        else: exp_stats[exp]['tn'] += 1

    exps = sorted(exp_stats.keys())
    tp_vals = [exp_stats[e]['tp'] for e in exps]
    fp_vals = [exp_stats[e]['fp'] for e in exps]
    fn_vals = [exp_stats[e]['fn'] for e in exps]
    tn_vals = [exp_stats[e]['tn'] for e in exps]
    accs = [(exp_stats[e]['tp'] + exp_stats[e]['tn']) /
            max(1, sum(exp_stats[e].values())) * 100 for e in exps]

    x = np.arange(len(exps))
    w = 0.2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(exps)*3), 10),
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#fafafa')

    ax1.bar(x - 1.5*w, tp_vals, w, label='TP', color='#4CAF50', alpha=0.85)
    ax1.bar(x - 0.5*w, tn_vals, w, label='TN', color='#2196F3', alpha=0.85)
    ax1.bar(x + 0.5*w, fp_vals, w, label='FP', color='#f44336', alpha=0.85)
    ax1.bar(x + 1.5*w, fn_vals, w, label='FN', color='#FF9800', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exps, fontsize=10)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Prediction Breakdown by Experiment', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Bottom: accuracy per experiment
    bars = ax2.bar(x, accs, color=['#4CAF50' if a >= 90 else '#FF9800' if a >= 80 else '#f44336'
                                    for a in accs], alpha=0.85, edgecolor='white')
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(exps, fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy by Experiment', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_experiment_metrics.png', dpi=200,
                bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved → per_experiment_metrics.png")


# ═══════════════════════════════════════════════════════
#  V_ARC VERIFICATION PLOTS  (I(t) + C2 from raw CSV)
# ═══════════════════════════════════════════════════════

DATA_ROOT = Path('/home/manip/pfe_salim_gouaied/Arc-Fault-Net/data/DataSet')
HEADER_LINES = 5   # LeCroy CSV header lines to skip


def _load_c2_segment(source_dir: str, exp_id: str, file_num: str,
                     start_sample: int, end_sample: int) -> Optional[np.ndarray]:
    """
    Load the V_arc (C2) segment from the original LeCroy CSV.

    Returns the raw amplitude array for the alternance [start_sample:end_sample],
    or None if the file cannot be found / read.
    """
    try:
        fnum = int(file_num)
        file_num_str = f"{fnum:05d}"
    except ValueError:
        file_num_str = str(file_num)

    csv_name = f"C2--{exp_id}--{file_num_str}.csv"
    csv_path = DATA_ROOT / source_dir / csv_name

    if not csv_path.exists():
        print(f"    ⚠ C2 file not found: {csv_path}")
        return None

    try:
        data = pd.read_csv(
            csv_path,
            skiprows=HEADER_LINES,
            header=0,
            names=['Time', 'Ampl'],
            dtype={'Ampl': np.float32},
            usecols=['Ampl'],
            engine='c'
        )
        ampl = data['Ampl'].values
        return ampl[start_sample:end_sample]
    except Exception as e:
        print(f"    ⚠ Error reading {csv_path}: {e}")
        return None


def plot_varc_verification(
    dataset: 'ArcFaultDataset',
    test_indices: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    threshold: float = 0.5
):
    """
    Generate 4 verification PNGs (one per category: TP, FP, FN, TN)
    showing the I(t) signal from the dataset alongside the raw V_arc
    (C2) loaded from the original CSV files.

    Purpose: visually verify whether the alternance labeling is correct
    by comparing the current waveform with the arc voltage oracle.
    """
    preds = (probs > threshold).astype(int)

    # Classify every test sample
    categories = {
        'TP': (labels == 1) & (preds == 1),
        'FP': (labels == 0) & (preds == 1),
        'FN': (labels == 1) & (preds == 0),
        'TN': (labels == 0) & (preds == 0),
    }

    cat_colors = {
        'TP': ('forestgreen',  'Arc correctly predicted as arc'),
        'FP': ('darkorange',   'Normal incorrectly predicted as arc'),
        'FN': ('red',          'Arc incorrectly predicted as normal'),
        'TN': ('steelblue',    'Normal correctly predicted as normal'),
    }

    print(f"\n  Generating V_arc verification plots ...")

    for cat_name, mask in categories.items():
        local_indices = np.where(mask)[0]
        if len(local_indices) == 0:
            print(f"    {cat_name}: no samples — skipping.")
            continue

        # Pick the most confident samples for this category (up to 4)
        if cat_name in ('TP', 'FP'):   # high prob = most confident
            sorted_local = local_indices[np.argsort(probs[local_indices])[::-1]]
        else:                           # FN, TN: low prob = most confident
            sorted_local = local_indices[np.argsort(probs[local_indices])]

        picks = sorted_local[:4]
        n_picks = len(picks)

        color, description = cat_colors[cat_name]

        # ── Plot: n_picks rows, 2 columns (I(t) | V_arc) ───────────────
        fig, axes = plt.subplots(n_picks, 2, figsize=(16, 4.5 * n_picks))
        if n_picks == 1:
            axes = axes.reshape(1, 2)
        fig.patch.set_facecolor('#fafafa')

        for i, pick in enumerate(picks):
            global_idx = test_indices[pick]
            meta_row = metadata.iloc[global_idx]

            source_dir   = str(meta_row.get('source_dir', ''))
            exp_id       = str(meta_row.get('exp_id', ''))
            file_num     = str(meta_row.get('file_num', ''))
            start_sample = int(meta_row.get('start_sample', 0))
            end_sample   = int(meta_row.get('end_sample', 0))
            arc_ratio    = meta_row.get('arc_ratio', '')
            prob_val     = float(probs[pick])

            # Get I(t) from the dataset (channel 1)
            x_1d, _, label_val, _ = dataset[global_idx]
            i_signal = x_1d[1].numpy()   # channel 1 = I(t)

            # Load V_arc from C2 CSV
            varc_raw = _load_c2_segment(source_dir, exp_id, file_num,
                                        start_sample, end_sample)

            ax1 = axes[i, 0]
            ax2 = axes[i, 1]

            # Left: I(t) — normalised signal from dataset
            ax1.plot(i_signal, linewidth=0.6, color=color)
            ax1.set_title(f'Sample {i+1} : I(t)  — normalised', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Sample')
            ax1.set_ylabel(
                f"Amplitude (z-score)\n\n"
                f"idx={global_idx}  p={prob_val:.3f}  label={int(label_val.item())}\n"
                f"ratio={arc_ratio}\n"
                f"{source_dir} / {exp_id} / file {file_num}\n"
                f"samples: {start_sample} -> {end_sample}",
                fontsize=9
            )
            ax1.set_xlim([0, len(i_signal)])
            ax1.grid(True, alpha=0.3, linestyle='--')

            # Right: V_arc (C2) — raw voltage
            if varc_raw is not None:
                ax2.plot(varc_raw, linewidth=0.6, color='purple')
                ax2.set_xlim([0, len(varc_raw)])
            else:
                ax2.text(0.5, 0.5, 'C2 File Not Found', ha='center', va='center')
            
            ax2.set_title(f'Sample {i+1} : V_arc (C2)  — raw voltage', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Sample')
            ax2.set_ylabel('Voltage (V)')
            ax2.grid(True, alpha=0.3, linestyle='--')

        # Overall title with metadata
        fig.suptitle(
            f"{cat_name} — {description} (Top {n_picks} most confident)",
            fontsize=14, fontweight='bold', y=1.02 if n_picks <= 2 else 1.01
        )

        plt.tight_layout()
        out_path = output_dir / f'varc_verification_{cat_name.lower()}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
        plt.close()
        print(f"    Saved → {out_path.name}")

    print(f"  V_arc verification plots complete.")


# ═══════════════════════════════════════════════════════
#  MAIN EVALUATION
# ═══════════════════════════════════════════════════════

def evaluate_model(
    model_path: Path,
    model_name: str,
    dataset: ArcFaultDataset,
    metadata: pd.DataFrame,
    device: torch.device,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    threshold: float = 0.5
) -> Dict:
    """
    Comprehensive model evaluation on the test split.

    Recreates the SAME random split that was used during training
    (same seed, same ratios) so we evaluate on the correct test set.
    """
    # ── Recreate split ───────────────────────────────────────────
    set_seed(seed)
    indices = np.random.permutation(len(dataset))
    n_train = int(len(dataset) * train_ratio)
    n_val   = int(len(dataset) * val_ratio)
    test_indices = indices[n_train + n_val:]

    print(f"  Recreated split (seed={seed}): "
          f"train={n_train}, val={n_val}, test={len(test_indices)}")

    # ── Load model ───────────────────────────────────────────────
    model = get_model(model_name, in_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── Get predictions on test set ──────────────────────────────
    labels, probs = get_predictions(model, dataset, test_indices, device)

    # ── Compute overall metrics ──────────────────────────────────
    metrics = compute_metrics(labels, probs, threshold)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {model_path}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {100*metrics['accuracy']:.2f}%")
    print(f"  F1 Score:  {100*metrics['f1']:.2f}%")
    print(f"  Precision: {100*metrics['precision']:.2f}%")
    print(f"  Recall:    {100*metrics['recall']:.2f}%")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  AP:        {metrics['average_precision']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN={metrics['tn']:4d}  FP={metrics['fp']:4d}")
    print(f"  FN={metrics['fn']:4d}  TP={metrics['tp']:4d}")

    # ── Generate visualizations ──────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = (probs > threshold).astype(int)

    plot_confusion_matrix(labels, preds, output_dir / 'confusion_matrix.png')
    plot_roc_curve(labels, probs, output_dir / 'roc_curve.png')
    plot_precision_recall_curve(labels, probs, output_dir / 'pr_curve.png')

    # ── Training curves (if history file exists) ─────────────────
    history_path = Path(model_path).parent / 'history_single.json'
    if not history_path.exists():
        history_path = Path(model_path).parent / 'history.json'
    if history_path.exists():
        print(f"\n  Plotting training curves from {history_path.name} ...")
        plot_training_curves(history_path, output_dir)
    else:
        print(f"  ⚠ No training history found, skipping training curves.")

    # ── False-negative / false-positive analysis ─────────────────
    analyse_false_negatives(
        dataset, test_indices, probs, labels, metadata,
        output_dir, threshold, n_plot=16
    )
    analyse_false_positives(
        dataset, test_indices, probs, labels, metadata,
        output_dir, threshold
    )
    analyse_true_positives(
        dataset, test_indices, probs, labels, metadata,
        output_dir, threshold, n_plot=16
    )

    # ── V_arc verification plots (I(t) + C2 from raw CSV) ────────
    plot_varc_verification(
        dataset, test_indices, probs, labels, metadata,
        output_dir, threshold
    )

    # ── Advanced analysis plots ──────────────────────────────────
    print(f"\n  Generating advanced analysis plots ...")
    plot_score_distribution(labels, probs, output_dir, threshold)
    plot_threshold_analysis(labels, probs, output_dir)
    plot_calibration_curve(labels, probs, output_dir)
    plot_confidence_distribution(labels, probs, output_dir, threshold)
    plot_per_experiment_metrics(test_indices, labels, probs, metadata, output_dir, threshold)

    # ── Save metrics JSON ────────────────────────────────────────
    results = {
        'model_name': model_name,
        'model_path': str(model_path),
        'seed': seed,
        'n_test': len(test_indices),
        'threshold': threshold,
        'overall_metrics': metrics,
    }
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Arc-FaultNet')

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--model', type=str, default='arcfaultnet',
                        choices=['arcfaultnet', '1d_only', 'no_attention',
                                 'standard_conv', 'independent_cbam', 'baseline_cnn'],
                        help='Model architecture')
    parser.add_argument('--data-dir', type=str,
                        default='/home/manip/pfe_salim_gouaied/Arc-Fault-Net/labeled_dataset',
                        help='Path to labeled dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualizations (default: <run_dir>/eval)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match training seed)')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU evaluation')

    args = parser.parse_args()

    # Device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Load dataset
    dataset = ArcFaultDataset(data_dir=args.data_dir)

    # Load metadata
    meta_path = Path(args.data_dir) / 'metadata.csv'
    if not meta_path.exists():
        print(f"ERROR: metadata.csv not found at {meta_path}")
        return
    metadata = pd.read_csv(meta_path)
    print(f"Metadata loaded: {len(metadata)} rows")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.model_path).parent / 'eval'

    evaluate_model(
        model_path=Path(args.model_path),
        model_name=args.model,
        dataset=dataset,
        metadata=metadata,
        device=device,
        output_dir=output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()
