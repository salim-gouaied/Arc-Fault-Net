#!/usr/bin/env python3
"""
ARC-FAULTNET — TestModel Evaluation
=====================================
Loads the trained model checkpoint and evaluates it on the
OthmaneSalim10052026 prepared dataset.

Outputs (saved to TestModel/results/):
  metrics.json          — all scalar metrics
  classification_report.txt — sklearn full report
  confusion_matrix.png  — colour-coded confusion matrix
  roc_curve.png         — ROC with AUC
  pr_curve.png          — Precision-Recall curve
  score_distribution.png — histogram of predicted probabilities
  training_curves.png   — training history from the run

Usage:
  python TestModel/run_test.py
  # or from the project root:
  python TestModel/run_test.py --model runs/arcfaultnet_single_20260513_115122/best_single.pt
"""

import sys, os
# Allow imports from project root (model.py, dataset.py, etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)

from mini_evaluate import build_model_from_checkpoint
from dataset import ArcFaultDataset

# ─────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_RUN    = PROJECT_ROOT / 'runs' / 'arcfaultnet_single_20260513_115122'
DATA_DIR     = PROJECT_ROOT / 'TestModel' / 'prepared_data'
RESULTS_DIR  = PROJECT_ROOT / 'TestModel' / 'results'


# ─────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load ArcFaultNet from a .pt checkpoint.
    
    Uses auto-detection from mini_evaluate.py to reconstruct the exact
    architecture (SE blocks, amplitude, deep classifier, etc.) that
    matches the checkpoint's state_dict keys.
    """
    return build_model_from_checkpoint(checkpoint_path, device)


# ─────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: nn.Module,
                  dataset: ArcFaultDataset,
                  device: torch.device,
                  batch_size: int = 64):
    """Run inference over the whole dataset. Returns (labels, probs)."""
    all_labels, all_probs = [], []
    indices = np.arange(len(dataset))

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        x1_list, x2_list, lab_list = [], [], []
        for i in batch_idx:
            x1, x2, lab, _ = dataset[i]
            x1_list.append(x1)
            x2_list.append(x2)
            lab_list.append(lab.item())

        x1 = torch.stack(x1_list).to(device)
        x2 = torch.stack(x2_list).to(device)
        logits = model(x1, x2)
        probs  = torch.sigmoid(logits).cpu().numpy()

        all_labels.extend(lab_list)
        all_probs.extend(probs.tolist())

    return np.array(all_labels), np.array(all_probs)


# ─────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    ap          = average_precision_score(labels, probs)

    total = len(labels)
    return {
        'accuracy':           float(accuracy_score(labels, preds)),
        'precision':          float(precision_score(labels, preds, zero_division=0)),
        'recall':             float(recall_score(labels, preds, zero_division=0)),
        'f1':                 float(f1_score(labels, preds, zero_division=0)),
        'specificity':        float(tn / (tn + fp + 1e-8)),
        'auc_roc':            float(roc_auc),
        'average_precision':  float(ap),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'total': int(total),
        'threshold': float(threshold),
    }


# ─────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────

def plot_confusion_matrix(labels, probs, threshold, save_path):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = cm / (row_sums + 1e-8) * 100
    total    = cm.sum()

    cell_labels = np.array([['TN', 'FP'], ['FN', 'TP']])

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    max_val = cm.max()
    for i in range(2):
        for j in range(2):
            intensity = 0.35 + 0.65 * (cm[i, j] / (max_val + 1e-8))
            color = ([0.18, 0.55 * intensity + 0.18, 0.22, 0.88]
                     if i == j else
                     [0.85 * intensity + 0.12, 0.15, 0.15, 0.78])
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  facecolor=color, edgecolor='white', linewidth=3)
            ax.add_patch(rect)
            ax.text(j, i - 0.15, f"{cm[i, j]}",
                    ha='center', va='center', fontsize=30,
                    fontweight='bold', color='white')
            ax.text(j, i + 0.12, f"({cm_pct[i, j]:.1f}%)",
                    ha='center', va='center', fontsize=15, color='white', alpha=0.95)
            ax.text(j, i + 0.35, cell_labels[i, j],
                    ha='center', va='center', fontsize=12,
                    fontweight='bold', color='white', alpha=0.8, fontstyle='italic')

    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Arc'], fontsize=13, fontweight='bold')
    ax.set_yticklabels(['Normal', 'Arc'], fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label',      fontsize=13, fontweight='bold', labelpad=10)

    acc  = (tp + tn) / total * 100
    prec = tp / (tp + fp + 1e-8) * 100
    rec  = tp / (tp + fn + 1e-8) * 100
    f1v  = 2 * prec * rec / (prec + rec + 1e-8)
    spec = tn / (tn + fp + 1e-8) * 100

    summary = (f"Accuracy {acc:.1f}%  |  Precision {prec:.1f}%  |  "
               f"Recall {rec:.1f}%  |  F1 {f1v:.1f}%  |  Specificity {spec:.1f}%")
    ax.text(0.5, 2.08, summary, ha='center', va='center', fontsize=10,
            transform=ax.get_yaxis_transform(),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50',
                      edgecolor='#34495e', alpha=0.9),
            color='white', fontweight='bold')

    run_name = getattr(plot_confusion_matrix, '_run_name', 'unknown')
    ax.set_title(f'Confusion Matrix — OthmaneSalim10052026\n'
                 f'Model: {run_name}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved → {save_path.name}")


def plot_roc(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#f8f9fa')
    ax.plot(fpr, tpr, color='#e74c3c', lw=2.5,
            label=f'AUC = {roc_auc:.4f}')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#e74c3c')
    ax.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved → {save_path.name}")


def plot_pr_curve(labels, probs, save_path):
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#f8f9fa')
    ax.plot(recall, precision, color='#2980b9', lw=2.5,
            label=f'AP = {ap:.4f}')
    ax.fill_between(recall, precision, alpha=0.08, color='#2980b9')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('Recall',    fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved → {save_path.name}")


def plot_score_distribution(labels, probs, threshold, save_path):
    normal_probs = probs[labels == 0]
    arc_probs    = probs[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#f8f9fa')
    bins = np.linspace(0, 1, 50)
    ax.hist(normal_probs, bins=bins, alpha=0.65, color='#27ae60',
            label=f'Normal (n={len(normal_probs)})', density=True)
    ax.hist(arc_probs,    bins=bins, alpha=0.65, color='#e74c3c',
            label=f'Arc (n={len(arc_probs)})',       density=True)
    ax.axvline(threshold, color='#2c3e50', lw=2, linestyle='--',
               label=f'Threshold = {threshold}')
    ax.set_xlabel('Predicted Probability (P(arc))', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Distribution by True Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved → {save_path.name}")


def plot_training_curves(history_path: Path, save_path: Path):
    """Reproduce training curves from the model run's history JSON."""
    if not history_path.exists():
        print(f"  SKIP training curves (history not found: {history_path})")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs     = list(range(1, len(history['train_loss']) + 1))
    best_epoch = history.get('best_epoch', None)
    colors     = {'train': '#2196F3', 'val': '#FF5722', 'best': '#4CAF50'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#fafafa')

    def style(ax, title, ylabel):
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=10)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color=colors['train'], lw=2,
            label='Train', marker='o', ms=2)
    ax.plot(epochs, history['val_loss'],   color=colors['val'],   lw=2,
            label='Val',   marker='s', ms=2)
    if best_epoch is not None:
        ax.axvline(best_epoch + 1, color=colors['best'], ls='--', alpha=0.7,
                   label=f'Best ({best_epoch+1})')
    ax.set_yscale('log')
    style(ax, 'Loss (log scale)', 'Loss')

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [a * 100 for a in history['train_acc']],
            color=colors['train'], lw=2, label='Train', marker='o', ms=2)
    ax.plot(epochs, [a * 100 for a in history['val_acc']],
            color=colors['val'],   lw=2, label='Val',   marker='s', ms=2)
    if best_epoch is not None:
        ax.axvline(best_epoch + 1, color=colors['best'], ls='--', alpha=0.7)
    style(ax, 'Accuracy', 'Accuracy (%)')

    # F1
    ax = axes[1, 0]
    if 'val_f1' in history:
        ax.plot(epochs, [f * 100 for f in history['val_f1']],
                color=colors['val'], lw=2, label='Val F1', marker='s', ms=2)
        if best_epoch is not None:
            ax.axvline(best_epoch + 1, color=colors['best'], ls='--', alpha=0.7)
    style(ax, 'F1 Score', 'F1 (%)')

    # LR
    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(epochs[:len(history['lr'])], history['lr'],
                color='#9C27B0', lw=2, label='LR', marker='D', ms=2)
        ax.set_yscale('log')
    style(ax, 'Learning Rate Schedule', 'LR')

    run_name = getattr(plot_training_curves, '_run_name', 'unknown')
    fig.suptitle(f'Training History — {run_name}',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved → {save_path.name}")


# ─────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Arc-FaultNet — TestModel evaluation')
    parser.add_argument('--model',     type=str,
                        default=str(MODEL_RUN / 'best_single.pt'),
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--data',      type=str,
                        default=str(DATA_DIR),
                        help='Path to prepared dataset directory')
    parser.add_argument('--results',   type=str,
                        default=str(RESULTS_DIR),
                        help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Inference batch size')
    args = parser.parse_args()

    model_path  = Path(args.model)
    data_path   = Path(args.data)
    results_dir = Path(args.results)
    threshold   = args.threshold

    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Checks ────────────────────────────────────────
    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found: {model_path}")
        print("  Run: python scripts/step4_prepare_othmanesalim.py first.")
        sys.exit(1)

    x_path = data_path / 'X_multi.npy'
    y_path = data_path / 'y.npy'
    if not x_path.exists() or not y_path.exists():
        print(f"ERROR: Prepared data not found in {data_path}")
        print("  Run: python scripts/step4_prepare_othmanesalim.py first.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"ARC-FAULTNET — TestModel Evaluation")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {model_path}")
    print(f"  Data       : {data_path}")
    print(f"  Results    : {results_dir}")
    print(f"  Threshold  : {threshold}")

    # ── Load model ────────────────────────────────────
    print("\n[1/5] Loading model …")
    model = load_model(model_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")

    # ── Load dataset ──────────────────────────────────
    print("\n[2/5] Loading dataset …")
    dataset = ArcFaultDataset(data_dir=str(data_path))

    # ── Inference ─────────────────────────────────────
    print(f"\n[3/5] Running inference (batch_size={args.batch_size}) …")
    labels, probs = run_inference(model, dataset, device, args.batch_size)
    print(f"  Processed {len(labels)} samples")

    # ── Metrics ───────────────────────────────────────
    print("\n[4/5] Computing metrics …")
    metrics = compute_metrics(labels, probs, threshold)
    preds   = (probs >= threshold).astype(int)

    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │  EVALUATION RESULTS         │")
    print(f"  ├─────────────────────────────┤")
    print(f"  │  Accuracy   : {metrics['accuracy']:.4f}          │")
    print(f"  │  Precision  : {metrics['precision']:.4f}          │")
    print(f"  │  Recall     : {metrics['recall']:.4f}          │")
    print(f"  │  F1 Score   : {metrics['f1']:.4f}          │")
    print(f"  │  Specificity: {metrics['specificity']:.4f}          │")
    print(f"  │  AUC-ROC    : {metrics['auc_roc']:.4f}          │")
    print(f"  │  Avg Prec   : {metrics['average_precision']:.4f}          │")
    print(f"  ├─────────────────────────────┤")
    print(f"  │  TP={metrics['tp']:4d}  FP={metrics['fp']:4d}           │")
    print(f"  │  FN={metrics['fn']:4d}  TN={metrics['tn']:4d}           │")
    print(f"  └─────────────────────────────┘")

    # Full sklearn report
    report = classification_report(labels, preds,
                                   target_names=['Normal', 'Arc'], digits=4)
    print(f"\n  Classification Report:\n{report}")

    # Save metrics JSON
    metrics['model_checkpoint'] = str(model_path)
    metrics['data_dir']         = str(data_path)
    metrics['n_params']         = n_params
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → metrics.json")

    # Save text report
    with open(results_dir / 'classification_report.txt', 'w') as f:
        f.write(f"Arc-FaultNet — TestModel Evaluation\n")
        f.write(f"Model     : {model_path}\n")
        f.write(f"Data      : {data_path}\n")
        f.write(f"Threshold : {threshold}\n")
        f.write(f"Samples   : {len(labels)}\n\n")
        f.write(report)
        f.write(f"\nMetrics JSON:\n{json.dumps(metrics, indent=2)}\n")
    print(f"  Saved → classification_report.txt")

    # ── Plots ─────────────────────────────────────────
    print("\n[5/5] Generating plots …")

    # Derive the run directory and run name from the checkpoint path
    model_run_dir = model_path.parent
    run_name = model_run_dir.name
    plot_confusion_matrix._run_name = run_name
    plot_training_curves._run_name  = run_name

    plot_confusion_matrix(labels, probs, threshold,
                          results_dir / 'confusion_matrix.png')
    plot_roc(labels, probs, results_dir / 'roc_curve.png')
    plot_pr_curve(labels, probs, results_dir / 'pr_curve.png')
    plot_score_distribution(labels, probs, threshold,
                            results_dir / 'score_distribution.png')
    plot_training_curves(model_run_dir / 'history_single.json',
                         results_dir / 'training_curves.png')

    print(f"\n{'='*60}")
    print(f"ALL DONE — results saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
