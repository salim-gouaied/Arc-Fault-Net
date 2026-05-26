#!/usr/bin/env python3
"""
mini_evaluate.py — Quick evaluation: confusion matrix, training curves, ROC.

Auto-detects the model architecture from the checkpoint state dict so it
works regardless of what version of model.py produced the checkpoint.

Usage:
  python mini_evaluate.py --run runs/arcfaultnet_single_20260521_121423
"""

import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score, precision_score, recall_score
)

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ArcFaultDataset


# ─────────────────────────────────────────────────────────────────
#  MODEL — imported from the single source of truth (model.py)
# ─────────────────────────────────────────────────────────────────

from model import build_model_from_checkpoint


# ─────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, dataset, indices, device, batch=64):
    labels, probs = [], []
    for i in range(0, len(indices), batch):
        idx = indices[i:i+batch]
        x1 = torch.stack([dataset[j][0] for j in idx]).to(device)
        x2 = torch.stack([dataset[j][1] for j in idx]).to(device)
        lb = [dataset[j][2].item() for j in idx]
        p  = torch.sigmoid(model(x1, x2)).cpu().numpy()
        labels.extend(lb); probs.extend(p.tolist())
    return np.array(labels), np.array(probs)


# ─────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, probs, threshold, out_dir):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    pct   = cm / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#f8f9fa')
    cell_labels = [['TN', 'FP'], ['FN', 'TP']]
    mx = cm.max()
    for i in range(2):
        for j in range(2):
            intens = 0.35 + 0.65 * cm[i,j] / (mx + 1e-8)
            col = [0.15, 0.55*intens+0.18, 0.22, 0.88] if i==j else [0.85*intens+0.12, 0.15, 0.15, 0.78]
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=col, edgecolor='white', lw=3))
            ax.text(j, i-0.15, str(cm[i,j]), ha='center', va='center',
                    fontsize=28, fontweight='bold', color='white')
            ax.text(j, i+0.13, f'({pct[i,j]:.1f}%)', ha='center', va='center',
                    fontsize=14, color='white', alpha=0.92)
            ax.text(j, i+0.35, cell_labels[i][j], ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white', fontstyle='italic')

    acc  = (tp+tn)/total*100
    prec = tp/(tp+fp+1e-8)*100
    rec  = tp/(tp+fn+1e-8)*100
    f1v  = 2*prec*rec/(prec+rec+1e-8)
    spec = tn/(tn+fp+1e-8)*100
    summary = f"Acc {acc:.1f}%  Prec {prec:.1f}%  Recall {rec:.1f}%  F1 {f1v:.1f}%  Spec {spec:.1f}%"
    ax.text(0.5, 2.08, summary, ha='center', va='center', fontsize=10,
            transform=ax.get_yaxis_transform(),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50', edgecolor='#34495e', alpha=0.9),
            color='white', fontweight='bold')

    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Normal','Arc'], fontsize=13, fontweight='bold')
    ax.set_yticklabels(['Normal','Arc'], fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    plt.tight_layout()
    p = out_dir / 'confusion_matrix.png'
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close(); print(f"  Saved → {p.name}")


def plot_roc(labels, probs, out_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#f8f9fa')
    ax.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'AUC = {roc_auc:.4f}')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#e74c3c')
    ax.plot([0,1],[0,1], color='#7f8c8d', lw=1.5, ls='--', label='Random')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = out_dir / 'roc_curve.png'
    plt.savefig(p, dpi=180, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close(); print(f"  Saved → {p.name}")


def plot_training_curves(history_path, out_dir):
    with open(history_path) as f:
        h = json.load(f)
    epochs = list(range(1, len(h['train_loss'])+1))
    best   = h.get('best_epoch')
    blue, orange, green = '#2196F3', '#FF5722', '#4CAF50'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#fafafa')

    # Loss
    ax = axes[0]
    ax.plot(epochs, h['train_loss'], color=blue, lw=2, label='Train', marker='o', ms=2)
    ax.plot(epochs, h['val_loss'],   color=orange, lw=2, label='Val', marker='s', ms=2)
    if best: ax.axvline(best+1, color=green, ls='--', alpha=0.7, label=f'Best ({best+1})')
    ax.set_yscale('log')
    ax.set_title('Loss (log)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, [a*100 for a in h['train_acc']], color=blue, lw=2, label='Train', marker='o', ms=2)
    ax.plot(epochs, [a*100 for a in h['val_acc']],   color=orange, lw=2, label='Val', marker='s', ms=2)
    if best: ax.axvline(best+1, color=green, ls='--', alpha=0.7, label=f'Best ({best+1})')
    ax.set_title('Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.suptitle('Training Curves', fontsize=15, fontweight='bold')
    plt.tight_layout()
    p = out_dir / 'training_curves.png'
    plt.savefig(p, dpi=180, bbox_inches='tight', facecolor='#fafafa')
    plt.close(); print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Path to run directory')
    parser.add_argument('--data-dir', default='labeled_dataset')
    parser.add_argument('--checkpoint', default='best_single.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    run_dir  = Path(args.run)
    ckpt     = run_dir / args.checkpoint
    out_dir  = run_dir / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f"  mini_evaluate — {run_dir.name}")
    print(f"  Device: {device}  |  Threshold: {args.threshold}")
    print(f"{'='*55}")

    # Build model
    print("\n[1/4] Building model from checkpoint …")
    model = build_model_from_checkpoint(ckpt, device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n:,}")

    # Dataset + split
    print("\n[2/4] Loading dataset …")
    ds = ArcFaultDataset(data_dir=args.data_dir)
    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    idx = np.random.permutation(len(ds))
    n_tr = int(len(ds) * args.train_ratio)
    n_vl = int(len(ds) * args.val_ratio)
    test_idx = idx[n_tr + n_vl:]
    print(f"  Test set: {len(test_idx)} samples")

    # Inference
    print("\n[3/4] Running inference …")
    labels, probs = get_predictions(model, ds, test_idx, device)
    preds = (probs >= args.threshold).astype(int)

    # Metrics
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    print(f"\n  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%")
    print(f"  Recall     : {rec*100:.2f}%")
    print(f"  F1         : {f1*100:.2f}%")
    print(f"  AUC-ROC    : {roc_auc:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'accuracy': acc, 'precision': prec, 'recall': rec,
                   'f1': f1, 'auc_roc': roc_auc,
                   'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}, f, indent=2)
    print(f"\n  Saved → metrics.json")

    # Plots
    print("\n[4/4] Generating plots …")
    plot_confusion_matrix(labels, probs, args.threshold, out_dir)
    plot_roc(labels, probs, out_dir)

    hist = run_dir / 'history_single.json'
    if not hist.exists():
        hist = run_dir / 'history.json'
    if hist.exists():
        plot_training_curves(hist, out_dir)
    else:
        print("  ⚠ No history file found, skipping training curves.")

    print(f"\n{'='*55}")
    print(f"  Done → {out_dir}")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
