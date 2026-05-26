#!/usr/bin/env python3
"""
kfold_evaluate.py — Evaluate a K-Fold cross-validation run.

Reads kfold_summary.json to reconstruct the exact test splits,
loads each fold's best checkpoint, runs inference, and produces:
  - Per-fold metrics table
  - Aggregated mean ± std summary
  - Confusion matrix, ROC curve, training curves per fold
  - A combined report saved to <run_dir>/eval/

Usage:
  python kfold_evaluate.py --run runs/arcfaultnet_kfold5_20260526_142337 \
                           --data-dir combined_dataset
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ArcFaultDataset
from model import build_model_from_checkpoint


# ─────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, dataset, indices, batch_size, device):
    """Return (labels, probs) numpy arrays for the given index subset."""
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    all_probs, all_labels = [], []
    model.eval()
    for x_1d, x_2d, label, _ in loader:
        x_1d   = x_1d.to(device)
        x_2d   = x_2d.to(device)
        logits = model(x_1d, x_2d)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(label.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy':    accuracy_score(labels, preds),
        'f1':          f1_score(labels, preds, zero_division=0),
        'precision':   precision_score(labels, preds, zero_division=0),
        'recall':      recall_score(labels, preds, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


# ─────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, probs, threshold, fold_idx, out_path):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ['Normal', 'Arc']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — Fold {fold_idx}')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc(labels, probs, fold_idx, out_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — Fold {fold_idx}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return roc_auc


def plot_training_curves(history_path, fold_idx, out_path):
    if not history_path.exists():
        return
    with open(history_path) as f:
        h = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if 'train_loss' in h:
        axes[0].plot(h['train_loss'], label='Train')
    if 'val_loss' in h:
        axes[0].plot(h['val_loss'],   label='Val')
    axes[0].set_title(f'Loss — Fold {fold_idx}')
    axes[0].set_xlabel('Epoch'); axes[0].legend()
    if 'val_f1' in h:
        axes[1].plot(h['val_f1'], label='Val F1', color='green')
    axes[1].set_title(f'Val F1 — Fold {fold_idx}')
    axes[1].set_xlabel('Epoch'); axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_summary_bars(fold_metrics, metric_keys, out_path):
    """Bar chart with error bars for each metric across folds."""
    labels = [m.replace('_', ' ').capitalize() for m in metric_keys]
    means  = [np.mean([fm[m] for fm in fold_metrics]) for m in metric_keys]
    stds   = [np.std( [fm[m] for fm in fold_metrics]) for m in metric_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, [m * 100 for m in means], yerr=[s * 100 for s in stds],
                  capsize=6, color='steelblue', alpha=0.85, edgecolor='navy')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(max(0, min(means) * 100 - 5), 101)
    ax.set_title('K-Fold CV — Mean ± Std per Metric')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std * 100 + 0.3,
                f'{mean*100:.2f}±{std*100:.2f}',
                ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',       required=True,
                        help='Path to kfold run directory')
    parser.add_argument('--data-dir',  default='combined_dataset')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int,  default=128)
    args = parser.parse_args()

    run_dir  = Path(args.run)
    eval_dir = run_dir / 'eval'
    eval_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Read kfold_summary ──────────────────────────────────────
    summary_path = run_dir / 'kfold_summary.json'
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Is this a kfold run directory?")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    n_folds = summary['n_folds']
    seed    = summary['seed']

    print(f"\n{'='*55}")
    print(f"  kfold_evaluate — {run_dir.name}")
    print(f"  Device: {device}  |  Folds: {n_folds}  |  Threshold: {args.threshold}")
    print(f"{'='*55}\n")

    # ── Load dataset ─────────────────────────────────────────────
    print("[1/3] Loading dataset …")
    dataset = ArcFaultDataset(data_dir=str(args.data_dir))
    labels  = dataset.y
    indices = np.arange(len(dataset))

    # Reconstruct the exact same splits used during training
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(indices, labels))

    # ── Evaluate each fold ───────────────────────────────────────
    print("[2/3] Evaluating each fold …\n")
    fold_metrics = []
    all_labels, all_probs = [], []   # for aggregate ROC

    for fold_idx, (_, test_idx) in enumerate(splits):
        fold_num  = fold_idx + 1
        fold_dir  = run_dir / f'fold_{fold_num}'
        ckpt_path = fold_dir / f'best_fold_{fold_num}.pt'

        if not ckpt_path.exists():
            print(f"  [Fold {fold_num}] WARNING: checkpoint not found → {ckpt_path}")
            continue

        print(f"  [Fold {fold_num}/{n_folds}] Loading {ckpt_path.name} …")
        model = build_model_from_checkpoint(str(ckpt_path), device)

        lbls, probs = run_inference(model, dataset, test_idx,
                                    args.batch_size, device)
        metrics = compute_metrics(lbls, probs, args.threshold)
        fold_metrics.append(metrics)
        all_labels.append(lbls)
        all_probs.append(probs)

        # Per-fold plots
        plot_confusion_matrix(lbls, probs, args.threshold, fold_num,
                               eval_dir / f'confusion_fold_{fold_num}.png')
        roc_auc = plot_roc(lbls, probs, fold_num,
                           eval_dir / f'roc_fold_{fold_num}.png')
        plot_training_curves(fold_dir / f'history_fold_{fold_num}.json',
                             fold_num, eval_dir / f'curves_fold_{fold_num}.png')

        print(f"         Acc={100*metrics['accuracy']:.2f}%  "
              f"F1={100*metrics['f1']:.2f}%  "
              f"Prec={100*metrics['precision']:.2f}%  "
              f"Rec={100*metrics['recall']:.2f}%  "
              f"AUC={roc_auc:.4f}")

    if not fold_metrics:
        print("No folds evaluated. Exiting.")
        sys.exit(1)

    # ── Aggregate ROC ────────────────────────────────────────────
    agg_labels = np.concatenate(all_labels)
    agg_probs  = np.concatenate(all_probs)
    plot_roc(agg_labels, agg_probs, 'Aggregate', eval_dir / 'roc_aggregate.png')

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n[3/3] Summary\n")
    metric_keys = ['accuracy', 'f1', 'precision', 'recall', 'specificity']
    agg_summary = {}

    print(f"  {'Metric':<14} {'Mean':>8}  {'Std':>6}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*50}")
    for m in metric_keys:
        vals = np.array([fm[m] for fm in fold_metrics])
        agg_summary[f'{m}_mean'] = float(vals.mean())
        agg_summary[f'{m}_std']  = float(vals.std())
        agg_summary[f'{m}_min']  = float(vals.min())
        agg_summary[f'{m}_max']  = float(vals.max())
        print(f"  {m.capitalize():<14} "
              f"{100*vals.mean():>7.2f}%  "
              f"{100*vals.std():>5.2f}%  "
              f"{100*vals.min():>7.2f}%  "
              f"{100*vals.max():>7.2f}%")

    plot_summary_bars(fold_metrics, metric_keys,
                      eval_dir / 'summary_bars.png')

    # Per-fold detail table
    agg_summary['per_fold'] = [
        {**{k: float(fm[k]) for k in metric_keys}, 'fold': i+1}
        for i, fm in enumerate(fold_metrics)
    ]
    agg_summary['n_folds']   = n_folds
    agg_summary['threshold'] = args.threshold

    with open(eval_dir / 'kfold_metrics.json', 'w') as f:
        json.dump(agg_summary, f, indent=2)

    print(f"\n  Saved plots + metrics → {eval_dir}/")
    print(f"\n{'='*55}")
    print(f"  Done → {run_dir}/eval/")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
