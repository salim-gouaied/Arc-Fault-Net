#!/usr/bin/env python3
"""
Evaluate the retrained model on the held-out OthmaneSalim samples
that were NEVER seen during training.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import json
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score 
)
from mini_evaluate import build_model_from_checkpoint
from dataset import ArcFaultDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT  = Path(__file__).parent
RUN_DIR  = PROJECT / 'runs' / 'arcfaultnet_single_20260528_114322'
CKPT     = RUN_DIR / 'best_single.pt'
DATA_DIR = PROJECT / 'combined_dataset'
OUT_DIR  = RUN_DIR / 'resultsOnHeldOut'

THRESHOLD = 0.5

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  HELD-OUT EVALUATION — OthmaneSalim (20% never-seen)")
    print(f"{'='*60}")
    print(f"  Model: {RUN_DIR.name}")
    print(f"  Device: {device}")

    # Load model
    print("\n[1/4] Loading model...")
    model = build_model_from_checkpoint(CKPT, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Load held-out data
    print("\n[2/4] Loading held-out data...")
    X = np.load(DATA_DIR / 'holdout_X.npy')
    y = np.load(DATA_DIR / 'holdout_y.npy')
    print(f"  Samples: {len(y)} ({int((y==0).sum())} normal, {int((y==1).sum())} arc)")

    # Create a temporary dataset-like structure for inference
    # We need to compute STFT on-the-fly like the dataset does
    ds = ArcFaultDataset.__new__(ArcFaultDataset)
    ds.X = X
    ds.y = y
    ds.n_samples = len(y)
    ds.n_channels = 2
    ds.seq_len = 20000
    ds.n_fft = 512
    ds.hop_length = 256
    ds.compute_stft = True
    ds.training = False
    ds.window = torch.hann_window(512)
    ds.n_freq = 257
    ds.n_time = 77
    ds.charges = np.zeros(len(y), dtype=np.int64)

    # Inference
    print("\n[3/4] Running inference...")
    all_labels, all_probs = [], []
    batch_size = 64

    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            end = min(start + batch_size, len(y))
            x1_list, x2_list = [], []
            for i in range(start, end):
                x1, x2, lab, _ = ds[i]
                x1_list.append(x1)
                x2_list.append(x2)
                all_labels.append(lab.item())

            x1 = torch.stack(x1_list).to(device)
            x2 = torch.stack(x2_list).to(device)
            logits = model(x1, x2)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    labels = np.array(all_labels)
    probs = np.array(all_probs)
    preds = (probs >= THRESHOLD).astype(int)

    # Metrics
    print("\n[4/4] Computing metrics...")
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(labels, probs)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    spec = tn / (tn + fp + 1e-8)

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  HELD-OUT RESULTS               │")
    print(f"  ├─────────────────────────────────┤")
    print(f"  │  Accuracy   : {acc:.4f}            │")
    print(f"  │  Precision  : {prec:.4f}            │")
    print(f"  │  Recall     : {rec:.4f}            │")
    print(f"  │  F1 Score   : {f1:.4f}            │")
    print(f"  │  Specificity: {spec:.4f}            │")
    print(f"  │  AUC-ROC    : {roc_auc:.4f}            │")
    print(f"  │  Avg Prec   : {ap:.4f}            │")
    print(f"  ├─────────────────────────────────┤")
    print(f"  │  TP={tp:4d}  FP={fp:4d}             │")
    print(f"  │  FN={fn:4d}  TN={tn:4d}             │")
    print(f"  └─────────────────────────────────┘")

    report = classification_report(labels, preds, target_names=['Normal', 'Arc'], digits=4)
    print(f"\n{report}")

    # Save metrics
    metrics = {
        'description': 'Held-out OthmaneSalim samples (20%, never seen during training)',
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'specificity': float(spec),
        'auc_roc': float(roc_auc),
        'average_precision': float(ap),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'total': int(len(labels)),
        'threshold': THRESHOLD,
        'model_checkpoint': str(CKPT),
        'n_params': n_params,
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save classification report
    with open(OUT_DIR / 'classification_report.txt', 'w') as f:
        f.write(f"Held-Out Evaluation — OthmaneSalim (20% never-seen)\n")
        f.write(f"Model: {RUN_DIR.name}\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")
        f.write(report)
        f.write(f"\nMetrics:\n{json.dumps(metrics, indent=2)}\n")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#f8f9fa')
    cell_labels = [['TN', 'FP'], ['FN', 'TP']]
    pct = cm / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
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

    summary = f"Acc {acc*100:.1f}%  Prec {prec*100:.1f}%  Recall {rec*100:.1f}%  F1 {f1*100:.1f}%  Spec {spec*100:.1f}%"
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
    ax.set_title(f'Held-Out Confusion Matrix — OthmaneSalim (n={len(labels)})\n'
                 f'Model: {RUN_DIR.name}', fontsize=13, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'confusion_matrix.png', dpi=200, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved → confusion_matrix.png")

    # Also run on the FULL OthmaneSalim dataset for comparison
    print(f"\n{'='*60}")
    print(f"  FULL OthmaneSalim EVALUATION (all 1842 samples)")
    print(f"{'='*60}")

    X_full = np.load(PROJECT / 'TestModel' / 'prepared_data' / 'X_multi.npy')
    y_full = np.load(PROJECT / 'TestModel' / 'prepared_data' / 'y.npy')

    ds2 = ArcFaultDataset.__new__(ArcFaultDataset)
    ds2.X = X_full; ds2.y = y_full; ds2.n_samples = len(y_full)
    ds2.n_channels = 2; ds2.seq_len = 20000
    ds2.n_fft = 512; ds2.hop_length = 256
    ds2.compute_stft = True; ds2.training = False
    ds2.window = torch.hann_window(512)
    ds2.n_freq = 257; ds2.n_time = 77
    ds2.charges = np.zeros(len(y_full), dtype=np.int64)

    all_labels2, all_probs2 = [], []
    with torch.no_grad():
        for start in range(0, len(y_full), batch_size):
            end = min(start + batch_size, len(y_full))
            x1_list, x2_list = [], []
            for i in range(start, end):
                x1, x2, lab, _ = ds2[i]
                x1_list.append(x1); x2_list.append(x2)
                all_labels2.append(lab.item())
            x1 = torch.stack(x1_list).to(device)
            x2 = torch.stack(x2_list).to(device)
            probs2 = torch.sigmoid(model(x1, x2)).cpu().numpy()
            all_probs2.extend(probs2.tolist())

    labels2 = np.array(all_labels2)
    probs2 = np.array(all_probs2)
    preds2 = (probs2 >= THRESHOLD).astype(int)
    cm2 = confusion_matrix(labels2, preds2)
    tn2, fp2, fn2, tp2 = cm2.ravel()

    acc2 = accuracy_score(labels2, preds2)
    f1_2 = f1_score(labels2, preds2, zero_division=0)
    prec2 = precision_score(labels2, preds2, zero_division=0)
    rec2 = recall_score(labels2, preds2, zero_division=0)
    spec2 = tn2 / (tn2 + fp2 + 1e-8)

    print(f"\n  Accuracy   : {acc2*100:.2f}%")
    print(f"  Precision  : {prec2*100:.2f}%")
    print(f"  Recall     : {rec2*100:.2f}%")
    print(f"  F1         : {f1_2*100:.2f}%")
    print(f"  Specificity: {spec2*100:.2f}%")
    print(f"  TP={tp2}  FP={fp2}  FN={fn2}  TN={tn2}")

    full_metrics = {
        'description': 'Full OthmaneSalim dataset (1842 samples, includes train+holdout)',
        'accuracy': float(acc2), 'precision': float(prec2),
        'recall': float(rec2), 'f1': float(f1_2), 'specificity': float(spec2),
        'tp': int(tp2), 'fp': int(fp2), 'fn': int(fn2), 'tn': int(tn2),
        'total': int(len(labels2)),
    }
    with open(OUT_DIR / 'full_othmanesalim_metrics.json', 'w') as f:
        json.dump(full_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ALL DONE — results in {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
