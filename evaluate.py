#!/usr/bin/env python3
"""
ARC-FAULTNET — Evaluation Script
=================================
Comprehensive evaluation with metrics, confusion matrix, and visualizations.

Features:
  - Per-charge accuracy breakdown
  - Confusion matrix (overall and per-charge)
  - ROC curve and AUC
  - Attention map visualization
  - Signal overlay with predictions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

from dataset import ArcFaultDataset, LeaveOneChargeOutSplitter
from model import get_model, ArcFaultNet


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predictions for a subset of the dataset.
    
    Returns:
        labels: Ground truth labels
        probs: Predicted probabilities
        charges: Charge indices
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    all_charges = []
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        
        x_1d_batch = []
        x_2d_batch = []
        labels_batch = []
        charges_batch = []
        
        for idx in batch_indices:
            x_1d, x_2d, label, charge = dataset[idx]
            x_1d_batch.append(x_1d)
            x_2d_batch.append(x_2d)
            labels_batch.append(label)
            charges_batch.append(charge)
        
        x_1d = torch.stack(x_1d_batch).to(device)
        x_2d = torch.stack(x_2d_batch).to(device)
        
        logits = model(x_1d, x_2d)
        probs = torch.sigmoid(logits)
        
        all_labels.extend([l.item() for l in labels_batch])
        all_probs.extend(probs.cpu().numpy())
        all_charges.extend([c.item() for c in charges_batch])
    
    return np.array(all_labels), np.array(all_probs), np.array(all_charges)


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


def compute_per_charge_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    charges: np.ndarray,
    charge_map: Dict[str, int]
) -> Dict:
    """Compute metrics per charge configuration."""
    idx_to_name = {v: k for k, v in charge_map.items()}
    
    per_charge = {}
    for charge_idx in np.unique(charges):
        mask = charges == charge_idx
        if mask.sum() > 0:
            charge_name = idx_to_name.get(charge_idx, f"charge_{charge_idx}")
            metrics = compute_metrics(labels[mask], probs[mask])
            metrics['n_samples'] = int(mask.sum())
            metrics['n_arc'] = int(labels[mask].sum())
            metrics['n_normal'] = int((1 - labels[mask]).sum())
            per_charge[charge_name] = metrics
    
    return per_charge


# ═══════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Arc'])
    ax.set_yticklabels(['Normal', 'Arc'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=14)
    
    plt.colorbar(im)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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


def plot_per_charge_accuracy(
    per_charge_metrics: Dict,
    save_path: Optional[Path] = None
):
    """Plot accuracy per charge configuration."""
    charges = list(per_charge_metrics.keys())
    accuracies = [per_charge_metrics[c]['accuracy'] * 100 for c in charges]
    f1_scores = [per_charge_metrics[c]['f1'] * 100 for c in charges]
    
    x = np.arange(len(charges))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='darkorange')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance per Charge Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(charges, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 105])
    ax.axhline(y=np.mean(accuracies), color='steelblue', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(f1_scores), color='darkorange', linestyle='--', alpha=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_sample_predictions(
    dataset: ArcFaultDataset,
    model: nn.Module,
    device: torch.device,
    indices: np.ndarray,
    n_samples: int = 4,
    save_path: Optional[Path] = None
):
    """Plot sample signals with predictions."""
    model.eval()
    
    # Select random samples
    np.random.shuffle(indices)
    selected = indices[:n_samples]
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    channel_names = ['V_ligne', 'I(t)']
    
    for i, idx in enumerate(selected):
        x_1d, x_2d, label, charge = dataset[idx]
        
        with torch.no_grad():
            x_1d_t = x_1d.unsqueeze(0).to(device)
            x_2d_t = x_2d.unsqueeze(0).to(device)
            logit = model(x_1d_t, x_2d_t)
            prob = torch.sigmoid(logit).item()
        
        pred = 1 if prob > 0.5 else 0
        correct = pred == label.item()
        
        for c in range(2):
            ax = axes[i, c]
            signal = x_1d[c].numpy()
            ax.plot(signal, linewidth=0.5, color='steelblue')
            ax.set_title(f"{channel_names[c]}")
            ax.set_xlim([0, len(signal)])
            
            if c == 0:
                status = "✓" if correct else "✗"
                color = "green" if correct else "red"
                ax.set_ylabel(
                    f"True: {'Arc' if label.item() else 'Normal'}\n"
                    f"Pred: {'Arc' if pred else 'Normal'} ({prob:.2f}) {status}",
                    color=color
                )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ═══════════════════════════════════════════════════════
#  MAIN EVALUATION
# ═══════════════════════════════════════════════════════

def evaluate_model(
    model_path: Path,
    model_name: str,
    dataset: ArcFaultDataset,
    device: torch.device,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Comprehensive model evaluation.
    """
    # Load model
    model = get_model(model_name, in_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get predictions on full dataset
    indices = np.arange(len(dataset))
    labels, probs, charges = get_predictions(model, dataset, indices, device)
    
    # Compute overall metrics
    metrics = compute_metrics(labels, probs)
    
    # Compute per-charge metrics
    per_charge = compute_per_charge_metrics(labels, probs, charges, dataset.charge_map)
    
    # Print results
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
    
    print(f"\nPer-Charge Performance:")
    for charge_name, charge_metrics in per_charge.items():
        print(f"  {charge_name}:")
        print(f"    Acc={100*charge_metrics['accuracy']:.1f}% "
              f"F1={100*charge_metrics['f1']:.1f}% "
              f"(n={charge_metrics['n_samples']})")
    
    # Generate visualizations
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        preds = (probs > 0.5).astype(int)
        
        plot_confusion_matrix(labels, preds, output_dir / 'confusion_matrix.png')
        plot_roc_curve(labels, probs, output_dir / 'roc_curve.png')
        plot_precision_recall_curve(labels, probs, output_dir / 'pr_curve.png')
        plot_per_charge_accuracy(per_charge, output_dir / 'per_charge_accuracy.png')
        plot_sample_predictions(dataset, model, device, indices, n_samples=6,
                               save_path=output_dir / 'sample_predictions.png')
        
        # Save metrics to JSON
        results = {
            'model_name': model_name,
            'model_path': str(model_path),
            'overall_metrics': metrics,
            'per_charge_metrics': per_charge
        }
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nVisualizations saved to: {output_dir}")
    
    return {
        'overall': metrics,
        'per_charge': per_charge
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Arc-FaultNet')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='arcfaultnet',
                        choices=['arcfaultnet', '1d_only', 'no_attention',
                                 'standard_conv', 'independent_cbam', 'baseline_cnn'],
                        help='Model architecture')
    parser.add_argument('--data-dir', type=str, default='/home/top/PFE/labeled_dataset',
                        help='Path to labeled dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualizations')
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
    
    # Evaluate
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    evaluate_model(
        model_path=Path(args.model_path),
        model_name=args.model,
        dataset=dataset,
        device=device,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()
