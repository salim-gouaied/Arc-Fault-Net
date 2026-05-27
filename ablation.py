#!/usr/bin/env python3
"""
ARC-FAULTNET — Ablation Study
==============================
Runs all model variants to measure the contribution of each component.

Variants:
  1. arcfaultnet      : Full model (dual-branch + Joint Attention + Gabor filters)
  2. standard_conv    : Standard Conv1d instead of ParametricConv1d (no Gabor)
  3. no_attention     : Dual-branch but simple concatenation, no Joint Attention
  4. 1d_only          : Only temporal branch (no STFT)
  5. independent_cbam : CBAM applied independently per branch (no cross-attention)
  6. baseline_cnn     : Simple Conv1d CNN baseline

Each variant is trained N times with a random 70/15/15 split using different seeds.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader, Subset

from dataset import ArcFaultDataset
from model import get_model
from train import set_seed, train_model, evaluate, compute_pos_weight


# ═══════════════════════════════════════════════════════
#  ABLATION VARIANTS
# ═══════════════════════════════════════════════════════

ABLATION_VARIANTS = [
    {
        'name':        'arcfaultnet',
        'description': 'Full model (dual-branch + Joint Attention + Gabor filters)',
        'category':    'full'
    },
    {
        'name':        'standard_conv',
        'description': 'Standard Conv1d instead of ParametricConv1d (Gabor disabled)',
        'category':    'no_parametric'
    },
    {
        'name':        'no_attention',
        'description': 'Dual-branch, simple concatenation, no Joint Attention',
        'category':    'no_attention'
    },
    {
        'name':        '1d_only',
        'description': 'Only temporal branch — no STFT spectrogram',
        'category':    'no_stft'
    },
    {
        'name':        'independent_cbam',
        'description': 'CBAM per branch independently (no cross-branch guidance)',
        'category':    'no_cross'
    },
    {
        'name':        'baseline_cnn',
        'description': 'Simple Conv1d CNN baseline (no attention, no Gabor, no STFT)',
        'category':    'baseline'
    },
]




def plot_cm_roc(model, data_loader, device, variant_name, fold_idx, output_dir: Path):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                x_1d, x_2d, labels = batch['x_1d'].to(device), batch['x_2d'].to(device), batch['label'].numpy()
            else:
                x_1d, x_2d, labels, _ = batch
                x_1d, x_2d, labels = x_1d.to(device), x_2d.to(device), labels.numpy()
            probs = torch.sigmoid(model(x_1d, x_2d)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
            
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    preds = (probs >= 0.5).astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Arc']); ax.set_yticklabels(['Normal', 'Arc'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {variant_name} (Rep {fold_idx})')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    fig.savefig(output_dir / f'cm_{variant_name}_rep{fold_idx}.png', dpi=120)
    plt.close(fig)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {variant_name} (Rep {fold_idx})')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(output_dir / f'roc_{variant_name}_rep{fold_idx}.png', dpi=120)
    plt.close(fig)

# ═══════════════════════════════════════════════════════
#  SINGLE VARIANT — RANDOM SPLIT
# ═══════════════════════════════════════════════════════

def evaluate_variant_random(
    variant_name: str,
    dataset: ArcFaultDataset,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 20,
    gradient_clip: float = 1.0,
    use_pos_weight: bool = False,
    num_workers: int = 4,
    seed: int = 42,
    output_dir: Path = None,
    rep_idx: int = 1,
    use_se: bool = False,
    se_reduction: int = 8,
    use_amplitude: bool = False,
    deep_classifier: bool = False
) -> Dict:
    """Train one variant with a random 70/15/15 split (fast, NOT for generalization)."""
    set_seed(seed)

    indices = np.random.permutation(len(dataset))
    n_train = int(len(dataset) * 0.70)
    n_val   = int(len(dataset) * 0.15)

    train_indices = indices[:n_train]
    val_indices   = indices[n_train:n_train + n_val]
    test_indices  = indices[n_train + n_val:]

    train_subset = Subset(dataset, train_indices)
    val_subset   = Subset(dataset, val_indices)
    test_subset  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    pw = None
    if use_pos_weight:
        train_labels = dataset.y[train_indices]
        pw = compute_pos_weight(train_labels, device)

    model    = get_model(variant_name, in_channels=2, use_se=use_se, se_reduction=se_reduction,
                         use_amplitude=use_amplitude, deep_classifier=deep_classifier).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    model, history = train_model(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        patience=patience, gradient_clip=gradient_clip,
        pos_weight=pw, checkpoint_dir=None, writer=None,
        fold_name=variant_name
    )


    criterion    = nn.BCEWithLogitsLoss()
    test_metrics = evaluate(model, test_loader, criterion, device, "Test")

    if output_dir is not None:
        plot_cm_roc(model, test_loader, device, variant_name, rep_idx, output_dir)


    return {
        'accuracy':   test_metrics['accuracy'],
        'f1':         test_metrics['f1'],
        'precision':  test_metrics['precision'],
        'recall':     test_metrics['recall'],
        'best_epoch': history['best_epoch'],
        'n_params':   n_params
    }




# ═══════════════════════════════════════════════════════
#  FULL ABLATION STUDY
# ═══════════════════════════════════════════════════════

def run_ablation_study(
    dataset: ArcFaultDataset,
    device: torch.device,
    mode: str = 'random',
    n_repetitions: int = 10,        # only used in 'random' mode
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 20,
    gradient_clip: float = 1.0,
    use_pos_weight: bool = False,
    output_dir: Path = Path('ablation_results'),
    num_workers: int = 4,
    base_seed: int = 42,
    use_se: bool = False,
    se_reduction: int = 8,
    use_amplitude: bool = False,
    deep_classifier: bool = False
) -> Dict:
    """Run full ablation study across all variants."""
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_dir / f"ablation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY")
    print(f"{'='*70}")
    print(f"Variants:     {len(ABLATION_VARIANTS)}")
    print(f"Repetitions:  {n_repetitions} per variant (random 70/15/15 split)")
    print(f"Output:       {output_dir}")
    print(f"{'='*70}\n")

    all_results = {}

    for variant in ABLATION_VARIANTS:
        variant_name = variant['name']
        print(f"\n{'─'*60}")
        print(f"Variant: {variant_name}")
        print(f"Desc:    {variant['description']}")
        print(f"{'─'*60}")

        variant_reps = []
        for rep in range(n_repetitions):
            seed = base_seed + rep
            print(f"\n  Rep {rep + 1}/{n_repetitions}  (seed={seed})")
            result = evaluate_variant_random(
                variant_name=variant_name, dataset=dataset, device=device,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, patience=patience,
                gradient_clip=gradient_clip, use_pos_weight=use_pos_weight,
                num_workers=num_workers, seed=seed, output_dir=output_dir, rep_idx=rep+1,
                use_se=use_se, se_reduction=se_reduction,
                use_amplitude=use_amplitude, deep_classifier=deep_classifier
            )
            variant_reps.append(result)
            print(f"    Acc={100*result['accuracy']:.2f}%  F1={100*result['f1']:.2f}%  epoch={result['best_epoch']}")

        accuracies = [r['accuracy'] for r in variant_reps]
        f1_scores  = [r['f1']       for r in variant_reps]

        all_results[variant_name] = {
            'description':  variant['description'],
            'category':     variant['category'],
            'n_params':     variant_reps[0]['n_params'],
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy':  float(np.std(accuracies)),
            'mean_f1':       float(np.mean(f1_scores)),
            'std_f1':        float(np.std(f1_scores)),
            'repetitions':  variant_reps
        }

        r = all_results[variant_name]
        print(f"\n  Summary: Acc = {100*r['mean_accuracy']:.2f}% ± {100*r['std_accuracy']:.2f}%")
        print(f"           F1  = {100*r['mean_f1']:.2f}% ± {100*r['std_f1']:.2f}%")

    # ── Comparison table ────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Model':<22} {'Accuracy':<22} {'F1 Score':<22} {'Params'}")
    print(f"{'─'*70}")

    sorted_results = sorted(all_results.items(),
                            key=lambda x: x[1]['mean_accuracy'], reverse=True)
    baseline_acc = all_results['arcfaultnet']['mean_accuracy']

    for name, result in sorted_results:
        acc_str   = f"{100*result['mean_accuracy']:.2f}% ± {100*result['std_accuracy']:.2f}%"
        f1_str    = f"{100*result['mean_f1']:.2f}% ± {100*result['std_f1']:.2f}%"
        delta     = result['mean_accuracy'] - baseline_acc
        delta_str = f" ({delta*100:+.2f}%)" if name != 'arcfaultnet' else " (ref)"
        print(f"{name:<22} {acc_str:<22} {f1_str:<22} {result['n_params']:,}{delta_str}")

    print(f"\n{'='*70}")
    print(f"COMPONENT CONTRIBUTION")
    print(f"{'='*70}")

    contributions = [
        ('Gabor filters (vs standard Conv)',   baseline_acc - all_results['standard_conv']['mean_accuracy']),
        ('Joint Attention (vs no attention)',   baseline_acc - all_results['no_attention']['mean_accuracy']),
        ('STFT branch (vs 1D only)',            baseline_acc - all_results['1d_only']['mean_accuracy']),
        ('Cross-attention (vs indep. CBAM)',    baseline_acc - all_results['independent_cbam']['mean_accuracy']),
        ('Full model vs Baseline CNN',          baseline_acc - all_results['baseline_cnn']['mean_accuracy']),
    ]
    for component, delta in contributions:
        sign = "+" if delta >= 0 else ""
        print(f"  {component:<40}: {sign}{100*delta:.2f}%")

    # ── Save results ────────────────────────────────────────────────
    results_summary = {
        'mode':           mode,
        'timestamp':      timestamp,
        'n_repetitions':  n_repetitions if mode == 'random' else None,
        'epochs':         epochs, 'lr': lr, 'weight_decay': weight_decay,
        'batch_size':     batch_size, 'patience': patience,
        'gradient_clip':  gradient_clip, 'use_pos_weight': use_pos_weight,
        'variants':       all_results,
        'contributions':  {name: float(delta) for name, delta in contributions}
    }
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    generate_ablation_plot(all_results, output_dir / 'ablation_comparison.png')
    generate_contribution_plot(contributions, output_dir / 'component_contributions.png')

    print(f"\nResults saved to: {output_dir}")
    return results_summary


def generate_ablation_plot(results: Dict, save_path: Path):
    """Bar chart comparing all variants."""
    import matplotlib.pyplot as plt

    sorted_items = sorted(results.items(),
                          key=lambda x: x[1]['mean_accuracy'], reverse=True)
    names  = [item[0]                              for item in sorted_items]
    accs   = [item[1]['mean_accuracy'] * 100       for item in sorted_items]
    stds   = [item[1]['std_accuracy']  * 100       for item in sorted_items]
    colors = ['#2ecc71' if n == 'arcfaultnet' else '#3498db' for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), accs, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study — Model Variant Comparison', fontsize=14)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([max(0, min(accs) - 10), 100])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_contribution_plot(contributions: List, save_path: Path):
    """Horizontal bar chart of component contributions."""
    import matplotlib.pyplot as plt

    names  = [c[0] for c in contributions]
    deltas = [c[1] * 100 for c in contributions]
    colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(names))
    bars  = ax.barh(y_pos, deltas, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Contribution to Accuracy (%)', fontsize=12)
    ax.set_title('Component Contribution Analysis', fontsize=14)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3)
    for bar, delta in zip(bars, deltas):
        x_pos = bar.get_width() + 0.1 if delta >= 0 else bar.get_width() - 0.1
        ha    = 'left' if delta >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{delta:+.2f}%', ha=ha, va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Run Arc-FaultNet Ablation Study')

    # --mode kept for backwards compatibility but only 'random' is supported
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random'],
                        help='random = multi-rep random split')
    parser.add_argument('--repetitions', type=int, default=10,
                        help='(random mode) Number of repetitions per variant')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--use-pos-weight', action='store_true')
    parser.add_argument('--data-dir', type=str, default='/home/manip/pfe_salim_gouaied/Arc-Fault-Net/labeled_dataset')
    parser.add_argument('--output-dir', type=str, default='/home/manip/pfe_salim_gouaied/Arc-Fault-Net/ablation_results')
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--seed', type=int, default=42, help='Base seed for experiments')

    
    # Architecture enhancement flags
    parser.add_argument('--use-se', action='store_true', help='Add Squeeze-and-Excitation blocks')
    parser.add_argument('--se-reduction', type=int, default=8, help='SE block reduction ratio')
    parser.add_argument('--use-amplitude', action='store_true', help='Add learnable amplitude to Gabor filters')
    parser.add_argument('--deep-clf', action='store_true', help='Use deeper classifier head')

    args = parser.parse_args()

    device = torch.device('cpu') if (args.cpu or not torch.cuda.is_available()) \
             else torch.device('cuda')
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == 'cuda' else ''))

    data_dir = Path(args.data_dir)
    if not (data_dir / 'X_multi.npy').exists():
        print(f"\nData not found at {data_dir}")
        print("Run: python step2_build_multichannel.py")
        return

    dataset = ArcFaultDataset(data_dir=str(data_dir))

    run_ablation_study(
        dataset=dataset,
        device=device,
        mode=args.mode,
        n_repetitions=args.repetitions,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        use_pos_weight=args.use_pos_weight,
        output_dir=Path(args.output_dir),
        num_workers=args.num_workers,
        base_seed=args.seed,
        use_se=args.use_se,
        se_reduction=args.se_reduction,
        use_amplitude=args.use_amplitude,
        deep_classifier=args.deep_clf
    )


if __name__ == '__main__':
    main()
