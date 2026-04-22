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

Modes:
  --mode random  : Each variant is trained N times with a random 70/15/15 split.
                   Fast, useful for sanity checks; NOT sufficient to claim generalization.
  --mode loco    : Leave-one-charge-out CV.  Proper evaluation protocol.
                   Use this for results reported in the thesis.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader, Subset

from dataset import ArcFaultDataset, LeaveOneChargeOutSplitter, create_dataloaders
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
    seed: int = 42
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

    model    = get_model(variant_name, in_channels=2).to(device)
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

    return {
        'accuracy':   test_metrics['accuracy'],
        'f1':         test_metrics['f1'],
        'precision':  test_metrics['precision'],
        'recall':     test_metrics['recall'],
        'best_epoch': history['best_epoch'],
        'n_params':   n_params
    }


# ═══════════════════════════════════════════════════════
#  SINGLE VARIANT — LEAVE-ONE-CHARGE-OUT
# ═══════════════════════════════════════════════════════

def evaluate_variant_loco(
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
    seed: int = 42
) -> Dict:
    """
    Evaluate one variant with LOCO CV (proper generalization metric).
    Returns per-fold results plus aggregated mean/std.
    """
    splitter    = LeaveOneChargeOutSplitter(dataset)
    fold_results = []

    for fold_idx, (train_indices, test_indices) in enumerate(splitter):
        fold_seed = seed + fold_idx
        set_seed(fold_seed)

        charge_name = splitter.get_fold_name(fold_idx)

        train_loader, val_loader, test_loader = create_dataloaders(
            dataset, train_indices.copy(), test_indices,
            batch_size=batch_size, num_workers=num_workers, val_split=0.15
        )

        pw = None
        if use_pos_weight:
            train_labels = dataset.y[train_indices]
            pw = compute_pos_weight(train_labels, device)

        model    = get_model(variant_name, in_channels=2).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        model, history = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            patience=patience, gradient_clip=gradient_clip,
            pos_weight=pw, checkpoint_dir=None, writer=None,
            fold_name=f"{variant_name}_fold{fold_idx}"
        )

        criterion    = nn.BCEWithLogitsLoss()
        test_metrics = evaluate(model, test_loader, criterion, device, f"Test fold{fold_idx}")

        fold_results.append({
            'fold_idx':    fold_idx,
            'charge_name': charge_name,
            'accuracy':    test_metrics['accuracy'],
            'f1':          test_metrics['f1'],
            'precision':   test_metrics['precision'],
            'recall':      test_metrics['recall'],
            'best_epoch':  history['best_epoch'],
            'n_params':    n_params
        })

    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores  = [r['f1']       for r in fold_results]

    return {
        'accuracy':    float(np.mean(accuracies)),
        'f1':          float(np.mean(f1_scores)),
        'std_accuracy':float(np.std(accuracies)),
        'std_f1':      float(np.std(f1_scores)),
        'precision':   float(np.mean([r['precision'] for r in fold_results])),
        'recall':      float(np.mean([r['recall']    for r in fold_results])),
        'best_epoch':  int(np.mean([r['best_epoch']  for r in fold_results])),
        'n_params':    fold_results[0]['n_params'],
        'fold_results': fold_results
    }


# ═══════════════════════════════════════════════════════
#  FULL ABLATION STUDY
# ═══════════════════════════════════════════════════════

def run_ablation_study(
    dataset: ArcFaultDataset,
    device: torch.device,
    mode: str = 'random',           # 'random' | 'loco'
    n_repetitions: int = 10,        # only used in 'random' mode
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 20,
    gradient_clip: float = 1.0,
    use_pos_weight: bool = False,
    output_dir: Path = Path('ablation_results'),
    num_workers: int = 4
) -> Dict:
    """Run full ablation study across all variants."""
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_dir / f"ablation_{mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY  [mode={mode}]")
    print(f"{'='*70}")
    print(f"Variants:     {len(ABLATION_VARIANTS)}")
    if mode == 'random':
        print(f"Repetitions:  {n_repetitions} per variant (random 70/15/15 split)")
        print(f"  WARNING: random split inflates metrics — use --mode loco for thesis results")
    else:
        print(f"Protocol:     Leave-one-charge-out CV (proper generalization evaluation)")
    print(f"Output:       {output_dir}")
    print(f"{'='*70}\n")

    all_results = {}

    for variant in ABLATION_VARIANTS:
        variant_name = variant['name']
        print(f"\n{'─'*60}")
        print(f"Variant: {variant_name}")
        print(f"Desc:    {variant['description']}")
        print(f"{'─'*60}")

        if mode == 'random':
            variant_reps = []
            for rep in range(n_repetitions):
                seed = 42 + rep
                print(f"\n  Rep {rep + 1}/{n_repetitions}  (seed={seed})")
                result = evaluate_variant_random(
                    variant_name=variant_name, dataset=dataset, device=device,
                    epochs=epochs, lr=lr, weight_decay=weight_decay,
                    batch_size=batch_size, patience=patience,
                    gradient_clip=gradient_clip, use_pos_weight=use_pos_weight,
                    num_workers=num_workers, seed=seed
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

        else:  # loco
            print(f"\n  Running LOCO CV …")
            result = evaluate_variant_loco(
                variant_name=variant_name, dataset=dataset, device=device,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, patience=patience,
                gradient_clip=gradient_clip, use_pos_weight=use_pos_weight,
                num_workers=num_workers, seed=42
            )
            all_results[variant_name] = {
                'description':   variant['description'],
                'category':      variant['category'],
                'n_params':      result['n_params'],
                'mean_accuracy': result['accuracy'],
                'std_accuracy':  result.get('std_accuracy', 0.0),
                'mean_f1':       result['f1'],
                'std_f1':        result.get('std_f1', 0.0),
                'loco_results':  result.get('fold_results', [])
            }

        r = all_results[variant_name]
        print(f"\n  Summary: Acc = {100*r['mean_accuracy']:.2f}% ± {100*r['std_accuracy']:.2f}%")
        print(f"           F1  = {100*r['mean_f1']:.2f}% ± {100*r['std_f1']:.2f}%")

    # ── Comparison table ────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"ABLATION RESULTS  [mode={mode}]")
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

    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'loco'],
                        help='random = fast multi-rep split | loco = leave-one-charge-out (thesis)')
    parser.add_argument('--repetitions', type=int, default=10,
                        help='(random mode) Number of repetitions per variant')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--use-pos-weight', action='store_true')
    parser.add_argument('--data-dir', type=str, default='/home/top/PFE/labeled_dataset')
    parser.add_argument('--output-dir', type=str, default='/home/top/PFE/ablation_results')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

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
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
