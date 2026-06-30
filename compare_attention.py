#!/usr/bin/env python3
"""
ARC-FAULTNET V2 — Attention Mechanism Comparison
================================================
Compares the old RevisedCrossAttention (gated fusion) against the new 
True Q/K/V Cross-Attention, with SE blocks and Deep Classifier DISABLED
to isolate the raw impact of the attention mechanism on the base architecture.
"""

import torch, torch.nn as nn, numpy as np, json, time
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import ArcFaultDataset
from model import ArcFaultNetV2
from ablation_v3 import train_variant, set_seed

VARIANTS = [
    {
        'key': 'old_gated_attention',
        'label': 'Old: Revised Cross-Attn (Gated)',
        'desc': 'No SE, No Deep Head, fusion_mode="gated"',
        'color': '#e74c3c',
        'build': lambda: ArcFaultNetV2(
            in_channels=4, spec_in_channels=1,
            fusion_mode='gated', use_se=False,
            deep_classifier=False, use_freq_gate=True),
    },
    {
        'key': 'new_true_attention',
        'label': 'New: True Q/K/V Cross-Attn',
        'desc': 'No SE, No Deep Head, fusion_mode="cross_attention"',
        'color': '#2ecc71',
        'build': lambda: ArcFaultNetV2(
            in_channels=4, spec_in_channels=1,
            fusion_mode='cross_attention', use_se=False,
            deep_classifier=False, use_freq_gate=True),
    }
]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    seed = 42
    set_seed(seed)

    dataset = ArcFaultDataset(
        data_dir='/home/manip/pfe_salim_gouaied/Arc-Fault-Net/combined_dataset_2048',
        n_fft=128, hop_length=64, channel_mode='i_derived4'
    )

    indices = np.random.permutation(len(dataset))
    n_train = int(len(dataset) * 0.70)
    n_val   = int(len(dataset) * 0.15)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]
    print(f"\nSplit: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    all_results = {}
    
    for v in VARIANTS:
        print(f"\n{'─'*60}")
        print(f"  Training: {v['label']}")
        print(f"  {v['desc']}")
        print(f"{'─'*60}")

        set_seed(seed)

        # Using train_variant from ablation_v3
        model, metrics = train_variant(
            variant=v, dataset=dataset,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            device=device, epochs=200, lr=3e-4, wd=5e-4, bs=64,
            patience=15, grad_clip=0.5, num_workers=4
        )
        
        all_results[v['key']] = metrics
        print(f"  → Acc={100*metrics['accuracy']:.2f}%  F1={100*metrics['f1']:.2f}%  Params={metrics['n_params']:,}")

    # Summarize directly in console
    old = all_results['old_gated_attention']
    new = all_results['new_true_attention']
    
    print("\n" + "="*70)
    print("  ATTENTION MECHANISM COMPARISON RESULTS")
    print("="*70)
    print(f"  {'Metric':<15} | {'Old (Gated)':<15} | {'New (True Q/K/V)':<15} | {'Delta':<10}")
    print("-" * 70)
    for m in ['accuracy', 'f1', 'precision', 'recall', 'specificity']:
        o_val = old[m] * 100
        n_val = new[m] * 100
        delta = n_val - o_val
        print(f"  {m.capitalize():<15} | {o_val:>14.2f}% | {n_val:>14.2f}% | {delta:>9.2f}%")
    print("-" * 70)

if __name__ == '__main__':
    main()
