#!/usr/bin/env python3
"""
Merge labeled_dataset + TestModel/prepared_data into a single combined dataset.

Strategy:
  - Combine both X_multi.npy and y.npy
  - Keep 20% of OthmaneSalim data held-out as a truly unseen test set
  - Save the rest merged with the original training data

Outputs (saved to combined_dataset/):
  X_multi.npy   — (N_total, 2, 20000)
  y.npy         — (N_total,)
  charges.npy   — (N_total,) — dummy single group (no LOCO needed)
  charge_map.json
  config.json
  metadata.csv
  held_out_indices.npy — indices of OthmaneSalim samples reserved for final test
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

PROJECT     = Path(__file__).parent
TRAIN_DIR   = PROJECT / 'labeled_dataset'
TEST_DIR    = PROJECT / 'TestModel' / 'prepared_data'
OUTPUT_DIR  = PROJECT / 'combined_dataset'
SEED        = 42
HOLDOUT_RATIO = 0.20  # 20% of OthmaneSalim reserved for final evaluation

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("=" * 60)
    print("  MERGE DATASETS FOR RETRAINING")
    print("=" * 60)

    # ── Load original training data ────────────────────────────
    print("\n[1] Loading original training data (labeled_dataset)...")
    X_orig = np.load(TRAIN_DIR / 'X_multi.npy')
    y_orig = np.load(TRAIN_DIR / 'y.npy')
    meta_orig = pd.read_csv(TRAIN_DIR / 'metadata.csv')
    print(f"    Shape: {X_orig.shape}")
    print(f"    Labels: {int((y_orig==0).sum())} normal, {int((y_orig==1).sum())} arc")

    # ── Load OthmaneSalim test data ────────────────────────────
    print("\n[2] Loading OthmaneSalim data (TestModel/prepared_data)...")
    X_new = np.load(TEST_DIR / 'X_multi.npy')
    y_new = np.load(TEST_DIR / 'y.npy')
    meta_new = pd.read_csv(TEST_DIR / 'metadata.csv')
    print(f"    Shape: {X_new.shape}")
    print(f"    Labels: {int((y_new==0).sum())} normal, {int((y_new==1).sum())} arc")

    # ── Hold out 20% of OthmaneSalim for truly unseen testing ──
    print(f"\n[3] Reserving {int(HOLDOUT_RATIO*100)}% of OthmaneSalim as held-out test...")
    n_new = len(y_new)
    perm = np.random.permutation(n_new)
    n_holdout = int(n_new * HOLDOUT_RATIO)
    holdout_idx = perm[:n_holdout]      # indices in OthmaneSalim
    merge_idx   = perm[n_holdout:]      # indices to merge with training

    X_holdout = X_new[holdout_idx]
    y_holdout = y_new[holdout_idx]

    X_merge = X_new[merge_idx]
    y_merge = y_new[merge_idx]

    print(f"    Held-out: {len(holdout_idx)} samples "
          f"({int((y_holdout==0).sum())}N / {int((y_holdout==1).sum())}A)")
    print(f"    To merge: {len(merge_idx)} samples "
          f"({int((y_merge==0).sum())}N / {int((y_merge==1).sum())}A)")

    # ── Combine ─────────────────────────────────────────────────
    print("\n[4] Combining datasets...")
    X_combined = np.concatenate([X_orig, X_merge], axis=0)
    y_combined = np.concatenate([y_orig, y_merge], axis=0)

    # Create source tracking: 0 = original, 1 = OthmaneSalim
    source = np.concatenate([
        np.zeros(len(y_orig), dtype=np.int64),
        np.ones(len(y_merge), dtype=np.int64)
    ])

    print(f"    Combined: {X_combined.shape}")
    print(f"    Labels: {int((y_combined==0).sum())} normal, {int((y_combined==1).sum())} arc")
    print(f"    From original: {len(y_orig)} | From OthmaneSalim: {len(y_merge)}")

    # ── Build metadata ──────────────────────────────────────────
    # Standardize columns
    meta_orig_std = meta_orig.copy()
    if 'source_dir' in meta_orig_std.columns:
        meta_orig_std['dataset'] = meta_orig_std['source_dir']
    else:
        meta_orig_std['dataset'] = 'juillet'
    if 'exp_id' in meta_orig_std.columns:
        meta_orig_std.rename(columns={'exp_id': 'exp_name'}, inplace=True)

    meta_new_merge = meta_new.iloc[merge_idx].copy()
    meta_new_merge['dataset'] = 'OthmaneSalim10052026'

    # Keep common columns
    common_cols = ['dataset', 'exp_name', 'alt_index', 'arc_ratio', 'label',
                   'start_sample', 'end_sample']
    for col in common_cols:
        if col not in meta_orig_std.columns:
            meta_orig_std[col] = ''
        if col not in meta_new_merge.columns:
            meta_new_merge[col] = ''

    meta_combined = pd.concat([
        meta_orig_std[common_cols],
        meta_new_merge[common_cols]
    ], ignore_index=True)

    # ── Save ────────────────────────────────────────────────────
    print("\n[5] Saving combined dataset...")
    np.save(OUTPUT_DIR / 'X_multi.npy', X_combined)
    np.save(OUTPUT_DIR / 'y.npy', y_combined)
    np.save(OUTPUT_DIR / 'source.npy', source)
    np.save(OUTPUT_DIR / 'holdout_X.npy', X_holdout)
    np.save(OUTPUT_DIR / 'holdout_y.npy', y_holdout)

    # Dummy charges (single group — use --mode single for training)
    charges = np.zeros(len(y_combined), dtype=np.int64)
    np.save(OUTPUT_DIR / 'charges.npy', charges)
    with open(OUTPUT_DIR / 'charge_map.json', 'w') as f:
        json.dump({'combined': 0}, f, indent=2)

    meta_combined.to_csv(OUTPUT_DIR / 'metadata.csv', index=False)

    config = {
        'description': 'Combined: labeled_dataset (juillet) + OthmaneSalim10052026 (80%)',
        'n_samples': int(len(y_combined)),
        'n_label0': int((y_combined == 0).sum()),
        'n_label1': int((y_combined == 1).sum()),
        'n_from_original': int(len(y_orig)),
        'n_from_othmanesalim': int(len(y_merge)),
        'n_holdout': int(len(y_holdout)),
        'holdout_ratio': HOLDOUT_RATIO,
        'seed': SEED,
        'FS': 1000000,
        'F0': 50,
        'SAMPLES_PER_CYCLE': 20000,
        'n_channels': 2,
        'channel_names': ['V_ligne', 'I'],
        'X_shape': list(X_combined.shape),
    }
    with open(OUTPUT_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SAVED TO {OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"  X_multi.npy     {X_combined.shape}")
    print(f"  y.npy           {y_combined.shape}")
    print(f"  source.npy      {source.shape} (0=juillet, 1=OthmaneSalim)")
    print(f"  holdout_X.npy   {X_holdout.shape} (reserved for final test)")
    print(f"  holdout_y.npy   {y_holdout.shape}")
    print(f"  metadata.csv    {meta_combined.shape}")
    print(f"  config.json")
    print(f"\n  Next: python train.py --mode single --data-dir combined_dataset --epochs 100")


if __name__ == '__main__':
    main()
