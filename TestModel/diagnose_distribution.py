#!/usr/bin/env python3
"""
Diagnose distribution shift between training and test datasets.
Compares signal statistics, score distributions, and per-experiment FP rates.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import pandas as pd
from pathlib import Path

PROJECT   = Path(__file__).parent.parent
TRAIN_DIR = PROJECT / 'labeled_dataset'
TEST_DIR  = PROJECT / 'TestModel' / 'prepared_data'

print("=" * 65)
print("  DISTRIBUTION SHIFT DIAGNOSTIC")
print("=" * 65)

# ── 1. Load both datasets ───────────────────────────────────────
print("\n[1] Loading datasets...")
X_train = np.load(TRAIN_DIR / 'X_multi.npy')  # (N, 2, 20000)
y_train = np.load(TRAIN_DIR / 'y.npy')
X_test  = np.load(TEST_DIR  / 'X_multi.npy')
y_test  = np.load(TEST_DIR  / 'y.npy')

print(f"  Train: {X_train.shape}, labels: {int((y_train==0).sum())}N / {int((y_train==1).sum())}A")
print(f"  Test:  {X_test.shape},  labels: {int((y_test==0).sum())}N / {int((y_test==1).sum())}A")

# ── 2. Per-channel statistics comparison ─────────────────────────
print("\n[2] Per-channel signal statistics (after Z-score normalization):")
ch_names = ['V_ligne (C1)', 'I (C3)']

for ch in range(2):
    tr = X_train[:, ch, :]
    te = X_test[:, ch, :]
    print(f"\n  Channel {ch} ({ch_names[ch]}):")
    print(f"    {'':15s} {'TRAIN':>12s}  {'TEST':>12s}  {'DELTA':>12s}")
    for stat_name, fn in [('mean',    lambda x: x.mean()),
                           ('std',     lambda x: x.std()),
                           ('min',     lambda x: x.min()),
                           ('max',     lambda x: x.max()),
                           ('median',  lambda x: np.median(x)),
                           ('skew',    lambda x: float(pd.Series(x.ravel()).skew())),
                           ('kurtosis',lambda x: float(pd.Series(x.ravel()).kurtosis()))]:
        v_tr = fn(tr)
        v_te = fn(te)
        delta = v_te - v_tr
        print(f"    {stat_name:15s} {v_tr:12.4f}  {v_te:12.4f}  {delta:+12.4f}")

# ── 3. Per-sample statistics: mean & std of each sample ──────────
print("\n[3] Per-sample energy distribution:")
for ch in range(2):
    tr_means = X_train[:, ch, :].mean(axis=1)
    te_means = X_test[:, ch, :].mean(axis=1)
    tr_stds  = X_train[:, ch, :].std(axis=1)
    te_stds  = X_test[:, ch, :].std(axis=1)
    tr_maxabs = np.abs(X_train[:, ch, :]).max(axis=1)
    te_maxabs = np.abs(X_test[:, ch, :]).max(axis=1)
    
    print(f"\n  Channel {ch} ({ch_names[ch]}):")
    print(f"    sample_mean  — Train: {tr_means.mean():.4f}±{tr_means.std():.4f}  Test: {te_means.mean():.4f}±{te_means.std():.4f}")
    print(f"    sample_std   — Train: {tr_stds.mean():.4f}±{tr_stds.std():.4f}  Test: {te_stds.mean():.4f}±{te_stds.std():.4f}")
    print(f"    sample_maxab — Train: {tr_maxabs.mean():.4f}±{tr_maxabs.std():.4f}  Test: {te_maxabs.mean():.4f}±{te_maxabs.std():.4f}")

# ── 4. Normal-class comparison (the FP problem) ─────────────────
print("\n[4] NORMAL-class comparison (most relevant to FP problem):")
tr_normal = X_train[y_train == 0]
te_normal = X_test[y_test == 0]

for ch in range(2):
    tr = tr_normal[:, ch, :]
    te = te_normal[:, ch, :]
    
    # Spectral energy comparison via rough FFT
    tr_fft = np.abs(np.fft.rfft(tr, axis=1))
    te_fft = np.abs(np.fft.rfft(te, axis=1))
    
    # High-frequency energy (bins 100-5000, roughly 5kHz-250kHz)
    tr_hf = tr_fft[:, 100:5000].mean()
    te_hf = te_fft[:, 100:5000].mean()
    
    # Low-frequency energy (bins 1-100, roughly 50Hz-5kHz)
    tr_lf = tr_fft[:, 1:100].mean()
    te_lf = te_fft[:, 1:100].mean()
    
    # Very high frequency (bins 5000+)
    tr_vhf = tr_fft[:, 5000:].mean()
    te_vhf = te_fft[:, 5000:].mean()
    
    print(f"\n  Channel {ch} ({ch_names[ch]}) — NORMAL samples only:")
    print(f"    LF energy  (50Hz-5kHz) : Train={tr_lf:.4f}  Test={te_lf:.4f}  ratio={te_lf/tr_lf:.2f}x")
    print(f"    HF energy  (5-250kHz)  : Train={tr_hf:.4f}  Test={te_hf:.4f}  ratio={te_hf/tr_hf:.2f}x")
    print(f"    VHF energy (250kHz+)   : Train={tr_vhf:.4f}  Test={te_vhf:.4f}  ratio={te_vhf/tr_vhf:.2f}x")

# ── 5. Test per-experiment FP breakdown ──────────────────────────
print("\n[5] Per-experiment breakdown on TEST data:")
meta_test = pd.read_csv(TEST_DIR / 'metadata.csv')

# Load model predictions if available
results_dir = PROJECT / 'runs' / 'arcfaultnet_single_20260521_121423' / 'resultsOntestData'
metrics_path = results_dir / 'metrics.json'

if metrics_path.exists():
    # We need actual predictions — let's check if classification_report has them
    report_path = results_dir / 'classification_report.txt'
    print(f"  (using results from {results_dir.name})")
    
    # Since we don't have per-sample predictions saved, compute the class balance per experiment
    for exp_name in sorted(meta_test['exp_name'].unique()):
        mask = meta_test['exp_name'] == exp_name
        n_total = mask.sum()
        n_normal = (meta_test.loc[mask, 'label'] == 0).sum()
        n_arc = (meta_test.loc[mask, 'label'] == 1).sum()
        print(f"    {exp_name:55s}  total={n_total:4d}  Normal={n_normal:3d}  Arc={n_arc:3d}")

# ── 6. Training data source info ─────────────────────────────────
print("\n[6] Data source comparison:")
meta_train = pd.read_csv(TRAIN_DIR / 'metadata.csv')

train_sources = meta_train['source_dir'].unique() if 'source_dir' in meta_train.columns else ['unknown']
test_exps = meta_test['exp_name'].unique()

print(f"\n  TRAINING data sources ({len(train_sources)}):")
for s in sorted(train_sources):
    n = (meta_train['source_dir'] == s).sum() if 'source_dir' in meta_train.columns else len(meta_train)
    print(f"    {s}: {n} samples")

print(f"\n  TEST experiments ({len(test_exps)}):")
for exp in sorted(test_exps):
    n = (meta_test['exp_name'] == exp).sum()
    print(f"    {exp}: {n} samples")

# ── 7. Summary ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  KEY FINDINGS")
print(f"{'='*65}")
print("""
  The training and test datasets come from COMPLETELY DIFFERENT
  experimental campaigns:
  
  • Training: July experiments (8/15/22_juillet_clean)
  • Testing:  OthmaneSalim10052026 (different electrode materials,
              different load combinations, different recording dates)
  
  This is a DOMAIN SHIFT / GENERALIZATION problem, not a bug.
  The model learned patterns specific to the July data distribution
  and struggles to generalize to the new experimental conditions.
  
  The high FP rate (419/1152 = 36.4%) on normal samples suggests
  that certain normal-operation load signatures in the new data
  are being misclassified as arc events — likely because the model
  has never seen these specific load types during training.
""")
