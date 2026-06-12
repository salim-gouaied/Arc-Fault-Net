#!/usr/bin/env python3
"""
Ensemble evaluation: combine predictions from multiple trained models
to reduce False Positives while maintaining or improving Recall.

Strategy 1: Threshold optimization on a single model
Strategy 2: Multi-model ensemble (majority vote / averaged probabilities)
"""

import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from dataset import ArcFaultDataset
from model import build_model_from_checkpoint

# ── Recreate split (mirrors train.py --mode single) ──────────────
def recreate_single_split(n, seed, train_ratio=0.7, val_ratio=0.15):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    indices = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]

# ── Inference ────────────────────────────────────────────────────
@torch.no_grad()
def get_probabilities(model, dataset, indices, device, batch_size=64):
    model.eval()
    loader = DataLoader(Subset(dataset, indices.tolist()),
                        batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    probs, labels = [], []
    for x_1d, x_2d, y, _ in loader:
        logits = model(x_1d.to(device), x_2d.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy().flatten())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)

# ── Metrics ──────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    eps = 1e-12
    return {
        'accuracy':    (tp+tn) / (tp+tn+fp+fn+eps),
        'precision':   tp / (tp+fp+eps),
        'recall':      tp / (tp+fn+eps),
        'f1':          2*tp / (2*tp+fp+fn+eps),
        'specificity': tn / (tn+fp+eps),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

def print_metrics(m, label=""):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Accuracy   : {m['accuracy']*100:.2f}%")
    print(f"  Precision  : {m['precision']*100:.2f}%")
    print(f"  Recall     : {m['recall']*100:.2f}%")
    print(f"  F1         : {m['f1']*100:.2f}%")
    print(f"  Specificity: {m['specificity']*100:.2f}%")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='List of run directories to ensemble')
    ap.add_argument('--data-dir', default='combined_dataset_2048')
    ap.add_argument('--n-fft', type=int, default=128)
    ap.add_argument('--hop-length', type=int, default=64)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = ArcFaultDataset(
        data_dir=args.data_dir, n_fft=args.n_fft,
        hop_length=args.hop_length, channel_mode='i_derived4'
    )

    # ─────────────────────────────────────────────────────────
    # Strategy 1: Threshold optimization on the FIRST model
    # ─────────────────────────────────────────────────────────
    first_run = Path(args.runs[0])
    with open(first_run / 'results.json') as f:
        cfg = json.load(f)
    seed = cfg['seed']

    ckpt = first_run / 'best_single.pt'
    model = build_model_from_checkpoint(str(ckpt), device=str(device),
                                            fs=cfg.get('fs', 102400),
                                            n_fft=cfg.get('n_fft', 128))
    _, val_idx, test_idx = recreate_single_split(len(dataset), seed)

    # Get val probabilities to find optimal threshold
    val_probs, val_labels = get_probabilities(model, dataset, val_idx, device)
    test_probs, test_labels = get_probabilities(model, dataset, test_idx, device)

    # Baseline (threshold=0.5)
    m_base = compute_metrics(test_labels, test_probs, 0.5)
    print_metrics(m_base, f"Baseline (threshold=0.5) — {first_run.name}")

    # Search for optimal threshold on VALIDATION set (minimize FP, keep recall high)
    best_t, best_score = 0.5, -1
    for t in np.arange(0.3, 0.95, 0.01):
        vm = compute_metrics(val_labels, val_probs, t)
        # Score: heavily penalize FP, reward recall
        score = vm['f1'] - 0.5 * (vm['fp'] / (len(val_labels)+1e-12))
        if score > best_score:
            best_score = score
            best_t = t

    m_opt = compute_metrics(test_labels, test_probs, best_t)
    print_metrics(m_opt, f"Optimized threshold={best_t:.2f}")

    # ─────────────────────────────────────────────────────────
    # Strategy 2: Model Ensemble (averaged probabilities)
    # ─────────────────────────────────────────────────────────
    if len(args.runs) > 1:
        print(f"\n\n{'#'*55}")
        print(f"  ENSEMBLE of {len(args.runs)} models")
        print(f"{'#'*55}")

        # We need a COMMON test set. Use the first model's seed for the split.
        all_test_probs = [test_probs]  # already have the first model

        for run_path in args.runs[1:]:
            run_dir = Path(run_path)
            with open(run_dir / 'results.json') as f:
                rcfg = json.load(f)
            ckpt_p = run_dir / 'best_single.pt'
            mdl = build_model_from_checkpoint(str(ckpt_p), device=str(device),
                                                  fs=rcfg.get('fs', 102400),
                                                  n_fft=rcfg.get('n_fft', 128))
            # Use the SAME test_idx (from seed of model 1) so all models predict on same samples
            tp, _ = get_probabilities(mdl, dataset, test_idx, device)
            all_test_probs.append(tp)
            print(f"  Loaded: {run_dir.name} (seed={rcfg['seed']})")

        # Average probabilities
        ensemble_probs = np.mean(all_test_probs, axis=0)

        # Ensemble at threshold=0.5
        m_ens_50 = compute_metrics(test_labels, ensemble_probs, 0.5)
        print_metrics(m_ens_50, "Ensemble — Averaged probs (threshold=0.5)")

        # Majority vote (predict arc only if >50% of models say arc)
        votes = np.array([(p >= 0.5).astype(int) for p in all_test_probs])
        majority = (votes.mean(axis=0) > 0.5).astype(float)
        m_maj = compute_metrics(test_labels, majority, 0.5)
        print_metrics(m_maj, "Ensemble — Strict majority vote")

        # Unanimous vote (predict arc only if ALL models say arc) → minimizes FP
        unanimous = (votes.min(axis=0)).astype(float)
        m_unan = compute_metrics(test_labels, unanimous, 0.5)
        print_metrics(m_unan, "Ensemble — Unanimous vote (ALL models agree → arc)")

        # Save results
        out_path = Path(args.runs[0]) / 'ensemble_results.json'
        results = {
            'models': args.runs,
            'n_models': len(args.runs),
            'baseline_t05': m_base,
            'optimized_threshold': {'threshold': float(best_t), **m_opt},
            'ensemble_avg_t05': m_ens_50,
            'ensemble_majority': m_maj,
            'ensemble_unanimous': m_unan,
        }
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out_path}")

if __name__ == '__main__':
    main()
