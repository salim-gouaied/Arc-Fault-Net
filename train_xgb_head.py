#!/usr/bin/env python3
"""
ARC-FAULTNET V2 — Stage 5: Hybrid tree classifier on the deep embedding
=======================================================================

Two-phase protocol from the V2 spec:

  Phase 1 (done elsewhere, in train.py):
      Train ArcFaultNetV2 end-to-end with its FC head (BCEWithLogitsLoss).
      Save the best checkpoint.

  Phase 2 (THIS script):
      1. Load the frozen ArcFaultNetV2 checkpoint.
      2. Recreate the SAME train/val/test split used at training time.
      3. Run every window through the frozen network and collect the fused
         128-dim embeddings (model.extract_embedding / return_embedding=True).
      4. Train an XGBoost (or RandomForest) classifier on those embeddings.
      5. Report metrics on the test split and save the tree model + a report.

This decouples the representation (deep net) from the decision (tree), which
tends to generalise better on small/medium datasets and yields calibrated
probabilities + feature importance over the 128 embedding dimensions.

NOTE: This script trains a model and therefore needs the dataset. It is meant
to run on the server where the data lives. It is intentionally NOT executed in
environments without the dataset.

Example
-------
    python train_xgb_head.py \
        --checkpoint runs/arcfaultnet_v2_single_XXXX/best_single.pt \
        --data-dir combined_dataset_2048 \
        --classifier xgboost --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import ArcFaultDataset
from model import ArcFaultNetV2


# ─────────────────────────────────────────────────────────────────────
#  SPLIT RECREATION (mirrors train.py --mode single)
# ─────────────────────────────────────────────────────────────────────

def recreate_single_split(n: int, seed: int,
                           train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Reproduce the random 70/15/15 split used by train.py --mode single."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    indices = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (indices[:n_train],
            indices[n_train:n_train + n_val],
            indices[n_train + n_val:])


# ─────────────────────────────────────────────────────────────────────
#  EMBEDDING EXTRACTION
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model: ArcFaultNetV2, dataset: ArcFaultDataset,
                       indices: np.ndarray, device: torch.device,
                       batch_size: int = 64, num_workers: int = 4):
    """Run the frozen network and collect (embeddings, labels) for `indices`."""
    model.eval()
    loader = DataLoader(Subset(dataset, indices.tolist()),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    embs, labels = [], []
    for x_1d, x_2d, y, _ in loader:
        x_1d = x_1d.to(device)
        x_2d = x_2d.to(device)
        _, emb = model(x_1d, x_2d, return_embedding=True)
        embs.append(emb.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labels)


# ─────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────

def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    from sklearn.metrics import roc_auc_score
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    eps = 1e-12
    precision   = tp / (tp + fp + eps)
    recall      = tp / (tp + fn + eps)
    f1          = 2 * precision * recall / (precision + recall + eps)
    specificity = tn / (tn + fp + eps)
    accuracy    = (tp + tn) / (tp + tn + fp + fn + eps)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float('nan')
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'specificity': specificity, 'auc_roc': auc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


# ─────────────────────────────────────────────────────────────────────
#  CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────

def train_xgboost(Xtr, ytr, Xval, yval, seed: int):
    import xgboost as xgb
    n_pos = float((ytr == 1).sum())
    n_neg = float((ytr == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return clf


def train_random_forest(Xtr, ytr, Xval, yval, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        class_weight='balanced',
        max_features='sqrt',
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    return clf


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="V2 Stage-5 tree head on deep embeddings")
    ap.add_argument('--checkpoint', required=True, help='Path to frozen ArcFaultNetV2 .pt')
    ap.add_argument('--data-dir', required=True, help='Dataset dir (e.g. combined_dataset_2048)')
    ap.add_argument('--classifier', choices=['xgboost', 'rf'], default='xgboost')
    ap.add_argument('--split', choices=['single'], default='single',
                    help='Split protocol used during Phase-1 training')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--n-fft', type=int, default=128)
    ap.add_argument('--hop-length', type=int, default=64)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--output-dir', type=str, default=None,
                    help='Where to save tree model + report (default: alongside checkpoint)')
    args = ap.parse_args()

    device = torch.device('cpu') if (args.cpu or not torch.cuda.is_available()) else torch.device('cuda')
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset (V2 uses the 4 I-derived channels) ──────────────────
    dataset = ArcFaultDataset(
        data_dir=args.data_dir, n_fft=args.n_fft, hop_length=args.hop_length,
        channel_mode='i_derived4'
    )

    # ── Frozen deep network ─────────────────────────────────────────
    model = ArcFaultNetV2(in_channels=4, spec_in_channels=1).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print(f"Loaded frozen checkpoint: {ckpt_path}")

    # ── Recreate the exact split ────────────────────────────────────
    tr_idx, val_idx, te_idx = recreate_single_split(len(dataset), args.seed)
    print(f"Split (seed={args.seed}): train={len(tr_idx)}, val={len(val_idx)}, test={len(te_idx)}")

    # ── Extract embeddings ──────────────────────────────────────────
    print("Extracting embeddings...")
    Xtr, ytr = extract_embeddings(model, dataset, tr_idx, device, args.batch_size, args.num_workers)
    Xval, yval = extract_embeddings(model, dataset, val_idx, device, args.batch_size, args.num_workers)
    Xte, yte = extract_embeddings(model, dataset, te_idx, device, args.batch_size, args.num_workers)
    print(f"  embeddings: train{Xtr.shape}, val{Xval.shape}, test{Xte.shape}")

    # ── Train tree classifier ───────────────────────────────────────
    print(f"Training {args.classifier} on 128-d embeddings...")
    if args.classifier == 'xgboost':
        clf = train_xgboost(Xtr, ytr, Xval, yval, args.seed)
    else:
        clf = train_random_forest(Xtr, ytr, Xval, yval, args.seed)

    # ── Evaluate on test ────────────────────────────────────────────
    yte_prob = clf.predict_proba(Xte)[:, 1]
    metrics = binary_metrics(yte, yte_prob, args.threshold)
    print("\nTest metrics (tree head):")
    for k in ('accuracy', 'precision', 'recall', 'f1', 'specificity', 'auc_roc'):
        print(f"  {k:<12s}: {metrics[k]*100:.2f}%" if k != 'auc_roc' else f"  {k:<12s}: {metrics[k]:.4f}")

    # ── Persist ─────────────────────────────────────────────────────
    report = {
        'checkpoint': str(ckpt_path),
        'data_dir': args.data_dir,
        'classifier': args.classifier,
        'split': args.split,
        'seed': args.seed,
        'threshold': args.threshold,
        'n_train': int(len(tr_idx)), 'n_val': int(len(val_idx)), 'n_test': int(len(te_idx)),
        'test_metrics': metrics,
    }
    (out_dir / f'xgb_head_report_{args.classifier}.json').write_text(json.dumps(report, indent=2))

    try:
        import joblib
        joblib.dump(clf, out_dir / f'tree_head_{args.classifier}.joblib')
        print(f"\nSaved tree model -> {out_dir / f'tree_head_{args.classifier}.joblib'}")
    except Exception as e:
        print(f"  (could not pickle tree model: {e})")

    print(f"Saved report     -> {out_dir / f'xgb_head_report_{args.classifier}.json'}")


if __name__ == '__main__':
    main()
