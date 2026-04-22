#!/usr/bin/env python3
"""
ARC-FAULTNET — Pipeline Sanity Check
======================================
Runs 5 fast checks to validate the full pipeline before a long training run.

Checks:
  1. Dataset loading      — shapes, label distribution, charge map
  2. Batch shapes         — x_1d (B,2,20000), x_2d (B,2,257,n_time)
  3. Forward pass         — logits shape, no NaN/Inf
  4. Loss + backward      — BCE loss, gradient norm non-zero
  5. Overfit mini-batch   — 100 gradient steps on 1 batch, loss must drop below 0.10

Usage:
  python sanity_check.py
  python sanity_check.py --data-dir /home/top/PFE/labeled_dataset --batch-size 8
  python sanity_check.py --cpu
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


PASS = "[ PASS ]"
FAIL = "[ FAIL ]"
INFO = "[  --  ]"


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def check(condition: bool, msg_ok: str, msg_fail: str) -> bool:
    if condition:
        print(f"  {PASS}  {msg_ok}")
    else:
        print(f"  {FAIL}  {msg_fail}")
    return condition


def main():
    parser = argparse.ArgumentParser(description='Arc-FaultNet Pipeline Sanity Check')
    parser.add_argument('--data-dir', type=str, default='/home/top/PFE/labeled_dataset')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--overfit-iters', type=int, default=100,
                        help='Number of gradient steps for overfit mini-batch test')
    parser.add_argument('--overfit-threshold', type=float, default=0.10,
                        help='Loss must drop below this value during overfit test')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    args = parser.parse_args()

    device = torch.device('cpu') if (args.cpu or not torch.cuda.is_available()) \
             else torch.device('cuda')

    print(f"\n{'='*60}")
    print(f"  ARC-FAULTNET — SANITY CHECK")
    print(f"{'='*60}")
    print(f"  device     : {device}")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  batch_size : {args.batch_size}")

    all_ok = True
    data_dir = Path(args.data_dir)

    # ──────────────────────────────────────────────────
    # CHECK 1 — Dataset loading
    # ──────────────────────────────────────────────────
    section("CHECK 1 — Dataset loading")

    try:
        from dataset import ArcFaultDataset
        dataset = ArcFaultDataset(data_dir=str(data_dir), device='cpu')
    except Exception as e:
        print(f"  {FAIL}  Cannot load ArcFaultDataset: {e}")
        sys.exit(1)

    ok = check(dataset.X.shape[1] == 2,
               f"X channels = {dataset.X.shape[1]} (expected 2)",
               f"X channels = {dataset.X.shape[1]} — expected 2 (V_ligne, I)")
    all_ok &= ok

    ok = check(dataset.X.shape[2] == 20000,
               f"X length = {dataset.X.shape[2]} (expected 20000)",
               f"X length = {dataset.X.shape[2]} — expected 20000 samples")
    all_ok &= ok

    ok = check(len(dataset.y) == dataset.X.shape[0],
               f"y aligned with X: {len(dataset.y)} samples",
               f"y/X length mismatch: {len(dataset.y)} vs {dataset.X.shape[0]}")
    all_ok &= ok

    ok = check(len(np.unique(dataset.y)) == 2,
               f"Binary labels: values = {np.unique(dataset.y).tolist()}",
               f"Labels not binary: {np.unique(dataset.y).tolist()}")
    all_ok &= ok

    n_arc    = int(np.sum(dataset.y == 1))
    n_normal = int(np.sum(dataset.y == 0))
    ratio    = n_arc / (n_arc + n_normal + 1e-8)
    print(f"  {INFO}  Label distribution: {n_normal} normal / {n_arc} arc "
          f"(arc ratio = {ratio:.3f})")
    if ratio < 0.1 or ratio > 0.9:
        print(f"  {INFO}  Class imbalance detected — consider --use-pos-weight in train.py")

    print(f"  {INFO}  Charges: {dataset.n_charges} unique configurations")
    for name, idx in dataset.charge_map.items():
        n = int(np.sum(dataset.charges == idx))
        print(f"             [{idx:2d}] {name}: {n} samples")

    # ──────────────────────────────────────────────────
    # CHECK 2 — Batch shapes
    # ──────────────────────────────────────────────────
    section("CHECK 2 — DataLoader batch shapes")

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)
    try:
        x_1d_b, x_2d_b, labels_b, charges_b = next(iter(loader))
    except Exception as e:
        print(f"  {FAIL}  DataLoader iteration failed: {e}")
        sys.exit(1)

    B = args.batch_size

    ok = check(x_1d_b.shape == torch.Size([B, 2, 20000]),
               f"x_1d shape = {tuple(x_1d_b.shape)}",
               f"x_1d shape = {tuple(x_1d_b.shape)} — expected ({B}, 2, 20000)")
    all_ok &= ok

    ok = check(x_2d_b.shape[0] == B and x_2d_b.shape[1] == 2 and x_2d_b.shape[2] == 257,
               f"x_2d shape = {tuple(x_2d_b.shape)}  [B=={B}, C==2, freq==257 ✓]",
               f"x_2d shape = {tuple(x_2d_b.shape)} — expected ({B}, 2, 257, n_time)")
    all_ok &= ok

    ok = check(labels_b.shape == torch.Size([B]),
               f"labels shape = {tuple(labels_b.shape)}",
               f"labels shape = {tuple(labels_b.shape)} — expected ({B},)")
    all_ok &= ok

    ok = check(charges_b.shape == torch.Size([B]),
               f"charges shape = {tuple(charges_b.shape)}",
               f"charges shape = {tuple(charges_b.shape)} — expected ({B},)")
    all_ok &= ok

    ok = check(not torch.isnan(x_1d_b).any() and not torch.isinf(x_1d_b).any(),
               "x_1d has no NaN/Inf",
               "x_1d contains NaN or Inf values — check normalization in step2")
    all_ok &= ok

    ok = check(not torch.isnan(x_2d_b).any() and not torch.isinf(x_2d_b).any(),
               "x_2d has no NaN/Inf",
               "x_2d contains NaN or Inf values — check STFT computation in dataset.py")
    all_ok &= ok

    # ──────────────────────────────────────────────────
    # CHECK 3 — Forward pass
    # ──────────────────────────────────────────────────
    section("CHECK 3 — Forward pass (ArcFaultNet, in_channels=2)")

    from model import get_model
    try:
        model = get_model('arcfaultnet', in_channels=2).to(device)
    except Exception as e:
        print(f"  {FAIL}  Model instantiation failed: {e}")
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {INFO}  Parameters: {n_params:,}")

    model.eval()
    x_1d_d = x_1d_b.to(device)
    x_2d_d = x_2d_b.to(device)

    try:
        with torch.no_grad():
            logits = model(x_1d_d, x_2d_d)
    except Exception as e:
        print(f"  {FAIL}  model(x_1d, x_2d) raised: {e}")
        sys.exit(1)

    ok = check(logits.shape == torch.Size([B]),
               f"logits shape = {tuple(logits.shape)}",
               f"logits shape = {tuple(logits.shape)} — expected ({B},)")
    all_ok &= ok

    ok = check(not torch.isnan(logits).any(),
               "logits has no NaN",
               f"logits contains NaN — check model architecture")
    all_ok &= ok

    ok = check(not torch.isinf(logits).any(),
               "logits has no Inf",
               f"logits contains Inf — check model architecture")
    all_ok &= ok

    probs = torch.sigmoid(logits)
    ok = check((probs >= 0).all() and (probs <= 1).all(),
               f"sigmoid(logits) ∈ [0,1]  (min={probs.min():.4f}, max={probs.max():.4f})",
               "sigmoid(logits) out of [0,1] range")
    all_ok &= ok

    # ──────────────────────────────────────────────────
    # CHECK 4 — Loss + backward
    # ──────────────────────────────────────────────────
    section("CHECK 4 — Loss computation + backward pass")

    model.train()
    criterion = nn.BCEWithLogitsLoss()

    try:
        logits = model(x_1d_d, x_2d_d)
        loss   = criterion(logits, labels_b.to(device))
    except Exception as e:
        print(f"  {FAIL}  Loss computation failed: {e}")
        sys.exit(1)

    ok = check(not torch.isnan(loss) and not torch.isinf(loss),
               f"loss = {loss.item():.6f}  (finite)",
               f"loss = {loss.item()} — NaN or Inf encountered")
    all_ok &= ok

    try:
        loss.backward()
    except Exception as e:
        print(f"  {FAIL}  loss.backward() failed: {e}")
        sys.exit(1)

    grad_norms = [p.grad.norm().item()
                  for p in model.parameters()
                  if p.grad is not None and p.requires_grad]

    ok = check(len(grad_norms) > 0 and max(grad_norms) > 0,
               f"Gradients computed for {len(grad_norms)} param tensors  "
               f"(max norm = {max(grad_norms):.6f})",
               "No gradients found — check that model has learnable parameters")
    all_ok &= ok

    ok = check(all(not np.isnan(g) for g in grad_norms),
               "No NaN in gradients",
               "NaN gradients detected — possible exploding gradient / bad init")
    all_ok &= ok

    # ──────────────────────────────────────────────────
    # CHECK 5 — Overfit mini-batch
    # ──────────────────────────────────────────────────
    section(f"CHECK 5 — Overfit mini-batch ({args.overfit_iters} iters, target loss < {args.overfit_threshold})")

    model_ov   = get_model('arcfaultnet', in_channels=2).to(device)
    optimizer  = optim.AdamW(model_ov.parameters(), lr=1e-3)
    criterion2 = nn.BCEWithLogitsLoss()

    x_1d_ov = x_1d_b.to(device)
    x_2d_ov = x_2d_b.to(device)
    y_ov    = labels_b.to(device)

    loss_start = None
    loss_end   = None

    model_ov.train()
    for i in range(args.overfit_iters):
        optimizer.zero_grad()
        logits_ov = model_ov(x_1d_ov, x_2d_ov)
        loss_ov   = criterion2(logits_ov, y_ov)
        loss_ov.backward()
        nn.utils.clip_grad_norm_(model_ov.parameters(), 1.0)
        optimizer.step()

        if i == 0:
            loss_start = loss_ov.item()
        if (i + 1) % 20 == 0:
            print(f"  {INFO}  iter {i+1:3d}/{args.overfit_iters}  loss = {loss_ov.item():.6f}")

    loss_end = loss_ov.item()

    ok = check(loss_end < args.overfit_threshold,
               f"Loss dropped to {loss_end:.6f} < {args.overfit_threshold} (start={loss_start:.4f}) ✓",
               f"Loss stuck at {loss_end:.6f} (start={loss_start:.4f}) — "
               f"model may not be learning; check architecture and loss")
    all_ok &= ok

    ok = check(loss_end < loss_start,
               f"Loss decreased: {loss_start:.6f} → {loss_end:.6f}",
               f"Loss did NOT decrease: {loss_start:.6f} → {loss_end:.6f}")
    all_ok &= ok

    # ──────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_ok:
        print("  ALL CHECKS PASSED — pipeline is ready for training.")
        print(f"\n  Suggested next step:")
        print(f"    python train.py --mode single --epochs 50 --batch-size 32 \\")
        print(f"      --num-workers 0 --seed 42")
    else:
        print("  SOME CHECKS FAILED — fix the issues above before training.")
        sys.exit(1)
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
