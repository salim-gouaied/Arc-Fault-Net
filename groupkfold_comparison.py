#!/usr/bin/env python3
"""
GroupKFold Comparatif — Full V2 vs No Attention
================================================
Lance un StratifiedGroupKFold (recording level) pour les deux variantes
afin de trancher statistiquement l'apport des mécanismes d'attention.
"""

import torch, torch.nn as nn, numpy as np, json, random, time, argparse
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGK = True
except ImportError:
    from sklearn.model_selection import GroupKFold
    _HAS_SGK = False

import warnings; warnings.filterwarnings('ignore')
from dataset import ArcFaultDataset
from model import get_model

# ── Seed ──────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Group IDs from metadata ──────────────────────────
def load_group_ids(data_dir, n_samples):
    """Load recording-level group IDs from metadata.csv."""
    import pandas as pd
    meta_path = Path(data_dir) / 'metadata.csv'
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {data_dir}")
    df = pd.read_csv(meta_path)
    if len(df) != n_samples:
        raise ValueError(f"metadata has {len(df)} rows but dataset has {n_samples} samples")
    if 'exp_name' in df.columns:
        groups = df['exp_name'].values
    elif 'recording' in df.columns:
        groups = df['recording'].values
    else:
        raise ValueError("No group column found in metadata.csv")
    return groups

# ── Train / Eval ─────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=0.5):
    model.train()
    if hasattr(loader.dataset, 'dataset'):
        loader.dataset.dataset.training = True
    total_loss, correct, total = 0., 0, 0
    for x1, x2, lab, _ in loader:
        x1, x2, lab = x1.to(device), x2.to(device), lab.to(device)
        smooth = lab * 0.9 + 0.05
        optimizer.zero_grad()
        logits = model(x1, x2)
        loss = criterion(logits, smooth)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(lab)
        correct += ((torch.sigmoid(logits) > 0.5).float() == lab).sum().item()
        total += len(lab)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    if hasattr(loader.dataset, 'dataset'):
        loader.dataset.dataset.training = False
    total_loss, correct, total = 0., 0, 0
    all_probs, all_labels = [], []
    for x1, x2, lab, _ in loader:
        x1, x2, lab = x1.to(device), x2.to(device), lab.to(device)
        logits = model(x1, x2)
        loss = criterion(logits, lab)
        probs = torch.sigmoid(logits)
        total_loss += loss.item() * len(lab)
        correct += ((probs > 0.5).float() == lab).sum().item()
        total += len(lab)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(lab.cpu().numpy())
    p, l = np.array(all_probs), np.array(all_labels)
    preds = (p >= 0.5).astype(int)
    tp = ((preds == 1) & (l == 1)).sum()
    fp = ((preds == 1) & (l == 0)).sum()
    fn = ((preds == 0) & (l == 1)).sum()
    tn = ((preds == 0) & (l == 0)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return {
        'loss': total_loss / total, 'accuracy': correct / total,
        'precision': float(prec), 'recall': float(rec),
        'f1': float(f1), 'specificity': float(spec),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }

def train_fold(model_name, dataset, train_idx, val_idx, test_idx, device,
               epochs=200, lr=3e-4, wd=5e-4, bs=64, patience=15, grad_clip=0.5, nw=4):
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=bs,
                              shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=bs,
                              shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=bs,
                              shuffle=False, num_workers=nw, pin_memory=True)

    model = get_model(model_name).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_f1, best_epoch, wait, best_sd = -1., 0, 0, None
    for ep in range(1, epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step(ep)
        if val_m['f1'] > best_f1:
            best_f1, best_epoch, wait = val_m['f1'], ep, 0
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience:
            break

    if best_sd:
        model.load_state_dict(best_sd)
    test_m = evaluate(model, test_loader, criterion, device)
    test_m['best_epoch'] = best_epoch
    test_m['n_params'] = sum(p.numel() for p in model.parameters())
    return test_m

# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='GroupKFold: Full V2 vs No Attention')
    parser.add_argument('--data-dir', type=str,
                        default='/home/manip/pfe_salim_gouaied/Arc-Fault-Net/combined_dataset_2048')
    parser.add_argument('--output-dir', type=str, default='runs')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-fft', type=int, default=128)
    parser.add_argument('--hop-length', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu') if (args.cpu or not torch.cuda.is_available()) \
             else torch.device('cuda')
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == 'cuda' else ''))

    set_seed(args.seed)

    # ── Dataset ──
    dataset = ArcFaultDataset(
        data_dir=args.data_dir,
        n_fft=args.n_fft, hop_length=args.hop_length,
        channel_mode='i_derived4'
    )
    group_ids = load_group_ids(args.data_dir, len(dataset))
    labels = dataset.y
    indices = np.arange(len(dataset))

    # ── Splitter ──
    n_unique = len(np.unique(group_ids))
    effective_folds = min(args.n_folds, n_unique)
    if _HAS_SGK:
        splitter = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=args.seed)
    else:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=effective_folds)
    splits = list(splitter.split(indices, labels, groups=group_ids))

    print(f"\nGroupKFold: {len(splits)} folds, {n_unique} unique groups")

    # ── Two variants to compare ──
    VARIANTS = [
        ('arcfaultnet_v2',  'Full V2 (cross-attn + freq gate)'),
        ('v2_no_attention', 'No Attention (concat simple + no freq gate)'),
    ]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) / f"groupkfold_comparison_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_variant_results = {}
    t0 = time.time()

    for var_name, var_label in VARIANTS:
        print(f"\n{'='*65}")
        print(f"  VARIANT: {var_label}  ({var_name})")
        print(f"{'='*65}")

        fold_metrics = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
            fold_seed = args.seed + fold_idx
            set_seed(fold_seed)

            # Anti-leakage check
            test_groups = set(group_ids[test_idx])
            tv_groups = set(group_ids[train_val_idx])
            assert test_groups.isdisjoint(tv_groups), "LEAKAGE!"

            # Sub-split train/val by group
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=fold_seed)
            tv_train, tv_val = next(gss.split(
                train_val_idx, labels[train_val_idx], groups=group_ids[train_val_idx]))
            train_idx = train_val_idx[tv_train]
            val_idx   = train_val_idx[tv_val]

            print(f"\n  Fold {fold_idx+1}/{len(splits)}  "
                  f"(train={len(train_idx)} val={len(val_idx)} test={len(test_idx)})  "
                  f"test_groups={sorted(test_groups)}")

            metrics = train_fold(
                var_name, dataset, train_idx, val_idx, test_idx, device,
                epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
                bs=args.batch_size, patience=args.patience, nw=args.num_workers
            )

            print(f"    → Acc={100*metrics['accuracy']:.2f}%  F1={100*metrics['f1']:.2f}%  "
                  f"Prec={100*metrics['precision']:.2f}%  Rec={100*metrics['recall']:.2f}%  "
                  f"(best_ep={metrics['best_epoch']})")

            fold_metrics.append(metrics)

        # ── Summary for this variant ──
        mk = ['accuracy', 'f1', 'precision', 'recall', 'specificity']
        summary = {}
        print(f"\n  {'─'*50}")
        print(f"  Résumé {var_label}:")
        for m in mk:
            vals = np.array([fm[m] for fm in fold_metrics])
            summary[f'{m}_mean'] = float(vals.mean())
            summary[f'{m}_std']  = float(vals.std())
            print(f"    {m:14s}: {100*vals.mean():.2f}% ± {100*vals.std():.2f}%")
        summary['fold_results'] = fold_metrics
        summary['n_params'] = fold_metrics[0]['n_params']
        all_variant_results[var_name] = summary

    duration = time.time() - t0

    # ═══════════════════════════════════════════════════════
    #  FINAL COMPARISON
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  COMPARAISON FINALE — GroupKFold ({len(splits)} folds)")
    print(f"{'='*65}")

    r_v2 = all_variant_results['arcfaultnet_v2']
    r_na = all_variant_results['v2_no_attention']

    print(f"\n  {'Métrique':<16} {'Full V2':>18} {'No Attention':>18} {'Δ':>10}")
    print(f"  {'─'*62}")
    for m in ['accuracy', 'f1', 'precision', 'recall', 'specificity']:
        v2_mean = r_v2[f'{m}_mean']
        v2_std  = r_v2[f'{m}_std']
        na_mean = r_na[f'{m}_mean']
        na_std  = r_na[f'{m}_std']
        delta   = (v2_mean - na_mean) * 100
        sign    = '+' if delta >= 0 else ''
        print(f"  {m:16s} {100*v2_mean:6.2f}% ±{100*v2_std:5.2f}%  "
              f"{100*na_mean:6.2f}% ±{100*na_std:5.2f}%  {sign}{delta:.2f}%")
    print(f"\n  Params: Full V2={r_v2['n_params']:,}  vs  No Attention={r_na['n_params']:,}")

    # ── Per-fold pairwise comparison ──
    v2_wins, na_wins, ties = 0, 0, 0
    print(f"\n  Comparaison fold par fold (F1):")
    for i, (fv2, fna) in enumerate(zip(r_v2['fold_results'], r_na['fold_results'])):
        d = fv2['f1'] - fna['f1']
        marker = '✓ V2' if d > 0 else ('✓ NA' if d < 0 else '= TIE')
        if d > 0: v2_wins += 1
        elif d < 0: na_wins += 1
        else: ties += 1
        print(f"    Fold {i+1}: V2={100*fv2['f1']:.2f}%  NA={100*fna['f1']:.2f}%  Δ={100*d:+.2f}%  {marker}")
    print(f"\n  Score: V2 gagne {v2_wins}/{len(splits)} folds, NA gagne {na_wins}/{len(splits)}")

    # ── Save ──
    output = {
        'timestamp': timestamp, 'seed': args.seed, 'n_folds': len(splits),
        'duration_seconds': duration,
        'variants': {k: {kk: vv for kk, vv in v.items() if kk != 'fold_results'}
                     for k, v in all_variant_results.items()},
        'fold_details': {k: v['fold_results'] for k, v in all_variant_results.items()},
        'pairwise': {'v2_wins': v2_wins, 'na_wins': na_wins, 'ties': ties},
    }
    with open(out_dir / 'comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Durée: {duration/60:.1f} min")
    print(f"  Résultats: {out_dir}")

if __name__ == '__main__':
    main()
