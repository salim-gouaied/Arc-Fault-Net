#!/usr/bin/env python3
"""
ARC-FAULTNET — Training Script
===============================
Trains Arc-FaultNet with leave-one-charge-out cross-validation.

Features:
  - set_seed for full reproducibility
  - AdamW optimizer with weight_decay
  - Gradient clipping
  - Optional pos_weight for class imbalance
  - Early stopping on val_f1 (max) instead of val_loss
  - Leave-one-charge-out CV for proper generalization testing
  - Per-fold history.json and config.json
  - Model checkpointing (best + last)
  - TensorBoard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
from pathlib import Path
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from dataset import ArcFaultDataset, LeaveOneChargeOutSplitter, create_dataloaders
from model import get_model


# ═══════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ═══════════════════════════════════════════════════════

def set_seed(seed: int):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════
#  TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_clip: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for x_1d, x_2d, labels, _ in pbar:
        x_1d   = x_1d.to(device)
        x_2d   = x_2d.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(x_1d, x_2d)
        loss   = criterion(logits, labels)

        loss.backward()

        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds   = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total   += len(labels)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc':  f"{100*correct/total:.1f}%"
        })

    return {
        'loss':     total_loss / total,
        'accuracy': correct / total
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval",
    threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate model on a dataloader."""
    model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    all_preds  = []
    all_labels = []

    for x_1d, x_2d, labels, _ in tqdm(loader, desc=desc, leave=False):
        x_1d   = x_1d.to(device)
        x_2d   = x_2d.to(device)
        labels = labels.to(device)

        logits = model(x_1d, x_2d)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        probs  = torch.sigmoid(logits)
        preds  = (probs > threshold).float()
        correct += (preds == labels).sum().item()
        total   += len(labels)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))

    precision    = tp / (tp + fp + 1e-8)
    recall       = tp / (tp + fn + 1e-8)
    f1           = 2 * precision * recall / (precision + recall + 1e-8)
    specificity  = tn / (tn + fp + 1e-8)

    return {
        'loss':        total_loss / total,
        'accuracy':    correct / total,
        'precision':   precision,
        'recall':      recall,
        'f1':          f1,
        'specificity': specificity,
        'tp': int(tp), 'fp': int(fp),
        'fn': int(fn), 'tn': int(tn)
    }


def compute_pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss from train labels only.
    pos_weight = n_negative / n_positive
    Never call this on validation or test labels.
    """
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    if n_pos == 0:
        return torch.tensor([1.0], device=device)
    weight = float(n_neg) / float(n_pos)
    return torch.tensor([weight], device=device)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    gradient_clip: float = 1.0,
    threshold: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
    checkpoint_dir: Optional[Path] = None,
    writer: Optional[SummaryWriter] = None,
    fold_name: str = ""
) -> Tuple[nn.Module, Dict]:
    """
    Train model with early stopping on val_f1.

    Returns:
        model:   Best checkpoint reloaded
        history: Full training history dict
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=max(5, patience // 4)
    )

    best_val_f1      = -1.0
    best_epoch       = 0
    patience_counter = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'val_f1':     [], 'val_precision': [], 'val_recall': [],
        'lr': []
    }

    best_ckpt_path = checkpoint_dir / f'best_{fold_name}.pt' if checkpoint_dir else None
    last_ckpt_path = checkpoint_dir / f'last_{fold_name}.pt' if checkpoint_dir else None

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, gradient_clip
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, "Val", threshold
        )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['f1'])

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['lr'].append(current_lr)

        if writer:
            writer.add_scalar(f'{fold_name}/train_loss',  train_metrics['loss'],      epoch)
            writer.add_scalar(f'{fold_name}/train_acc',   train_metrics['accuracy'],  epoch)
            writer.add_scalar(f'{fold_name}/val_loss',    val_metrics['loss'],        epoch)
            writer.add_scalar(f'{fold_name}/val_acc',     val_metrics['accuracy'],    epoch)
            writer.add_scalar(f'{fold_name}/val_f1',      val_metrics['f1'],          epoch)
            writer.add_scalar(f'{fold_name}/lr',          current_lr,                 epoch)

        # Early stopping on val_f1 (max)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1      = val_metrics['f1']
            best_epoch       = epoch
            patience_counter = 0
            if best_ckpt_path:
                torch.save(model.state_dict(), best_ckpt_path)
        else:
            patience_counter += 1

        # Save last checkpoint every epoch
        if last_ckpt_path:
            torch.save(model.state_dict(), last_ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: "
                  f"train_loss={train_metrics['loss']:.4f}  "
                  f"val_loss={val_metrics['loss']:.4f}  "
                  f"val_acc={100*val_metrics['accuracy']:.1f}%  "
                  f"val_f1={100*val_metrics['f1']:.1f}%  "
                  f"lr={current_lr:.2e}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch}, best_val_f1={100*best_val_f1:.2f}%)")
            break

    # Reload best weights
    if best_ckpt_path and best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    history['best_epoch']   = best_epoch
    history['best_val_f1']  = best_val_f1

    # Save history JSON
    if checkpoint_dir:
        with open(checkpoint_dir / f'history_{fold_name}.json', 'w') as f:
            json.dump(history, f, indent=2)

    return model, history


# ═══════════════════════════════════════════════════════
#  LEAVE-ONE-CHARGE-OUT CROSS-VALIDATION
# ═══════════════════════════════════════════════════════

def run_leave_one_charge_out_cv(
    model_name: str,
    dataset: ArcFaultDataset,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 20,
    gradient_clip: float = 1.0,
    threshold: float = 0.5,
    use_pos_weight: bool = False,
    output_dir: Path = Path('runs'),
    num_workers: int = 4,
    seed: int = 42,
    fold_filter: Optional[int] = None
) -> Dict:
    """
    Run leave-one-charge-out cross-validation.

    fold_filter: if set, run only that fold index (0-based).
    """
    splitter = LeaveOneChargeOutSplitter(dataset)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = output_dir / f"{model_name}_loco_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(run_dir / 'tensorboard')

    all_results = []

    print(f"\n{'='*60}")
    print(f"LEAVE-ONE-CHARGE-OUT CROSS-VALIDATION")
    print(f"Model: {model_name}  |  seed={seed}")
    print(f"{'='*60}")

    for fold_idx, (train_indices, test_indices) in enumerate(splitter):
        if fold_filter is not None and fold_idx != fold_filter:
            continue

        charge_name = splitter.get_fold_name(fold_idx)
        fold_seed   = seed + fold_idx
        set_seed(fold_seed)

        print(f"\n--- Fold {fold_idx + 1}/{len(splitter)}: Test on '{charge_name}' (seed={fold_seed}) ---")
        print(f"    Train: {len(train_indices)} samples")
        print(f"    Test:  {len(test_indices)} samples")

        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            train_indices.copy(),
            test_indices,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=0.15
        )

        # pos_weight computed from train labels only
        pw = None
        if use_pos_weight:
            train_labels = dataset.y[train_indices]
            pw = compute_pos_weight(train_labels, device)
            print(f"    pos_weight = {pw.item():.3f}")

        model = get_model(model_name, in_channels=2).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        model, history = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            patience=patience, gradient_clip=gradient_clip,
            threshold=threshold, pos_weight=pw,
            checkpoint_dir=run_dir, writer=writer,
            fold_name=f"fold{fold_idx}_{charge_name}"
        )

        # Test on held-out charge
        criterion = nn.BCEWithLogitsLoss()
        test_metrics = evaluate(model, test_loader, criterion, device, "Test", threshold)

        print(f"    Test results on '{charge_name}':")
        print(f"      Accuracy:    {100*test_metrics['accuracy']:.2f}%")
        print(f"      F1 Score:    {100*test_metrics['f1']:.2f}%")
        print(f"      Precision:   {100*test_metrics['precision']:.2f}%")
        print(f"      Recall:      {100*test_metrics['recall']:.2f}%")
        print(f"      Specificity: {100*test_metrics['specificity']:.2f}%")

        fold_result = {
            'fold_idx':       fold_idx,
            'charge_name':    charge_name,
            'fold_seed':      fold_seed,
            'n_train':        len(train_indices),
            'n_test':         len(test_indices),
            'n_params':       n_params,
            'best_epoch':     history['best_epoch'],
            'test_accuracy':  test_metrics['accuracy'],
            'test_f1':        test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall':    test_metrics['recall'],
            'test_specificity': test_metrics['specificity'],
            'test_tp': test_metrics['tp'], 'test_fp': test_metrics['fp'],
            'test_fn': test_metrics['fn'], 'test_tn': test_metrics['tn'],
        }
        all_results.append(fold_result)

        writer.add_scalar('test/accuracy',    test_metrics['accuracy'],  fold_idx)
        writer.add_scalar('test/f1',          test_metrics['f1'],        fold_idx)
        writer.add_scalar('test/precision',   test_metrics['precision'], fold_idx)
        writer.add_scalar('test/recall',      test_metrics['recall'],    fold_idx)

        # Save per-fold config
        fold_config = {
            'fold_idx': fold_idx, 'charge_name': charge_name,
            'model_name': model_name, 'fold_seed': fold_seed,
            'n_params': n_params,
            'epochs': epochs, 'lr': lr, 'weight_decay': weight_decay,
            'batch_size': batch_size, 'patience': patience,
            'gradient_clip': gradient_clip, 'threshold': threshold,
            'use_pos_weight': use_pos_weight,
        }
        with open(run_dir / f'config_fold{fold_idx}.json', 'w') as f:
            json.dump(fold_config, f, indent=2)

    writer.close()

    if not all_results:
        print("No folds were run (check --fold argument).")
        return {}

    avg_accuracy = np.mean([r['test_accuracy']  for r in all_results])
    std_accuracy = np.std( [r['test_accuracy']  for r in all_results])
    avg_f1       = np.mean([r['test_f1']        for r in all_results])
    std_f1       = np.std( [r['test_f1']        for r in all_results])
    avg_recall   = np.mean([r['test_recall']    for r in all_results])
    avg_precision= np.mean([r['test_precision'] for r in all_results])

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Mean Accuracy  : {100*avg_accuracy:.2f}% ± {100*std_accuracy:.2f}%")
    print(f"  Mean F1        : {100*avg_f1:.2f}% ± {100*std_f1:.2f}%")
    print(f"  Mean Precision : {100*avg_precision:.2f}%")
    print(f"  Mean Recall    : {100*avg_recall:.2f}%")

    results_summary = {
        'model_name':     model_name,
        'n_folds':        len(all_results),
        'global_seed':    seed,
        'timestamp':      timestamp,
        'epochs':         epochs, 'lr': lr, 'weight_decay': weight_decay,
        'batch_size':     batch_size, 'patience': patience,
        'gradient_clip':  gradient_clip, 'threshold': threshold,
        'use_pos_weight': use_pos_weight,
        'mean_accuracy':  float(avg_accuracy),
        'std_accuracy':   float(std_accuracy),
        'mean_f1':        float(avg_f1),
        'std_f1':         float(std_f1),
        'mean_precision': float(avg_precision),
        'mean_recall':    float(avg_recall),
        'fold_results':   all_results
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {run_dir}")
    return results_summary


# ═══════════════════════════════════════════════════════
#  SINGLE TRAINING RUN
# ═══════════════════════════════════════════════════════

def run_single_training(
    model_name: str,
    dataset: ArcFaultDataset,
    device: torch.device,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 20,
    gradient_clip: float = 1.0,
    threshold: float = 0.5,
    use_pos_weight: bool = False,
    output_dir: Path = Path('runs'),
    num_workers: int = 4,
    seed: int = 42
) -> Dict:
    """
    Single training run with random train/val/test split.

    NOTE: Does NOT test generalization to unseen charges.
          Use for quick smoke tests only.
    """
    set_seed(seed)

    indices = np.random.permutation(len(dataset))
    n_train = int(len(dataset) * train_ratio)
    n_val   = int(len(dataset) * val_ratio)

    train_indices = indices[:n_train]
    val_indices   = indices[n_train:n_train + n_val]
    test_indices  = indices[n_train + n_val:]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = output_dir / f"{model_name}_single_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(run_dir / 'tensorboard')

    print(f"\n{'='*60}")
    print(f"SINGLE TRAINING RUN (random split — NOT for generalization)")
    print(f"Model: {model_name}  |  seed={seed}")
    print(f"{'='*60}")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val:   {len(val_indices)} samples")
    print(f"  Test:  {len(test_indices)} samples")

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
        print(f"  pos_weight = {pw.item():.3f}")

    model = get_model(model_name, in_channels=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    model, history = train_model(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        patience=patience, gradient_clip=gradient_clip,
        threshold=threshold, pos_weight=pw,
        checkpoint_dir=run_dir, writer=writer,
        fold_name="single"
    )

    criterion    = nn.BCEWithLogitsLoss()
    test_metrics = evaluate(model, test_loader, criterion, device, "Test", threshold)

    writer.close()

    print(f"\nTest Results:")
    print(f"  Accuracy:    {100*test_metrics['accuracy']:.2f}%")
    print(f"  F1 Score:    {100*test_metrics['f1']:.2f}%")
    print(f"  Precision:   {100*test_metrics['precision']:.2f}%")
    print(f"  Recall:      {100*test_metrics['recall']:.2f}%")
    print(f"  Specificity: {100*test_metrics['specificity']:.2f}%")

    results = {
        'model_name':     model_name,
        'seed':           seed,
        'n_params':       n_params,
        'timestamp':      timestamp,
        'epochs':         epochs, 'lr': lr, 'weight_decay': weight_decay,
        'batch_size':     batch_size, 'patience': patience,
        'gradient_clip':  gradient_clip, 'threshold': threshold,
        'best_epoch':     history['best_epoch'],
        'test_accuracy':  float(test_metrics['accuracy']),
        'test_f1':        float(test_metrics['f1']),
        'test_precision': float(test_metrics['precision']),
        'test_recall':    float(test_metrics['recall']),
        'test_specificity': float(test_metrics['specificity']),
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), run_dir / 'final_model.pt')
    print(f"\nResults saved to: {run_dir}")
    return results


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train Arc-FaultNet')

    parser.add_argument('--model', type=str, default='arcfaultnet',
                        choices=['arcfaultnet', '1d_only', 'no_attention',
                                 'standard_conv', 'independent_cbam', 'baseline_cnn'],
                        help='Model to train')
    parser.add_argument('--mode', type=str, default='cv',
                        choices=['cv', 'single'],
                        help='cv = leave-one-charge-out | single = random split')
    parser.add_argument('--fold', type=int, default=None,
                        help='(cv mode) Run only this fold index (0-based)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Max gradient norm (0 = disabled)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for sigmoid output')
    parser.add_argument('--use-pos-weight', action='store_true',
                        help='Use pos_weight in BCEWithLogitsLoss for class imbalance')
    parser.add_argument('--data-dir', type=str, default='/home/top/PFE/labeled_dataset')
    parser.add_argument('--output-dir', type=str, default='/home/top/PFE/runs')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cpu') if (args.cpu or not torch.cuda.is_available()) \
             else torch.device('cuda')
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == 'cuda' else ''))

    data_dir = Path(args.data_dir)
    if not (data_dir / 'X_multi.npy').exists():
        print(f"\nData not found at {data_dir}")
        print("Run: python step2_build_multichannel.py")
        return

    dataset    = ArcFaultDataset(data_dir=str(data_dir))
    output_dir = Path(args.output_dir)

    if args.mode == 'cv':
        run_leave_one_charge_out_cv(
            model_name=args.model,
            dataset=dataset,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            patience=args.patience,
            gradient_clip=args.gradient_clip,
            threshold=args.threshold,
            use_pos_weight=args.use_pos_weight,
            output_dir=output_dir,
            num_workers=args.num_workers,
            seed=args.seed,
            fold_filter=args.fold
        )
    else:
        run_single_training(
            model_name=args.model,
            dataset=dataset,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            patience=args.patience,
            gradient_clip=args.gradient_clip,
            threshold=args.threshold,
            use_pos_weight=args.use_pos_weight,
            output_dir=output_dir,
            num_workers=args.num_workers,
            seed=args.seed
        )


if __name__ == '__main__':
    main()
