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
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, GroupShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    from sklearn.model_selection import GroupKFold
    _HAS_STRATIFIED_GROUP_KFOLD = False

import numpy as np
import pandas as pd
import random
import re
from pathlib import Path
import json
import argparse
from datetime import datetime
import time
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
    gradient_clip: float = 0.5,
    label_smoothing: float = 0.05
) -> Dict[str, float]:
    """Train for one epoch with label smoothing."""
    model.train()
    # Enable augmentation on the underlying dataset
    if hasattr(loader.dataset, 'dataset'):
        loader.dataset.dataset.training = True

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for x_1d, x_2d, labels, _ in pbar:
        x_1d   = x_1d.to(device)
        x_2d   = x_2d.to(device)
        labels = labels.to(device)

        # Binary label smoothing: 0 -> 0.05, 1 -> 0.95
        smoothed_labels = labels * (1.0 - 2 * label_smoothing) + label_smoothing

        optimizer.zero_grad()

        logits = model(x_1d, x_2d)
        loss   = criterion(logits, smoothed_labels)

        loss.backward()

        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds   = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()  # Accuracy vs hard labels
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
    # Disable augmentation during evaluation
    if hasattr(loader.dataset, 'dataset'):
        loader.dataset.dataset.training = False
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
    epochs: int = 80,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    patience: int = 10,
    gradient_clip: float = 0.5,
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
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_f1      = -1.0
    best_epoch       = 0
    patience_counter = 0
    best_state_dict  = None  # Always keep best weights in memory

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
        scheduler.step(epoch)

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
            # Always save best weights in memory
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
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

    # Reload best weights (from memory — works even without checkpoint_dir)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

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
    epochs: int = 80,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    patience: int = 10,
    gradient_clip: float = 0.5,
    threshold: float = 0.5,
    use_pos_weight: bool = False,
    output_dir: Path = Path('runs'),
    num_workers: int = 4,
    seed: int = 42,
    fold_filter: Optional[int] = None,
    use_se: bool = False,
    se_reduction: int = 8,
    use_amplitude: bool = False,
    deep_classifier: bool = False,
    fs: float = 1_000_000,
    n_fft: int = 512
) -> Dict:
    """
    Run leave-one-charge-out cross-validation.

    fold_filter: if set, run only that fold index (0-based).
    """
    start_time = time.time()
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

        model = get_model(model_name, in_channels=2,
                      use_se=use_se, se_reduction=se_reduction,
                      use_amplitude=use_amplitude,
                      deep_classifier=deep_classifier,
                      fs=fs, n_fft=n_fft).to(device)
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
        'data_dir':       str(dataset.data_dir),
        'fs':             int(fs),
        'n_fft':          n_fft,
        'hop_length':     dataset.hop_length,
        'use_se':         use_se,
        'se_reduction':   se_reduction,
        'use_amplitude':  use_amplitude,
        'deep_classifier': deep_classifier,
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
        'training_duration_seconds': time.time() - start_time,
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
    epochs: int = 80,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    patience: int = 10,
    gradient_clip: float = 0.5,
    threshold: float = 0.5,
    use_pos_weight: bool = False,
    output_dir: Path = Path('runs'),
    num_workers: int = 4,
    seed: int = 42,
    use_se: bool = False,
    se_reduction: int = 8,
    use_amplitude: bool = False,
    deep_classifier: bool = False,
    fs: float = 1_000_000,
    n_fft: int = 512
) -> Dict:
    """
    Single training run with random train/val/test split.

    NOTE: Does NOT test generalization to unseen charges.
          Use for quick smoke tests only.
    """
    start_time = time.time()
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

    model = get_model(model_name, in_channels=2,
                      use_se=use_se, se_reduction=se_reduction,
                      use_amplitude=use_amplitude,
                      deep_classifier=deep_classifier,
                      fs=fs, n_fft=n_fft).to(device)
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
        'data_dir':       str(dataset.data_dir),
        'fs':             int(fs),
        'n_fft':          n_fft,
        'hop_length':     dataset.hop_length,
        'use_se':         use_se,
        'se_reduction':   se_reduction,
        'use_amplitude':  use_amplitude,
        'deep_classifier': deep_classifier,
        'epochs':         epochs, 'lr': lr, 'weight_decay': weight_decay,
        'batch_size':     batch_size, 'patience': patience,
        'gradient_clip':  gradient_clip, 'threshold': threshold,
        'best_epoch':     history['best_epoch'],
        'test_accuracy':  float(test_metrics['accuracy']),
        'test_f1':        float(test_metrics['f1']),
        'test_precision': float(test_metrics['precision']),
        'test_recall':    float(test_metrics['recall']),
        'test_specificity': float(test_metrics['specificity']),
        'training_duration_seconds': time.time() - start_time,
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), run_dir / 'final_model.pt')
    print(f"\nResults saved to: {run_dir}")
    return results


# ═══════════════════════════════════════════════════════
#  K-FOLD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════

def run_kfold_cv(
    model_name: str,
    dataset: ArcFaultDataset,
    device: torch.device,
    n_folds: int = 5,
    epochs: int = 80,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    patience: int = 10,
    gradient_clip: float = 0.5,
    threshold: float = 0.5,
    use_pos_weight: bool = False,
    output_dir: Path = Path('runs'),
    num_workers: int = 4,
    seed: int = 42,
    use_se: bool = False,
    se_reduction: int = 8,
    use_amplitude: bool = False,
    deep_classifier: bool = False,
    fs: float = 1_000_000,
    n_fft: int = 512
) -> Dict:
    """
    Stratified K-Fold cross-validation.

    Each fold trains a fresh model on (K-1) folds and evaluates on the held-out fold.
    At the end, reports mean ± std across all folds to give a confidence interval.
    Each fold uses a different seed so the data split AND weight init differ.
    """
    start_time = time.time()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = output_dir / f"{model_name}_kfold{n_folds}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = dataset.y  # (N,) numpy array of 0/1
    indices = np.arange(len(dataset))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    print(f"\n{'='*60}")
    print(f"STRATIFIED {n_folds}-FOLD CROSS-VALIDATION")
    print(f"Model: {model_name}  |  seed={seed}")
    print(f"{'='*60}")

    fold_results = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels)):
        fold_seed = seed + fold_idx
        set_seed(fold_seed)

        # Split train_val further into train/val (85%/15%)
        n_val = int(len(train_val_idx) * 0.15)
        np.random.shuffle(train_val_idx)
        val_idx   = train_val_idx[:n_val]
        train_idx = train_val_idx[n_val:]

        print(f"\n--- Fold {fold_idx+1}/{n_folds}  (seed={fold_seed}) ---")
        print(f"    Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

        fold_dir = run_dir / f"fold_{fold_idx+1}"
        fold_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(fold_dir / 'tensorboard')

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, pin_memory=True)

        pw = None
        if use_pos_weight:
            pw = compute_pos_weight(dataset.y[train_idx], device)
            print(f"    pos_weight = {pw.item():.3f}")

        model = get_model(model_name, in_channels=2,
                          use_se=use_se, se_reduction=se_reduction,
                          use_amplitude=use_amplitude,
                          deep_classifier=deep_classifier,
                          fs=fs, n_fft=n_fft).to(device)

        if fold_idx == 0:
            print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

        model, history = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            patience=patience, gradient_clip=gradient_clip,
            threshold=threshold, pos_weight=pw,
            checkpoint_dir=fold_dir, writer=writer,
            fold_name=f"fold_{fold_idx+1}"
        )

        criterion    = nn.BCEWithLogitsLoss()
        test_metrics = evaluate(model, test_loader, criterion, device,
                                f"Fold {fold_idx+1} Test", threshold)
        writer.close()

        fold_result = {
            'fold':       fold_idx + 1,
            'seed':       fold_seed,
            'best_epoch': history['best_epoch'],
            **{f'test_{k}': float(v) for k, v in test_metrics.items()
               if k in ('accuracy', 'f1', 'precision', 'recall', 'specificity')}
        }
        fold_results.append(fold_result)

        print(f"    → Acc={100*test_metrics['accuracy']:.2f}%  "
              f"F1={100*test_metrics['f1']:.2f}%  "
              f"AUC (val)={history.get('best_val_f1', 0)*100:.2f}%")

        with open(fold_dir / 'fold_results.json', 'w') as f:
            json.dump(fold_result, f, indent=2)

    # ── Summary ──────────────────────────────────────────────
    metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_specificity']
    summary = {}
    print(f"\n{'='*60}")
    print(f"K-FOLD SUMMARY ({n_folds} folds)")
    print(f"{'='*60}")
    for m in metrics:
        vals = np.array([r[m] for r in fold_results])
        mean, std = vals.mean(), vals.std()
        label = m.replace('test_', '').capitalize()
        print(f"  {label:12s}: {100*mean:.2f}% ± {100*std:.2f}%")
        summary[f'{m}_mean'] = float(mean)
        summary[f'{m}_std']  = float(std)

    summary.update({
        'model_name':  model_name,
        'n_folds':     n_folds,
        'seed':        seed,
        'epochs':      epochs,
        'lr':          lr,
        'weight_decay': weight_decay,
        'batch_size':  batch_size,
        'patience':    patience,
        'fold_results': fold_results,
        'training_duration_seconds': time.time() - start_time,
    })

    with open(run_dir / 'kfold_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {run_dir}")
    return summary


# ═══════════════════════════════════════════════════════
#  GROUP K-FOLD CROSS-VALIDATION (ANTI-LEAKAGE)
# ═══════════════════════════════════════════════════════

def load_group_ids(data_dir: Path, n_samples: int, group_level: str = 'recording') -> np.ndarray:
    """
    Derive group IDs from metadata.csv for group-based cross-validation.
    Returns string array aligned with X_multi.npy / y.npy.

    recording: each unique exp_name is a group.
    session:   regex exp(\d+) extraction, fallback 'other'.
    """
    metadata_path = data_dir / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    meta = pd.read_csv(metadata_path)

    if len(meta) != n_samples:
        raise ValueError(
            f"metadata.csv has {len(meta)} rows but dataset has {n_samples} samples. "
            f"They must be aligned row-by-row. Aborting."
        )

    # Determine experiment name column
    if 'exp_name' in meta.columns:
        exp_names = meta['exp_name'].values.astype(str)
    elif 'exp_id' in meta.columns:
        # labeled_dataset format: combine exp_id + file_num for per-file grouping
        exp_names = (meta['exp_id'] + '--' + meta['file_num'].astype(str).str.zfill(5)).values
    else:
        raise ValueError("metadata.csv must have 'exp_name' or 'exp_id' column")

    if group_level == 'recording':
        group_ids = exp_names
    elif group_level == 'session':
        def _extract_session(name):
            m = re.search(r'exp(\d+)', name)
            return m.group(0) if m else 'other'
        group_ids = np.array([_extract_session(n) for n in exp_names])
    else:
        raise ValueError(f"Unknown group_level: {group_level}")

    unique_groups, counts = np.unique(group_ids, return_counts=True)
    print(f"\n  Group level '{group_level}': {len(unique_groups)} unique groups")
    for g, c in zip(unique_groups, counts):
        print(f"    {g}: {c} alternances")

    return group_ids


def run_groupkfold_cv(
    model_name: str,
    dataset: ArcFaultDataset,
    device: torch.device,
    group_level: str = 'recording',
    n_folds: int = 5,
    epochs: int = 80,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    patience: int = 10,
    gradient_clip: float = 0.5,
    threshold: float = 0.5,
    use_pos_weight: bool = False,
    output_dir: Path = Path('runs'),
    num_workers: int = 4,
    seed: int = 42,
    use_se: bool = False,
    se_reduction: int = 8,
    use_amplitude: bool = False,
    deep_classifier: bool = False,
    fs: float = 1_000_000,
    n_fft: int = 512
) -> Dict:
    """
    Group-based K-Fold cross-validation preventing data leakage.

    - recording level: StratifiedGroupKFold on exp_name groups.
      Measures generalization to unseen recordings (LOCO substitute).
    - session level: LeaveOneGroupOut on exp11/exp12/exp13/other.
      Measures inter-session shift.

    All alternances from the same group stay in a single fold.
    Val set is also split by group to prevent train/val leakage.
    """
    start_time = time.time()

    data_dir = Path(dataset.data_dir)
    group_ids = load_group_ids(data_dir, len(dataset), group_level)
    labels = dataset.y
    indices = np.arange(len(dataset))

    # ── Choose splitter ──────────────────────────────────
    if group_level == 'session':
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(indices, labels, groups=group_ids))
        n_actual_folds = len(splits)
        print(f"\n  LeaveOneGroupOut: {n_actual_folds} folds (--n-folds ignored)")
    else:
        n_unique = len(np.unique(group_ids))
        effective_folds = min(n_folds, n_unique)
        if effective_folds < n_folds:
            print(f"\n  WARNING: requested {n_folds} folds but only {n_unique} groups. "
                  f"Using {effective_folds} folds.")
        if _HAS_STRATIFIED_GROUP_KFOLD:
            splitter = StratifiedGroupKFold(
                n_splits=effective_folds, shuffle=True, random_state=seed)
        else:
            warnings.warn(
                "StratifiedGroupKFold unavailable; using GroupKFold (no label stratification)")
            splitter = GroupKFold(n_splits=effective_folds)
        splits = list(splitter.split(indices, labels, groups=group_ids))
        n_actual_folds = len(splits)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"{model_name}_groupkfold_{group_level}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    level_desc = ("unseen recordings (LOCO substitute)" if group_level == 'recording'
                  else "inter-session shift")
    print(f"\n{'='*60}")
    print(f"GROUP K-FOLD CROSS-VALIDATION  (anti-leakage)")
    print(f"Model: {model_name}  |  group_level={group_level}  |  seed={seed}")
    print(f"Measures: {level_desc}")
    print(f"{'='*60}")

    fold_results = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        fold_seed = seed + fold_idx
        set_seed(fold_seed)

        # ── Anti-leakage: test vs train+val ──
        test_groups = set(group_ids[test_idx])
        train_val_groups = set(group_ids[train_val_idx])
        assert test_groups.isdisjoint(train_val_groups), (
            f"LEAKAGE fold {fold_idx}: groups in both train_val and test: "
            f"{test_groups & train_val_groups}")

        # ── Sub-split train_val → train + val BY GROUP (~15% groups to val) ──
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=fold_seed)
        tv_train_sub, tv_val_sub = next(gss.split(
            train_val_idx, labels[train_val_idx], groups=group_ids[train_val_idx]))
        train_idx = train_val_idx[tv_train_sub]
        val_idx = train_val_idx[tv_val_sub]

        # ── Anti-leakage: pairwise disjoint ──
        train_groups = set(group_ids[train_idx])
        val_groups = set(group_ids[val_idx])
        assert train_groups.isdisjoint(val_groups), (
            f"LEAKAGE fold {fold_idx}: groups shared train↔val: "
            f"{train_groups & val_groups}")
        assert train_groups.isdisjoint(test_groups), (
            f"LEAKAGE fold {fold_idx}: groups shared train↔test: "
            f"{train_groups & test_groups}")
        assert val_groups.isdisjoint(test_groups), (
            f"LEAKAGE fold {fold_idx}: groups shared val↔test: "
            f"{val_groups & test_groups}")

        # ── Log fold info ──
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

        if group_level == 'session':
            fold_name = f"fold_{fold_idx+1}_test_{'_'.join(sorted(test_groups))}"
        else:
            fold_name = f"fold_{fold_idx+1}"

        print(f"\n--- Fold {fold_idx+1}/{n_actual_folds}  (seed={fold_seed}) ---")
        print(f"    Train: {len(train_idx):5d} samples ({len(train_groups):2d} groups)  "
              f"[{int(np.sum(train_labels==0))} N / {int(np.sum(train_labels==1))} A]")
        print(f"    Val:   {len(val_idx):5d} samples ({len(val_groups):2d} groups)  "
              f"[{int(np.sum(val_labels==0))} N / {int(np.sum(val_labels==1))} A]")
        print(f"    Test:  {len(test_idx):5d} samples ({len(test_groups):2d} groups)  "
              f"[{int(np.sum(test_labels==0))} N / {int(np.sum(test_labels==1))} A]")
        print(f"    Test groups: {sorted(test_groups)}")

        # ── DataLoaders ──
        fold_dir = run_dir / fold_name
        fold_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(fold_dir / 'tensorboard')

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)

        # ── pos_weight from train labels only ──
        pw = None
        if use_pos_weight:
            pw = compute_pos_weight(train_labels, device)
            print(f"    pos_weight = {pw.item():.3f}")

        # ── Model ──
        model = get_model(model_name, in_channels=2,
                          use_se=use_se, se_reduction=se_reduction,
                          use_amplitude=use_amplitude,
                          deep_classifier=deep_classifier,
                          fs=fs, n_fft=n_fft).to(device)

        if fold_idx == 0:
            print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # ── Train ──
        model, history = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            patience=patience, gradient_clip=gradient_clip,
            threshold=threshold, pos_weight=pw,
            checkpoint_dir=fold_dir, writer=writer,
            fold_name=fold_name
        )

        # ── Evaluate on held-out test groups ──
        criterion = nn.BCEWithLogitsLoss()
        test_metrics = evaluate(model, test_loader, criterion, device,
                                f"Fold {fold_idx+1} Test", threshold)
        writer.close()

        print(f"    → Acc={100*test_metrics['accuracy']:.2f}%  "
              f"F1={100*test_metrics['f1']:.2f}%  "
              f"Prec={100*test_metrics['precision']:.2f}%  "
              f"Rec={100*test_metrics['recall']:.2f}%  "
              f"Spec={100*test_metrics['specificity']:.2f}%")

        fold_result = {
            'fold': fold_idx + 1,
            'fold_name': fold_name,
            'seed': fold_seed,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'n_groups_train': len(train_groups),
            'n_groups_val': len(val_groups),
            'n_groups_test': len(test_groups),
            'test_groups': sorted(test_groups),
            'best_epoch': history['best_epoch'],
            'test_accuracy': float(test_metrics['accuracy']),
            'test_f1': float(test_metrics['f1']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_specificity': float(test_metrics['specificity']),
            'test_tp': test_metrics['tp'], 'test_fp': test_metrics['fp'],
            'test_fn': test_metrics['fn'], 'test_tn': test_metrics['tn'],
        }
        fold_results.append(fold_result)

        # Per-fold config
        fold_config = {
            'fold': fold_idx + 1, 'fold_name': fold_name,
            'model_name': model_name, 'group_level': group_level,
            'n_folds': n_actual_folds, 'global_seed': seed,
            'fold_seed': fold_seed,
            'epochs': epochs, 'lr': lr, 'weight_decay': weight_decay,
            'batch_size': batch_size, 'patience': patience,
            'gradient_clip': gradient_clip, 'threshold': threshold,
            'use_pos_weight': use_pos_weight,
        }
        with open(fold_dir / 'config.json', 'w') as f:
            json.dump(fold_config, f, indent=2)

    # ── Summary across folds ─────────────────────────────
    if not fold_results:
        print("No folds were run.")
        return {}

    metrics_keys = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_specificity']
    summary = {}

    print(f"\n{'='*60}")
    print(f"GROUP K-FOLD SUMMARY  ({n_actual_folds} folds, group_level={group_level})")
    print(f"{'='*60}")

    for m in metrics_keys:
        vals = np.array([r[m] for r in fold_results])
        mean, std = vals.mean(), vals.std()
        label = m.replace('test_', '').capitalize()
        print(f"  {label:12s}: {100*mean:.2f}% ± {100*std:.2f}%")
        summary[f'{m}_mean'] = float(mean)
        summary[f'{m}_std'] = float(std)

    print(f"\n  Interpretation:")
    if group_level == 'recording':
        print(f"    group-level=recording measures generalization to NEW RECORDINGS")
        print(f"    (substitute for leave-one-charge-out when charge info is unavailable)")
    else:
        print(f"    group-level=session measures INTER-SESSION shift")
        print(f"    (exp11=8 juil, exp12=15 juil, exp13=22 juil, other=OthmaneSalim)")

    duration = time.time() - start_time
    summary.update({
        'model_name': model_name,
        'group_level': group_level,
        'n_folds': n_actual_folds,
        'seed': seed,
        'timestamp': timestamp,
        'epochs': epochs, 'lr': lr, 'weight_decay': weight_decay,
        'batch_size': batch_size, 'patience': patience,
        'gradient_clip': gradient_clip, 'threshold': threshold,
        'use_pos_weight': use_pos_weight,
        'fold_results': fold_results,
        'training_duration_seconds': duration,
    })

    with open(run_dir / 'groupkfold_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Duration: {duration/60:.1f} min")
    print(f"  Results saved to: {run_dir}")
    return summary


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train Arc-FaultNet')

    parser.add_argument('--model', type=str, default='arcfaultnet',
                        choices=['arcfaultnet', '1d_only', 'no_attention',
                                 'standard_conv', 'independent_cbam', 'baseline_cnn',
                                 'arcfaultnet_v2'],
                        help='Model to train')
    parser.add_argument('--channel-mode', type=str, default='auto',
                        choices=['auto', 'raw2', 'i_derived4'],
                        help="1D front-end channels. 'auto' = i_derived4 for "
                             "arcfaultnet_v2, else raw2.")
    parser.add_argument('--mode', type=str, default='single',
                        choices=['cv', 'single', 'kfold', 'groupkfold'],
                        help='cv = leave-one-charge-out | single = random split | kfold = stratified K-fold | groupkfold = group-based K-fold')
    parser.add_argument('--fold', type=int, default=None,
                        help='(cv mode) Run only this fold index (0-based)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='(kfold/groupkfold mode) Number of folds (default: 5)')
    parser.add_argument('--group-level', type=str, default='recording',
                        choices=['recording', 'session'],
                        help='(groupkfold mode) Group level for splits (default: recording)')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gradient-clip', type=float, default=0.5,
                        help='Max gradient norm (0 = disabled)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for sigmoid output')
    parser.add_argument('--use-pos-weight', action='store_true',
                        help='Use pos_weight in BCEWithLogitsLoss for class imbalance')
    parser.add_argument('--data-dir', type=str, default='/home/manip/pfe_salim_gouaied/Arc-Fault-Net/labeled_dataset')
    parser.add_argument('--output-dir', type=str, default='/home/manip/pfe_salim_gouaied/Arc-Fault-Net/runs')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    # Signal / STFT parameters (for decimated datasets)
    parser.add_argument('--fs', type=int, default=None,
                        help='Sampling frequency in Hz (auto-detected from config.json if not set)')
    parser.add_argument('--n-fft', type=int, default=512,
                        help='FFT size for STFT (default: 512, use 128 for decimated 2048-point data)')
    parser.add_argument('--hop-length', type=int, default=256,
                        help='STFT hop length (default: 256, use 64 for decimated 2048-point data)')
    # Architecture enhancement flags
    parser.add_argument('--use-se', action='store_true',
                        help='Add Squeeze-and-Excitation blocks to conv layers')
    parser.add_argument('--se-reduction', type=int, default=8,
                        help='SE block reduction ratio')
    parser.add_argument('--use-amplitude', action='store_true',
                        help='Add learnable amplitude to Gabor filters')
    parser.add_argument('--deep-clf', action='store_true',
                        help='Use deeper classifier head with BatchNorm')

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

    # Resolve channel mode (V2 uses the 4 I-derived channels by default)
    if args.channel_mode == 'auto':
        channel_mode = 'i_derived4' if args.model == 'arcfaultnet_v2' else 'raw2'
    else:
        channel_mode = args.channel_mode

    dataset    = ArcFaultDataset(data_dir=str(data_dir),
                                  n_fft=args.n_fft,
                                  hop_length=args.hop_length,
                                  channel_mode=channel_mode)
    output_dir = Path(args.output_dir)

    # Resolve fs: CLI override > auto-detected from config.json
    fs = args.fs if args.fs is not None else dataset.fs
    print(f"Signal: fs={fs:,} Hz  |  n_fft={args.n_fft}  |  hop_length={args.hop_length}")

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
            fold_filter=args.fold,
            use_se=args.use_se,
            se_reduction=args.se_reduction,
            use_amplitude=args.use_amplitude,
            deep_classifier=args.deep_clf,
            fs=fs,
            n_fft=args.n_fft
        )
    elif args.mode == 'kfold':
        run_kfold_cv(
            model_name=args.model,
            dataset=dataset,
            device=device,
            n_folds=args.n_folds,
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
            use_se=args.use_se,
            se_reduction=args.se_reduction,
            use_amplitude=args.use_amplitude,
            deep_classifier=args.deep_clf,
            fs=fs,
            n_fft=args.n_fft
        )
    elif args.mode == 'groupkfold':
        run_groupkfold_cv(
            model_name=args.model,
            dataset=dataset,
            device=device,
            group_level=args.group_level,
            n_folds=args.n_folds,
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
            use_se=args.use_se,
            se_reduction=args.se_reduction,
            use_amplitude=args.use_amplitude,
            deep_classifier=args.deep_clf,
            fs=fs,
            n_fft=args.n_fft
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
            seed=args.seed,
            use_se=args.use_se,
            se_reduction=args.se_reduction,
            use_amplitude=args.use_amplitude,
            deep_classifier=args.deep_clf,
            fs=fs,
            n_fft=args.n_fft
        )


if __name__ == '__main__':
    main()
