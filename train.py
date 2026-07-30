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
  - Configurable early-stopping metric: val_f1 (default), val_precision,
    val_recall, val_specificity, or val_fbeta (weighted F-score, β tunable)
    via --monitor / --fbeta.  For arc-fault detection where false positives
    (nuisance trips) are more costly than false negatives, use
    --monitor val_fbeta --fbeta 0.5 to weight precision 4× over recall.
  - Leave-one-charge-out CV for proper generalization testing
  - Per-fold history.json and config.json
  - Model checkpointing (best + last)
  - TensorBoard logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, GroupShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    from sklearn.model_selection import GroupKFold
    _HAS_STRATIFIED_GROUP_KFOLD = False
from sklearn.metrics import roc_auc_score, average_precision_score

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

def _coral_penalty(emb: torch.Tensor, dom: torch.Tensor) -> torch.Tensor:
    """
    Deep-CORAL alignment across training campaigns (Sun & Saenko 2016), extended
    with a mean term. Pulls the per-campaign embedding distributions (1st + 2nd
    moments) together so the 128-d embedding — hence the logit — stops sliding
    from one campaign to the next. Zero parameters. Needs >=2 samples per domain.
    """
    stats = []
    d_dim = emb.shape[1]
    for d in torch.unique(dom):
        e = emb[dom == d]
        if e.shape[0] < 2:
            continue
        mu = e.mean(0)
        ec = e - mu
        cov = (ec.t() @ ec) / (e.shape[0] - 1)
        stats.append((mu, cov))
    if len(stats) < 2:
        return emb.new_zeros(())
    loss = emb.new_zeros(())
    n = 0
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            mu_i, cov_i = stats[i]; mu_j, cov_j = stats[j]
            loss = loss + ((cov_i - cov_j) ** 2).sum() / (4 * d_dim * d_dim) \
                        + ((mu_i - mu_j) ** 2).mean()
            n += 1
    return loss / max(n, 1)


def _dg_step(model, x_1d, x_2d, targets, dom, dg):
    """
    One domain-generalization training step (0 extra params). Returns (loss, logits).
      - GroupDRO (Sagawa 2020): weight the per-campaign losses toward the WORST
        campaign via an online exponentiated update, instead of the plain average.
      - CORAL: optional embedding-alignment penalty (see _coral_penalty).
    dg['group_weights'] is a persistent dict (owned by train_model) carrying the
    DRO weights across batches/epochs.
    """
    need_emb = dg['coral_weight'] > 0
    if need_emb:
        logits, emb = model(x_1d, x_2d, return_embedding=True)
    else:
        logits, emb = model(x_1d, x_2d), None
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=dg.get('pos_weight'))

    if dg['group_dro']:
        q = dg['group_weights']
        per = {int(d): bce[dom == d].mean() for d in torch.unique(dom)}
        for d, Ld in per.items():                       # online DRO weight update
            q[d] = q.get(d, 1.0) * math.exp(dg['dro_eta'] * float(Ld.detach()))
        Z = sum(q[d] for d in per) + 1e-12
        loss = sum((q[d] / Z) * Ld for d, Ld in per.items())
    else:
        loss = bce.mean()

    if need_emb:
        loss = loss + dg['coral_weight'] * _coral_penalty(emb, dom)
    return loss, logits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_clip: float = 0.5,
    label_smoothing: float = 0.05,
    channel_dropout: float = 0.0,
    dg: Optional[dict] = None
) -> Dict[str, float]:
    """Train for one epoch with label smoothing and optional channel dropout.

    Args:
        channel_dropout: Probability of dropping each temporal channel independently.
                         E.g. 0.3 means each of the 4 channels has a 30% chance of
                         being zeroed out per batch. Set to 0.0 to disable.
    """
    model.train()
    # Enable augmentation on the underlying dataset
    if hasattr(loader.dataset, 'dataset'):
        loader.dataset.dataset.training = True

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for x_1d, x_2d, labels, dom in pbar:
        x_1d   = x_1d.to(device)
        x_2d   = x_2d.to(device)
        labels = labels.to(device)

        # ── Channel dropout: zero out random temporal channels ──
        if channel_dropout > 0.0 and x_1d.shape[1] > 1:
            n_ch = x_1d.shape[1]
            # Each channel dropped independently with prob=channel_dropout
            # But always keep at least 1 channel alive
            mask = (torch.rand(n_ch, device=device) >= channel_dropout).float()
            if mask.sum() == 0:  # ensure at least 1 survives
                mask[random.randint(0, n_ch - 1)] = 1.0
            x_1d = x_1d * mask.view(1, n_ch, 1)

        # Binary label smoothing: 0 -> 0.05, 1 -> 0.95
        smoothed_labels = labels * (1.0 - 2 * label_smoothing) + label_smoothing

        optimizer.zero_grad()

        if dg is None:
            logits = model(x_1d, x_2d)
            loss   = criterion(logits, smoothed_labels)
        else:
            loss, logits = _dg_step(model, x_1d, x_2d, smoothed_labels,
                                     dom.to(device), dg)

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


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Predict"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (probs, labels) for a loader, in loader order.
    Requires shuffle=False so the output can be aligned with dataset indices.
    """
    if hasattr(loader.dataset, 'dataset'):
        loader.dataset.dataset.training = False
    model.eval()

    probs, labels = [], []
    for x_1d, x_2d, y, _ in tqdm(loader, desc=desc, leave=False):
        p = torch.sigmoid(model(x_1d.to(device), x_2d.to(device)))
        probs.append(p.cpu().numpy().ravel())
        labels.append(y.numpy().ravel())
    return np.concatenate(probs), np.concatenate(labels)


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


# ── Monitored-metric helper ────────────────────────────────────────────────
_VALID_MONITORS = ('val_f1', 'val_precision', 'val_recall',
                   'val_specificity', 'val_fbeta')

def _monitor_score(val_metrics: Dict[str, float],
                   monitor: str,
                   fbeta: float = 1.0) -> float:
    """Return the scalar value of the chosen early-stopping metric.

    Args:
        val_metrics: dict returned by evaluate().
        monitor:     one of 'val_f1', 'val_precision', 'val_recall',
                     'val_specificity', 'val_fbeta'.
        fbeta:       β for F-beta score (only used when monitor='val_fbeta').
                     β < 1 → precision-weighted (fewer false positives);
                     β > 1 → recall-weighted.
    """
    if monitor == 'val_f1':
        return val_metrics['f1']
    elif monitor == 'val_precision':
        return val_metrics['precision']
    elif monitor == 'val_recall':
        return val_metrics['recall']
    elif monitor == 'val_specificity':
        return val_metrics['specificity']
    elif monitor == 'val_fbeta':
        p  = val_metrics['precision']
        r  = val_metrics['recall']
        b2 = fbeta ** 2
        return (1 + b2) * p * r / (b2 * p + r + 1e-8)
    else:
        raise ValueError(f"Unknown monitor metric '{monitor}'. "
                         f"Choose from: {_VALID_MONITORS}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 80,
    lr: float = 3e-4,
    lr_scheduler: str = 'warm_restarts',
    weight_decay: float = 5e-4,
    patience: int = 10,
    gradient_clip: float = 0.5,
    threshold: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
    checkpoint_dir: Optional[Path] = None,
    writer: Optional[SummaryWriter] = None,
    fold_name: str = "",
    channel_dropout: float = 0.0,
    monitor: str = 'val_f1',
    fbeta: float = 1.0,
    group_dro: bool = False,
    coral_weight: float = 0.0,
    dro_eta: float = 0.05
) -> Tuple[nn.Module, Dict]:
    """
    Train model with configurable early stopping.

    monitor: metric to track — 'val_f1' | 'val_precision' | 'val_recall' |
             'val_specificity' | 'val_fbeta'.
    fbeta:   β for F-beta score (only when monitor='val_fbeta').
    lr_scheduler: 'warm_restarts' retains the historical cosine-restart
        schedule. 'cosine' performs one uninterrupted cosine decay over the
        requested number of epochs, without raising the learning rate again.

    Returns:
        model:   Best checkpoint reloaded
        history: Full training history dict
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Domain-generalization state (0 extra params). dg=None => standard training.
    dg = None
    if group_dro or coral_weight > 0:
        dg = {'group_dro': group_dro, 'coral_weight': float(coral_weight),
              'dro_eta': float(dro_eta), 'group_weights': {}, 'pos_weight': pos_weight}
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler == 'warm_restarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
    elif lr_scheduler == 'cosine':
        # Decays from `lr` to 0 once over the complete training run.  This is
        # intentionally not restarted, so late epochs cannot receive a large
        # learning-rate jump.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=0.0
        )
    else:
        raise ValueError("Unknown lr_scheduler "
                         f"{lr_scheduler!r}; choose 'warm_restarts' or 'cosine'.")

    print(f"  LR scheduler: {lr_scheduler}")

    if monitor not in _VALID_MONITORS:
        raise ValueError(f"Unknown monitor '{monitor}'. Choose from: {_VALID_MONITORS}")
    best_monitor_val = -1.0
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
            model, train_loader, criterion, optimizer, device, epoch, gradient_clip,
            channel_dropout=channel_dropout, dg=dg
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, "Val", threshold
        )

        current_lr = optimizer.param_groups[0]['lr']
        # Step after the optimizer updates.  Both schedules therefore set the
        # learning rate to be used by the *next* epoch.
        scheduler.step()

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

        # Early stopping on chosen monitor metric (max)
        score = _monitor_score(val_metrics, monitor, fbeta)
        if score > best_monitor_val:
            best_monitor_val = score
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
            print(f"  Early stopping at epoch {epoch} "
                  f"(best epoch: {best_epoch}, "
                  f"best {monitor}={100*best_monitor_val:.2f}%)")
            break

    # Reload best weights (from memory — works even without checkpoint_dir)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history['best_epoch']       = best_epoch
    history['monitor']          = monitor
    history['fbeta']            = fbeta
    history['best_monitor_val'] = best_monitor_val
    # Keep legacy key for backward-compat with evaluate.py / kfold_evaluate.py
    history['best_val_f1']      = best_monitor_val if monitor == 'val_f1' \
                                  else val_metrics.get('f1', float('nan'))

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
    lr_scheduler: str = 'warm_restarts',
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
    fusion_mode: str = 'gated',
    use_channel_attn: bool = True,
    fs: float = 1_000_000,
    n_fft: int = 512,
    monitor: str = 'val_f1',
    fbeta: float = 1.0
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
                      fusion_mode=fusion_mode,
                      use_channel_attn=use_channel_attn,
                      fs=fs, n_fft=n_fft).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        model, history = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, lr_scheduler=lr_scheduler, weight_decay=weight_decay,
            patience=patience, gradient_clip=gradient_clip,
            threshold=threshold, pos_weight=pw,
            checkpoint_dir=run_dir, writer=writer,
            fold_name=f"fold{fold_idx}_{charge_name}",
            monitor=monitor, fbeta=fbeta
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
            'epochs': epochs, 'lr': lr, 'lr_scheduler': lr_scheduler,
            'weight_decay': weight_decay,
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
        'fusion_mode':    fusion_mode,
        'epochs':         epochs, 'lr': lr, 'lr_scheduler': lr_scheduler,
        'weight_decay': weight_decay,
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
    lr_scheduler: str = 'warm_restarts',
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
    fusion_mode: str = 'gated',
    use_channel_attn: bool = True,
    fs: float = 1_000_000,
    n_fft: int = 512,
    channel_dropout: float = 0.0,
    ssm_backbone: str = 's4d',
    ssm_layers: int = 4,
    fas_k: int = 0,
    fas_channels: tuple = (1, 2),
    use_voltage: bool = False,
    monitor: str = 'val_f1',
    fbeta: float = 1.0
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
                      fusion_mode=fusion_mode,
                      use_channel_attn=use_channel_attn,
                      ssm_backbone=ssm_backbone, ssm_layers=ssm_layers,
                      fas_k=fas_k, fas_channels=fas_channels, use_voltage=use_voltage,
                      fs=fs, n_fft=n_fft).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    model, history = train_model(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr, lr_scheduler=lr_scheduler, weight_decay=weight_decay,
        patience=patience, gradient_clip=gradient_clip,
        threshold=threshold, pos_weight=pw,
        checkpoint_dir=run_dir, writer=writer,
        fold_name="single", channel_dropout=channel_dropout,
        monitor=monitor, fbeta=fbeta
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
        'fusion_mode':    fusion_mode,
        'ssm_backbone':   ssm_backbone,
        'ssm_layers':     ssm_layers,
        'fas_k':          fas_k,
        'fas_channels':   list(fas_channels),
        'epochs':         epochs, 'lr': lr, 'lr_scheduler': lr_scheduler,
        'weight_decay': weight_decay,
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
    lr_scheduler: str = 'warm_restarts',
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
    fusion_mode: str = 'gated',
    use_channel_attn: bool = True,
    fs: float = 1_000_000,
    n_fft: int = 512,
    monitor: str = 'val_f1',
    fbeta: float = 1.0
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
                          fusion_mode=fusion_mode,
                          use_channel_attn=use_channel_attn,
                          fs=fs, n_fft=n_fft).to(device)

        if fold_idx == 0:
            print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

        model, history = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, lr_scheduler=lr_scheduler, weight_decay=weight_decay,
            patience=patience, gradient_clip=gradient_clip,
            threshold=threshold, pos_weight=pw,
            checkpoint_dir=fold_dir, writer=writer,
            fold_name=f"fold_{fold_idx+1}",
            monitor=monitor, fbeta=fbeta
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
        'lr_scheduler': lr_scheduler,
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

def load_metadata(data_dir: Path, n_samples: int) -> pd.DataFrame:
    """Load metadata.csv and assert row-by-row alignment with X_multi.npy / y.npy."""
    metadata_path = data_dir / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    meta = pd.read_csv(metadata_path)

    if len(meta) != n_samples:
        raise ValueError(
            f"metadata.csv has {len(meta)} rows but dataset has {n_samples} samples. "
            f"They must be aligned row-by-row. Aborting."
        )
    return meta


def load_alternance_ids(data_dir: Path, n_samples: int) -> np.ndarray:
    """
    Sub-group IDs at the *alternance* level: one arc burst / one normal window.

    Several consecutive 20 ms cycles are cut from the same alternance, so those
    cycles are strongly correlated. Any split that is meant to be honest — even a
    validation split used only for early stopping — must keep a whole alternance
    on one side. In the 2024 campaigns an alternance holds up to 88 cycles; in the
    2026 campaign each recording contributes a single cycle, so the alternance ID
    degenerates to the cycle ID (which is correct: those recordings are independent).
    """
    meta = load_metadata(data_dir, n_samples)
    for col in ('dataset', 'exp_name', 'alt_index'):
        if col not in meta.columns:
            raise ValueError(f"metadata.csv must have '{col}' for alternance-level splits")
    return (meta['dataset'].astype(str) + '|' + meta['exp_name'].astype(str)
            + '|' + meta['alt_index'].astype(str)).values


def load_group_ids(data_dir: Path, n_samples: int, group_level: str = 'recording') -> np.ndarray:
    r"""
    Derive group IDs from metadata.csv for group-based cross-validation.
    Returns string array aligned with X_multi.npy / y.npy.

    campaign:  each acquisition campaign is a group (metadata 'dataset' column):
               8_juillet / 15_juillet / 22_juillet (IJL 2024) + OthmaneSalim (2026).
               This is the coarsest and most honest level available: the dataset
               carries no per-experiment load information, so leave-one-charge-out
               is impossible and leave-one-campaign-out replaces it.
    recording: each unique exp_name is a group. Unbalanced here — the three 2024
               campaigns are one monolithic recording each (2.7k-3.8k cycles) while
               the 2026 campaign splits into 22 small recordings (40-83 cycles).
    session:   regex exp(\d+) extraction, fallback 'other'. Equivalent to campaign
               on the current dataset, kept for backwards compatibility.
    """
    meta = load_metadata(data_dir, n_samples)

    if group_level == 'campaign':
        if 'dataset' not in meta.columns:
            raise ValueError("metadata.csv must have a 'dataset' column for campaign-level CV")
        group_ids = meta['dataset'].values.astype(str)
        unique_groups, counts = np.unique(group_ids, return_counts=True)
        print(f"\n  Group level 'campaign': {len(unique_groups)} unique groups")
        for g, c in zip(unique_groups, counts):
            print(f"    {g}: {c} cycles")
        return group_ids

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


def load_recording_ids(data_dir: Path, n_samples: int) -> np.ndarray:
    r"""
    Recording-level group IDs (one physical CSV / one continuous LeCroy capture),
    aligned with X_multi.npy / y.npy. A recording is the honest anti-leakage unit
    for the validation split: its consecutive périodes all evolve from ONE arc
    event, so they must never be split across train and val.

    The recording id is not stored as a column; it is recovered from metadata.
    Two storage conventions coexist and are auto-detected per (dataset, exp_name):
      - 2024 IJL July campaigns: one exp_name packs many recordings, stored in
        temporal order; a new recording begins wherever alt_index resets
        (sawtooth). ~65-88 recordings per campaign.
      - 2026 OthmaneSalim: each exp_name (load config) IS one recording of ~76
        périodes stored shuffled; alt_index is nearly unique within it, so the
        exp_name itself is the recording id. 20 recordings.
    """
    meta = load_metadata(data_dir, n_samples)
    for col in ('dataset', 'exp_name', 'alt_index'):
        if col not in meta.columns:
            raise ValueError(f"metadata.csv must have '{col}' for recording-level splits")
    ds  = meta['dataset'].astype(str).values
    exp = meta['exp_name'].astype(str).values
    alt = meta['alt_index'].astype(int).values

    blocks = {}
    for i in range(n_samples):                       # group rows by (dataset,exp), keep order
        blocks.setdefault((ds[i], exp[i]), []).append(i)

    rec = np.empty(n_samples, dtype=object)
    for (d, e), idxs in blocks.items():
        alts = [alt[i] for i in idxs]
        if len(set(alts)) < 0.5 * len(idxs):         # repeats => many recordings (sawtooth)
            run, prev = 0, None
            for i in idxs:
                if prev is not None and alt[i] <= prev:
                    run += 1
                rec[i] = f"{d}|{e}|run{run}"
                prev = alt[i]
        else:                                        # nearly unique => exp_name is one recording
            for i in idxs:
                rec[i] = f"{d}|{e}"
    return rec.astype(str)


def _auc_scores(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """(ROC-AUC, PR-AUC) — threshold-free. NaN if only one class is present."""
    labels = np.asarray(labels).astype(int)
    probs  = np.asarray(probs, dtype=float)
    if len(np.unique(labels)) < 2:
        return float('nan'), float('nan')
    try:
        return (float(roc_auc_score(labels, probs)),
                float(average_precision_score(labels, probs)))
    except Exception:
        return float('nan'), float('nan')


def _metrics_from_probs(labels: np.ndarray, probs: np.ndarray,
                        threshold: float) -> Dict[str, float]:
    """Confusion-matrix metrics at a given threshold, computed from stored probs."""
    labels = np.asarray(labels).astype(int)
    pred = (np.asarray(probs, dtype=float) > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    return {
        'accuracy': (tp + tn) / max(len(labels), 1),
        'precision': prec, 'recall': rec,
        'f1': 2 * prec * rec / (prec + rec + 1e-8),
        'specificity': tn / (tn + fp + 1e-8),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def _select_threshold(labels: np.ndarray, probs: np.ndarray,
                      beta: float = 1.0) -> Tuple[float, float]:
    """
    Threshold that maximises F-beta on the given probs. MUST be called on
    VALIDATION data only — never on the test campaign. Returns (threshold,
    best_fbeta); falls back to 0.5 if the split has a single class.
    """
    labels = np.asarray(labels).astype(int)
    probs  = np.asarray(probs, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5, float('nan')
    b2 = beta * beta
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 197):
        pred = (probs > t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum(); fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        s = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-8)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, float(best_s)


def run_groupkfold_cv(
    model_name: str,
    dataset: ArcFaultDataset,
    device: torch.device,
    group_level: str = 'recording',
    val_mode: str = 'auto',
    n_folds: int = 5,
    epochs: int = 80,
    lr: float = 3e-4,
    lr_scheduler: str = 'warm_restarts',
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
    fusion_mode: str = 'gated',
    use_channel_attn: bool = True,
    fs: float = 1_000_000,
    n_fft: int = 512,
    channel_dropout: float = 0.0,
    ssm_backbone: str = 's4d',
    ssm_layers: int = 4,
    fas_k: int = 0,
    fas_channels: tuple = (1, 2),
    use_voltage: bool = False,
    monitor: str = 'val_f1',
    fbeta: float = 1.0,
    group_dro: bool = False,
    coral_weight: float = 0.0,
    dro_eta: float = 0.05,
    dg_balanced_sampler: bool = False
) -> Dict:
    """
    Group-based cross-validation preventing data leakage.

    - campaign level: LeaveOneGroupOut on acquisition campaigns
      (8/15/22 juillet IJL 2024 + OthmaneSalim 2026). Each fold trains on three
      campaigns and tests on the fourth. This is the primary generalization
      protocol: the dataset has no per-experiment load labels, so
      leave-one-charge-out cannot be run and leave-one-campaign-out replaces it.
    - session level: same thing via regex on exp_name (legacy alias).
    - recording level: StratifiedGroupKFold on exp_name groups. Folds are very
      unbalanced on this dataset (one 2024 recording = up to 3820 cycles vs
      ~80 for a 2026 recording), so read its mean ± std with care.

    All cycles from the same group stay in a single fold. The validation split is
    also leakage-free: `val_mode` controls how it is carved out of the training
    groups (see --val-mode).
    """
    start_time = time.time()

    data_dir = Path(dataset.data_dir)
    group_ids = load_group_ids(data_dir, len(dataset), group_level)
    # Domain generalization needs a per-cycle campaign id. The dataset returns
    # self.charges as the 4th batch element (dummy here), so we repurpose it to
    # carry the integer campaign id — no dataset/interface change needed.
    if group_dro or coral_weight > 0 or dg_balanced_sampler:
        _c2i = {c: i for i, c in enumerate(sorted(np.unique(group_ids)))}
        dataset.charges = np.array([_c2i[g] for g in group_ids], dtype=np.int64)
    labels = dataset.y
    indices = np.arange(len(dataset))

    # ── Choose splitter ──────────────────────────────────
    if group_level in ('session', 'campaign'):
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

    # ── Resolve how the validation set is carved out of the training groups ──
    # Honest unit is the RECORDING (one CSV / one continuous capture): all its
    # consecutive périodes stay on one side. 'group' spends whole campaign groups
    # (too costly with 4 campaigns); 'alternance' groups période-slots and still
    # splits a recording across train/val; 'random' is cycle-level (leaky).
    try:
        recording_ids = load_recording_ids(data_dir, len(dataset))
        _rec_err = None
    except Exception as e:
        recording_ids, _rec_err = None, str(e)
    n_train_groups_typical = len(np.unique(group_ids)) - 1
    if val_mode == 'auto':
        if recording_ids is not None:
            val_mode_eff = 'recording'
        elif n_train_groups_typical >= 6:
            val_mode_eff = 'group'
        else:
            val_mode_eff = 'alternance'
    else:
        val_mode_eff = val_mode
    if val_mode_eff == 'recording' and recording_ids is None:
        raise ValueError(f"--val-mode recording unavailable (recording ids: {_rec_err})")
    alternance_ids = (load_alternance_ids(data_dir, len(dataset))
                      if val_mode_eff == 'alternance' else None)

    level_desc = {
        'campaign':  "generalization to an UNSEEN ACQUISITION CAMPAIGN (LOCO substitute)",
        'session':   "inter-session shift",
        'recording': "generalization to unseen recordings",
    }.get(group_level, group_level)
    val_desc = {
        'recording':  "~1/7 of training RECORDINGS held out, label-stratified "
                      "(one CSV/capture kept whole — no période of a recording "
                      "is split across train/val)",
        'alternance': "15% of training-campaign ALTERNANCES (label-stratified, "
                      "no arc burst shared with train)",
        'group':      "whole training groups (~15%)",
        'random':     "15% of training cycles, label-stratified (cycle-level, "
                      "correlated with train — use only for smoke tests)",
    }[val_mode_eff]
    print(f"\n{'='*60}")
    print(f"GROUP CROSS-VALIDATION  (anti-leakage)")
    print(f"Model: {model_name}  |  group_level={group_level}  |  seed={seed}")
    print(f"Measures: {level_desc}")
    print(f"Val split ({val_mode_eff}): {val_desc}")
    print(f"{'='*60}")

    fold_results = []
    # Out-of-fold predictions: with LeaveOneGroupOut every cycle is tested exactly
    # once, by a model that never saw its group → they pool into one honest matrix.
    oof_probs = np.full(len(dataset), np.nan, dtype=np.float64)
    oof_fold = np.full(len(dataset), -1, dtype=np.int64)
    # Calibrated OOF preds: each fold's test cycles predicted at THAT fold's
    # val-chosen threshold (Phase 1). Pooled → the honest deployable operating point.
    oof_pred_cal = np.full(len(dataset), -1, dtype=np.int64)

    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        fold_seed = seed + fold_idx
        set_seed(fold_seed)

        # ── Anti-leakage: test vs train+val ──
        test_groups = set(group_ids[test_idx])
        train_val_groups = set(group_ids[train_val_idx])
        assert test_groups.isdisjoint(train_val_groups), (
            f"LEAKAGE fold {fold_idx}: groups in both train_val and test: "
            f"{test_groups & train_val_groups}")

        # ── Sub-split train_val → train + val (leakage-free, see val_mode_eff) ──
        if val_mode_eff == 'group':
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=fold_seed)
            tv_train_sub, tv_val_sub = next(gss.split(
                train_val_idx, labels[train_val_idx], groups=group_ids[train_val_idx]))
        elif val_mode_eff == 'recording':
            # ~1/7 of the training RECORDINGS to val, label-stratified, each
            # recording (one CSV/capture) kept intact on one side.
            sub_groups = recording_ids[train_val_idx]
            if _HAS_STRATIFIED_GROUP_KFOLD:
                sub_splitter = StratifiedGroupKFold(
                    n_splits=7, shuffle=True, random_state=fold_seed)
            else:
                sub_splitter = GroupKFold(n_splits=7)
            tv_train_sub, tv_val_sub = next(sub_splitter.split(
                train_val_idx, labels[train_val_idx], groups=sub_groups))
        elif val_mode_eff == 'alternance':
            # ~1/7 of the alternances to val, label-stratified, alternances intact
            sub_groups = alternance_ids[train_val_idx]
            if _HAS_STRATIFIED_GROUP_KFOLD:
                sub_splitter = StratifiedGroupKFold(
                    n_splits=7, shuffle=True, random_state=fold_seed)
            else:
                sub_splitter = GroupKFold(n_splits=7)
            tv_train_sub, tv_val_sub = next(sub_splitter.split(
                train_val_idx, labels[train_val_idx], groups=sub_groups))
        elif val_mode_eff == 'random':
            skf_sub = StratifiedKFold(n_splits=7, shuffle=True, random_state=fold_seed)
            tv_train_sub, tv_val_sub = next(skf_sub.split(
                train_val_idx, labels[train_val_idx]))
        else:
            raise ValueError(f"Unknown val_mode: {val_mode_eff}")

        train_idx = train_val_idx[tv_train_sub]
        val_idx = train_val_idx[tv_val_sub]

        # ── Anti-leakage: pairwise disjoint ──
        train_groups = set(group_ids[train_idx])
        val_groups = set(group_ids[val_idx])
        if val_mode_eff == 'group':
            assert train_groups.isdisjoint(val_groups), (
                f"LEAKAGE fold {fold_idx}: groups shared train↔val: "
                f"{train_groups & val_groups}")
        elif val_mode_eff == 'recording':
            train_recs = set(recording_ids[train_idx])
            val_recs = set(recording_ids[val_idx])
            assert train_recs.isdisjoint(val_recs), (
                f"LEAKAGE fold {fold_idx}: recordings shared train↔val: "
                f"{sorted(train_recs & val_recs)[:5]}")
        elif val_mode_eff == 'alternance':
            train_alts = set(alternance_ids[train_idx])
            val_alts = set(alternance_ids[val_idx])
            assert train_alts.isdisjoint(val_alts), (
                f"LEAKAGE fold {fold_idx}: alternances shared train↔val: "
                f"{sorted(train_alts & val_alts)[:5]}")
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

        # Background-load donors for the strong augmentation must come from this
        # fold's training split only — never from the held-out campaign.
        if getattr(dataset, 'strong_augment', False):
            dataset.set_donor_pool(train_idx)
            print(f"    strong augmentation ON — donor pool: "
                  f"{len(dataset._donor_pool)} normal training cycles")

        # Campaign-balanced sampling (0 params): each training campaign is equally
        # likely per batch, so no campaign dominates the gradient and the per-campaign
        # DRO/CORAL estimates stay stable.
        if dg_balanced_sampler:
            _dom = dataset.charges[train_idx]
            _w = 1.0 / np.bincount(_dom)[_dom]
            _sampler = WeightedRandomSampler(
                torch.as_tensor(_w, dtype=torch.double), len(train_idx), replacement=True)
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                                      sampler=_sampler, num_workers=num_workers,
                                      pin_memory=True, drop_last=True)
        else:
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
                          fusion_mode=fusion_mode,
                          use_channel_attn=use_channel_attn,
                          ssm_backbone=ssm_backbone, ssm_layers=ssm_layers,
                          fas_k=fas_k, fas_channels=fas_channels, use_voltage=use_voltage,
                          fs=fs, n_fft=n_fft).to(device)

        if fold_idx == 0:
            print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # ── Train ──
        model, history = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, lr_scheduler=lr_scheduler, weight_decay=weight_decay,
            patience=patience, gradient_clip=gradient_clip,
            threshold=threshold, pos_weight=pw,
            checkpoint_dir=fold_dir, writer=writer,
            fold_name=fold_name, channel_dropout=channel_dropout,
            monitor=monitor, fbeta=fbeta,
            group_dro=group_dro, coral_weight=coral_weight, dro_eta=dro_eta
        )

        # ── Evaluate on held-out test groups ──
        criterion = nn.BCEWithLogitsLoss()
        test_metrics = evaluate(model, test_loader, criterion, device,
                                f"Fold {fold_idx+1} Test", threshold)
        writer.close()

        # ── Store raw test probabilities (threshold-free re-analysis later) ──
        fold_probs, fold_labels = predict_probs(model, test_loader, device,
                                                f"Fold {fold_idx+1} Probs")
        assert np.array_equal(fold_labels.astype(int), labels[test_idx].astype(int)), \
            f"fold {fold_idx}: prediction order does not match test_idx order"
        np.savez(fold_dir / 'test_predictions.npz',
                 idx=test_idx, probs=fold_probs, labels=fold_labels,
                 groups=np.array([str(g) for g in group_ids[test_idx]]))
        oof_probs[test_idx] = fold_probs
        oof_fold[test_idx] = fold_idx + 1

        # ── Phase 0/1: honest val-based threshold + threshold-free AUC ─────────
        # Pick the operating threshold on VALIDATION probs only (never on test),
        # using the same F-beta objective as the early-stopping monitor, then
        # apply it once to the held-out campaign. ROC-AUC / PR-AUC are reported
        # threshold-free (immune to the per-campaign logit drift).
        val_probs, val_labels_arr = predict_probs(model, val_loader, device,
                                                  f"Fold {fold_idx+1} Val-probs")
        assert np.array_equal(val_labels_arr.astype(int), labels[val_idx].astype(int)), \
            f"fold {fold_idx}: val prediction order does not match val_idx order"
        np.savez(fold_dir / 'val_predictions.npz',
                 idx=val_idx, probs=val_probs, labels=val_labels_arr)
        beta_sel = fbeta if monitor == 'val_fbeta' else 1.0
        thr_star, val_fbeta_star = _select_threshold(val_labels_arr, val_probs, beta_sel)
        test_cal = _metrics_from_probs(fold_labels, fold_probs, thr_star)
        test_roc_auc, test_pr_auc = _auc_scores(fold_labels, fold_probs)
        oof_pred_cal[test_idx] = (fold_probs > thr_star).astype(int)

        print(f"    → @0.50   Acc={100*test_metrics['accuracy']:.2f}%  "
              f"F1={100*test_metrics['f1']:.2f}%  "
              f"Prec={100*test_metrics['precision']:.2f}%  "
              f"Rec={100*test_metrics['recall']:.2f}%  "
              f"Spec={100*test_metrics['specificity']:.2f}%")
        print(f"    → @{thr_star:.2f}*  Acc={100*test_cal['accuracy']:.2f}%  "
              f"F1={100*test_cal['f1']:.2f}%  "
              f"Prec={100*test_cal['precision']:.2f}%  "
              f"Rec={100*test_cal['recall']:.2f}%  "
              f"Spec={100*test_cal['specificity']:.2f}%   (*threshold from val)")
        print(f"      ROC-AUC={test_roc_auc:.4f}  PR-AUC={test_pr_auc:.4f}  (threshold-free)")

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
            # Phase 0/1: threshold-free + val-calibrated
            'val_threshold': thr_star,
            'val_fbeta_star': val_fbeta_star,
            'test_roc_auc': test_roc_auc,
            'test_pr_auc': test_pr_auc,
            'test_accuracy_cal': float(test_cal['accuracy']),
            'test_f1_cal': float(test_cal['f1']),
            'test_precision_cal': float(test_cal['precision']),
            'test_recall_cal': float(test_cal['recall']),
            'test_specificity_cal': float(test_cal['specificity']),
        }
        fold_results.append(fold_result)

        # Per-fold config
        fold_config = {
            'fold': fold_idx + 1, 'fold_name': fold_name,
            'model_name': model_name, 'group_level': group_level,
            'n_folds': n_actual_folds, 'global_seed': seed,
            'fold_seed': fold_seed,
            'epochs': epochs, 'lr': lr, 'lr_scheduler': lr_scheduler,
            'weight_decay': weight_decay,
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

    # ── Threshold-free + val-calibrated fold metrics (Phase 0/1) ──
    print(f"\n  Threshold-free (mean ± std over folds):")
    for m, lab in [('test_roc_auc', 'ROC-AUC'), ('test_pr_auc', 'PR-AUC')]:
        vals = np.array([r[m] for r in fold_results], dtype=float)
        print(f"    {lab:11s}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")
        summary[f'{m}_mean'] = float(np.nanmean(vals)); summary[f'{m}_std'] = float(np.nanstd(vals))
    print(f"\n  At val-chosen threshold (per fold — honest operating point):")
    for m, lab in [('test_f1_cal', 'F1'), ('test_precision_cal', 'Precision'),
                   ('test_recall_cal', 'Recall'), ('test_specificity_cal', 'Specificity')]:
        vals = np.array([r[m] for r in fold_results], dtype=float)
        print(f"    {lab:12s}: {100*np.nanmean(vals):.2f}% ± {100*np.nanstd(vals):.2f}%")
        summary[f'{m}_mean'] = float(np.nanmean(vals)); summary[f'{m}_std'] = float(np.nanstd(vals))
    print(f"    thresholds : {[round(r['val_threshold'], 2) for r in fold_results]}")

    # ── Pooled out-of-fold metrics ───────────────────────
    # Mean ± std over folds weights a 60-cycle fold like a 3820-cycle one. When the
    # folds partition the dataset (LeaveOneGroupOut), pooling every out-of-fold
    # prediction into a single matrix is the number to quote as "the" test score.
    tested = ~np.isnan(oof_probs)
    pooled = None
    if tested.sum() > 0:
        p = oof_probs[tested]
        l = labels[tested].astype(int)
        pred = (p > threshold).astype(int)
        tp = int(((pred == 1) & (l == 1)).sum()); fp = int(((pred == 1) & (l == 0)).sum())
        fn = int(((pred == 0) & (l == 1)).sum()); tn = int(((pred == 0) & (l == 0)).sum())
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        roc_auc, pr_auc = _auc_scores(l, p)
        pooled = {
            'n_tested': int(tested.sum()),
            'covers_full_dataset': bool(tested.all()),
            'accuracy': (tp + tn) / max(tested.sum(), 1),
            'precision': prec, 'recall': rec,
            'f1': 2 * prec * rec / (prec + rec + 1e-8),
            'specificity': tn / (tn + fp + 1e-8),
            'roc_auc': roc_auc, 'pr_auc': pr_auc,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }
        # Pooled at per-fold val-chosen thresholds (honest deployable operating point)
        cal_mask = tested & (oof_pred_cal >= 0)
        if cal_mask.sum() > 0:
            lc = labels[cal_mask].astype(int); pc = oof_pred_cal[cal_mask].astype(int)
            tpc = int(((pc == 1) & (lc == 1)).sum()); fpc = int(((pc == 1) & (lc == 0)).sum())
            fnc = int(((pc == 0) & (lc == 1)).sum()); tnc = int(((pc == 0) & (lc == 0)).sum())
            precc = tpc / (tpc + fpc + 1e-8); recc = tpc / (tpc + fnc + 1e-8)
            pooled['calibrated'] = {
                'accuracy': (tpc + tnc) / max(int(cal_mask.sum()), 1),
                'precision': precc, 'recall': recc,
                'f1': 2 * precc * recc / (precc + recc + 1e-8),
                'specificity': tnc / (tnc + fpc + 1e-8),
                'tp': tpc, 'fp': fpc, 'fn': fnc, 'tn': tnc,
            }
        np.savez(run_dir / 'oof_predictions.npz',
                 probs=oof_probs, labels=labels.astype(int), fold=oof_fold,
                 pred_cal=oof_pred_cal,
                 groups=np.array([str(g) for g in group_ids]))
        print(f"\n  POOLED OUT-OF-FOLD ({pooled['n_tested']}/{len(dataset)} cycles, "
              f"each predicted by a model that never saw its {group_level}):")
        print(f"    Accuracy    : {100*pooled['accuracy']:.2f}%")
        print(f"    F1          : {100*pooled['f1']:.2f}%")
        print(f"    Precision   : {100*pooled['precision']:.2f}%")
        print(f"    Recall      : {100*pooled['recall']:.2f}%")
        print(f"    Specificity : {100*pooled['specificity']:.2f}%")
        print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"    ROC-AUC     : {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}  (threshold-free)")
        if 'calibrated' in pooled:
            c = pooled['calibrated']
            print(f"\n  POOLED at val-chosen thresholds (honest deployable operating point):")
            print(f"    Accuracy    : {100*c['accuracy']:.2f}%")
            print(f"    F1          : {100*c['f1']:.2f}%")
            print(f"    Precision   : {100*c['precision']:.2f}%")
            print(f"    Recall      : {100*c['recall']:.2f}%")
            print(f"    Specificity : {100*c['specificity']:.2f}%")
            print(f"    TP={c['tp']}  FP={c['fp']}  FN={c['fn']}  TN={c['tn']}")

    print(f"\n  Interpretation:")
    if group_level == 'recording':
        print(f"    group-level=recording measures generalization to NEW RECORDINGS")
        print(f"    (folds are size-unbalanced on this dataset — prefer 'campaign')")
    else:
        print(f"    group-level={group_level} measures generalization to an UNSEEN CAMPAIGN")
        print(f"    (8 juil / 15 juil / 22 juil 2024 IJL + OthmaneSalim 2026)")
        print(f"    Stands in for leave-one-charge-out, which the dataset cannot support:")
        print(f"    the 2024 campaigns carry no per-experiment load labels.")

    duration = time.time() - start_time
    summary.update({
        'model_name': model_name,
        'group_level': group_level,
        'val_mode': val_mode_eff,
        'pooled_oof': pooled,
        'channel_dropout': channel_dropout,
        'strong_augment': bool(getattr(dataset, 'strong_augment', False)),
        'ssm_backbone': ssm_backbone,
        'ssm_layers': ssm_layers,
        'fas_k': fas_k,
        'fas_channels': list(fas_channels),
        'use_voltage': use_voltage,
        'group_dro': group_dro,
        'coral_weight': coral_weight,
        'dro_eta': dro_eta,
        'dg_balanced_sampler': dg_balanced_sampler,
        'n_folds': n_actual_folds,
        'seed': seed,
        'timestamp': timestamp,
        'epochs': epochs, 'lr': lr, 'lr_scheduler': lr_scheduler,
        'weight_decay': weight_decay,
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
                                 'arcfaultnet_v2',
                                 'arcssm', 'arcssm_selective'],
                        help='Model to train')
    parser.add_argument('--channel-mode', type=str, default='auto',
                        choices=['auto', 'raw2', 'i_derived4'],
                        help="1D front-end channels. 'auto' = i_derived4 for "
                             "arcfaultnet_v2 / arcssm, else raw2.")
    parser.add_argument('--mode', type=str, default='single',
                        choices=['cv', 'single', 'kfold', 'groupkfold'],
                        help='cv = leave-one-charge-out | single = random split | kfold = stratified K-fold | groupkfold = group-based K-fold')
    parser.add_argument('--fold', type=int, default=None,
                        help='(cv mode) Run only this fold index (0-based)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='(kfold/groupkfold mode) Number of folds (default: 5)')
    parser.add_argument('--group-level', type=str, default='recording',
                        choices=['recording', 'session', 'campaign'],
                        help='(groupkfold mode) Group level for splits (default: recording). '
                             'campaign = leave-one-acquisition-campaign-out (4 folds, '
                             'recommended: the honest generalization protocol on this dataset)')
    parser.add_argument('--val-mode', type=str, default='auto',
                        choices=['auto', 'recording', 'alternance', 'group', 'random'],
                        help='(groupkfold mode) How the val set is taken from the training '
                             'groups. recording = ~1/7 of training RECORDINGS held out, one '
                             'CSV/capture kept whole (the honest default via auto). '
                             'alternance = 15%% of training période-slots (still splits a '
                             'recording across train/val). group = whole campaign groups. '
                             'random = cycle-level, leaky, smoke tests only.')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr-scheduler', type=str, default='warm_restarts',
                        choices=['warm_restarts', 'cosine'],
                        help='Learning-rate schedule: warm_restarts = the previous '
                             'cosine schedule with restarts; cosine = one smooth '
                             'cosine decay over --epochs, with no LR jumps.')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gradient-clip', type=float, default=0.5,
                        help='Max gradient norm (0 = disabled)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for sigmoid output')
    parser.add_argument('--use-pos-weight', action='store_true',
                        help='Use pos_weight in BCEWithLogitsLoss for class imbalance')
    parser.add_argument('--data-dir', type=str, default='home/top/Arc-Fault-Net/labeled_dataset')
    parser.add_argument('--output-dir', type=str, default='runs')
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
    parser.add_argument('--no-channel-attn', action='store_true',
                        help='Disable the temporal-branch channel attention '
                             '(DescriptorChannelAttention). It is ON by default; '
                             'use this flag only for the channel-attention ablation.')
    parser.add_argument('--fusion-mode', type=str, default='gated',
                        choices=['gated', 'cross_attention', 'concat'],
                        help='V2 fusion: gated (cross-conditioned gating), '
                             'cross_attention (true Q/K/V), concat (simple concat)')
    parser.add_argument('--strong-aug', action='store_true',
                        help='Cross-campaign robustness augmentation on the raw cycle: '
                             'pink noise at random SNR, spectral tilt, band limiting, '
                             '±0.5 Hz mains jitter, time shift, half-cycle+polarity flip, '
                             'and background-load mixing with normal cycles of the '
                             'TRAINING split only. Off by default (keeps older runs '
                             'reproducible).')
    parser.add_argument('--channel-dropout', type=float, default=0.0,
                        help='Per-channel dropout probability for temporal branch (0.0=off, '
                             '0.3=each ch has 30%% chance of being zeroed per batch). '
                             'Forces model to learn from all channels.')
    parser.add_argument('--ssm-backbone', choices=['s4d', 'mamba'], default='s4d',
                        help='ArcSSM sequence backbone (only --model arcssm): s4d = LTI '
                             'diagonal-complex resonator bank (default); mamba = selective '
                             'S6 (DCAMamba-style, causal scan).')
    parser.add_argument('--ssm-layers', type=int, default=4,
                        help='Number of stacked SSM blocks in the arcssm track '
                             '(DCAMamba ablation: 2 near-optimal, >4 overfits).')
    parser.add_argument('--fas-k', type=int, default=0,
                        help='FAS (Feature Amplification Strategy) K: per channel keep '
                             'top-K + bottom-K values over time (2K total); 0 = off. '
                             'Order-statistic front-end adapted from DCAMamba (DC->AC).')
    parser.add_argument('--fas-channels', type=str, default='1,2',
                        help='Comma-separated channel indices FAS applies to (default '
                             '"1,2" = |dI|,TKEO, fundamental-suppressed; 0=I_norm, '
                             '3=RMS_slide). Used only when --fas-k > 0.')
    # ── Domain generalization (groupkfold only; 0 extra parameters) ──────────
    parser.add_argument('--group-dro', action='store_true',
                        help='GroupDRO (Sagawa 2020): optimise the WORST training '
                             'campaign instead of the average. Targets cross-campaign '
                             'generalization directly. groupkfold only.')
    parser.add_argument('--coral-weight', type=float, default=0.0,
                        help='Deep-CORAL penalty weight: align per-campaign embedding '
                             'mean+covariance to reduce the per-campaign score slide. '
                             '0=off. Try 0.1-1.0. groupkfold only.')
    parser.add_argument('--dro-eta', type=float, default=0.05,
                        help='GroupDRO step size for the online worst-group weighting.')
    parser.add_argument('--dg-balanced-sampler', action='store_true',
                        help='Sample each training campaign equally per batch '
                             '(recommended with --group-dro / --coral-weight).')
    parser.add_argument('--use-voltage', action='store_true',
                        help='ArcSSM dual-branch: add a lighter S4D branch on '
                             'v_derived4 (voltage) fused with the current branch. '
                             "v(t)'s HF arc signature is more bench-consistent, so "
                             'this mainly stabilises specificity on unseen campaigns. '
                             'Switches channel-mode to iv_derived4 (8 channels).')
    # ── Early-stopping monitor ──────────────────────────────────────────────
    parser.add_argument('--monitor', type=str, default='val_f1',
                        choices=list(_VALID_MONITORS),
                        help='Metric to maximise for early stopping and model '
                             'checkpointing. Choices: '
                             'val_f1 (default, balanced F1); '
                             'val_precision (minimises false positives / nuisance '
                             'trips — recommended for arc-fault disjoncteur context); '
                             'val_recall (minimises missed arcs); '
                             'val_specificity (TN rate — how often normal cycles '
                             'are correctly left alone); '
                             'val_fbeta (weighted F-score, use --fbeta to set β: '
                             'β<1 → precision-heavy, β>1 → recall-heavy).')
    parser.add_argument('--fbeta', type=float, default=1.0,
                        help='β for the F-beta score used when --monitor val_fbeta. '
                             'β=0.5 weights precision 4× over recall (fewer false '
                             'alarms). β=2 weights recall 4× over precision (fewer '
                             'missed arcs). Default: 1.0 (= F1).')

    args = parser.parse_args()
    fas_channels = tuple(int(c) for c in str(args.fas_channels).split(',') if c.strip())

    set_seed(args.seed)

    device = torch.device('cpu') if (args.cpu or not torch.cuda.is_available()) \
             else torch.device('cuda')
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == 'cuda' else ''))

    data_dir = Path(args.data_dir)
    if not (data_dir / 'X_multi.npy').exists():
        print(f"\nData not found at {data_dir}")
        print("Run: python step2_build_multichannel.py")
        return

    # Resolve channel mode (V2 and the SSM track use the 4 I-derived channels)
    if args.channel_mode == 'auto':
        if args.model in ('arcssm', 'arcssm_selective') and args.use_voltage:
            channel_mode = 'iv_derived4'          # dual-branch: I + V derived channels
        elif args.model in ('arcfaultnet_v2', 'arcssm', 'arcssm_selective'):
            channel_mode = 'i_derived4'
        else:
            channel_mode = 'raw2'
    else:
        channel_mode = args.channel_mode

    dataset    = ArcFaultDataset(data_dir=str(data_dir),
                                  n_fft=args.n_fft,
                                  hop_length=args.hop_length,
                                  channel_mode=channel_mode,
                                  strong_augment=args.strong_aug)
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
            lr_scheduler=args.lr_scheduler,
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
            fusion_mode=args.fusion_mode,
            use_channel_attn=not args.no_channel_attn,
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
            lr_scheduler=args.lr_scheduler,
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
            fusion_mode=args.fusion_mode,
            use_channel_attn=not args.no_channel_attn,
            fs=fs,
            n_fft=args.n_fft,
            monitor=args.monitor,
            fbeta=args.fbeta
        )
    elif args.mode == 'groupkfold':
        run_groupkfold_cv(
            model_name=args.model,
            dataset=dataset,
            device=device,
            group_level=args.group_level,
            val_mode=args.val_mode,
            n_folds=args.n_folds,
            epochs=args.epochs,
            lr=args.lr,
            lr_scheduler=args.lr_scheduler,
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
            fusion_mode=args.fusion_mode,
            use_channel_attn=not args.no_channel_attn,
            fs=fs,
            n_fft=args.n_fft,
            channel_dropout=args.channel_dropout,
            ssm_backbone=args.ssm_backbone,
            ssm_layers=args.ssm_layers,
            fas_k=args.fas_k,
            fas_channels=fas_channels,
            use_voltage=args.use_voltage,
            monitor=args.monitor,
            fbeta=args.fbeta,
            group_dro=args.group_dro,
            coral_weight=args.coral_weight,
            dro_eta=args.dro_eta,
            dg_balanced_sampler=args.dg_balanced_sampler
        )
    else:
        run_single_training(
            model_name=args.model,
            dataset=dataset,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            lr_scheduler=args.lr_scheduler,
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
            fusion_mode=args.fusion_mode,
            use_channel_attn=not args.no_channel_attn,
            fs=fs,
            n_fft=args.n_fft,
            channel_dropout=args.channel_dropout,
            ssm_backbone=args.ssm_backbone,
            ssm_layers=args.ssm_layers,
            fas_k=args.fas_k,
            fas_channels=fas_channels,
            use_voltage=args.use_voltage,
            monitor=args.monitor,
            fbeta=args.fbeta
        )


if __name__ == '__main__':
    main()
