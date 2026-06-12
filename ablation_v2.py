#!/usr/bin/env python3
"""
ARC-FAULTNET V2 — Critical Ablation Study
==========================================
Single-mode (random split) ablation comparing:
  1. arcfaultnet_v2     : Full model (temporal + spectral + cross-attention)
  2. v2_no_attention    : Dual-branch, simple concat (no attention)
  3. v2_no_chan_gate     : Dual-branch, MLP fusion (no channel gating)
  4. v2_temporal_only   : Temporal branch only (no STFT)
  5. v2_spectral_only   : Spectral branch only (no Conv1d)
  6. v2_baseline_cnn    : Plain CNN baseline (no dual-branch, no attention)

Generates: confusion matrices, ROC curves, comparison bar charts,
radar plot, component contribution chart, and JSON summary.
"""

import torch, torch.nn as nn, numpy as np, json, argparse, random, time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import warnings; warnings.filterwarnings('ignore')

from dataset import ArcFaultDataset
from model import get_model

# ═══════════════════════════════════════════════════════
#  VARIANTS
# ═══════════════════════════════════════════════════════

VARIANTS = [
    {'name': 'arcfaultnet_v2',    'label': 'Full V2',
     'desc': 'Temporal + Spectral + CrossAttention (reference)', 'color': '#2ecc71'},
    {'name': 'v2_no_attention',   'label': 'Sans Attention',
     'desc': 'Dual-branch, concat simple sans attention',        'color': '#3498db'},
    {'name': 'v2_no_chan_gate',   'label': 'Sans Channel Gate',
     'desc': 'Dual-branch, MLP fusion sans gating sigmoid',      'color': '#9b59b6'},
    {'name': 'v2_temporal_only',  'label': 'Temporel seul',
     'desc': 'Branche temporelle uniquement (pas de STFT)',       'color': '#e67e22'},
    {'name': 'v2_spectral_only',  'label': 'Spectral seul',
     'desc': 'Branche spectrale uniquement (pas de Conv1d)',      'color': '#e74c3c'},
    {'name': 'v2_baseline_cnn',   'label': 'CNN Classique',
     'desc': 'CNN simple sans dual-branch ni attention',          'color': '#95a5a6'},
]

# ═══════════════════════════════════════════════════════
#  SEED
# ═══════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ═══════════════════════════════════════════════════════
#  TRAIN / EVAL (self-contained, no import from train.py)
# ═══════════════════════════════════════════════════════

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
        preds = (probs > 0.5).float()
        total_loss += loss.item() * len(lab)
        correct += (preds == lab).sum().item()
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
        'probs': p, 'labels': l,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }

def train_variant(name, dataset, train_idx, val_idx, test_idx, device,
                  epochs=200, lr=3e-4, wd=5e-4, bs=64, patience=15,
                  grad_clip=0.5, num_workers=4):
    """Train a single variant and return test metrics + model."""
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=bs,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=bs,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=bs,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    model = get_model(name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_f1, best_epoch, wait = -1., 0, 0
    best_sd = None

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step(ep)
        if val_m['f1'] > best_f1:
            best_f1, best_epoch, wait = val_m['f1'], ep, 0
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 20 == 0 or ep == 1:
            print(f"    Ep {ep:3d}: trL={tr_loss:.4f} vAcc={100*val_m['accuracy']:.1f}% vF1={100*val_m['f1']:.1f}%")
        if wait >= patience:
            print(f"    Early stop ep {ep} (best={best_epoch}, F1={100*best_f1:.1f}%)")
            break

    if best_sd:
        model.load_state_dict(best_sd)
    test_m = evaluate(model, test_loader, criterion, device)
    test_m['n_params'] = n_params
    test_m['best_epoch'] = best_epoch
    return model, test_m

# ═══════════════════════════════════════════════════════
#  VISUAL EVALUATION PLOTS
# ═══════════════════════════════════════════════════════

def plot_confusion_matrix(labels, probs, variant, out_dir):
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Normal','Arc']); ax.set_yticklabels(['Normal','Arc'])
    ax.set_xlabel('Prédit'); ax.set_ylabel('Vrai')
    ax.set_title(f"Matrice de Confusion — {variant['label']}", fontsize=11)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > thresh else 'black', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / f"cm_{variant['name']}.png", dpi=150)
    plt.close(fig)

def plot_roc_curve(labels, probs, variant, out_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, color=variant['color'], label=f"AUC = {roc_auc:.4f}")
    ax.plot([0,1],[0,1],'k--', lw=0.8)
    ax.set_xlabel('Taux de Faux Positifs'); ax.set_ylabel('Taux de Vrais Positifs')
    ax.set_title(f"Courbe ROC — {variant['label']}", fontsize=11)
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"roc_{variant['name']}.png", dpi=150)
    plt.close(fig)
    return roc_auc

def plot_all_roc_overlay(all_results, out_dir):
    """Overlay all ROC curves on one figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for v in VARIANTS:
        r = all_results[v['name']]
        fpr, tpr, _ = roc_curve(r['labels'], r['probs'])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=v['color'], label=f"{v['label']} (AUC={a:.3f})")
    ax.plot([0,1],[0,1],'k--', lw=0.8)
    ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs', fontsize=12)
    ax.set_title('Comparaison ROC — Étude d\'Ablation V2', fontsize=13)
    ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'roc_overlay.png', dpi=150)
    plt.close(fig)

def plot_comparison_bars(all_results, out_dir):
    """Grouped bar chart: Accuracy + F1 for all variants."""
    names  = [v['label'] for v in VARIANTS]
    accs   = [all_results[v['name']]['accuracy'] * 100 for v in VARIANTS]
    f1s    = [all_results[v['name']]['f1'] * 100 for v in VARIANTS]
    colors = [v['color'] for v in VARIANTS]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, accs, w, color=colors, edgecolor='black', linewidth=0.5, label='Accuracy')
    b2 = ax.bar(x + w/2, f1s,  w, color=colors, edgecolor='black', linewidth=0.5, alpha=0.65, label='F1-Score')
    # Add hatching to F1 bars
    for bar in b2:
        bar.set_hatch('//')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Étude d\'Ablation — Comparaison des Variantes V2', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax.set_ylim([max(0, min(min(accs), min(f1s)) - 10), 102])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(b1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(b2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / 'comparison_bars.png', dpi=150)
    plt.close(fig)

def plot_radar(all_results, out_dir):
    """Radar / spider plot comparing all variants on 5 metrics."""
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'specificity']
    labels_fr = ['Accuracy', 'F1-Score', 'Précision', 'Rappel', 'Spécificité']

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for v in VARIANTS:
        r = all_results[v['name']]
        vals = [r[m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, color=v['color'], label=v['label'])
        ax.fill(angles, vals, alpha=0.08, color=v['color'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_fr, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title('Profil Multi-Métrique — Ablation V2', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / 'radar_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_contributions(all_results, out_dir):
    """Horizontal bar chart showing the Δ each component contributes."""
    ref_acc = all_results['arcfaultnet_v2']['accuracy']
    contributions = [
        ('Cross-Attention\n(vs concat simple)',
         ref_acc - all_results['v2_no_attention']['accuracy']),
        ('Channel Gating\n(vs MLP fusion)',
         ref_acc - all_results['v2_no_chan_gate']['accuracy']),
        ('Branche Spectrale\n(vs temporel seul)',
         ref_acc - all_results['v2_temporal_only']['accuracy']),
        ('Branche Temporelle\n(vs spectral seul)',
         ref_acc - all_results['v2_spectral_only']['accuracy']),
        ('Architecture Complète\n(vs CNN classique)',
         ref_acc - all_results['v2_baseline_cnn']['accuracy']),
    ]

    names  = [c[0] for c in contributions]
    deltas = [c[1] * 100 for c in contributions]
    colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), deltas, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Contribution à l\'Accuracy (%)', fontsize=12)
    ax.set_title('Apport de Chaque Composant — Ablation V2', fontsize=14)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    for bar, d in zip(bars, deltas):
        x = bar.get_width() + 0.1 if d >= 0 else bar.get_width() - 0.1
        ha = 'left' if d >= 0 else 'right'
        ax.text(x, bar.get_y() + bar.get_height()/2, f'{d:+.2f}%',
                ha=ha, va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'component_contributions.png', dpi=150)
    plt.close(fig)

def plot_params_vs_accuracy(all_results, out_dir):
    """Scatter: param count vs accuracy for each variant."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in VARIANTS:
        r = all_results[v['name']]
        ax.scatter(r['n_params'] / 1000, r['accuracy'] * 100,
                   s=150, color=v['color'], edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(v['label'], (r['n_params']/1000, r['accuracy']*100),
                    textcoords='offset points', xytext=(8, 5), fontsize=9)
    ax.set_xlabel('Paramètres (×1000)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Complexité vs Performance', fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'params_vs_accuracy.png', dpi=150)
    plt.close(fig)

# ═══════════════════════════════════════════════════════
#  SUMMARY TABLE (console)
# ═══════════════════════════════════════════════════════

def print_summary(all_results):
    ref = all_results['arcfaultnet_v2']['accuracy']
    print(f"\n{'='*80}")
    print(f"  RÉSULTATS DE L'ÉTUDE D'ABLATION — ARC-FAULTNET V2")
    print(f"{'='*80}")
    print(f"\n  {'Variante':<22} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Spec':>8} {'Params':>10} {'Δ Acc':>8}")
    print(f"  {'─'*76}")
    for v in VARIANTS:
        r = all_results[v['name']]
        delta = (r['accuracy'] - ref) * 100
        tag = '(ref)' if v['name'] == 'arcfaultnet_v2' else f'{delta:+.2f}%'
        print(f"  {v['label']:<22} {100*r['accuracy']:>7.2f}% {100*r['f1']:>7.2f}% "
              f"{100*r['precision']:>7.2f}% {100*r['recall']:>7.2f}% "
              f"{100*r['specificity']:>7.2f}% {r['n_params']:>10,} {tag:>8}")
    print(f"  {'─'*76}\n")

# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Arc-FaultNet V2 Ablation Study (single mode)')
    parser.add_argument('--data-dir', type=str,
                        default='/home/manip/pfe_salim_gouaied/Arc-Fault-Net/combined_dataset_2048')
    parser.add_argument('--output-dir', type=str,
                        default='/home/manip/pfe_salim_gouaied/Arc-Fault-Net/ablation_results')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--gradient-clip', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--n-fft', type=int, default=128)
    parser.add_argument('--hop-length', type=int, default=64)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Run only specific variants (e.g. arcfaultnet_v2 v2_baseline_cnn)')
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

    # ── Single random split 70/15/15 ──
    indices = np.random.permutation(len(dataset))
    n_train = int(len(dataset) * 0.70)
    n_val   = int(len(dataset) * 0.15)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"\nSplit: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # ── Output directory ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) / f"ablation_v2_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Filter variants if requested ──
    variants_to_run = VARIANTS
    if args.variants:
        variants_to_run = [v for v in VARIANTS if v['name'] in args.variants]

    # ── Train all variants ──
    all_results = {}
    t0 = time.time()

    for v in variants_to_run:
        print(f"\n{'─'*60}")
        print(f"  Variante: {v['label']}  ({v['name']})")
        print(f"  {v['desc']}")
        print(f"{'─'*60}")

        set_seed(args.seed)  # Same init for fair comparison

        model, metrics = train_variant(
            name=v['name'], dataset=dataset,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            device=device, epochs=args.epochs, lr=args.lr,
            wd=args.weight_decay, bs=args.batch_size,
            patience=args.patience, grad_clip=args.gradient_clip,
            num_workers=args.num_workers,
        )

        all_results[v['name']] = metrics
        print(f"  → Acc={100*metrics['accuracy']:.2f}%  F1={100*metrics['f1']:.2f}%  "
              f"Prec={100*metrics['precision']:.2f}%  Rec={100*metrics['recall']:.2f}%  "
              f"Params={metrics['n_params']:,}")

        # Per-variant visuals
        plot_confusion_matrix(metrics['labels'], metrics['probs'], v, out_dir)
        plot_roc_curve(metrics['labels'], metrics['probs'], v, out_dir)

    duration = time.time() - t0

    # ── Summary ──
    if len(all_results) == len(VARIANTS):
        print_summary(all_results)
        plot_all_roc_overlay(all_results, out_dir)
        plot_comparison_bars(all_results, out_dir)
        plot_radar(all_results, out_dir)
        plot_contributions(all_results, out_dir)
        plot_params_vs_accuracy(all_results, out_dir)

    # ── Save JSON ──
    save_data = {}
    for k, v in all_results.items():
        save_data[k] = {kk: vv for kk, vv in v.items() if kk not in ('probs', 'labels')}
    summary = {
        'timestamp': timestamp, 'seed': args.seed,
        'split': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)},
        'epochs': args.epochs, 'lr': args.lr, 'weight_decay': args.weight_decay,
        'batch_size': args.batch_size, 'patience': args.patience,
        'duration_seconds': duration,
        'variants': save_data,
    }
    with open(out_dir / 'ablation_v2_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDurée totale: {duration/60:.1f} min")
    print(f"Résultats sauvegardés: {out_dir}")


if __name__ == '__main__':
    main()
