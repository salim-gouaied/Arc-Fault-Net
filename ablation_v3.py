#!/usr/bin/env python3
"""
ARC-FAULTNET V2 — Component-Level Ablation Study (V3)
=====================================================
Removes ONE component at a time from the full model to measure its contribution.

Full reference: ArcFaultNetV2(cross_attention + SE + deep_classifier + freq_gate + dual-branch + 4 derived channels)

Variants (each removes exactly one component):
  1. full_model          : Reference (all components)
  2. wo_cross_attention  : Replace cross-attention with simple concat
  3. wo_se_blocks        : Remove Squeeze-and-Excitation blocks
  4. wo_deep_classifier  : Use shallow classifier head
  5. wo_freq_gate        : Remove learnable frequency gate
  6. wo_spectral_branch  : Temporal branch only (no STFT)
  7. wo_temporal_branch  : Spectral branch only (no Conv1d)
  8. wo_derived_channels : Use raw I(t) only (1ch) instead of 4 derived
  9. baseline_cnn        : Plain CNN (no dual-branch, no attention)
"""

import torch, torch.nn as nn, numpy as np, json, argparse, random, time
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings; warnings.filterwarnings('ignore')

from dataset import ArcFaultDataset
from model import (
    ArcFaultNetV2, ArcFaultNetV2_TemporalOnly, ArcFaultNetV2_SpectralOnly,
    ArcFaultNetV2_BaselineCNN
)

# ═══════════════════════════════════════════════════════
#  VARIANT DEFINITIONS
# ═══════════════════════════════════════════════════════

VARIANTS = [
    {
        'key': 'full_model',
        'label': 'Full Model',
        'desc': 'All components (reference)',
        'color': '#2ecc71',
        'build': lambda: ArcFaultNetV2(
            in_channels=4, spec_in_channels=1,
            fusion_mode='cross_attention', use_se=True,
            deep_classifier=True, use_freq_gate=True),
    },
    {
        'key': 'wo_cross_attention',
        'label': 'w/o Cross-Attention',
        'desc': 'Simple concat fusion instead of Q/K/V cross-attention',
        'color': '#3498db',
        'build': lambda: ArcFaultNetV2(
            in_channels=4, spec_in_channels=1,
            fusion_mode='concat', use_se=True,
            deep_classifier=True, use_freq_gate=True),
    },
    {
        'key': 'wo_se_blocks',
        'label': 'w/o SE Blocks',
        'desc': 'No Squeeze-and-Excitation channel recalibration',
        'color': '#9b59b6',
        'build': lambda: ArcFaultNetV2(
            in_channels=4, spec_in_channels=1,
            fusion_mode='cross_attention', use_se=False,
            deep_classifier=True, use_freq_gate=True),
    },
    {
        'key': 'wo_deep_classifier',
        'label': 'w/o Deep Classifier',
        'desc': 'Shallow 2-layer head instead of deep 3-layer + BN',
        'color': '#e67e22',
        'build': lambda: ArcFaultNetV2(
            in_channels=4, spec_in_channels=1,
            fusion_mode='cross_attention', use_se=True,
            deep_classifier=False, use_freq_gate=True),
    },
    {
        'key': 'wo_freq_gate',
        'label': 'w/o Frequency Gate',
        'desc': 'No learnable frequency attention in spectral branch',
        'color': '#1abc9c',
        'build': lambda: ArcFaultNetV2(
            in_channels=4, spec_in_channels=1,
            fusion_mode='cross_attention', use_se=True,
            deep_classifier=True, use_freq_gate=False),
    },
    {
        'key': 'wo_spectral_branch',
        'label': 'w/o Spectral Branch',
        'desc': 'Temporal branch only — no STFT',
        'color': '#e74c3c',
        'build': lambda: ArcFaultNetV2_TemporalOnly(
            in_channels=4, use_se=True, deep_classifier=True),
    },
    {
        'key': 'wo_temporal_branch',
        'label': 'w/o Temporal Branch',
        'desc': 'Spectral branch only — no Conv1d',
        'color': '#f39c12',
        'build': lambda: ArcFaultNetV2_SpectralOnly(
            spec_in_channels=1, use_se=True,
            deep_classifier=True, use_freq_gate=True),
    },
    {
        'key': 'wo_derived_channels',
        'label': 'w/o Derived Channels',
        'desc': 'Raw I(t) only (1ch) instead of [I, |dI|, TKEO, RMS]',
        'color': '#8e44ad',
        'build': lambda: ArcFaultNetV2(
            in_channels=1, spec_in_channels=1,
            fusion_mode='cross_attention', use_se=True,
            deep_classifier=True, use_freq_gate=True),
    },
    {
        'key': 'baseline_cnn',
        'label': 'Baseline CNN',
        'desc': 'Plain CNN — no dual-branch, no attention',
        'color': '#95a5a6',
        'build': lambda: ArcFaultNetV2_BaselineCNN(in_channels=4),
    },
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
#  TRAIN / EVAL
# ═══════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, variant, grad_clip=0.5):
    model.train()
    if hasattr(loader.dataset, 'dataset'):
        loader.dataset.dataset.training = True
    total_loss, correct, total = 0., 0, 0
    for x1, x2, lab, _ in loader:
        if variant['key'] == 'wo_derived_channels':
            x1 = x1[:, 0:1, :]  # keep only raw I(t)
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
def evaluate(model, loader, criterion, device, variant):
    model.eval()
    if hasattr(loader.dataset, 'dataset'):
        loader.dataset.dataset.training = False
    total_loss, correct, total = 0., 0, 0
    all_probs, all_labels = [], []
    for x1, x2, lab, _ in loader:
        if variant['key'] == 'wo_derived_channels':
            x1 = x1[:, 0:1, :]  # keep only raw I(t)
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
        'probs': p, 'labels': l,
    }


def train_variant(variant, dataset, train_idx, val_idx, test_idx, device,
                  epochs=200, lr=3e-4, wd=5e-4, bs=64, patience=15,
                  grad_clip=0.5, num_workers=4, output_dir=None):
    """
    Train one model variant and return (model, metrics).

    Args:
        output_dir: Optional Path. If provided, saves:
                    - best_model.pt  (best checkpoint by val F1)
                    - metrics.json   (test metrics + config)
    """
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=bs,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=bs,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(Subset(dataset, test_idx), batch_size=bs,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    model = variant['build']().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_f1, best_epoch, wait, best_sd = -1., 0, 0, None

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, variant, grad_clip)
        val_m = evaluate(model, val_loader, criterion, device, variant)
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
    test_m = evaluate(model, test_loader, criterion, device, variant)
    test_m['n_params'] = n_params
    test_m['best_epoch'] = best_epoch

    # ── Persist checkpoint + metrics if output_dir is given ──
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / 'best_model.pt')
        save_m = {k: v for k, v in test_m.items() if k not in ('probs', 'labels')}
        save_m.update({'key': variant['key'], 'label': variant['label'],
                       'desc': variant['desc'], 'best_epoch': best_epoch})
        with open(out / 'metrics.json', 'w') as f:
            import json as _json
            _json.dump(save_m, f, indent=2)
        print(f"    Saved: {out / 'best_model.pt'}")

    return model, test_m

# ═══════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════

def plot_comparison_bars(results, variants_run, out_dir):
    names  = [v['label'] for v in variants_run]
    accs   = [results[v['key']]['accuracy'] * 100 for v in variants_run]
    f1s    = [results[v['key']]['f1'] * 100 for v in variants_run]
    colors = [v['color'] for v in variants_run]

    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w/2, accs, w, color=colors, edgecolor='black', lw=0.5, label='Accuracy')
    b2 = ax.bar(x + w/2, f1s,  w, color=colors, edgecolor='black', lw=0.5, alpha=0.65, label='F1-Score')
    for bar in b2: bar.set_hatch('//')
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Ablation Study — Component Contributions (Arc-FaultNet V2)', fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax.set_ylim([max(0, min(min(accs), min(f1s)) - 10), 102])
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(b1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar, val in zip(b2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / 'comparison_bars.png', dpi=150); plt.close(fig)


def plot_contributions(results, variants_run, out_dir):
    ref_acc = results['full_model']['accuracy']
    items = [(v['label'], (results[v['key']]['accuracy'] - ref_acc) * 100)
             for v in variants_run if v['key'] != 'full_model']
    items.sort(key=lambda x: x[1])
    names = [i[0] for i in items]; deltas = [i[1] for i in items]
    colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), deltas, color=colors, edgecolor='black', lw=0.5)
    ax.set_xlabel('Δ Accuracy vs Full Model (%)', fontsize=12)
    ax.set_title('Impact of Removing Each Component', fontsize=14)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10)
    ax.axvline(x=0, color='black', lw=0.8); ax.grid(axis='x', alpha=0.3)
    for bar, d in zip(bars, deltas):
        x = bar.get_width() + 0.1 if d >= 0 else bar.get_width() - 0.1
        ha = 'left' if d >= 0 else 'right'
        ax.text(x, bar.get_y() + bar.get_height()/2, f'{d:+.2f}%',
                ha=ha, va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'component_contributions.png', dpi=150); plt.close(fig)


def plot_radar(results, variants_run, out_dir):
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'specificity']
    labels_fr = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Specificity']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for v in variants_run:
        r = results[v['key']]
        vals = [r[m] for m in metrics] + [r[metrics[0]]]
        ax.plot(angles, vals, 'o-', lw=2, color=v['color'], label=v['label'])
        ax.fill(angles, vals, alpha=0.05, color=v['color'])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_fr, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title('Multi-Metric Radar — Ablation V3', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / 'radar_plot.png', dpi=150, bbox_inches='tight'); plt.close(fig)


def plot_roc_overlay(results, variants_run, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for v in variants_run:
        r = results[v['key']]
        fpr, tpr, _ = roc_curve(r['labels'], r['probs'])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=v['color'], label=f"{v['label']} (AUC={a:.3f})")
    ax.plot([0,1],[0,1],'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Ablation Study V3', fontsize=13)
    ax.legend(loc='lower right', fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'roc_overlay.png', dpi=150); plt.close(fig)


def plot_params_vs_perf(results, variants_run, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in variants_run:
        r = results[v['key']]
        ax.scatter(r['n_params']/1000, r['accuracy']*100,
                   s=150, color=v['color'], edgecolors='black', lw=0.8, zorder=5)
        ax.annotate(v['label'], (r['n_params']/1000, r['accuracy']*100),
                    textcoords='offset points', xytext=(8, 5), fontsize=8)
    ax.set_xlabel('Parameters (×1000)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Complexity vs Performance', fontsize=13); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'params_vs_accuracy.png', dpi=150); plt.close(fig)

# ═══════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════

def print_summary(results, variants_run):
    ref = results['full_model']['accuracy']
    print(f"\n{'='*95}")
    print(f"  ABLATION STUDY RESULTS — ARC-FAULTNET V2 (Component-Level)")
    print(f"{'='*95}")
    print(f"\n  {'Variant':<24} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Spec':>8} {'Params':>10} {'Δ Acc':>8}")
    print(f"  {'─'*90}")
    for v in variants_run:
        r = results[v['key']]
        delta = (r['accuracy'] - ref) * 100
        tag = '(ref)' if v['key'] == 'full_model' else f'{delta:+.2f}%'
        print(f"  {v['label']:<24} {100*r['accuracy']:>7.2f}% {100*r['f1']:>7.2f}% "
              f"{100*r['precision']:>7.2f}% {100*r['recall']:>7.2f}% "
              f"{100*r['specificity']:>7.2f}% {r['n_params']:>10,} {tag:>8}")
    print(f"  {'─'*90}\n")

# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Arc-FaultNet V2 Component Ablation (V3)')
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
                        help='Run specific variants by key (e.g. full_model wo_se_blocks)')
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
    out_dir = Path(args.output_dir) / f"ablation_v3_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Filter variants ──
    variants_to_run = VARIANTS
    if args.variants:
        variants_to_run = [v for v in VARIANTS if v['key'] in args.variants]

    # ── Train all variants ──
    all_results = {}
    t0 = time.time()

    for v in variants_to_run:
        print(f"\n{'─'*60}")
        print(f"  Variant: {v['label']}  ({v['key']})")
        print(f"  {v['desc']}")
        print(f"{'─'*60}")

        set_seed(args.seed)

        variant_dir = out_dir / v['key']
        model, metrics = train_variant(
            variant=v, dataset=dataset,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            device=device, epochs=args.epochs, lr=args.lr,
            wd=args.weight_decay, bs=args.batch_size,
            patience=args.patience, grad_clip=args.gradient_clip,
            num_workers=args.num_workers,
            output_dir=variant_dir,
        )

        all_results[v['key']] = metrics
        print(f"  → Acc={100*metrics['accuracy']:.2f}%  F1={100*metrics['f1']:.2f}%  "
              f"Prec={100*metrics['precision']:.2f}%  Rec={100*metrics['recall']:.2f}%  "
              f"Params={metrics['n_params']:,}")

    duration = time.time() - t0

    # ── Plots ──
    if len(all_results) >= 2:
        print_summary(all_results, variants_to_run)
        plot_comparison_bars(all_results, variants_to_run, out_dir)
        plot_roc_overlay(all_results, variants_to_run, out_dir)
        plot_radar(all_results, variants_to_run, out_dir)
        plot_params_vs_perf(all_results, variants_to_run, out_dir)
        if 'full_model' in all_results:
            plot_contributions(all_results, variants_to_run, out_dir)

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
    with open(out_dir / 'ablation_v3_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal duration: {duration/60:.1f} min")
    print(f"Results saved to: {out_dir}")


if __name__ == '__main__':
    main()
