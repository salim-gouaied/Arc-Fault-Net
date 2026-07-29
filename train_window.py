#!/usr/bin/env python3
"""
train_window.py — Multi-cycle ("window") ArcSSM for cross-campaign generalization.

Idea (see docs/arcssm_groupkfold_generalization.md): a single cycle's *style* is
bench-specific, which is why per-cycle detection over-detects on an unseen bench.
But the arc's HF content is measurably LESS repetitive cycle-to-cycle than a normal
load's, CONSISTENTLY across every campaign (a bench-invariant signal). So we classify
a WINDOW of K consecutive cycles and let the model use inter-cycle (non-)repetitivity.

Model:
    per-cycle S4D encoder (identical to B1's i_derived4 front-end) → K embeddings
    → aggregate over the K cycles as [mean ⊕ std]   (std = the non-repetitivity cue)
    → shallow classifier → window logit
The current branch (S4D) is unchanged; the only addition is the mean⊕std head, so the
SSM's memory requirement is unchanged (it still processes ONE cycle).

Protocol: leave-one-campaign-out over WINDOWS (windows never cross a recording, and
train/test are campaign-disjoint → leakage-free). Compared against B1 (per-cycle,
pooled acc 81.28 / F1 79.63 / spec 82.30) — note the decision unit differs (window).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from arc_ssm import S4Block
from dataset import ArcFaultDataset


# ─────────────────────────── model ───────────────────────────
class ArcSSMWindow(nn.Module):
    """Per-cycle S4D encoder (B1 front-end) + mean⊕std aggregation over K cycles."""

    def __init__(self, in_ch=4, d_model=128, d_state=64, n_layers=4, embed_kernel=7,
                 block_dropout=0.1, clf_hidden=64, dropout=0.3):
        super().__init__()
        self.encoder = nn.Conv1d(in_ch, d_model, embed_kernel, padding=embed_kernel // 2)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList(
            [S4Block(d_model, d_state, True, False, block_dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.to_embed = nn.Linear(d_model, 128)
        # head consumes [mean(128) ⊕ std(128)] = 256; std carries the non-repetitivity.
        self.classifier = nn.Sequential(
            nn.Linear(256, clf_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(clf_hidden, 1))

    def encode_cycle(self, x):                         # (N, 4, L) -> (N, 128)
        z = self.act(self.encoder(x)).transpose(1, 2)
        for b in self.blocks:
            z = b(z)
        return self.to_embed(self.norm(z).mean(dim=1))

    def forward(self, xw):                             # (B, K, 4, L) -> (B,)
        B, K, C, L = xw.shape
        e = self.encode_cycle(xw.reshape(B * K, C, L)).reshape(B, K, 128)
        agg = torch.cat([e.mean(dim=1), e.std(dim=1, unbiased=False)], dim=-1)  # (B,256)
        return self.classifier(agg).squeeze(-1)


# ─────────────────────────── data ───────────────────────────
class WindowDataset(Dataset):
    """Serves K-consecutive-cycle windows, deriving i_derived4 per cycle (B1 front-end)."""

    def __init__(self, base: ArcFaultDataset, windows):
        self.base = base
        self.windows = windows                          # list of (idx_tuple, label, campaign)
        self.y = np.array([w[1] for w in windows], dtype=np.int64)
        self.groups = np.array([w[2] for w in windows])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, w):
        idxs, lab, _ = self.windows[w]
        cyc = [self.base._derive_i_channels(
                   torch.from_numpy(self.base.X[gi, self.base.i_channel]).float())
               for gi in idxs]
        return torch.stack(cyc, 0), torch.tensor(lab, dtype=torch.float32)  # (K,4,L), ()


def build_windows(base, meta, K, stride):
    """Non-overlapping (stride=K) K-cycle windows within each recording, ordered by time."""
    windows = []
    for exp in meta['exp_name'].unique():
        sub = meta[meta['exp_name'] == exp].sort_values('start_sample')
        gi = sub.index.values                            # global cycle rows (aligned to base.X)
        lab = sub['label'].values
        camp = sub['dataset'].iloc[0]
        for s in range(0, len(gi) - K + 1, stride):
            w = gi[s:s + K]
            windows.append((tuple(int(x) for x in w), int(lab[s:s + K].max()), camp))
    return windows


# ─────────────────────────── train / eval ───────────────────────────
def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / max(len(y), 1); pr = tp / (tp + fp + 1e-9); re = tp / (tp + fn + 1e-9)
    sp = tn / (tn + fp + 1e-9); f1 = 2 * pr * re / (pr + re + 1e-9)
    return dict(accuracy=acc, f1=f1, precision=pr, recall=re, specificity=sp,
                tp=tp, fp=fp, fn=fn, tn=tn)


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ps, ys = [], []
    for xw, y in loader:
        p = torch.sigmoid(model(xw.to(device))).cpu().numpy()
        ps.append(p); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys).astype(int)


def train_fold(model, tr, va, device, epochs, lr, wd, patience, clip):
    crit = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    best_f1, best_ep, best_state, wait = -1.0, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, nseen = 0.0, 0
        for xw, y in tr:
            xw, y = xw.to(device), y.to(device)
            ys = y * 0.9 + 0.05                          # label smoothing (as B1)
            opt.zero_grad(); loss = crit(model(xw), ys); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
            tot += loss.item() * len(y); nseen += len(y)
        lr_now = opt.param_groups[0]['lr']
        sched.step(ep)
        p, yv = predict(model, va, device)
        m = metrics(yv, (p > 0.5).astype(int))
        improved = m['f1'] > best_f1
        if improved:
            best_f1, best_ep = m['f1'], ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        # every epoch: follow live (flush so it appears immediately, not buffered)
        print(f"  ep {ep:3d}/{epochs}: train_loss={tot/max(nseen,1):.4f}  "
              f"val_acc={100*m['accuracy']:.1f}%  val_f1={100*m['f1']:.1f}%  "
              f"val_spec={100*m['specificity']:.1f}%  lr={lr_now:.2e}"
              f"{'   *best' if improved else ''}", flush=True)
        # every 10 epochs: fuller recap
        if ep % 10 == 0:
            print(f"  ── after {ep} epochs ── val: P={100*m['precision']:.1f}%  "
                  f"R={100*m['recall']:.1f}%  Spec={100*m['specificity']:.1f}%  "
                  f"F1={100*m['f1']:.1f}%   |   best F1={100*best_f1:.1f}% @ ep{best_ep}", flush=True)
        if wait >= patience:
            print(f"  early stop ep {ep} (best ep {best_ep}, val_f1={100*best_f1:.1f}%)", flush=True); break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--data-dir', default='combined_dataset_2048')
    ap.add_argument('--output-dir', default='runs')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=16,
                    help='Windows per batch. NOTE: the S4D encodes batch*K cycles at '
                         'once, so effective S4D batch = batch*K. Keep batch*K ~32 to '
                         'match B1 memory (K=2 -> 16, K=4 -> 8) on a ~8 GB GPU.')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--gradient-clip', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-workers', type=int, default=4)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  window K={args.K}")
    dd = Path(args.data_dir)
    base = ArcFaultDataset(data_dir=str(dd), n_fft=512, hop_length=256,
                           channel_mode='i_derived4')
    meta = pd.read_csv(dd / 'metadata.csv')
    windows = build_windows(base, meta, args.K, stride=args.K)
    wds = WindowDataset(base, windows)
    print(f"Windows (K={args.K}, non-overlapping): {len(wds)}  "
          f"[{int((wds.y==0).sum())} normal / {int((wds.y==1).sum())} arc]")
    for c in sorted(np.unique(wds.groups)):
        m = wds.groups == c
        print(f"    {c:22s}: {m.sum():4d} windows  [{int((wds.y[m]==0).sum())} N / {int((wds.y[m]==1).sum())} A]")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.output_dir) / f"arcssm_window{args.K}_groupkfold_campaign_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = list(LeaveOneGroupOut().split(np.arange(len(wds)), wds.y, groups=wds.groups))
    oof_p = np.full(len(wds), np.nan); oof_y = wds.y.copy()
    fold_results = []
    t0 = time.time()
    for fi, (tr_idx, te_idx) in enumerate(splits):
        torch.manual_seed(args.seed + fi); np.random.seed(args.seed + fi)
        test_camp = sorted(set(wds.groups[te_idx]))
        # in-domain val: stratified 15% of train windows (early-stopping signal)
        skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=args.seed + fi)
        sub_tr, sub_va = next(skf.split(tr_idx, wds.y[tr_idx]))
        tr_i, va_i = tr_idx[sub_tr], tr_idx[sub_va]
        print(f"\n--- Fold {fi+1}/{len(splits)}  test={test_camp}  "
              f"(train {len(tr_i)} / val {len(va_i)} / test {len(te_idx)} windows) ---")
        dl = lambda idx, sh: DataLoader(Subset(wds, idx), batch_size=args.batch_size,
                                        shuffle=sh, num_workers=args.num_workers,
                                        pin_memory=True, drop_last=sh)
        model = ArcSSMWindow().to(device)
        if fi == 0:
            print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
        model, best_ep = train_fold(model, dl(tr_i, True), dl(va_i, False), device,
                                    args.epochs, args.lr, args.weight_decay,
                                    args.patience, args.gradient_clip)
        p, y = predict(model, dl(te_idx, False), device)
        oof_p[te_idx] = p
        m = metrics(y, (p > 0.5).astype(int))
        print(f"    → Acc={100*m['accuracy']:.2f}%  F1={100*m['f1']:.2f}%  "
              f"Prec={100*m['precision']:.2f}%  Rec={100*m['recall']:.2f}%  Spec={100*m['specificity']:.2f}%")
        fold_results.append(dict(fold=fi + 1, test_groups=test_camp, best_epoch=best_ep,
                                 n_test=len(te_idx), **{k: float(v) for k, v in m.items()}))

    pooled = metrics(oof_y, (oof_p > 0.5).astype(int))
    print("\n" + "=" * 60)
    print(f"WINDOW (K={args.K}) GROUP K-FOLD SUMMARY")
    print("=" * 60)
    print(f"  POOLED (window-level): Acc={100*pooled['accuracy']:.2f}%  F1={100*pooled['f1']:.2f}%  "
          f"Prec={100*pooled['precision']:.2f}%  Rec={100*pooled['recall']:.2f}%  Spec={100*pooled['specificity']:.2f}%")
    print(f"    FP={pooled['fp']}  FN={pooled['fn']}")
    print(f"  (B1 per-cycle ref: Acc 81.28 / F1 79.63 / Spec 82.30 — different decision unit)")
    summary = dict(model='arcssm_window', K=args.K, seed=args.seed, epochs=args.epochs,
                   n_windows=len(wds), pooled=pooled, fold_results=fold_results,
                   duration_seconds=time.time() - t0, timestamp=ts)
    with open(run_dir / 'window_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    np.savez(run_dir / 'oof_predictions.npz', probs=oof_p, labels=oof_y, groups=wds.groups)
    print(f"  Duration: {(time.time()-t0)/60:.1f} min  |  saved: {run_dir}")


if __name__ == '__main__':
    main()
