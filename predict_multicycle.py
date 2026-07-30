#!/usr/bin/env python3
"""
predict_multicycle.py — Multi-cycle decision + per-installation calibration.

NO TRAINING. This post-processes the per-cycle out-of-fold scores that a groupkfold
run already saved (`oof_predictions.npz`) and applies the two decision-layer steps
that actually improved cross-installation performance:

  1. aggregate the score over K consecutive cycles (an arc spans many cycles; an
     AFDD decides over several half-cycles — IEC 62606),
  2. set the threshold per installation from the *unlabelled* score histogram (Otsu),
     which removes the per-campaign score shift.

On the B1 reference run this takes 81.28 % acc / 82.30 % spec (per cycle, thr 0.5)
to 88.3 % / 91.8 % at K=6 — see docs/arcssm_groupkfold_generalization.md §6.1.

Usage:
    python predict_multicycle.py --run runs/arcssm_groupkfold_campaign_20260726_195946
    python predict_multicycle.py --run <dir> --K 4          # single operating point
    python predict_multicycle.py --run <dir> --calibrate none
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def otsu_threshold(p: np.ndarray, bins: int = 256) -> float:
    """Histogram-valley threshold from UNLABELLED scores (Otsu 1979).

    Deployable: it needs scores from the target installation, not labels.
    """
    h, edges = np.histogram(p, bins=bins, range=(0.0, 1.0))
    h = h.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h)
    w1 = h.sum() - w0
    cs = np.cumsum(h * centers)
    m0 = cs / np.maximum(w0, 1e-9)
    m1 = (cs[-1] - cs) / np.maximum(w1, 1e-9)
    return float(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def gmm_threshold(p: np.ndarray, seed: int = 0) -> float:
    """Threshold where two fitted Gaussians cross, in logit space. Also UNLABELLED.

    More robust than Otsu when the two score clusters have unequal spread or size —
    which is the case here, so this is the better default.
    """
    from sklearn.mixture import GaussianMixture
    q = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q)).reshape(-1, 1)                 # logit: makes them Gaussian-ish
    g = GaussianMixture(2, random_state=seed, n_init=5).fit(z)
    grid = np.linspace(z.min(), z.max(), 4000).reshape(-1, 1)
    lo, hi = np.argsort(g.means_.ravel())                  # low cluster = normal, high = arc
    d = g.predict_proba(grid)[:, hi] - g.predict_proba(grid)[:, lo]
    zt = grid[np.argmin(np.abs(d)), 0]                     # crossing point
    return float(1 / (1 + np.exp(-zt)))


def pick_threshold(p, mode):
    if mode == 'gmm':
        return gmm_threshold(p)
    if mode == 'otsu':
        return otsu_threshold(p)
    return 0.5


def aggregate(probs, labels, meta, K):
    """Mean score over K consecutive cycles within a recording (time-ordered).

    Windows never cross a recording boundary. A window is 'arc' if it contains at
    least one arc cycle.
    """
    idx = []
    for exp in meta['exp_name'].unique():
        sub = meta[meta['exp_name'] == exp].sort_values('start_sample')
        gi = sub.index.values
        for s in range(0, len(gi) - K + 1, K):
            idx.append(gi[s:s + K])
    idx = np.array(idx)
    return (probs[idx].mean(1),
            meta['label'].values[idx].max(1),
            meta['dataset'].values[idx[:, 0]])


def score(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    pr = tp / max(tp + fp, 1); re = tp / max(tp + fn, 1)
    return dict(acc=100 * (tp + tn) / max(len(y), 1), spec=100 * tn / max(tn + fp, 1),
                rec=100 * re, prec=100 * pr, f1=200 * pr * re / max(pr + re, 1e-9),
                fp=fp, fn=fn)


def evaluate(probs, labels, meta, K, calibrate):
    p, y, g = aggregate(probs, labels, meta, K)
    pred = np.zeros(len(p), dtype=int)
    per_campaign = []
    for c in sorted(np.unique(g)):
        m = g == c
        thr = pick_threshold(p[m], calibrate)
        pred[m] = (p[m] > thr).astype(int)
        s = score(y[m], pred[m])
        s.update(campaign=c, thr=thr, n=int(m.sum()),
                 auc=roc_auc_score(y[m], p[m]) if len(np.unique(y[m])) == 2 else float('nan'))
        per_campaign.append(s)
    pooled = score(y, pred)
    pooled['auc'] = roc_auc_score(y, p)
    pooled['n'] = len(y)
    return pooled, per_campaign


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='run dir containing oof_predictions.npz')
    ap.add_argument('--data-dir', default='combined_dataset_2048')
    ap.add_argument('--K', type=int, default=None,
                    help='cycles per decision. Omit to sweep K=1,2,3,4,6,8.')
    ap.add_argument('--calibrate', choices=['gmm', 'otsu', 'none'], default='gmm',
                    help="per-installation threshold from UNLABELLED scores: 'gmm' "
                         "(default, 2-Gaussian crossing — best), 'otsu' (histogram "
                         "valley), or 'none' for a fixed 0.5")
    args = ap.parse_args()

    npz = Path(args.run) / 'oof_predictions.npz'
    o = np.load(npz, allow_pickle=True)
    probs, labels = o['probs'].astype(float), o['labels'].astype(int)
    meta = pd.read_csv(Path(args.data_dir) / 'metadata.csv')
    if len(probs) != len(meta):
        raise SystemExit(
            f"{npz} holds {len(probs)} predictions but metadata has {len(meta)} cycles.\n"
            "This tool needs PER-CYCLE out-of-fold scores (a groupkfold run of train.py).")
    if np.isnan(probs).any():
        raise SystemExit("oof_predictions.npz contains NaN — the run did not cover every cycle.")

    print(f"Run: {args.run}\nCycles: {len(probs)}  |  calibration: {args.calibrate}")
    Ks = [args.K] if args.K else [1, 2, 3, 4, 6, 8]

    if args.K:
        pooled, per_c = evaluate(probs, labels, meta, args.K, args.calibrate)
        print(f"\n=== K={args.K} ({args.K * 20} ms @50Hz) ===")
        print(f"{'campaign':22s} | {'n':>5} | {'acc':>6} {'spec':>6} {'rec':>6} {'F1':>6} | "
              f"{'AUC':>5} | thr")
        for s in per_c:
            print(f"{s['campaign']:22s} | {s['n']:5d} | {s['acc']:5.1f}% {s['spec']:5.1f}% "
                  f"{s['rec']:5.1f}% {s['f1']:5.1f}% | {s['auc']:.3f} | {s['thr']:.2f}")
        print(f"{'POOLED':22s} | {pooled['n']:5d} | {pooled['acc']:5.1f}% {pooled['spec']:5.1f}% "
              f"{pooled['rec']:5.1f}% {pooled['f1']:5.1f}% | {pooled['auc']:.3f} |")
        print(f"    FP={pooled['fp']}  FN={pooled['fn']}")
        return

    print(f"\n{'K':>2} | {'decision':>9} | {'units':>6} | {'acc':>6} {'spec':>6} "
          f"{'rec':>6} {'F1':>6} | recall per campaign")
    print("-" * 104)
    for K in Ks:
        pooled, per_c = evaluate(probs, labels, meta, K, args.calibrate)
        recs = "  ".join(f"{s['rec']:5.1f}%" for s in per_c)
        print(f"{K:>2} | {K * 20:6d} ms | {pooled['n']:6d} | {pooled['acc']:5.1f}% "
              f"{pooled['spec']:5.1f}% {pooled['rec']:5.1f}% {pooled['f1']:5.1f}% | {recs}")
    print("-" * 104)
    print("campaigns, in order: " + ", ".join(s['campaign'] for s in per_c))
    print("K=1 is the per-cycle baseline. Larger K trades time resolution for accuracy;\n"
          "it also dilutes ISOLATED single-cycle arcs (see 8_juillet's recall).")


if __name__ == '__main__':
    main()
