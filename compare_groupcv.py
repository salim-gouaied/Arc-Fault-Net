#!/usr/bin/env python3
"""
Compare group-CV runs against a baseline.
=========================================
Takes two or more run directories produced by `train.py --mode groupkfold` and
reports what changed: pooled metrics, per-campaign metrics, per-campaign AUC, and
a paired significance test.

Because every run under the same protocol predicts the *same* 10 860 cycles
out-of-fold, the comparison is paired: McNemar's test on the discordant pairs
answers "did this change really do something" far better than comparing two
accuracies that each carry their own sampling noise.

Usage:
  python compare_groupcv.py runs/<baseline_run> runs/<variant_run> [more...] \
      --labels baseline "no-STFT-leak" --out docs/baselines/comparison.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.metrics import roc_curve, auc

RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7",
              "#eda100", "#e87ba4", "#008300", "#e34948"]
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def load(run_dir: Path):
    z = np.load(run_dir / 'oof_predictions.npz', allow_pickle=False)
    summary_path = run_dir / 'groupkfold_summary.json'
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    keep = z['fold'] > 0
    return {
        'dir': run_dir,
        'probs': z['probs'][keep],
        'labels': z['labels'][keep].astype(int),
        'groups': z['groups'][keep],
        'summary': summary,
    }


def metrics(labels, probs, thr):
    pred = (probs > thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    try:
        a = auc(*roc_curve(labels, probs)[:2])
    except ValueError:
        a = float('nan')
    return {
        'n': len(labels), 'accuracy': (tp + tn) / len(labels),
        'precision': prec, 'recall': rec,
        'f1': 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        'specificity': tn / (tn + fp) if tn + fp else 0.0,
        'auc': a, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray):
    """Paired test on per-cycle correctness. Returns (b, c, chi2, p)."""
    b = int((correct_a & ~correct_b).sum())   # baseline right, variant wrong
    c = int((~correct_a & correct_b).sum())   # baseline wrong, variant right
    if b + c == 0:
        return b, c, 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)    # continuity-corrected
    return b, c, stat, float(chi2.sf(stat, 1))


def md_table(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def delta(v, base):
    d = 100 * (v - base)
    return f"{d:+.2f}"


def fig_compare(runs, labels, campaigns, thr, out_path):
    x = np.arange(len(campaigns) + 1)     # campaigns + pooled
    width = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(campaigns)), 4.4))
    for k, (r, name) in enumerate(zip(runs, labels)):
        vals = []
        for c in campaigns:
            s = r['groups'] == c
            vals.append(100 * metrics(r['labels'][s], r['probs'][s], thr)['f1'])
        vals.append(100 * metrics(r['labels'], r['probs'], thr)['f1'])
        bars = ax.bar(x + k * width, vals, width * 0.9,
                      color=RUN_COLORS[k % len(RUN_COLORS)], zorder=3, label=name)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.0f}",
                    ha='center', fontsize=7, color=INK_SOFT)
    ax.set_xticks(x + (len(runs) - 1) * width / 2)
    ax.set_xticklabels([c.replace('_clean', '') for c in campaigns] + ['POOLED'],
                       fontsize=8, color=INK)
    ax.set_ylabel("F1 %", color=INK_SOFT); ax.set_ylim(0, 108)
    ax.set_title("Out-of-fold F1 per held-out campaign", fontsize=11, color=INK)
    ax.grid(axis='y', color=GRID, lw=0.6, alpha=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_SOFT, length=0)
    ax.legend(fontsize=8, frameon=False, ncol=min(4, len(runs)), labelcolor=INK)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compare group-CV runs against a baseline")
    ap.add_argument('run_dirs', nargs='+', type=str,
                    help='first one is the baseline, the rest are variants')
    ap.add_argument('--labels', nargs='*', default=None,
                    help='display names, same order as run_dirs')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--out', type=str, default=None,
                    help='write a markdown report here (figure goes next to it)')
    args = ap.parse_args()

    runs = [load(Path(d)) for d in args.run_dirs]
    labels = args.labels or [Path(d).name for d in args.run_dirs]
    if len(labels) != len(runs):
        raise SystemExit("--labels must have one entry per run dir")
    thr = args.threshold

    base = runs[0]
    if not all(np.array_equal(r['labels'], base['labels']) for r in runs[1:]):
        raise SystemExit("runs do not cover the same cycles in the same order — "
                         "they must use the same dataset and protocol to be paired")

    campaigns = sorted(set(base['groups']))
    base_m = metrics(base['labels'], base['probs'], thr)

    lines = ["# Group-CV comparison", "",
             f"- Baseline: `{base['dir']}` ({labels[0]})",
             f"- Threshold: {thr}", f"- Cycles compared: {base_m['n']}", ""]

    # ── pooled ──
    rows = []
    for r, name in zip(runs, labels):
        m = metrics(r['labels'], r['probs'], thr)
        is_base = r is base
        rows.append([
            name,
            f"{100*m['accuracy']:.2f}" + ("" if is_base else f" ({delta(m['accuracy'], base_m['accuracy'])})"),
            f"{100*m['f1']:.2f}" + ("" if is_base else f" ({delta(m['f1'], base_m['f1'])})"),
            f"{100*m['recall']:.2f}" + ("" if is_base else f" ({delta(m['recall'], base_m['recall'])})"),
            f"{100*m['specificity']:.2f}" + ("" if is_base else f" ({delta(m['specificity'], base_m['specificity'])})"),
            f"{m['auc']:.4f}" + ("" if is_base else f" ({m['auc']-base_m['auc']:+.4f})"),
        ])
    lines += ["## Pooled out-of-fold", "",
              md_table(rows, ["Run", "Acc %", "F1 %", "Recall %", "Spec %", "AUC"]), ""]

    # ── paired significance ──
    if len(runs) > 1:
        base_ok = (base['probs'] > thr).astype(int) == base['labels']
        sig_rows = []
        for r, name in zip(runs[1:], labels[1:]):
            ok = (r['probs'] > thr).astype(int) == r['labels']
            b, c, stat, p = mcnemar(base_ok, ok)
            verdict = ("variant better" if c > b else "baseline better") if p < 0.05 else "no significant difference"
            sig_rows.append([name, b, c, f"{stat:.2f}", f"{p:.2e}", verdict])
        lines += ["## Paired test vs baseline (McNemar, same cycles)", "",
                  md_table(sig_rows, ["Run", "baseline✓ variant✗", "baseline✗ variant✓",
                                      "χ²", "p", "verdict"]),
                  "", "Both runs classify the same cycles, so only the discordant pairs "
                  "carry information; a change that moves the accuracy but has p > 0.05 "
                  "is inside run-to-run noise.", ""]

    # ── per campaign ──
    lines += ["## Per held-out campaign", ""]
    for c in campaigns:
        s_base = base['groups'] == c
        bm = metrics(base['labels'][s_base], base['probs'][s_base], thr)
        rows = []
        for r, name in zip(runs, labels):
            s = r['groups'] == c
            m = metrics(r['labels'][s], r['probs'][s], thr)
            is_base = r is base
            rows.append([
                name, int(m['n']),
                f"{100*m['accuracy']:.2f}" + ("" if is_base else f" ({delta(m['accuracy'], bm['accuracy'])})"),
                f"{100*m['f1']:.2f}" + ("" if is_base else f" ({delta(m['f1'], bm['f1'])})"),
                f"{100*m['recall']:.2f}" + ("" if is_base else f" ({delta(m['recall'], bm['recall'])})"),
                f"{100*m['specificity']:.2f}" + ("" if is_base else f" ({delta(m['specificity'], bm['specificity'])})"),
                f"{m['auc']:.4f}" + ("" if is_base else f" ({m['auc']-bm['auc']:+.4f})"),
            ])
        lines += [f"### {c}", "",
                  md_table(rows, ["Run", "n", "Acc %", "F1 %", "Recall %", "Spec %", "AUC"]), ""]

    # ── worst-campaign summary: the number that matters for deployment ──
    rows = []
    for r, name in zip(runs, labels):
        f1s = [metrics(r['labels'][r['groups'] == c], r['probs'][r['groups'] == c], thr)['f1']
               for c in campaigns]
        aucs = [metrics(r['labels'][r['groups'] == c], r['probs'][r['groups'] == c], thr)['auc']
                for c in campaigns]
        rows.append([name, f"{100*min(f1s):.2f}", f"{100*np.mean(f1s):.2f}",
                     f"{100*np.std(f1s):.2f}", f"{min(aucs):.4f}", f"{np.mean(aucs):.4f}"])
    lines += ["## Worst-campaign summary", "",
              md_table(rows, ["Run", "worst F1 %", "mean F1 %", "std F1 %",
                              "worst AUC", "mean AUC"]),
              "", "A change that lifts the mean but not the worst campaign has not "
              "improved generalization — it has improved the campaigns that already worked.", ""]

    report = "\n".join(lines) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig_name = out_path.stem + '_f1.png'
        fig_compare(runs, labels, campaigns, thr, out_path.parent / fig_name)
        out_path.write_text(report + f"\n![comparison]({fig_name})\n")
        print(f"Report written to {out_path}")
    print(report)


if __name__ == '__main__':
    main()
