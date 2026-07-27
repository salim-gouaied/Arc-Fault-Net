#!/usr/bin/env python3
"""
Evaluation report for a group cross-validation run (train.py --mode groupkfold).
===============================================================================
Consumes a run directory produced by run_groupkfold_cv and writes, into
<run_dir>/eval/:

  results.md              per-fold + pooled tables, ready to paste into the report
  fold_metrics.csv        same numbers, machine-readable
  confusion_pooled.png    pooled out-of-fold confusion matrix (the headline figure)
  confusion_per_fold.png  one matrix per held-out group
  roc_folds.png           per-fold ROC + pooled ROC with AUC
  metrics_per_fold.png    accuracy / F1 / precision / recall / specificity per fold
  threshold_sweep.png     pooled recall vs false-alarm rate against the threshold
  false_positives.csv     pooled FP forensics joined with metadata
  false_negatives.csv     pooled FN forensics joined with metadata

Why pooled matters: with leave-one-group-out the folds partition the dataset, so
every cycle is predicted exactly once by a model that never saw its group. Pooling
those predictions gives one confusion matrix over the whole dataset, unlike a
mean-over-folds which weights a 60-cycle fold like a 3820-cycle one.

Usage:
  python eval_groupcv.py runs/arcssm_groupkfold_campaign_YYYYMMDD_HHMMSS \
      --data-dir combined_dataset_2048/combined_dataset_2048
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ── Palette (validated: light mode, all-pairs CVD ΔE 9.2, normal-vision 16.3) ──
FOLD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7",
               "#eda100", "#e87ba4", "#008300", "#e34948"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"
CM_CMAP = "Blues"

METRICS = ["accuracy", "f1", "precision", "recall", "specificity"]


# ═══════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════

def confusion(labels: np.ndarray, probs: np.ndarray, threshold: float):
    pred = (probs > threshold).astype(int)
    l = labels.astype(int)
    return {
        'tp': int(((pred == 1) & (l == 1)).sum()),
        'fp': int(((pred == 1) & (l == 0)).sum()),
        'fn': int(((pred == 0) & (l == 1)).sum()),
        'tn': int(((pred == 0) & (l == 0)).sum()),
    }


def metrics_from_confusion(c: dict) -> dict:
    tp, fp, fn, tn = c['tp'], c['fp'], c['fn'], c['tn']
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        'n': n,
        'accuracy': (tp + tn) / n if n else 0.0,
        'precision': prec,
        'recall': rec,
        'f1': 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        'specificity': tn / (tn + fp) if tn + fp else 0.0,
        'fpr': fp / (tn + fp) if tn + fp else 0.0,
        **c,
    }


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion — honest CI at these sample sizes."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ═══════════════════════════════════════════════════════
#  LOADING
# ═══════════════════════════════════════════════════════

def load_run(run_dir: Path):
    """Return (summary, oof DataFrame with probs/labels/fold/group)."""
    summary_path = run_dir / 'groupkfold_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found — is this a groupkfold run?")
    summary = json.load(open(summary_path))

    oof_path = run_dir / 'oof_predictions.npz'
    if oof_path.exists():
        z = np.load(oof_path, allow_pickle=False)
        df = pd.DataFrame({
            'idx': np.arange(len(z['probs'])),
            'prob': z['probs'], 'label': z['labels'],
            'fold': z['fold'], 'group': z['groups'],
        })
        df = df[df.fold > 0].reset_index(drop=True)
        return summary, df

    # Fallback: stitch the per-fold prediction files (older runs, or a partial run)
    rows = []
    for fold in sorted(run_dir.glob('fold_*/test_predictions.npz')):
        z = np.load(fold, allow_pickle=False)
        fold_no = int(str(fold.parent.name).split('_')[1])
        rows.append(pd.DataFrame({
            'idx': z['idx'], 'prob': z['probs'], 'label': z['labels'],
            'fold': fold_no, 'group': z['groups'],
        }))
    if not rows:
        raise FileNotFoundError(
            f"No oof_predictions.npz and no fold_*/test_predictions.npz in {run_dir}. "
            f"Re-run training with the current train.py to save raw probabilities.")
    return summary, pd.concat(rows, ignore_index=True)


def fold_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-fold metrics recomputed from raw probabilities."""
    rows = []
    for fold, g in df.groupby('fold'):
        m = metrics_from_confusion(confusion(g.label.values, g.prob.values, threshold))
        try:
            m['auc'] = auc(*roc_curve(g.label.values, g.prob.values)[:2])
        except ValueError:
            m['auc'] = float('nan')
        held_out = sorted(set(g.group))
        m['fold'] = int(fold)
        m['held_out'] = held_out[0] if len(held_out) == 1 else f"{len(held_out)} groups"
        m['held_out_full'] = ', '.join(held_out)
        rows.append(m)
    return pd.DataFrame(rows).set_index('fold').sort_index()


# ═══════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════

def _cm_axes(ax, c: dict, title: str, subtitle: str = "", fontsize: int = 15):
    cm = np.array([[c['tn'], c['fp']], [c['fn'], c['tp']]], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    ax.imshow(row_pct, cmap=CM_CMAP, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            dark = row_pct[i, j] > 0.5
            ax.text(j, i, f"{int(cm[i, j])}", ha='center', va='center',
                    fontsize=fontsize, color='white' if dark else INK)
            ax.text(j, i + 0.28, f"{100*row_pct[i, j]:.1f}%", ha='center', va='center',
                    fontsize=max(7, fontsize - 7),
                    color='#e8f0fa' if dark else INK_SOFT)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Normal', 'Arc'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Normal', 'Arc'])
    ax.set_xlabel('Predicted', color=INK_SOFT)
    ax.set_ylabel('True', color=INK_SOFT)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK_SOFT)


def fig_pooled_confusion(pooled: dict, model: str, level: str, out: Path):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    _cm_axes(ax, pooled,
             f"Pooled out-of-fold — {model}",
             f"leave-one-{level}-out · n={pooled['n']} cycles\n"
             f"acc={100*pooled['accuracy']:.1f}%  F1={100*pooled['f1']:.1f}%  "
             f"rec={100*pooled['recall']:.1f}%  spec={100*pooled['specificity']:.1f}%")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_per_fold_confusion(tbl: pd.DataFrame, out: Path):
    n = len(tbl)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.8 * nrow), squeeze=False)
    for ax, (fold, r) in zip(axes.ravel(), tbl.iterrows()):
        _cm_axes(ax, {k: int(r[k]) for k in ('tp', 'fp', 'fn', 'tn')},
                 f"Fold {fold} — held out: {r.held_out}",
                 f"n={int(r.n)} · F1={100*r.f1:.1f}% · rec={100*r.recall:.1f}%",
                 fontsize=12)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.suptitle("Per-fold confusion matrices (each model never saw its test group)",
                 fontsize=11, color=INK)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_roc(df: pd.DataFrame, tbl: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], ls='--', lw=1, color=GRID, zorder=1)

    for k, (fold, g) in enumerate(df.groupby('fold')):
        fpr, tpr, _ = roc_curve(g.label.values, g.prob.values)
        a = auc(fpr, tpr)
        color = FOLD_COLORS[k % len(FOLD_COLORS)]
        label = f"fold {fold} · {tbl.loc[fold, 'held_out']} (AUC {a:.3f})"
        ax.plot(fpr, tpr, lw=2, color=color, label=label, zorder=3)

    fpr, tpr, _ = roc_curve(df.label.values, df.prob.values)
    pooled_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2.6, color=INK, label=f"pooled (AUC {pooled_auc:.3f})", zorder=4)

    ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False-alarm rate (1 − specificity)", color=INK_SOFT)
    ax.set_ylabel("Arc detection rate (recall)", color=INK_SOFT)
    ax.set_title("ROC — out-of-fold predictions", fontsize=11, color=INK)
    ax.grid(color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_SOFT)
    ax.legend(loc='lower right', fontsize=8, frameon=False, labelcolor=INK)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    return pooled_auc


def fig_metrics_per_fold(tbl: pd.DataFrame, pooled: dict, out: Path):
    n = len(tbl)
    x = np.arange(len(METRICS))
    width = 0.8 / (n + 1)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.5 * len(METRICS) + n), 4.4))

    for k, (fold, r) in enumerate(tbl.iterrows()):
        vals = [100 * r[m] for m in METRICS]
        color = FOLD_COLORS[k % len(FOLD_COLORS)]
        bars = ax.bar(x + k * width, vals, width * 0.9, color=color, zorder=3,
                      label=f"fold {fold} · {r.held_out}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}",
                    ha='center', fontsize=7, color=INK_SOFT)

    vals = [100 * pooled[m] for m in METRICS]
    bars = ax.bar(x + n * width, vals, width * 0.9, color=INK, zorder=3, label="pooled")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}",
                ha='center', fontsize=7, color=INK)

    ax.set_xticks(x + (n * width) / 2)
    ax.set_xticklabels([m.capitalize() for m in METRICS], color=INK)
    ax.set_ylim(0, 108)
    ax.set_ylabel("%", color=INK_SOFT)
    ax.set_title("Out-of-fold performance per held-out group", fontsize=11, color=INK)
    ax.grid(axis='y', color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_SOFT, length=0)
    ax.legend(fontsize=8, frameon=False, ncol=min(3, n + 1), loc='lower left',
              bbox_to_anchor=(0, -0.28), labelcolor=INK)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def threshold_sweep(df: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = metrics_from_confusion(confusion(df.label.values, df.prob.values, t))
        m['threshold'] = t
        rows.append(m)
    return pd.DataFrame(rows)


def fig_threshold(sweep: pd.DataFrame, picks: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(sweep.threshold, 100 * sweep.recall, lw=2, color=FOLD_COLORS[0],
            label="Arc detection rate (recall)", zorder=3)
    ax.plot(sweep.threshold, 100 * sweep.fpr, lw=2, color=FOLD_COLORS[1],
            label="False-alarm rate on normal", zorder=3)
    ax.plot(sweep.threshold, 100 * sweep.f1, lw=2, color=FOLD_COLORS[3],
            label="F1", zorder=3)

    # Several criteria can land on the same threshold — merge them into one label
    merged = {}
    for name, t in picks.items():
        merged.setdefault(round(float(t), 3), []).append(name)
    for k, (t, names) in enumerate(sorted(merged.items())):
        ax.axvline(t, color=GRID, lw=1, ls='--', zorder=1)
        ax.text(t - 0.008, 103 - 14 * (k % 2), " / ".join(names), rotation=90,
                fontsize=7, color=INK_SOFT, ha='right', va='top')

    ax.set_xlabel("Decision threshold on P(arc)", color=INK_SOFT)
    ax.set_ylabel("%", color=INK_SOFT)
    ax.set_ylim(0, 105)
    ax.set_title("Pooled out-of-fold operating curve", fontsize=11, color=INK)
    ax.grid(color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_SOFT)
    ax.legend(fontsize=8, frameon=False, loc='center left', labelcolor=INK)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


# ═══════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════

def md_table(rows, header) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Evaluation report for a group-CV run")
    ap.add_argument('run_dir', type=str)
    ap.add_argument('--data-dir', type=str,
                    default='combined_dataset_2048/combined_dataset_2048',
                    help='dataset dir (for metadata.csv used in FP/FN forensics)')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--max-fpr', type=float, default=0.01,
                    help='target false-alarm rate for the fixed-FPR operating point')
    ap.add_argument('--compare-single', type=str, default=None,
                    help='results.json of a random-split single run, for the '
                         'optimism-of-the-random-split comparison')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = run_dir / 'eval'
    out.mkdir(parents=True, exist_ok=True)

    summary, df = load_run(run_dir)
    model = summary.get('model_name', '?')
    level = summary.get('group_level', 'group')
    thr = args.threshold

    tbl = fold_table(df, thr)
    pooled = metrics_from_confusion(confusion(df.label.values, df.prob.values, thr))

    # ── figures ──
    fig_pooled_confusion(pooled, model, level, out / 'confusion_pooled.png')
    fig_per_fold_confusion(tbl, out / 'confusion_per_fold.png')
    pooled_auc = fig_roc(df, tbl, out / 'roc_folds.png')
    fig_metrics_per_fold(tbl, pooled, out / 'metrics_per_fold.png')

    sweep = threshold_sweep(df, np.round(np.arange(0.02, 0.99, 0.01), 3))
    best_f1_row = sweep.loc[sweep.f1.idxmax()]
    ok = sweep[sweep.fpr <= args.max_fpr]
    fixed_fpr_row = ok.loc[ok.recall.idxmax()] if len(ok) else None
    picks = {f"default {thr:g}": thr, f"max F1 {best_f1_row.threshold:g}": best_f1_row.threshold}
    if fixed_fpr_row is not None:
        picks[f"FPR≤{100*args.max_fpr:g}% @ {fixed_fpr_row.threshold:g}"] = fixed_fpr_row.threshold
    fig_threshold(sweep, picks, out / 'threshold_sweep.png')
    sweep.to_csv(out / 'threshold_sweep.csv', index=False)

    # ── per-fold csv ──
    tbl.assign(threshold=thr).to_csv(out / 'fold_metrics.csv')

    # ── FP/FN forensics ──
    forensics_note = ""
    meta_path = Path(args.data_dir) / 'metadata.csv'
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if len(meta) >= df.idx.max() + 1:
            j = meta.iloc[df.idx.values].reset_index(drop=True).copy()
            j['prob'] = df.prob.values
            j['true'] = df.label.values.astype(int)
            j['pred'] = (df.prob.values > thr).astype(int)
            j['fold'] = df.fold.values
            fps = j[(j.pred == 1) & (j.true == 0)].sort_values('prob', ascending=False)
            fns = j[(j.pred == 0) & (j.true == 1)].sort_values('prob')
            fps.to_csv(out / 'false_positives.csv', index=False)
            fns.to_csv(out / 'false_negatives.csv', index=False)
            err_rows = []
            for ds, gg in j.groupby('dataset'):
                n_fp = int(((gg.pred == 1) & (gg.true == 0)).sum())
                n_fn = int(((gg.pred == 0) & (gg.true == 1)).sum())
                n_norm = int((gg.true == 0).sum())
                n_arc = int((gg.true == 1).sum())
                err_rows.append([ds, n_arc, n_norm, n_fn,
                                 f"{100*n_fn/max(n_arc,1):.2f}", n_fp,
                                 f"{100*n_fp/max(n_norm,1):.2f}"])
            forensics_note = (
                "\n### Errors by campaign\n\n" +
                md_table(err_rows, ["Campaign", "Arc cycles", "Normal cycles",
                                    "Missed arcs (FN)", "Miss rate %",
                                    "False alarms (FP)", "False-alarm rate %"]) +
                "\n\nPer-cycle forensics are in `false_positives.csv` / "
                "`false_negatives.csv` (experiment, alternance index, arc_ratio, "
                "predicted probability, fold).\n")
        else:
            forensics_note = "\n_metadata.csv shorter than the prediction set — forensics skipped._\n"

    # ── per-campaign breakdown (independent of fold layout) ──
    camp_rows = []
    for grp, g in df.groupby('group'):
        m = metrics_from_confusion(confusion(g.label.values, g.prob.values, thr))
        camp_rows.append([grp, int(m['n']), f"{100*m['accuracy']:.2f}", f"{100*m['f1']:.2f}",
                          f"{100*m['recall']:.2f}", f"{100*m['specificity']:.2f}",
                          m['tp'], m['fp'], m['fn'], m['tn']])

    # ── markdown report ──
    acc_lo, acc_hi = wilson(pooled['tp'] + pooled['tn'], pooled['n'])
    rec_lo, rec_hi = wilson(pooled['tp'], pooled['tp'] + pooled['fn'])
    spec_lo, spec_hi = wilson(pooled['tn'], pooled['tn'] + pooled['fp'])

    lines = [
        f"# Group cross-validation report — {model}",
        "",
        f"- Run: `{run_dir}`",
        f"- Protocol: leave-one-{level}-out, {len(tbl)} folds "
        f"(`--mode groupkfold --group-level {level}`)",
        f"- Validation split inside the training groups: `{summary.get('val_mode', 'n/a')}`",
        f"- Decision threshold: {thr}",
        f"- Epochs (max): {summary.get('epochs')}, patience {summary.get('patience')}, "
        f"lr {summary.get('lr')}, batch {summary.get('batch_size')}, seed {summary.get('seed')}",
        "",
        "## 1. Headline — pooled out-of-fold",
        "",
        f"Every one of the {pooled['n']} cycles was classified exactly once, by a model "
        f"trained without its {level}. No cycle is scored by a model that saw its own "
        f"{level}, so this matrix is the honest performance estimate.",
        "",
        md_table([
            ["Accuracy", f"{100*pooled['accuracy']:.2f}%", f"[{100*acc_lo:.2f}, {100*acc_hi:.2f}]"],
            ["F1", f"{100*pooled['f1']:.2f}%", "—"],
            ["Precision", f"{100*pooled['precision']:.2f}%", "—"],
            ["Recall (arc detection)", f"{100*pooled['recall']:.2f}%", f"[{100*rec_lo:.2f}, {100*rec_hi:.2f}]"],
            ["Specificity", f"{100*pooled['specificity']:.2f}%", f"[{100*spec_lo:.2f}, {100*spec_hi:.2f}]"],
            ["ROC AUC", f"{pooled_auc:.4f}", "—"],
        ], ["Metric", "Value", "95% CI (Wilson)"]),
        "",
        f"Confusion counts: TP={pooled['tp']}  FP={pooled['fp']}  "
        f"FN={pooled['fn']}  TN={pooled['tn']}",
        "",
        "![pooled](confusion_pooled.png)",
        "",
        "## 2. Per-fold results",
        "",
        md_table([[f, r.held_out, int(r.n), f"{100*r.accuracy:.2f}", f"{100*r.f1:.2f}",
                   f"{100*r.precision:.2f}", f"{100*r.recall:.2f}", f"{100*r.specificity:.2f}",
                   f"{r.auc:.4f}"] for f, r in tbl.iterrows()],
                 ["Fold", "Held out", "n", "Acc %", "F1 %", "Prec %", "Rec %", "Spec %", "AUC"]),
        "",
        md_table([["mean ± std"] + [f"{100*tbl[m].mean():.2f} ± {100*tbl[m].std(ddof=0):.2f}"
                                    for m in METRICS]],
                 ["Across folds"] + [m.capitalize() + " %" for m in METRICS]),
        "",
        "Mean ± std weights each fold equally regardless of size; prefer the pooled "
        "numbers in section 1 as the headline and read the spread here as the "
        "campaign-to-campaign variability.",
        "",
        "![per fold](metrics_per_fold.png)",
        "",
        "![per fold cm](confusion_per_fold.png)",
        "",
        "## 3. Per-held-out-group breakdown",
        "",
        md_table(camp_rows, ["Group", "n", "Acc %", "F1 %", "Rec %", "Spec %",
                             "TP", "FP", "FN", "TN"]),
        "",
        "## 4. Operating point",
        "",
        md_table(
            [[f"{thr:g} (default)", f"{100*pooled['recall']:.2f}", f"{100*pooled['fpr']:.2f}",
              f"{100*pooled['precision']:.2f}", f"{100*pooled['f1']:.2f}"],
             [f"{best_f1_row.threshold:g} (max F1)", f"{100*best_f1_row.recall:.2f}",
              f"{100*best_f1_row.fpr:.2f}", f"{100*best_f1_row.precision:.2f}",
              f"{100*best_f1_row.f1:.2f}"]] +
            ([[f"{fixed_fpr_row.threshold:g} (FPR≤{100*args.max_fpr:g}%)",
               f"{100*fixed_fpr_row.recall:.2f}", f"{100*fixed_fpr_row.fpr:.2f}",
               f"{100*fixed_fpr_row.precision:.2f}", f"{100*fixed_fpr_row.f1:.2f}"]]
             if fixed_fpr_row is not None else []),
            ["Threshold", "Recall %", "False-alarm %", "Precision %", "F1 %"]),
        "",
        "The thresholds above are picked on the pooled out-of-fold predictions, so "
        "treat any non-default choice as a *reported trade-off*, not a tuned model: "
        "tuning it on the same predictions you report would leak the test set.",
        "",
        "![roc](roc_folds.png)",
        "",
        "![threshold](threshold_sweep.png)",
        forensics_note,
    ]

    if args.compare_single:
        s = json.load(open(args.compare_single))
        lines += [
            "## 5. Random-split comparison (optimism of the in-distribution number)",
            "",
            md_table([
                ["Random 70/15/15 split (cycle level)",
                 f"{100*s.get('test_accuracy', float('nan')):.2f}",
                 f"{100*s.get('test_f1', float('nan')):.2f}",
                 f"{100*s.get('test_recall', float('nan')):.2f}",
                 f"{100*s.get('test_specificity', float('nan')):.2f}"],
                [f"Leave-one-{level}-out (pooled)",
                 f"{100*pooled['accuracy']:.2f}", f"{100*pooled['f1']:.2f}",
                 f"{100*pooled['recall']:.2f}", f"{100*pooled['specificity']:.2f}"],
            ], ["Protocol", "Acc %", "F1 %", "Rec %", "Spec %"]),
            "",
            "The random split lets cycles of the same arc burst and the same recording "
            "sit in train and test at once, so it measures in-distribution fit — the "
            "gap between the two rows is the cost of that leakage.",
            "",
        ]

    (out / 'results.md').write_text("\n".join(lines) + "\n")

    print(f"Pooled out-of-fold ({pooled['n']} cycles): "
          f"acc={100*pooled['accuracy']:.2f}%  F1={100*pooled['f1']:.2f}%  "
          f"rec={100*pooled['recall']:.2f}%  spec={100*pooled['specificity']:.2f}%  "
          f"AUC={pooled_auc:.4f}")
    print(f"Per-fold F1: " + "  ".join(f"{f}:{100*r.f1:.1f}%" for f, r in tbl.iterrows()))
    print(f"\nReport written to {out/'results.md'}")


if __name__ == '__main__':
    main()
