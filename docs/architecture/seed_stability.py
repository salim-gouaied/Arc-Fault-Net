#!/usr/bin/env python3
"""
Seed-stability study for Arc-FaultNet (mode `single`).
======================================================

Reads every comparable run (same architecture + hyper-parameters) from
``runs/arcfaultnet_single_*`` and aggregates the test-set metrics across
seeds to answer two questions:

  1. Is the model stable when the random seed changes?
     -> per-metric mean, std, sem, 95% CI (Student-t)
  2. How much do the test sets actually overlap between seeds?
     -> exact index overlap matrix (Jaccard + raw count)

For each metric (accuracy, F1, precision, recall, specificity) we compute:

  - mean   : average over seeds
  - std    : sample standard deviation (n-1)
  - sem    : standard error of the mean = std / sqrt(n)
  - 95% CI : mean +/- t_{n-1, .975} * sem  (Student-t)
  - CV     : coefficient of variation = std / mean

Outputs (relative to docs/architecture/):

  diagrams/seed_stability/seed_<S>.png         one PNG per seed
  diagrams/seed_stability/summary_forest.png   95% CI summary across seeds
  diagrams/seed_stability/overlap_matrix.png   test-set overlap heatmap
  modules/17_seed_stability.md                 human-readable report
  seed_stability/per_seed.csv                  one row per (seed, ...)
  seed_stability/summary.csv                   one row per metric
  seed_stability/overlap_jaccard.csv           Jaccard between test sets
  seed_stability/overlap_count.csv             |test_i intersect test_j|

The script only reads ``results.json`` files and dataset config – it does
*not* need the raw .npy data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR     = PROJECT_ROOT / "runs"
DOC_DIR      = PROJECT_ROOT / "docs" / "architecture"
FIG_DIR      = DOC_DIR / "diagrams" / "seed_stability"
MOD_DIR      = DOC_DIR / "modules"
OUT_DIR      = DOC_DIR / "seed_stability"

# Only aggregate runs that share this exact training recipe.
REFERENCE_CONFIG = {
    "model_name":   "arcfaultnet",
    "n_params":     344409,
    "epochs":       200,
    "lr":           3e-4,
    "weight_decay": 5e-4,
    "batch_size":   64,
    "patience":     10,
    "gradient_clip": 0.5,
    "threshold":    0.5,
}

# Split ratios used by `train.py --mode single`
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15

METRICS = [
    ("test_accuracy",    "Accuracy"),
    ("test_f1",          "F1 score"),
    ("test_precision",   "Precision"),
    ("test_recall",      "Recall"),
    ("test_specificity", "Specificity"),
]

# Stability thresholds (on the 0..1 metric scale)
THRESH_STD_STABLE = 0.01   # std <= 1 pp  -> stable
THRESH_STD_OK     = 0.02   # std <= 2 pp  -> acceptable
THRESH_CV_STABLE  = 0.02   # CV  <= 2 %   -> stable


# ─────────────────────────────────────────────────────────────────────
#  DISCOVERY
# ─────────────────────────────────────────────────────────────────────

def discover_runs() -> pd.DataFrame:
    """Find every results.json that matches REFERENCE_CONFIG."""
    rows = []
    for results_path in sorted(RUNS_DIR.glob("arcfaultnet_single_*/results.json")):
        try:
            data = json.loads(results_path.read_text())
        except Exception as e:
            print(f"  ! skipping {results_path}: {e}")
            continue

        match = all(
            data.get(k) == v
            or (isinstance(v, float) and isinstance(data.get(k), (int, float))
                and abs(data.get(k) - v) < 1e-9)
            for k, v in REFERENCE_CONFIG.items()
        )
        if not match:
            continue

        rows.append({
            "seed":      int(data["seed"]),
            "timestamp": data.get("timestamp", results_path.parent.name),
            "best_epoch": int(data.get("best_epoch", -1)),
            "duration_s": float(data.get("training_duration_seconds", float("nan"))),
            **{m: float(data[m]) for m, _ in METRICS},
            "run_dir": results_path.parent.name,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No comparable runs found.  "
                         "Check REFERENCE_CONFIG against your results.json files.")

    df = df.sort_values(["seed", "timestamp"]).drop_duplicates(
        subset="seed", keep="first").reset_index(drop=True)
    return df


def detect_dataset_size() -> tuple[int, str]:
    """Read N from the dataset config (combined_dataset preferred)."""
    for name in ("combined_dataset", "labeled_dataset"):
        cfg = PROJECT_ROOT / name / "config.json"
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text())
                if "n_samples" in data:
                    return int(data["n_samples"]), name
                if "X_shape" in data:
                    return int(data["X_shape"][0]), name
            except Exception:
                continue
        cfg = PROJECT_ROOT / name / "config_multi.json"
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text())
                if "n_samples" in data:
                    return int(data["n_samples"]), name
                if "X_shape" in data:
                    return int(data["X_shape"][0]), name
            except Exception:
                continue
    raise SystemExit("Could not detect dataset size from any config.json")


# ─────────────────────────────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Per-metric mean, std, sem, 95% CI, min, max, CV."""
    n = len(df)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else float("nan")

    rows = []
    for key, pretty in METRICS:
        x = df[key].to_numpy(dtype=float)
        mean = x.mean()
        std  = x.std(ddof=1) if n > 1 else 0.0
        sem  = std / np.sqrt(n) if n > 1 else 0.0
        ci_half = t_crit * sem if n > 1 else 0.0
        cv = std / mean if mean > 0 else float("nan")

        rows.append({
            "metric":   pretty,
            "key":      key,
            "n":        n,
            "mean":     mean,
            "std":      std,
            "sem":      sem,
            "ci_low":   mean - ci_half,
            "ci_high":  mean + ci_half,
            "ci_half":  ci_half,
            "min":      x.min(),
            "max":      x.max(),
            "cv":       cv,
        })
    return pd.DataFrame(rows)


def stability_verdict(summary: pd.DataFrame) -> tuple[str, list[str]]:
    notes = []
    statuses = []
    for _, r in summary.iterrows():
        if r["std"] <= THRESH_STD_STABLE and r["cv"] <= THRESH_CV_STABLE:
            status = "STABLE"
        elif r["std"] <= THRESH_STD_OK:
            status = "ACCEPTABLE"
        else:
            status = "UNSTABLE"
        statuses.append(status)
        notes.append(
            f"- **{r['metric']}**: mean = {r['mean']:.4f}, "
            f"std = {r['std']:.4f} ({r['cv']*100:.2f}%), "
            f"95% CI = [{r['ci_low']:.4f}, {r['ci_high']:.4f}] -> {status}"
        )

    if all(s == "STABLE" for s in statuses):
        overall = "STABLE"
    elif any(s == "UNSTABLE" for s in statuses):
        overall = "UNSTABLE on at least one metric"
    else:
        overall = "ACCEPTABLE (low-to-moderate variance)"
    return overall, notes


# ─────────────────────────────────────────────────────────────────────
#  SPLIT RECOMPUTATION  (no .npy needed)
# ─────────────────────────────────────────────────────────────────────

def recompute_test_indices(seed: int, n: int) -> np.ndarray:
    """
    Reproduce the test indices used by `train.py --mode single`.

    Matches exactly:
        set_seed(seed)
        indices = np.random.permutation(N)
        n_train = int(N * train_ratio)
        n_val   = int(N * val_ratio)
        test_indices = indices[n_train + n_val:]
    """
    # `set_seed` calls random.seed, np.random.seed, torch.manual_seed in
    # this order, but the FIRST RNG consumed by the split is np.random,
    # so only np.random.seed matters for reproducing the test indices.
    np.random.seed(seed)
    indices = np.random.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return indices[n_train + n_val:]


def overlap_matrices(test_sets: dict[int, set[int]]
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise intersection count + Jaccard between seeds' test sets."""
    seeds = sorted(test_sets.keys())
    n = len(seeds)
    inter = np.zeros((n, n), dtype=int)
    jac   = np.zeros((n, n), dtype=float)
    for i, si in enumerate(seeds):
        ti = test_sets[si]
        for j, sj in enumerate(seeds):
            tj = test_sets[sj]
            inter[i, j] = len(ti & tj)
            jac[i, j]   = len(ti & tj) / len(ti | tj)
    idx = [f"seed {s}" for s in seeds]
    return (pd.DataFrame(inter, index=idx, columns=idx),
            pd.DataFrame(jac,   index=idx, columns=idx))


# ─────────────────────────────────────────────────────────────────────
#  PLOTTING - independent figure per seed
# ─────────────────────────────────────────────────────────────────────

def plot_one_seed(seed: int, row: pd.Series, summary: pd.DataFrame,
                  out_path: Path):
    """One PNG showing this seed's 5 metrics vs the group mean +/- 95% CI."""
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)

    keys     = [k for k, _ in METRICS]
    labels   = [p for _, p in METRICS]
    vals     = np.array([row[k] for k in keys]) * 100
    means    = summary["mean"].to_numpy() * 100
    ci_low   = summary["ci_low"].to_numpy() * 100
    ci_high  = summary["ci_high"].to_numpy() * 100

    x = np.arange(len(METRICS))
    bar_w = 0.55
    colors = ["#3a7ca5" if v >= m else "#c25450"
              for v, m in zip(vals, means)]
    bars = ax.bar(x, vals, bar_w, color=colors, edgecolor="black",
                  linewidth=0.7, zorder=3,
                  label=f"seed {seed} value")

    # 95% CI band (per metric) drawn as short horizontal segments
    for xi, mu, lo, hi in zip(x, means, ci_low, ci_high):
        ax.add_patch(plt.Rectangle((xi - bar_w/2, lo), bar_w, hi - lo,
                                   color="tab:red", alpha=0.13, zorder=1))
        ax.hlines(mu, xi - bar_w/2, xi + bar_w/2,
                  color="tab:red", linestyles="--", linewidth=1.6, zorder=4)

    # legend proxies
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elems = [
        Patch(facecolor="#3a7ca5", edgecolor="black",
              label=f"seed {seed}  (>= group mean)"),
        Patch(facecolor="#c25450", edgecolor="black",
              label=f"seed {seed}  (<  group mean)"),
        Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.6,
               label="group mean over all seeds"),
        Patch(facecolor="tab:red", alpha=0.13, label="95% CI band"),
    ]
    ax.legend(handles=legend_elems, fontsize=8.5, loc="lower right")

    # annotate each bar with value and delta from mean
    for xi, v, mu in zip(x, vals, means):
        delta = v - mu
        sign = "+" if delta >= 0 else ""
        ax.text(xi, v + 0.18, f"{v:.2f}%\n({sign}{delta:.2f} pp)",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value (%)")

    ymin = max(0, min(vals.min(), ci_low.min()) - 4)
    ymax = min(100, max(vals.max(), ci_high.max()) + 4)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    n_total = int(summary["n"].iloc[0])
    ax.set_title(
        f"Arc-FaultNet - seed {seed}  vs  group mean +/- 95% CI "
        f"(n = {n_total} seeds, best epoch = {row['best_epoch']})",
        fontsize=11, fontweight="bold")

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def pick_top3_seeds(df: pd.DataFrame, key: str = "test_f1"
                    ) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Pick 3 representative seeds: worst, median, best on `key`.
    Returns the sub-DataFrame (sorted worst -> best) and a role map.
    """
    sorted_df = df.sort_values(key, ascending=True).reset_index(drop=True)
    n = len(sorted_df)
    worst_idx  = 0
    best_idx   = n - 1
    # closest to the population mean = robust "typical" seed
    mean_val = sorted_df[key].mean()
    median_idx = (sorted_df[key] - mean_val).abs().idxmin()
    # avoid duplicates if median collides with worst/best
    if median_idx in (worst_idx, best_idx):
        for cand in sorted_df.index:
            if cand not in (worst_idx, best_idx):
                median_idx = cand
                break
    picks = sorted_df.iloc[[worst_idx, median_idx, best_idx]].reset_index(drop=True)
    roles = {
        "worst":  int(sorted_df.iloc[worst_idx]["seed"]),
        "median": int(sorted_df.iloc[median_idx]["seed"]),
        "best":   int(sorted_df.iloc[best_idx]["seed"]),
    }
    return picks, roles


def plot_top3_combined(df_top3: pd.DataFrame, summary: pd.DataFrame,
                       roles: dict[str, int], n_all_seeds: int,
                       out_path: Path):
    """
    2x3 grid restricted to 3 representative seeds (worst / median / best
    by F1). CI band still reflects all n seeds in the population.
    """
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    seeds_in_order = df_top3["seed"].to_numpy()
    role_lookup = {v: k for k, v in roles.items()}
    role_labels = {"worst": "worst", "median": "median", "best": "best"}

    palette = {"worst": "#c25450", "median": "#d4a017", "best": "#3a7ca5"}
    colors = [palette[role_lookup[int(s)]] for s in seeds_in_order]

    x = np.arange(len(seeds_in_order))

    for i, (key, pretty) in enumerate(METRICS):
        ax = fig.add_subplot(gs[i // 3, i % 3])

        vals = df_top3[key].to_numpy() * 100
        row  = summary.iloc[i]
        mean_pc   = row["mean"] * 100
        ci_lo_pc  = row["ci_low"] * 100
        ci_hi_pc  = row["ci_high"] * 100

        bars = ax.bar(x, vals, color=colors, edgecolor="black",
                      linewidth=0.7, alpha=0.92, zorder=2)

        ax.axhspan(ci_lo_pc, ci_hi_pc, color="tab:red", alpha=0.13, zorder=1,
                   label=f"95% CI [{ci_lo_pc:.2f}, {ci_hi_pc:.2f}]")
        ax.axhline(mean_pc, color="tab:red", linestyle="--", linewidth=1.5,
                   zorder=3, label=f"mean = {mean_pc:.2f}%")

        for xi, v in zip(x, vals):
            delta = v - mean_pc
            sign = "+" if delta >= 0 else ""
            ax.text(xi, v + 0.18, f"{v:.2f}%\n({sign}{delta:.2f} pp)",
                    ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {int(s)}\n({role_labels[role_lookup[int(s)]]})"
                            for s in seeds_in_order], fontsize=9)
        ax.set_ylabel(f"{pretty} (%)")
        ax.set_title(f"{pretty}  -  std={row['std']*100:.2f} pp, "
                     f"CV={row['cv']*100:.2f}%", fontsize=10)
        ymin = max(0, min(vals.min(), ci_lo_pc) - 4)
        ymax = min(100, max(vals.max(), ci_hi_pc) + 4)
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=7.5, loc="lower right")

    # Forest plot panel (bottom-right) - over ALL seeds
    ax = fig.add_subplot(gs[1, 2])
    pretty_labels = [m[1] for m in METRICS]
    y = np.arange(len(METRICS))[::-1]
    means    = summary["mean"].to_numpy() * 100
    ci_low   = summary["ci_low"].to_numpy() * 100
    ci_high  = summary["ci_high"].to_numpy() * 100
    err_low  = means - ci_low
    err_high = ci_high - means

    # plot top-3 seeds as coloured dots
    for _, r in df_top3.iterrows():
        s = int(r["seed"])
        role = role_lookup[s]
        col = palette[role]
        for k, (key, _) in enumerate(METRICS):
            ax.scatter(r[key] * 100, y[k], color=col, s=55, zorder=3,
                       edgecolor="black", linewidth=0.5,
                       label=f"seed {s} ({role})" if k == 0 else None)

    ax.errorbar(means, y, xerr=[err_low, err_high], fmt="D",
                color="tab:red", ecolor="tab:red", capsize=6,
                markersize=8, linewidth=1.8, zorder=4,
                label=f"mean +/- 95% CI  (n = {n_all_seeds})")

    for yi, m_val, lo, hi in zip(y, means, ci_low, ci_high):
        ax.text(hi + 0.5, yi, f"{m_val:.2f}% [{lo:.2f}, {hi:.2f}]",
                va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(pretty_labels)
    ax.set_xlabel("metric value (%)")
    ax.set_title("95% CI (all seeds) + top-3 seeds", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    xmin = max(0, ci_low.min() - 3)
    xmax = min(100, ci_high.max() + 9)
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=8, loc="lower left")

    title = (f"Arc-FaultNet - seed stability  (3 representative seeds: "
             f"worst={roles['worst']}, median={roles['median']}, "
             f"best={roles['best']}, by F1   |   "
             f"CI over all {n_all_seeds} seeds)")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_forest_summary(df: pd.DataFrame, summary: pd.DataFrame,
                        out_path: Path):
    """Mean +/- 95% CI across seeds for every metric, on one PNG."""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    pretty_labels = [m[1] for m in METRICS]
    y = np.arange(len(METRICS))[::-1]
    means    = summary["mean"].to_numpy() * 100
    ci_low   = summary["ci_low"].to_numpy() * 100
    ci_high  = summary["ci_high"].to_numpy() * 100
    err_low  = means - ci_low
    err_high = ci_high - means

    # individual seed dots overlaid
    seeds = df["seed"].to_numpy()
    seed_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(seeds)))
    for j, (_, r) in enumerate(df.iterrows()):
        for k, (key, _) in enumerate(METRICS):
            ax.scatter(r[key] * 100, y[k] + (j - (len(seeds)-1)/2) * 0.07,
                       color=seed_colors[j], s=22, alpha=0.85, zorder=2,
                       edgecolor="black", linewidth=0.4)

    ax.errorbar(means, y, xerr=[err_low, err_high], fmt="D",
                color="tab:red", ecolor="tab:red", capsize=6,
                markersize=8, linewidth=1.8, zorder=3,
                label="mean +/- 95% CI")

    for yi, m_val, lo, hi in zip(y, means, ci_low, ci_high):
        ax.text(hi + 0.5, yi, f"{m_val:.2f}% [{lo:.2f}, {hi:.2f}]",
                va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(pretty_labels, fontsize=10)
    ax.set_xlabel("metric value (%)")

    # seed legend
    seed_handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                               color=seed_colors[j], markersize=7,
                               markeredgecolor="black", markeredgewidth=0.4,
                               label=f"seed {s}")
                    for j, s in enumerate(seeds)]
    ax.legend(handles=seed_handles + [plt.Line2D([0], [0], marker="D",
                                                  linestyle="",
                                                  color="tab:red",
                                                  markersize=8,
                                                  label="mean +/- 95% CI")],
              fontsize=8.5, loc="lower left", ncol=2)

    ax.set_title(f"Arc-FaultNet - seed stability summary  (n = {len(df)} seeds)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    xmin = max(0, ci_low.min() - 3)
    xmax = min(100, ci_high.max() + 9)
    ax.set_xlim(xmin, xmax)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_heatmap(jac: pd.DataFrame, inter: pd.DataFrame,
                         test_size: int, out_path: Path):
    """Heatmap of pairwise Jaccard with intersection count annotation."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    im = ax.imshow(jac.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jaccard index   |A inter B| / |A union B|")

    n = jac.shape[0]
    for i in range(n):
        for j in range(n):
            v = jac.iat[i, j]
            count = int(inter.iat[i, j])
            text_col = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.3f}\n({count})",
                    ha="center", va="center", color=text_col, fontsize=9)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(jac.columns, rotation=30, ha="right")
    ax.set_yticklabels(jac.index)
    ax.set_title(
        f"Test-set overlap between seeds\n"
        f"(test size per seed = {test_size}; cell = Jaccard / count)",
        fontsize=11, fontweight="bold")

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, summary: pd.DataFrame,
                 overall: str, notes: list[str],
                 jac: pd.DataFrame, inter: pd.DataFrame,
                 n_dataset: int, n_test: int, dataset_name: str,
                 roles: dict[str, int],
                 report_path: Path):
    seeds_str = ", ".join(str(s) for s in df["seed"])
    hp = REFERENCE_CONFIG

    # average off-diagonal jaccard / intersection
    mat = jac.to_numpy()
    off = mat[~np.eye(mat.shape[0], dtype=bool)]
    mean_jac = float(off.mean())
    mean_int = float(inter.to_numpy()[
        ~np.eye(inter.shape[0], dtype=bool)].mean())

    lines = [
        "# Seed Stability Study",
        "",
        f"**Seeds analysed:** {len(df)}  -  {seeds_str}",
        f"**Overall stability:** **{overall}**",
        f"**Dataset:** `{dataset_name}` (N = {n_dataset} samples, "
        f"test split = {n_test}, ratios 70/15/15)",
        "",
        "## 1. Why this study?",
        "",
        "Each `--mode single` training run re-rolls *two* sources of "
        "randomness through `--seed`:",
        "",
        "1. the **random 70/15/15 train-val-test split** "
        "(`np.random.permutation` in `train.py`), so the *identity* of the "
        "test samples changes across seeds;",
        "2. the **optimisation randomness** - weight init, batch order, "
        "augmentation, dropout.",
        "",
        "The architecture and hyper-parameters are held **constant**. The "
        "spread of test metrics therefore measures the *combined* "
        "sensitivity of the model to split + training randomness.",
        "",
        "## 2. Reference configuration",
        "",
        "| Hyper-parameter | Value |",
        "|---|---|",
        f"| model         | `{hp['model_name']}` |",
        f"| parameters    | {hp['n_params']:,} |",
        f"| epochs (max)  | {hp['epochs']} |",
        f"| learning rate | {hp['lr']} |",
        f"| weight decay  | {hp['weight_decay']} |",
        f"| batch size    | {hp['batch_size']} |",
        f"| patience      | {hp['patience']} |",
        f"| grad clip     | {hp['gradient_clip']} |",
        f"| threshold     | {hp['threshold']} |",
        "",
        "## 3. Per-seed raw results",
        "",
        "| seed | best ep. | accuracy | F1 | precision | recall | specificity | run dir |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['seed']} | {r['best_epoch']} | "
            f"{r['test_accuracy']*100:.2f} | "
            f"{r['test_f1']*100:.2f} | "
            f"{r['test_precision']*100:.2f} | "
            f"{r['test_recall']*100:.2f} | "
            f"{r['test_specificity']*100:.2f} | "
            f"`{r['run_dir']}` |"
        )

    lines += [
        "",
        "## 4. Aggregate statistics (Student-t, 95% CI)",
        "",
        "| metric | mean | std (pp) | CV | 95% CI | min | max |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['metric']} | "
            f"{r['mean']*100:.2f}% | "
            f"{r['std']*100:.2f} | "
            f"{r['cv']*100:.2f}% | "
            f"[{r['ci_low']*100:.2f}, {r['ci_high']*100:.2f}]% | "
            f"{r['min']*100:.2f}% | "
            f"{r['max']*100:.2f}% |"
        )

    lines += [
        "",
        "## 5. Stability verdict",
        "",
        *notes,
        "",
        "**Decision rule**",
        "",
        f"- `STABLE`     - std <= {THRESH_STD_STABLE*100:.0f} pp **and** "
        f"CV <= {THRESH_CV_STABLE*100:.0f} %",
        f"- `ACCEPTABLE` - std <= {THRESH_STD_OK*100:.0f} pp",
        "- `UNSTABLE`   - otherwise",
        "",
        f"**Overall: {overall}**",
        "",
        "## 6. How to read the 95% CI",
        "",
        "For each metric we estimate the population mean over all possible "
        "seeds with the Student-t 95% confidence interval:",
        "",
        "$$\\bar{x} \\pm t_{n-1,\\,0.975}\\,\\frac{s}{\\sqrt{n}}$$",
        "",
        "A **narrow CI** means the model behaves consistently across seeds; "
        "a **wide CI** means a single training run is not representative and "
        "you should report mean +/- CI, not the score of one favourable "
        "seed.",
        "",
        "## 7. Test-set overlap between seeds",
        "",
        f"Each seed produces a test set of {n_test} samples drawn from "
        f"N = {n_dataset}. With a 70/15/15 split, two independent seeds "
        f"share on average about "
        f"{(n_test/n_dataset)*n_test:.0f} samples just by chance "
        f"(approx. {n_test/n_dataset*100:.2f}% of the test set).",
        "",
        f"Observed mean pairwise overlap "
        f"(off-diagonal, exact reproduction of `np.random.permutation`):",
        "",
        f"- **mean intersection count:** {mean_int:.1f} samples / "
        f"{n_test} ({mean_int/n_test*100:.2f}%)",
        f"- **mean Jaccard index:**     {mean_jac:.4f}",
        "",
        "Pairwise Jaccard matrix (also in `seed_stability/overlap_jaccard.csv`):",
        "",
        "| | " + " | ".join(jac.columns) + " |",
        "|" + "---|" * (len(jac.columns) + 1),
    ]
    for idx, row in jac.iterrows():
        lines.append(f"| **{idx}** | " +
                     " | ".join(f"{v:.3f}" for v in row.values) + " |")

    lines += [
        "",
        "Interpretation: a low Jaccard (~ 0.08-0.10) confirms each seed "
        "evaluates on a **largely different test set**. So the variance "
        "across seeds is not a pure 'training noise' figure - it also "
        "absorbs the **split-roulette** effect (some test samples are "
        "intrinsically harder than others).",
        "",
        "## 8. Figures",
        "",
        "Per-seed plots (one PNG each):",
        "",
    ]
    for _, r in df.iterrows():
        rel = Path("..") / "diagrams" / "seed_stability" / f"seed_{int(r['seed'])}.png"
        lines.append(f"- seed {int(r['seed'])}: ![]({rel.as_posix()})")

    lines += [
        "",
        "Summary across seeds (forest plot with mean +/- 95% CI):",
        "",
        "![]( ../diagrams/seed_stability/summary_forest.png )",
        "",
        f"Top-3 representative seeds (worst = {roles['worst']}, "
        f"median = {roles['median']}, best = {roles['best']}, ranked by F1; "
        f"CI band still computed over all {len(df)} seeds):",
        "",
        "![]( ../diagrams/seed_stability/top3_seeds.png )",
        "",
        "Pairwise test-set overlap (Jaccard heatmap):",
        "",
        "![]( ../diagrams/seed_stability/overlap_matrix.png )",
        "",
        "## 9. Files",
        "",
        "- `docs/architecture/seed_stability/per_seed.csv`",
        "- `docs/architecture/seed_stability/summary.csv`",
        "- `docs/architecture/seed_stability/overlap_jaccard.csv`",
        "- `docs/architecture/seed_stability/overlap_count.csv`",
        "- `docs/architecture/diagrams/seed_stability/seed_<S>.png`",
        "- `docs/architecture/diagrams/seed_stability/summary_forest.png`",
        "- `docs/architecture/diagrams/seed_stability/overlap_matrix.png`",
        "",
    ]

    report_path.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MOD_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Seed-stability study")
    print("=" * 60)

    df = discover_runs()
    seeds_str = ", ".join(str(s) for s in df["seed"])
    print(f"  found {len(df)} comparable runs  (seeds: {seeds_str})")

    n_dataset, dataset_name = detect_dataset_size()
    n_train = int(n_dataset * TRAIN_RATIO)
    n_val   = int(n_dataset * VAL_RATIO)
    n_test  = n_dataset - n_train - n_val
    print(f"  dataset = {dataset_name}  (N = {n_dataset}, test split = {n_test})")

    # ── stats ──────────────────────────────────────────────────
    summary = summarise(df)
    overall, notes = stability_verdict(summary)

    df.to_csv(OUT_DIR / "per_seed.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    # ── test-set overlap ───────────────────────────────────────
    test_sets = {int(s): set(recompute_test_indices(int(s), n_dataset).tolist())
                 for s in df["seed"]}
    inter, jac = overlap_matrices(test_sets)
    inter.to_csv(OUT_DIR / "overlap_count.csv")
    jac.to_csv(OUT_DIR / "overlap_jaccard.csv")
    print(f"  csv     -> {(OUT_DIR / 'per_seed.csv').relative_to(PROJECT_ROOT)}")
    print(f"  csv     -> {(OUT_DIR / 'summary.csv').relative_to(PROJECT_ROOT)}")
    print(f"  csv     -> {(OUT_DIR / 'overlap_jaccard.csv').relative_to(PROJECT_ROOT)}")
    print(f"  csv     -> {(OUT_DIR / 'overlap_count.csv').relative_to(PROJECT_ROOT)}")

    # ── one PNG per seed ───────────────────────────────────────
    for _, row in df.iterrows():
        seed = int(row["seed"])
        out  = FIG_DIR / f"seed_{seed}.png"
        plot_one_seed(seed, row, summary, out)
        print(f"  figure  -> {out.relative_to(PROJECT_ROOT)}")

    forest_path = FIG_DIR / "summary_forest.png"
    plot_forest_summary(df, summary, forest_path)
    print(f"  figure  -> {forest_path.relative_to(PROJECT_ROOT)}")

    df_top3, roles = pick_top3_seeds(df, key="test_f1")
    top3_path = FIG_DIR / "top3_seeds.png"
    plot_top3_combined(df_top3, summary, roles, n_all_seeds=len(df),
                       out_path=top3_path)
    print(f"  figure  -> {top3_path.relative_to(PROJECT_ROOT)}  "
          f"(worst={roles['worst']}, median={roles['median']}, best={roles['best']})")

    overlap_path = FIG_DIR / "overlap_matrix.png"
    plot_overlap_heatmap(jac, inter, n_test, overlap_path)
    print(f"  figure  -> {overlap_path.relative_to(PROJECT_ROOT)}")

    # ── markdown report ────────────────────────────────────────
    report_path = MOD_DIR / "17_seed_stability.md"
    write_report(df, summary, overall, notes, jac, inter,
                 n_dataset, n_test, dataset_name, roles, report_path)
    print(f"  report  -> {report_path.relative_to(PROJECT_ROOT)}")

    print("=" * 60)
    print(f"  overall verdict: {overall}")
    print()
    print("  per-metric summary:")
    for _, r in summary.iterrows():
        print(f"    {r['metric']:<13s} "
              f"mean={r['mean']*100:6.2f}%  "
              f"std={r['std']*100:5.2f}pp  "
              f"CV={r['cv']*100:5.2f}%  "
              f"95% CI=[{r['ci_low']*100:6.2f}, {r['ci_high']*100:6.2f}]%")

    print()
    off = jac.to_numpy()[~np.eye(jac.shape[0], dtype=bool)]
    off_count = inter.to_numpy()[~np.eye(inter.shape[0], dtype=bool)]
    print(f"  test-set overlap (off-diagonal): "
          f"Jaccard mean={off.mean():.4f}  "
          f"intersection mean={off_count.mean():.1f}/{n_test} "
          f"({off_count.mean()/n_test*100:.2f}%)")


if __name__ == "__main__":
    main()
