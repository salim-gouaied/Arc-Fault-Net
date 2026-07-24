"""
stability_eval.py — Multi-seed stability evaluation (random single-split).

Why this exists
---------------
Leave-one-charge-out (LOCO) cross-validation is NOT usable on this dataset:
most of the arc-production recordings were saved WITHOUT their load
configuration, so charge-based splits cannot be formed. Instead we assess an
architecture by its **stability across many random splits**: run ``--mode
single`` for a list of seeds (each seed reshuffles the train/val/test split AND
re-initialises the weights) and report the mean ± std of the test metrics.

Low variance across seeds ⇒ the architecture is robust to the particular split
and initialisation; high variance ⇒ the reported single-run numbers are luck.

This script does not reimplement training — it calls ``run_single_training``
from ``train.py`` once per seed and aggregates the returned metrics. Works for
any model registered in ``get_model`` (``arcssm``, ``arcssm_selective``,
``arcfaultnet_v2``, ...), so the SSM track and V2 can be compared on the exact
same protocol.

Example
-------
    python stability_eval.py --model arcssm --n-seeds 20 --epochs 60 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import ArcFaultDataset
from train import run_single_training


# Metrics returned by run_single_training (keys) and their display labels.
METRICS = [
    ("test_accuracy", "Acc"),
    ("test_f1", "F1"),
    ("test_precision", "Prec"),
    ("test_recall", "Rec"),
    ("test_specificity", "Spec"),
]


def parse_seeds(spec: str) -> list[int]:
    """Expand a seed spec like '0-19,42,100' into a list of ints."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    p = argparse.ArgumentParser(description="Multi-seed stability evaluation")
    p.add_argument("--model", type=str, default="arcssm",
                   choices=["arcssm", "arcssm_selective", "arcfaultnet_v2",
                            "arcfaultnet", "1d_only", "no_attention",
                            "standard_conv", "independent_cbam", "baseline_cnn"])
    # Seeds: either an explicit spec, or --n-seeds N (=> 0..N-1).
    p.add_argument("--seeds", type=str, default="0-19",
                   help="Seed spec, e.g. '0-19,42,100'. Ignored if --n-seeds is set.")
    p.add_argument("--n-seeds", type=int, default=None,
                   help="Shortcut: use seeds 0..N-1 (overrides --seeds).")
    # Training hyper-parameters (mirror train.py defaults).
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--gradient-clip", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--use-pos-weight", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    # Data / signal.
    p.add_argument("--data-dir", type=str,
                   default="/home/manip/pfe_salim_gouaied/Arc-Fault-Net/labeled_dataset")
    p.add_argument("--output-dir", type=str,
                   default="/home/manip/pfe_salim_gouaied/Arc-Fault-Net/runs")
    p.add_argument("--channel-mode", type=str, default="auto",
                   choices=["auto", "raw2", "i_derived4"])
    p.add_argument("--fs", type=int, default=None)
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--hop-length", type=int, default=256)
    # Model flags (mostly for V2 comparability; ignored by the SSM models).
    p.add_argument("--deep-clf", action="store_true")
    p.add_argument("--use-se", action="store_true")
    p.add_argument("--se-reduction", type=int, default=8)
    p.add_argument("--fusion-mode", type=str, default="gated",
                   choices=["gated", "cross_attention", "concat"])
    p.add_argument("--no-channel-attn", action="store_true")
    args = p.parse_args()

    seeds = list(range(args.n_seeds)) if args.n_seeds else parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds to run.")

    device = torch.device("cpu") if (args.cpu or not torch.cuda.is_available()) \
        else torch.device("cuda")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    if not (data_dir / "X_multi.npy").exists():
        raise SystemExit(f"Data not found at {data_dir}")

    # Channel mode: V2 and the SSM track use the 4 I-derived channels.
    if args.channel_mode == "auto":
        channel_mode = "i_derived4" if args.model in (
            "arcfaultnet_v2", "arcssm", "arcssm_selective") else "raw2"
    else:
        channel_mode = args.channel_mode

    # Build the dataset ONCE; each seed re-splits it inside run_single_training.
    dataset = ArcFaultDataset(data_dir=str(data_dir), n_fft=args.n_fft,
                              hop_length=args.hop_length, channel_mode=channel_mode)
    fs = args.fs if args.fs is not None else dataset.fs
    output_dir = Path(args.output_dir)

    print(f"\nStability run: model={args.model}  seeds={seeds}  "
          f"channel_mode={channel_mode}  fs={fs:,}\n")

    runs = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}\n  SEED {seed}  ({i + 1}/{len(seeds)})\n{'='*70}")
        res = run_single_training(
            model_name=args.model, dataset=dataset, device=device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            batch_size=args.batch_size, patience=args.patience,
            gradient_clip=args.gradient_clip, threshold=args.threshold,
            use_pos_weight=args.use_pos_weight, output_dir=output_dir,
            num_workers=args.num_workers, seed=seed,
            use_se=args.use_se, se_reduction=args.se_reduction,
            deep_classifier=args.deep_clf, fusion_mode=args.fusion_mode,
            use_channel_attn=not args.no_channel_attn, fs=fs, n_fft=args.n_fft,
        )
        runs.append({"seed": seed, **{k: float(res[k]) for k, _ in METRICS}})
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate ──────────────────────────────────────────────────────────
    def col(key):
        return np.array([r[key] for r in runs], dtype=float)

    print(f"\n\n{'='*70}\n  STABILITY SUMMARY — {args.model}  ({len(seeds)} seeds)\n{'='*70}")
    header = "  seed  " + "".join(f"{lbl:>9}" for _, lbl in METRICS) + f"{'FPR':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in runs:
        fpr = 1.0 - r["test_specificity"]
        row = f"  {r['seed']:>4}  " + "".join(f"{100*r[k]:>8.2f} " for k, _ in METRICS)
        row += f"{100*fpr:>8.2f}"
        print(row)

    print("  " + "-" * (len(header) - 2))
    summary = {"model": args.model, "seeds": seeds, "n_seeds": len(seeds),
               "epochs": args.epochs, "per_seed": runs, "aggregate": {}}
    for k, lbl in METRICS:
        v = col(k)
        summary["aggregate"][k] = {
            "mean": float(v.mean()), "std": float(v.std(ddof=1) if len(v) > 1 else 0.0),
            "min": float(v.min()), "max": float(v.max()),
        }
    fpr = 1.0 - col("test_specificity")
    summary["aggregate"]["test_fpr"] = {
        "mean": float(fpr.mean()), "std": float(fpr.std(ddof=1) if len(fpr) > 1 else 0.0),
        "min": float(fpr.min()), "max": float(fpr.max()),
    }

    def line(label, stat):
        cells = "".join(f"{100*summary['aggregate'][k][stat]:>8.2f} " for k, _ in METRICS)
        cells += f"{100*summary['aggregate']['test_fpr'][stat]:>8.2f}"
        print(f"  {label:>4}  {cells}")

    line("mean", "mean")
    line("std", "std")
    line("min", "min")
    line("max", "max")

    out = output_dir / f"stability_{args.model}_{len(seeds)}seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {out}")


if __name__ == "__main__":
    main()
