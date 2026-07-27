# Baseline B1 — ArcSSM under leave-one-campaign-out

**Frozen reference point.** Every later change to the SSM track is measured against
this run with `compare_groupcv.py`. Do not edit the numbers below; add a new
baseline file (B2, B3, …) when a change is adopted.

- Date: 2026-07-26
- Code: commit `73c4b5c` + the group-CV additions to `train.py`
  (campaign grouping, `--val-mode`, out-of-fold prediction saving)
- Run: `runs/arcssm_groupkfold_campaign_20260726_195946`
- Duration: 124.5 min (RTX 3050 Laptop, 4 GB)

## 1. Architecture (`model_ssm.py: ArcSSMNet`, 359 553 parameters)

| Stage | Definition |
|---|---|
| Input | `x_1d` (B, 4, 2048) — `x_2d` (STFT) is **ignored**: this track is SSM-only, no spectral branch |
| Front-end | `i_derived4`: per-cycle RMS-normalised `[I, \|ΔI\|, TKEO, RMS_slide]` derived from I(t) only (`dataset.py:_derive_i_channels`) |
| Encoder | `Conv1d(4 → 128, kernel=7, padding=3)` + GELU, full temporal resolution |
| Backbone | 4 × `S4Block(d_model=128, d_state=64, bidirectional=True, selective=False, block_dropout=0.1)` |
| Head | `LayerNorm(128)` → mean-pool over time → `Linear(128, 128)` → classifier `Linear(128, 64)`+ReLU+`Dropout(0.3)`+`Linear(64, 1)` |
| Output | single logit, `BCEWithLogitsLoss`, threshold 0.5 |

Signal: 2048 samples/cycle at 102.4 kHz (decimated from 1 MHz), n_fft 512 / hop 256
(unused by this model). Augmentation active during training: additive Gaussian noise
at 0.005·std per channel, spectrogram frequency masking (also unused here),
`--channel-dropout 0.0` (**off**).

## 2. Training configuration

```
python train.py --model arcssm --mode groupkfold --group-level campaign \
  --data-dir combined_dataset_2048/combined_dataset_2048 \
  --output-dir runs --epochs 60 --patience 10 --batch-size 32 \
  --n-fft 512 --hop-length 256 --num-workers 4 --seed 42
```

AdamW, lr 3e-4, weight decay 5e-4, `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`,
gradient clip 0.5, label smoothing 0.05, early stopping on val F1 with patience 10,
`--val-mode alternance` (auto). Fold seeds 42–45.

## 3. Results

Protocol: leave-one-campaign-out, 4 folds, threshold 0.5. Full report with figures:
`runs/arcssm_groupkfold_campaign_20260726_195946/eval/results.md`.

**Pooled out-of-fold (all 10 860 cycles, each scored by a model that never saw its campaign)**

| Metric | Value |
|---|---|
| Accuracy | 81.28 % |
| F1 | 79.63 % |
| Precision | 79.19 % |
| Recall | 80.07 % |
| Specificity | 82.30 % |
| ROC AUC | 0.8872 |
| Counts | TP 3973 · FP 1044 · FN 989 · TN 4854 |

**Per fold**

| Fold | Held out | n | Acc % | F1 % | Prec % | Rec % | Spec % | AUC | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 15_juillet_clean | 2820 | 73.16 | 77.23 | 64.01 | 97.35 | 51.90 | 0.9119 | 22 |
| 2 | 22_juillet_clean | 3820 | 88.85 | 87.52 | 91.26 | 84.07 | 93.00 | 0.9076 | 17 |
| 3 | 8_juillet_clean | 2746 | 75.71 | 65.46 | 100.00 | 48.65 | 100.00 | 0.8804 | 26 |
| 4 | OthmaneSalim10052026 | 1474 | 87.58 | 86.02 | 75.88 | 99.29 | 80.26 | 0.9963 | 16 |

Mean ± std across folds: acc 81.32 ± 6.96, F1 79.06 ± 8.78, recall 82.34 ± 20.31,
specificity 81.29 ± 18.39. **Worst campaign: 8_juillet, F1 65.46 %.**

Reference ceiling — same architecture, random 70/15/15 cycle-level split
(`runs/arcssm_single_20260726_150603`): acc 98.47 %, F1 98.33 %. The 17-point gap is
the cost of the leakage in that split plus the campaign shift.

## 4. Diagnostics recorded with this baseline

| Campaign | AUC | acc @ 0.5 | acc @ oracle thr | oracle thr | mean p(arc) on normal | mean p(arc) on arc |
|---|---|---|---|---|---|---|
| 15_juillet | 0.9119 | 73.16 % | 88.40 % | 0.87 | 0.450 | 0.935 |
| 22_juillet | 0.9076 | 88.85 % | 90.42 % | 0.27 | 0.114 | 0.791 |
| 8_juillet | 0.8804 | 75.71 % | 84.20 % | 0.06 | 0.047 | 0.497 |
| 2026 | 0.9963 | 87.58 % | 99.46 % | 0.95 | 0.226 | 0.973 |

- Mean per-fold AUC **0.9240** vs pooled AUC **0.8872** — pooling *lowers* AUC, the
  signature of score distributions that are offset between campaigns.
- Pooled accuracy with a per-campaign oracle threshold: **89.55 %** vs 81.28 % at a
  fixed 0.5. So ≈ 8 of the 17 lost points are decision-boundary placement, and
  ≈ 9 points are genuine representation shift.
- Early-stopping val F1 was 90.6–99.1 % while held-out campaign F1 was 65–88 %:
  the selection signal is in-domain and does not track cross-campaign quality.
- Label protocol is not a confound: every arc-labelled cycle has `arc_ratio ≥ 0.8`
  in all four campaigns.

## 5. Artifacts

| Path | In git? |
|---|---|
| `runs/.../groupkfold_summary.json` | yes |
| `runs/.../oof_predictions.npz`, `fold_*/test_predictions.npz` | yes |
| `runs/.../eval/*` (results.md, figures, FP/FN CSVs) | yes |
| `runs/.../fold_*/best_fold_*.pt` | **no** — `*.pt` is gitignored; the weights live only on the machine that trained them |

The saved probabilities are enough to recompute every metric, ROC and threshold in
this file without the checkpoints.

## 6. Comparing a change against this baseline

```bash
# after training a variant with the SAME protocol command
python eval_groupcv.py runs/<variant_run> --data-dir combined_dataset_2048/combined_dataset_2048
python compare_groupcv.py \
  runs/arcssm_groupkfold_campaign_20260726_195946 runs/<variant_run> \
  --labels B1-baseline "<what-changed>" \
  --out docs/baselines/<variant>_vs_B1.md
```

Judge a change on **worst-campaign F1** and the **McNemar p-value**, not on pooled
accuracy alone. Before chasing deltas smaller than a few points, measure the seed
noise floor (`stability_eval.py`) — a change inside that band is not a result.
