# Arc-FaultNet V2 — cross-campaign generalization: status & plan

**Scope:** `--model arcfaultnet_v2`, `--mode groupkfold --group-level campaign`
(leave-one-acquisition-campaign-out). Last updated **2026-07-29**.

Companion doc: [`generalization_plan.md`](generalization_plan.md) /
[`docs/generalization_plan.md`](docs/generalization_plan.md) cover the **ArcSSM**
track against baseline B1. This file is the V2 track and supersedes nothing there —
the two models are evaluated separately.

---

## 0. TL;DR — where you are right now

| | state |
|---|---|
| Honest evaluation harness | ✅ **built and validated** (Phase 0+1 in `train.py`) |
| Full provenance logging | ✅ **added 2026-07-30** — `fs`, `n_fft`, `hop_length`, `channel_mode`, `data_dir`, `monitor`, `fbeta`, `n_recordings` now in `groupkfold_summary.json` |
| Honest baseline | ✅ pooled ROC-AUC **0.906–0.914**, pooled F1 **83 %** (4 configs all land here) |
| Single-mode 98 % | ❌ **discarded — leaky**, do not report |
| Unsafe recall (`--fbeta 0.5`) | ✅ **fixed** by β = 1.0 — worst-fold recall 62.9 % → 67.3 % |
| Cross-campaign stability | ✅ **best yet in R1** — F1 std 7.5 % → **4.85 %**, least score drift of any run |
| Domain-generalization stack | ❌ **no gain** (run C), and it caused under-training |
| Decision-layer smoothing / Otsu | ❌ **measured and ruled out** for V2 (§3.5) |
| Label noise | ❌ **ruled out** — `arc_ratio` is cleanly separated (§3.5) |
| Noise floor (seed variance) | ✅ **measured, then fixed: σ 0.0271 → 0.0072** via `--lr-scheduler cosine --monitor val_pr_auc` (§7 N1b) |
| `fs` fix | ⚠️ **not measurable** (+0.0044 ≪ 2σ) — keep for correctness, not as a result |
| **Best honest config** | ✅ `cosine` + `val_pr_auc`, **3-seed ensemble** — see below |
| **Discrimination ceiling** | 🔴 **stuck**: pooled AUC ≈ 0.915 single seed / **0.935 ensemble**, flat across six configs |
| Operating point | ✅ **recall ≥ 88 % AND spec ≥ 88 % is reachable** (89.55 / 88.88 / 90.12 at thr 0.34) — but the threshold must come from val against a stated requirement (§7 N2b) |

**Two findings that reorganise everything:**

1. **σ = 0.0271 on pooled AUC.** A seed swing (0.8824 → 0.9364) is bigger than every
   config effect we chased, so none of the earlier A/B/C/R1 comparisons were real (§7 N1).
   Its root cause is early stopping landing anywhere from epoch 7 to 182.
2. The residual error is **not** a campaign-level shift — it sits in **22 recordings =
   8.8 % of the data across all four campaigns**, holding ~⅓ of all errors (§3.6). That is
   why CORAL/DRO, smoothing and threshold adaptation all produced nothing.

**Immediate next action:** **N1b** (§7) — stabilise early stopping (smoother LR schedule or
fixed epoch budget). It is the only change that shrinks σ, and until σ shrinks no further
experiment can be interpreted.

---

## 1. Dataset ground truth (settled — do not re-derive)

Terminology was a source of error; this is the verified hierarchy.

| term | definition |
|---|---|
| **alternance** (électrique) | half a période. **Not stored as a unit anywhere.** |
| **cycle = période = segment** | **one model input** = 2048 pts = one ~20 ms mains period = 2 alternances |
| **recording** | one CSV / one continuous LeCroy capture (arc created → captured: first part no-arc, later part arc) |
| **campaign** | the `dataset` column — one acquisition day/bench |

- `X_multi.npy` = `(10860, 2, 2048)`, channels `[V_ligne, I]`; `y` = 5898 normal / 4962 arc.
- **`alt_index` is misnamed**: it is the *période-slot index inside a recording*, **not**
  an electrical alternance. `load_alternance_ids()` groups "same slot across recordings",
  which does **not** keep a recording intact → that was the val leak (§4.2).
- Pipeline: create arc → LeCroy capture → clean ambiguous parts → segment into périodes
  → label via the `Varc > seuil` quotient (doubtful cycles **removed**) → decimate
  20000 → 2048 pts (`resample_poly`, up=64 down=625).

**Campaigns and recovered recordings** (238 total, via `load_recording_ids()`):

| campaign | cycles | recordings | storage convention |
|---|---|---|---|
| `15_juillet_clean` | 2820 | 65 | one `exp_name` (`exp12--IJL--LR`), many recordings in temporal order → `alt_index` sawtooth |
| `22_juillet_clean` | 3820 | 88 | same |
| `8_juillet_clean` | 2746 | 65 | same |
| `OthmaneSalim10052026` | 1474 | 20 | 20 `exp_name`s = 20 load configs = 20 recordings, périodes stored **shuffled** → `exp_name` *is* the recording id |

> Note: the folder is `8_juillet_clean` (sometimes referred to verbally as "18 July") —
> confirm the true date before it goes in the paper.

---

## 2. Run ledger (all numbers verified from saved artifacts)

| id | run dir | val split | `fs` | monitor | patience | DG stack |
|---|---|---|---|---|---|---|
| **S** | `arcfaultnet_v2_single_*` ×5 seeds | random cycle split | 1 MHz | — | — | no |
| **A** | `..._groupkfold_campaign_20260729_115802` | `alternance` (leaky) | 1 MHz | β 0.5 | 15 | no |
| **B** | `..._groupkfold_campaign_20260729_172929` | `recording` ✅ | 1 MHz ❌ | β 0.5 | 15 | no |
| **C** | `..._groupkfold_campaign_20260729_182029` | `recording` ✅ | 102.4 kHz ✅ | β 0.5 | 15 | **yes** |
| **R1** | `..._groupkfold_campaign_20260729_184723` | `recording` ✅ | **unknown** 🔴 | β 1.0 | 25 | no |

### Headline metrics

| metric | S (leaky) | A | B | C | **R1** |
|---|---|---|---|---|---|
| pooled accuracy | 98.65 % | 86.17 % | 84.01 % | 84.93 % | 84.02 % |
| pooled F1 @0.5 | 98.54 % | 85.10 % | 83.56 % | 82.05 % | 83.27 % |
| pooled precision | 99.51 % | 83.80 % | 78.82 % | 89.99 % | 79.83 % |
| pooled recall | 97.59 % | 86.44 % | 88.90 % | 75.39 % | 87.02 % |
| pooled specificity | — | 85.94 % | 79.91 % | 92.95 % | 81.50 % |
| pooled ROC-AUC | — | 0.9010 | **0.9135** | 0.9085 | 0.9063 |
| mean per-fold ROC-AUC | — | 0.8966 | 0.9408 | **0.9448** | 0.9176 |
| **AUC std across campaigns** | — | 0.045 | 0.045 | 0.044 | **0.0212** ✅ |
| **F1 std across campaigns** | — | 7.55 % | 7.52 % | 7.03 % | **4.85 %** ✅ |
| worst-fold recall | — | — | — | 62.9 % 🔴 | **67.3 %** |
| drift (z-scoring recovers) | — | +0.0300 | +0.0257 | — | **+0.0129** ✅ |
| best epochs | — | 55/66/26/28 | 29/27/33/30 | 21/11/45/10 | 54/90/37/26 |

### Per-campaign ROC-AUC

| held-out campaign | A | B | C | **R1** |
|---|---|---|---|---|
| `15_juillet` | 0.9621 | **0.9948** | 0.9753 | 0.9252 |
| `22_juillet` | 0.9689 | 0.9321 | 0.9239 | 0.9200 |
| `8_juillet` | **0.7707** | 0.8723 | 0.8830 | **0.8835** |
| `OthmaneSalim` | 0.8847 | 0.9641 | **0.9968** | 0.9416 |

### Are these differences real? (paired tests, same 10 860 cycles, identical folds)

| comparison | McNemar @0.5 | paired bootstrap on pooled AUC |
|---|---|---|
| A vs B | A better, p = 4.2e-12 | B − A = **+0.0125** [+0.0078, +0.0174] |
| **B vs R1** | **tie: 526 vs 527, p = 1.00** | R1 − B = **−0.0072** [−0.0123, −0.0017] |
| A vs R1 | A better, p = 7.8e-16 | R1 − A = **+0.0053** [+0.0012, +0.0095] |

⚠️ These CIs capture *sampling* noise only. **The seed has never been varied** (all runs use
fold seeds 42–45 with `cudnn.deterministic=True`), so we cannot tell whether re-running an
unchanged config lands within ±0.005 or ±0.04. Meanwhile the same campaign's AUC moves by
**median 0.043 / max 0.113** between configs — roughly 6× the effects being compared.
**Nothing is attributable until N1 runs.**

---

## 3. Diagnosis — three established facts

### 3.1 Single mode leaks ~15 points. Never report it.
`run_single_training()` ([train.py:741](train.py:741)) shuffles all 10 860 cycles and
deals them into train/val/test. Near-duplicate périodes from the same recording land on
both sides, so the model recognises twins rather than detecting arcs.
**98.5 % → 83.6 %** is the leakage being removed, not the model degrading.

### 3.2 The leaky val was *causing* overfitting (A → B)
With `val-mode alternance`, early stopping ran to epochs 55/66 on folds 1–2 while the
inflated val still said "improving". With `recording`, best epochs tighten to 29/27/33/30
and **mean AUC rises 0.8966 → 0.9408**; `8_juillet` gains **+0.10** (0.771 → 0.872).
So a large part of what looked like a *capacity* failure on 8 July was an
early-stopping artifact.

### 3.3 Val-based threshold calibration cannot work here — structural, not a bug
Measured on run B: val-chosen thresholds vs test-optimal thresholds are essentially
uncorrelated.

| campaign | val-chosen thr | test-optimal thr | F1 left on table |
|---|---|---|---|
| `15_juillet` | 0.38 | 0.82 | +0.042 |
| `22_juillet` | 0.57 | 0.86 | +0.043 |
| `8_juillet` | 0.67 | 0.07 | +0.051 |
| `OthmaneSalim` | 0.40 | 0.92 | **+0.218** |

Pooled F1 @0.5 = **83.56 %**, @val-chosen = **83.56 %** → **zero gain**.

**Why:** the threshold drift is *caused by* the shift between training campaigns and the
unseen campaign. Val is drawn from the **training** campaigns, i.e. in-distribution, so it
cannot observe the drift it is meant to correct. Keep reporting the val-chosen operating
point (it is the honest deployable protocol, and it is free) but **claim no gain from it**.

### 3.4 The drift is a pure per-campaign offset, and the headroom is quantified
On run B:

| quantity | value |
|---|---|
| mean per-fold AUC (each campaign on its own scale) | 0.9408 |
| pooled AUC (all campaigns on one shared scale) | 0.9135 |
| pooled AUC after **per-campaign z-scoring** | **0.9392** |

z-scoring recovers almost the whole gap ⇒ the loss is an **offset/scale slide, not a
ranking failure**. And:

- pooled F1 honest (@0.5 or @val) — **0.8356**
- pooled F1 @test-optimal threshold (**oracle, never reportable**) — **0.8987**

⇒ **≈ 6.3 F1 points are locked behind the operating point alone.**

---

### 3.5 What has been measured and RULED OUT (do not re-attempt without new evidence)

All measured on V2's own saved predictions, no retraining.

| lever | result | verdict |
|---|---|---|
| **Val-based threshold selection** | pooled F1 83.56 % @0.5 vs 83.56 % @val (run B); R1: 83.27 % vs 82.74 % | ❌ zero gain — structural (§3.3) |
| **Training-side DG stack** (`--group-dro --coral-weight 0.5 --strong-aug --channel-dropout 0.2 --dg-balanced-sampler`) | run C: pooled AUC 0.9085 vs B 0.9135; drift *widened*; under-training, fold-2 recall 62.9 % | ❌ no gain + unsafe side-effect |
| **Multi-cycle score smoothing** (trailing mean of K consecutive périodes within a recording, ordered by `alt_index`) | pooled AUC K=1 **0.9063** → K=2 0.9068 → K=4 0.8999 → K=6 0.8910 → K=8 0.8816; accuracy 84.02 % → 79.91 % | ❌ **degrades** past K=2 |
| **Unsupervised per-campaign Otsu threshold** | pooled acc 84.06 % vs 84.02 % @0.5 | ❌ +0.04 pt — nothing to fix, R1's scores are already centred |
| **Label ambiguity / borderline cycles** | `arc_ratio`: label 0 ∈ [0, 0.041], label 1 ∈ [0.950, 0.997] — **no overlap at all** | ❌ ruled out; the `Varc > seuil` cleaning was thorough |

**Why smoothing and threshold tricks cannot work here** — the errors are not per-cycle noise:

- **57.4 %** of recordings (≥5 cycles) are almost entirely right or entirely wrong.
- lag-1 autocorrelation of the error signal inside a recording = **0.423** → errors arrive
  in long runs, so averaging a cycle with its neighbours averages *equally wrong* neighbours.
- mean within-recording error rate 0.159.

### 3.6 Where the residual error actually lives — the concrete target

**22 recordings (≥5 cycles) have >50 % error: 959 cycles = 8.8 % of the dataset, holding
roughly a third of all errors.** They are spread across **every** campaign
(`22_juillet` 10, `8_juillet` 6, `15_juillet` 4, `OthmaneSalim` 2) — so this is a
**recording-level** failure mode, **not** a bench/campaign shift. Two are 100 % wrong on
*pure, strong* arc recordings (48 and 47 cycles, `arc_ratio` ≈ 0.97–0.98):

| err | n | arc frac | recording |
|---|---|---|---|
| 100 % | 48 | 1.00 | `8_juillet_clean\|exp11--IJL--LR\|run54` |
| 100 % | 47 | 1.00 | `22_juillet_clean\|exp13--IJL--LR\|run17` |
| 85 % | 26 | 0.08 | `22_juillet_clean\|exp13--IJL--LR\|run67` |
| 72 % | 32 | 0.28 | `15_juillet_clean\|exp12--IJL--LR\|run1` |
| 70 % | 76 | 0.24 | `OthmaneSalim10052026\|AcierCu_Kettle+Halogene` |
| 68 % | 37 | 1.00 | `8_juillet_clean\|exp11--IJL--LR\|run56` |

**This reframes the whole problem.** Every lever tried so far targets *campaign-level*
shift (align campaigns, adapt the per-campaign threshold, smooth within a campaign). The
actual residual is a *subpopulation of recordings* the model fails on in all four
campaigns. That is why they all returned nothing — and it is where the remaining ~16 points
of accuracy live.

### 3.7 The discarded HF band does carry signal (partial support for the `fs` fix)

Energy fraction of the current spectrum **above 10.4 kHz** — the band the wrong `fs = 1 MHz`
gate threw away (§4.1):

| group | mean HF fraction |
|---|---|
| arc cycles, well-classified recordings | 0.0167 |
| normal cycles, well-classified recordings | 0.0006 |
| arc cycles, systematically-missed recordings | 0.0328 |
| normal cycles, systematically-missed recordings | 0.0018 |

Using **only** that discarded band as a one-feature classifier gives
**AUC 0.7745** on well-classified recordings (0.6074 on the missed ones).

Two conclusions: (a) the band the mis-set `fs` discarded holds genuine, unexploited
discriminative information → **fixing `fs` is a real lever**; (b) it does **not** explain
the 22 missed recordings — there HF is *elevated* but *less* class-discriminative, so
restoring the band alone will not rescue them.

### 3.8 The ceiling, located exactly (2026-07-30) — and why no big lever exists

Run on the best 3-seed ensemble. Two corrections to earlier reasoning first: multi-cycle
aggregation had only been tested with **mean** pooling, and score normalisation had only been
tested **per campaign**. Both gaps are now closed.

**(a) The residual failure is a per-RECORDING score offset, not a discrimination failure.**
For the 15 systematically-failing recordings the model assigns a nearly *constant* score to
every cycle regardless of label — e.g. `22_juillet|run22` gives p(arc)=0.949 / p(norm)=0.956
(all predicted arc), `8_juillet|run53` gives 0.055 / 0.060 (all predicted normal). Yet
**8 of the 13 mixed-label failures rank correctly inside the recording** (AUC 0.74, 0.83,
0.87, 0.87, 0.90, 0.90, 0.93, **1.000**). `run20` has *perfect* within-recording ranking and
still 53 % error, purely because both classes sit above 0.5. This is the same offset drift
found at campaign level in §3.4, but at recording level and far more severe.

**(b) The model's TRUE discriminative power: mean within-recording AUC = 0.9478**
(226 mixed-label recordings, offset-free by construction). *That is the real ceiling.* It
caps accuracy at roughly **91–92 %** even with a perfect offset-correction oracle.

**(c) Every lever, measured:**

| approach | AUC | balanced acc | verdict |
|---|---|---|---|
| raw ensemble (current best) | 0.9349 | **89.24 %** | baseline |
| multi-cycle **mean** pool, K=2 | 0.9369 | 89.31 % | +0.1 — nothing |
| multi-cycle **max** pool, K=4 | 0.9392 | 89.30 % | +0.1 — nothing |
| multi-cycle **p75** pool, K=2 | 0.9378 | 89.51 % | +0.3 — inside σ |
| **per-recording z-score (full recording)** | **0.9568** | **91.85 %** | **+2.6 — but not deployable, see below** |
| per-recording median-centred | 0.9564 | 88.58 % | ranking up, accuracy down |
| rolling causal median, W=100 | 0.9215 | 88.08 % | **worse than baseline** |
| rolling causal median, W=50 | 0.9151 | 86.94 % | worse |
| rolling causal median, W=20 | 0.7335 | 64.21 % | catastrophic |

**(d) Why the one apparent win does not count.** Per-recording z-scoring lifts the failing
subset from AUC 0.483 → 0.7375 and pooled accuracy to 91.85 %. But it (i) needs the *entire*
recording including future cycles, and (ii) implicitly injects the per-recording class
balance — most recordings are ≈ 50/50 arc/normal (e.g. 24 arc / 23 normal), so centring
forces about half the cycles to each side of the threshold. That is label-prior leakage, and
it would *destroy* a pure-arc recording the model currently gets right. The honest, causal
version is the rolling-median row — and **it performs worse than doing nothing**.

**Conclusion.** Remaining headroom is ≈ 2–3 accuracy points (89.2 → 91–92 %), it is entirely
per-recording offset, and no deployable correction for it exists in this data. Fixing the 15
failing recordings outright would give **93.44 %** — that is the absolute ceiling. There is
no training-recipe, aggregation, or calibration lever left worth more than ≈ 0.3 points.
**Phase G (more benches) is the only remaining path to a large gain**, because the binding
constraint is now the model's within-recording AUC of 0.948, which is a data problem.

Report (a) and (b) in the paper — they are a genuine diagnostic contribution: the failure is
localised, characterised, and shown not to be a threshold or aggregation artifact.

## 4. Open blockers

### 4.1 🔴 `fs` / FrequencyGate bug — invalidates the spectral branch of runs S, A, B
`combined_dataset_2048/config.json` is **0 bytes (empty)**, so `ArcFaultDataset` fell back
to the `fs = 1_000_000` default. The real config lives in the **nested**
`combined_dataset_2048/combined_dataset_2048/config.json`: `FS: 102400`,
`SAMPLES_PER_CYCLE: 2048`, decimated from `original_fs: 1000000` / 20000 samples.
**102 400 Hz is correct** (2048 pts per 20 ms période).

`FrequencyGate` ([model.py:318](model.py:318)) does `bin_res = fs / n_fft` and keeps
2 kHz–100 kHz:

| run | assumed bin_res | bins kept | band actually seen |
|---|---|---|---|
| S, A, B (`fs`=1 MHz) | 7812 Hz | **13 of 65** | 0.8 – 10.4 kHz |
| C (`fs`=102.4 kHz) | 800 Hz | **63 of 65** | 1.6 – 51.2 kHz (Nyquist) |

Every pre-fix run fed the spectral branch **13 of 65 bins**, blind above 10.4 kHz where
much of the arc HF signature lives.

**Mitigation (mandatory from now on):** always pass `--fs 102400` **explicitly** and use
the nested data dir. Do not rely on config auto-detection.

### 4.2 ✅ Val leak — fixed
`--val-mode recording` (now the `auto` default) holds out whole recordings.

### 4.3 🔴 Under-training caused by `--fbeta 0.5`
In run C best epochs collapsed to **21 / 11 / 45 / 10** (train_loss still 0.57–0.59).
Fold 2 became pathological: **recall 62.86 %**, precision 99.64 % — an AFDD missing 37 %
of real arcs, i.e. failing in the *unsafe* direction.

Cause: β = 0.5 weights precision 4×, so a barely-firing model scores high `val_fbeta`
very early; the CORAL/DRO terms slow convergence; and
`CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` produces val crashes that trip
`patience=15`. **The apparent "precision gain" in run C is substantially under-training,
not robustness.**

### 4.4 ⚠️ Run C is confounded — two variables changed at once
C changed **both** `fs` (bug fix) **and** the DG stack. Its deltas cannot be attributed.
Also, within C the drift did **not** shrink: mean−pooled AUC gap **0.036** (B: 0.027), so
**CORAL did not achieve its stated goal** on that run.

---

## 5. What is already implemented in `train.py`

Phase 0 + 1, validated end-to-end (1-epoch × 4-fold smoke test + helper unit checks):

- `load_recording_ids()` — recovers the true recording id, auto-detecting both storage
  conventions (§1). Verified: 238 recordings, none spanning a campaign.
- `--val-mode recording` (+ `auto` now selects it) with a hard leakage assertion.
- `_auc_scores()`, `_metrics_from_probs()`, `_select_threshold()`.
- Per-fold `val_predictions.npz`; `oof_predictions.npz` gains `pred_cal`.
- ROC-AUC + PR-AUC logged per fold and pooled; per-fold output prints `@0.50`,
  `@thr*` (val-chosen), and both AUCs; summary adds threshold-free and
  val-calibrated mean ± std blocks plus a pooled calibrated operating point.

**Pending code change (proposed, ~10 lines):** add **`val_pr_auc`** to `_VALID_MONITORS`
so early stopping is threshold-free — it stops being hostage to the operating-point drift
proven in §3.3. Optionally make `fs` fail loudly instead of silently defaulting.

---

## 6. Success criteria

| metric | B | R1 | N1 single seed | **N1 3-seed ensemble** | target | met? |
|---|---|---|---|---|---|---|
| pooled ROC-AUC | 0.9135 | 0.9063 | 0.9107 ± 0.0271 | **0.9365** | ≥ 0.94 | ~0.94, essentially met |
| worst-campaign ROC-AUC | 0.8723 | 0.8835 | 0.8406 ± 0.0515 | **0.890** | ≥ 0.85 | ✅ |
| pooled F1 | 83.56 % | 83.27 % | 84.15 ± 3.38 % | **86.06 %** | ≥ 90 % | ✗ |
| pooled accuracy | 84.01 % | 84.02 % | 84.62 ± 4.06 % | **86.66 %** | — | — |
| recall **and** specificity | 88.9 / 79.9 | 87.0 / 81.5 | 88.7 / 81.2 | **90.1 / 83.7** | both ≥ 88 % | ✅ / ✗ |

The ensemble is the best honest configuration to date and clears the worst-campaign target
with every campaign ≥ 0.89. The remaining gaps are **specificity** (83.7 % vs 88 %) and
**F1** (86.1 % vs 90 %) — both bounded by the 22 hard recordings (§3.6), not by the recipe:
single-seed pooled AUC is flat at 0.91 ± 0.03 across five very different configs.

Report pooled **and** per-campaign. A mean alone hides the weak campaign.

---

## 7. Next phases

Order revised 2026-07-30 after §3.5–3.7. Rationale: campaign-level levers are exhausted;
the target is now the noise floor, the `fs` fix, and the 22 hard recordings.

### N1 — noise floor + `fs` fix, in one batch (**do this first**, ~75 min)
Three seeds of one fixed config with `fs` **explicit**. This is the gate for everything
else: it produces the error bar that makes all future comparisons interpretable, *and* the
`fs`-fixed number, *and* resolves R1's unknown `fs` (§4.1). Provenance is now logged, so
these runs are self-documenting.

```bash
for S in 42 142 242; do python train.py --model arcfaultnet_v2 --mode groupkfold --group-level campaign --data-dir combined_dataset_2048 --fs 102400 --n-fft 128 --hop-length 64 --deep-clf --fusion-mode cross_attention --monitor val_fbeta --fbeta 1.0 --epochs 200 --patience 25 --seed $S; done
```

> **`--data-dir` differs per machine.** On `ijl-expe-209` the arrays live in
> `combined_dataset_2048/` (its `config.json` is absent/empty, which is why `fs` silently
> defaulted to 1 MHz); on the laptop they live in the nested
> `combined_dataset_2048/combined_dataset_2048/`. Point `--data-dir` at whichever holds
> `X_multi.npy`, and **always pass `--fs 102400`** — the CLI value overrides `config.json`,
> so the command is correct on both. Confirm the header prints `Signal: fs=102,400 Hz`.

**✅ RUN 2026-07-30.** `115553` (s42), `123602` (s142), `130648` (s242) — all with
`fs=102400` logged, 238 recordings.

| | seed 42 | seed 142 | seed 242 | mean ± σ |
|---|---|---|---|---|
| pooled ROC-AUC | 0.8824 | 0.9132 | 0.9364 | **0.9107 ± 0.0271** |
| pooled F1 @0.5 | 80.38 % | 85.17 % | 86.90 % | 84.15 ± 3.38 % |
| pooled accuracy | 80.04 % | 86.08 % | 87.74 % | 84.62 ± 4.06 % |
| pooled recall / spec | 89.50 / 72.08 | 87.48 / 84.89 | 89.00 / 86.69 | 88.66 ± 1.05 / 81.22 ± 7.97 |
| worst-campaign AUC | 0.8613 | 0.7819 | 0.8786 | 0.8406 ± 0.0515 |
| best epochs | 62/182/33/46 | 26/110/60/36 | **7/8**/40/67 | — |

Per-campaign AUC (mean ± σ): `15_juillet` 0.9929 ± 0.0061 · `22_juillet` 0.9252 ± 0.0569 ·
`8_juillet` 0.8442 ± 0.0541 · `OthmaneSalim` 0.9504 ± 0.0377.

### 🔴 N1 verdict: **σ = 0.0271, and it voids every config comparison made so far**

| comparison | Δ pooled AUC | vs 2σ = 0.0541 |
|---|---|---|
| `fs` fix (N1 mean 0.9107 − R1 0.9063) | **+0.0044** | ❌ not significant |
| val-leak fix (B − A) | +0.0125 | ❌ inside σ |
| β/patience (R1 − B) | −0.0072 | ❌ inside σ |
| DG stack (C − B) | −0.0050 | ❌ inside σ |

**One seed swing (0.8824 → 0.9364, spread 0.054) is larger than every config effect we
chased.** Keep the val-leak fix and `--fs 102400` on **correctness** grounds — a recording
split across train/val is wrong, and 102 400 Hz is the true rate — but neither can be
claimed as a measured improvement. **Report ± σ on every number.**

### Root cause of σ: early stopping is effectively random

Best epoch for the *same fold* ranges **7 → 182** across seeds (fold 2: 182 / 110 / 8).
`CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` restarts at epochs 10/30/70/150 and drives
val crashes (val_acc dips to ~50 %), so "best epoch" is whichever restart the seed happened
to land on. Fixing this is now the highest-value change: it shrinks σ, which both raises the
floor and makes every future experiment interpretable.

### ✅ The one real, reproducible win: 3-seed probability ensemble

| | pooled AUC | acc | F1 | recall | spec |
|---|---|---|---|---|---|
| single seed (expected) | 0.9107 ± 0.0271 | 84.62 % | 84.15 % | 88.66 % | 81.22 % |
| **3-seed ensemble** | **0.9365** | **86.66 %** | **86.06 %** | **90.12 %** | 83.74 % |
| luckiest seed (242) — *not selectable* | 0.9364 | 87.74 % | 86.90 % | 89.00 % | 86.69 % |

Per-campaign AUC: `15_juillet` **0.998** · `22_juillet` **0.973** · `8_juillet` **0.890**
(best ever) · `OthmaneSalim` **0.984** — the best profile of any run, every campaign ≥ 0.89.

**Honest reading:** the ensemble does **not** beat the luckiest seed (McNemar: seed 242 wins
493 vs 375, p = 7e-5). But **you cannot pick the best seed** — that is selection on the test
result. The ensemble's value is that it delivers ≈ the luckiest seed's score
**deterministically, without the lottery**, beating the *expected* single seed by +0.026 AUC
and +2.0 accuracy. That is the number to report.
Caveat: with only 3 seeds the ensemble's *own* variance is unmeasured. Also note 2 seeds
(142+242) scored 0.9409 / 88.14 % — but that pairing was chosen post-hoc, so it is not
reportable.

**Threshold note:** the val-chosen threshold *hurts* the ensemble (84.49 % vs 86.66 % @0.5),
re-confirming §3.3. Report the ensemble **at 0.5**.

### N1b — stabilise early stopping (**now the top priority**, ~25 min per variant)
σ = 0.0271 is driven by "best epoch" landing anywhere from 7 to 182 (§ above). Until this
shrinks, no experiment is interpretable and the floor stays low. Three candidates, cheapest
first — each needs a small code change in `train_model`:

1. **Drop the warm restarts.** Replace `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` with
   plain `CosineAnnealingLR` (or `ReduceLROnPlateau`). The restarts at 10/30/70/150 are what
   create the val crashes that randomise the checkpoint.
2. **Fixed epoch budget, no early stopping.** Pick one budget from the val curves *once*
   (≈ 40–60 epochs), apply to every fold. Removes the checkpoint lottery entirely; the
   honest option when val cannot rank checkpoints reliably (§3.3).
3. **Monitor `val_pr_auc`** (threshold-free, §5) instead of F1 at 0.5 — far less jumpy,
   since val F1 saturates at 97–99 % where tiny fluctuations decide the checkpoint.

**Success = σ shrinks.** Re-measure with the same 3 seeds; that is the only acceptance test.

#### ✅ N1b RESULT (2026-07-30) — **ADOPTED, but only for σ**

Runs `151449` (s42), `135419` (s142), `141958` (s242) with
`--lr-scheduler cosine --monitor val_pr_auc`. (`cosine` was already implemented and
committed; only the `val_pr_auc` monitor was new. Both knobs moved together, so their
individual contributions are still unattributed.)

**Part 1 — σ: PASS, decisively.**

| | warm_restarts + fbeta | cosine + val_pr_auc | change |
|---|---|---|---|
| pooled AUC per seed | 0.8824 / 0.9132 / 0.9364 | 0.9086 / 0.9157 / 0.9231 | floor +0.026 |
| **σ (AUC)** | 0.0271 | **0.0072** | **3.7× tighter** |
| **σ (accuracy)** | 4.06 pt | **1.38 pt** | **2.9× tighter** |

2σ is now **0.014** instead of 0.054 — experiments are finally interpretable.

**Part 2 — hard-campaign regression: PASS (no campaign moved beyond σ),** though both
generalization campaigns drifted consistently negative and deserve monitoring:
`15_juillet` −0.0036 · `22_juillet` +0.0107 · `8_juillet` **−0.0374** · `OthmaneSalim`
**−0.0390**.

Per the pre-registered rule → **adopt** `--lr-scheduler cosine --monitor val_pr_auc`.

#### 🔴 …but the accuracy gain is a THRESHOLD SHIFT, not better discrimination

3-seed means: accuracy 84.62 → **88.12 %**, specificity 81.22 → **91.30 %**, recall
88.66 → **84.33 %**. That looks like a large win. It is not:

| ensemble | AUC | acc | F1 | recall | spec |
|---|---|---|---|---|---|
| old (warm_restarts+fbeta) @0.50 | **0.9365** | 86.66 % | 86.06 % | 90.12 % | 83.74 % |
| new (cosine+val_pr_auc) @0.50 | **0.9349** | 89.44 % | 87.90 % | 83.94 % | 94.07 % |
| **old @0.72 — specificity-matched to new** | 0.9365 | **89.67 %** | **88.19 %** | 84.44 % | 94.07 % |

AUC differs by **−0.0016**, far inside σ = 0.0072 → the two ensembles are
**ranking-equivalent**. At matched specificity the *old* one is marginally better
(+0.23 acc). Every bit of the apparent gain was purchasable from the old config by moving
the threshold 0.5 → 0.72. **Discrimination did not improve.**

#### ✅ Good news: the §6 dual target is reachable — it was a threshold problem all along

Searching each ensemble's ROC curve for a point with **recall ≥ 88 % AND specificity ≥ 88 %**:

| ensemble | threshold | accuracy | recall | specificity |
|---|---|---|---|---|
| old | 0.61 | 88.52 % | 88.03 % | 88.93 % |
| **new** | **0.34** | **89.55 %** | **88.88 %** | **90.12 %** |

Both clear it; the new ensemble more comfortably. We had only ever evaluated at 0.5, which
is why this looked unreachable.

⚠️ **Reading a threshold off the test ROC is oracle selection — not reportable.** The honest
procedure is different from §3.3's failed attempt: do **not** tune the threshold to maximise
a test metric. Instead **state the requirement first** (e.g. "recall ≥ 88 %", justified from
IEC 62606), then pick the threshold on the **validation** split that meets it, and report
whatever the held-out campaign gives. That is a design decision, not a fitted parameter —
and it is the one legitimate way to bank this. **Not yet tested; see N2b.**

### ❌ N2b RESULT (2026-07-30) — requirement-driven val threshold FAILS. Report at 0.5.

Procedure: state the requirement first (recall ≥ 90 %), pick on each model's **own**
validation split the most specific threshold meeting it, apply once to the held-out campaign.
Fully inductive. *(Note: the 3 seeds have different val partitions — `random_state=fold_seed`
— so a cycle in seed 42's val is seed 142's **training** data. There is therefore no clean
shared val set for a probability ensemble; each model must be calibrated on its own val and
the **decisions** aggregated by majority vote.)*

| stated requirement (on val) | test acc | test F1 | **test recall** | test spec |
|---|---|---|---|---|
| val recall ≥ 85 % | 87.07 % | 84.18 % | 75.27 % | 97.00 % |
| val recall ≥ 88 % | 87.49 % | 84.98 % | 77.47 % | 95.91 % |
| val recall ≥ 90 % | 87.36 % | 84.98 % | **78.27 %** | 95.00 % |
| val recall ≥ 92 % | 87.53 % | 85.45 % | 80.15 % | 93.74 % |
| val recall ≥ 95 % | 87.55 % | 85.93 % | 83.21 % | 91.20 % |
| **fixed 0.50, probability ensemble** | **89.44 %** | **87.90 %** | **83.94 %** | 94.07 % |
| oracle balanced (NOT reportable) | 89.31 % | 88.41 % | 89.26 % | 89.35 % |

**Asking for 90 % recall on val delivers 78 % on the unseen campaign — a 12-point
shortfall**, and the selected thresholds are incoherent across seeds/folds
(`[0.05, 0.65, 0.89, 0.90]` vs `[0.34, 0.94, 0.94, 0.93]` vs `[0.72, 0.92, 0.87, 0.94]`).
Every requirement level is *worse than simply using 0.5*.

**This is the third independent confirmation of §3.3** (after F-β val thresholds → 0 gain,
and per-campaign Otsu → +0.04 pt): under campaign shift, validation data cannot set the
operating point. The balanced 89.3 / 89.4 point is **real but not honestly reachable**.

**Consequence — and it is a good one for the defence:** the best defensible operating point
is the plain **fixed threshold 0.5**, with no tuning whatsoever. That removes any suspicion
of fitting the test set. Present **ROC-AUC 0.9349 ± 0.0072** as the threshold-free headline
plus the ROC curve so the jury can see the achievable trade-offs, and report the 0.5
operating point as the deployed one. The balanced point belongs in *future work*, conditional
on a per-installation calibration procedure (which §3.8 shows does not yet exist in a
deployable form).

### N2 — characterise the 22 hard recordings (**zero GPU**, highest information per minute)
They hold ~⅓ of all errors in 8.8 % of the data and appear in all four campaigns (§3.6).
Work from `oof_predictions.npz` + `metadata.csv` + `X_multi.npy`:

- What do `run54` (8 juil) and `run17` (22 juil) — 100 % missed, pure strong arcs — share?
- Load config / electrode (the `OthmaneSalim` names carry `AcierCu` vs `GraphAcier`).
- Position in the session (high `run` index ⇒ electrode wear?).
- Current amplitude, RMS, crest factor, spectral centroid vs the well-classified set.
- Check the **per-cycle RMS normalisation** in `_derive_i_channels` ([dataset.py:220](dataset.py:220)):
  it makes the descriptors load-invariant *by construction*, which may erase exactly the
  amplitude cue these recordings need.

Outcome decides the fix: a *feature/front-end* gap (addressable) vs *intrinsically
ambiguous* physics (then report it as a documented limitation, stratified by recording).

### N3 — retune the FrequencyGate band (hyperparameter, not architecture)
At the correct `fs`, `freq_max_hz = 100000` exceeds Nyquist (51.2 kHz) so the gate keeps
**63 of 65 bins** — it is no longer gating anything. Given §3.7 shows the >10.4 kHz band is
informative, sweep `freq_min_hz` / `freq_max_hz` to actually select a band. These are
constructor arguments, so this is hyperparameter tuning and does **not** touch the locked
architecture. *(Needs a CLI flag — currently not exposed.)*

### N4 — variance reduction (only after N1 gives σ)
- **Seed ensemble within each fold** (average probabilities over 3 seeds) — directly
  attacks the cross-campaign std; fully honest per fold. Cheap if N1 has already trained
  the seeds.
- **SWA** — *to build, ~15 lines*.
- Replace `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` with a smoother schedule: its
  restarts cause the val crashes at epochs 10/20/30 that destabilise early stopping.
- `--weight-decay 1e-3`; 309 833 parameters is a lot for 238 independent recordings.
- Monitor: **`val_pr_auc`** (§5) or β = 1.0. **β = 0.5 is retired** (§4.3).

### Retired — do not re-run without new evidence
`--group-dro` / `--coral-weight` / multi-cycle smoothing / Otsu or val threshold
adaptation. All measured, all flat or negative (§3.5). The reason is structural: they
target campaign-level shift, and the residual is recording-level (§3.6).

Still open as a **declared-transductive** option if N1–N4 stall: **AdaBN** (recompute
BatchNorm statistics on the unlabelled target campaign). Untested here, and unlike the
threshold tricks it changes the *representation* rather than the operating point. It
would move the claim from inductive to transductive — legitimate for a commissioned AFDD,
but it must be declared, and test labels must never be touched.

### Phase 5 — capacity / architecture (**last resort**)
The paper's architecture is locked (`DescriptorChannelAttention` + `FrequencyGate` +
Sequential Cross-Attention — see the paper-scope memo). Any change here must be framed as
a **separate ablation**, not the main model. 309 833 parameters is a lot for 238
independent recordings, so a *smaller* model may transfer better: try reduced width,
higher weight decay, and the `--no-channel-attn` / fusion-mode ablations under the same
protocol.

### Phase G — more benches (the real fix)
Three of four campaigns are the same IJL bench; the only genuinely different setup (2026)
is also the easiest fold (AUC 0.996). No training trick substitutes for a fifth and sixth
campaign on different installations, electrodes and load mixes. If new acquisition is
possible before the deadline, **this outranks everything above.**

---

## 8. Anti-cheating protocol (non-negotiable)

**The fold's test campaign is sacred: no weight, hyperparameter, threshold, or
early-stopping decision may ever see it.**

With only **4 campaigns**, the fatal trap is: try many configs → pick the best 4-fold
LOCO mean → report it. That leaks the test campaigns into model selection.

- **Rigorous:** nested LOCO — outer fold reports, inner leave-one-*training*-campaign-out
  selects hyperparameters.
- **Pragmatic (time-constrained):** *pre-register* a small config set (R1, R2, +SWA,
  +ensemble), select among them on the **recording-grouped val only**, then run LOCO
  **once** and report it. Commit in writing not to re-pick based on the test result.

Standing guardrails: recording-grouped val · threshold from val only · early-stop on val ·
train-only normalization and augmentation donor pool (`set_donor_pool(train_idx)`,
[train.py:1322](train.py:1322)) · test seen exactly once · oracle/test-optimal thresholds
are **diagnostics only**, never headline numbers.

---

## 9. What to report in the paper

1. **Both numbers, framed as a finding.** Random split = "seen-bench / in-distribution";
   leave-one-campaign-out = "cross-campaign generalization". The gap **is** a
   contribution — reporting only the 98 % is the single biggest red flag reviewers look
   for in fault-detection papers.
2. **ROC-AUC and PR-AUC as primary metrics** — threshold-free, so they are not hostage to
   the operating-point drift. Pooled AUC ≈ 0.91 and mean per-fold ≈ 0.94 are far more
   defensible than an F1 that swings 15 points with the threshold.
3. **Per-campaign table, not just the mean** — the weak campaign is the story.
4. **State the operating-point protocol explicitly**: threshold chosen on the
   training-campaign validation split, applied once to the held-out campaign.
5. **Declare the `fs` correction** and re-run any number that appears in the paper — every
   pre-2026-07-29 result used 13 of 65 STFT bins (§4.1).
6. Note that leave-one-campaign-out **stands in for leave-one-charge-out**, which this
   dataset cannot support: the 2024 campaigns carry no per-experiment load labels.

---

## 10. Artifact map

| path | contents |
|---|---|
| `runs/<run>/groupkfold_summary.json` | all fold results + pooled + calibrated blocks + config |
| `runs/<run>/oof_predictions.npz` | `probs`, `labels`, `fold`, `pred_cal`, `groups` — every cycle predicted by a model that never saw its campaign |
| `runs/<run>/<fold>/test_predictions.npz` | per-fold test `idx`, `probs`, `labels`, `groups` |
| `runs/<run>/<fold>/val_predictions.npz` | per-fold val `idx`, `probs`, `labels` (threshold selection) |
| `runs/<run>/<fold>/history_<fold>.json` | per-epoch curves, `best_epoch`, monitor config |

Any threshold-free re-analysis (AUC, per-campaign z-scoring, oracle headroom) can be
redone post-hoc from `oof_predictions.npz` with **no retraining**.
