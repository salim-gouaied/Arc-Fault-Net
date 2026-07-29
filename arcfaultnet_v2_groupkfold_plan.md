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
| Honest baseline number | ✅ pooled **F1 83.6 % / ROC-AUC 0.914**, mean per-fold AUC **0.941** |
| Single-mode 98 % | ❌ **discarded — leaky**, do not report |
| Domain-generalization stack | ⚠️ **run once but CONFOUNDED** — no valid measurement yet |
| `fs` / FrequencyGate bug | 🔴 **confirmed, unfixed in workflow** — invalidates the spectral branch of every run before 2026-07-29 18:20 |
| Under-training from `--fbeta 0.5` | 🔴 **new problem**, fold 2 recall fell to 62.9 % |

**Immediate next action:** run **R1** (§7) — clean reference with correct `fs`, no DG
stack. Until R1 exists, no DG/architecture claim can be made.

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

| id | run dir | val split | `fs` | DG stack |
|---|---|---|---|---|
| **S** | `arcfaultnet_v2_single_*` ×5 seeds | random cycle split | 1 MHz | no |
| **A** | `..._groupkfold_campaign_20260729_115802` | `alternance` (leaky) | 1 MHz | no |
| **B** | `..._groupkfold_campaign_20260729_172929` | `recording` ✅ | 1 MHz ❌ | no |
| **C** | `..._groupkfold_campaign_20260729_182029` | `recording` ✅ | 102.4 kHz ✅ | **yes** |

### Headline metrics

| metric | S (leaky) | A | **B (reference)** | C (confounded) |
|---|---|---|---|---|
| pooled accuracy | 98.65 % | 86.17 % | 84.01 % | 84.93 % |
| pooled F1 @0.5 | 98.54 % | 85.10 % | **83.56 %** | 82.05 % |
| pooled precision | 99.51 % | 83.80 % | 78.82 % | 89.99 % |
| pooled recall | 97.59 % | 86.44 % | 88.90 % | 75.39 % |
| pooled specificity | — | 85.94 % | 79.91 % | 92.95 % |
| pooled ROC-AUC | — | 0.9010 | **0.9135** | 0.9085 |
| mean per-fold ROC-AUC | — | 0.8966 | **0.9408** | 0.9448 |
| mean F1 ± std | — | 82.78 ± 7.55 | 82.06 ± 7.52 | 83.21 ± 7.03 |
| best epochs | — | 55/66/26/28 | 29/27/33/30 | **21/11/45/10** |

### Per-campaign ROC-AUC

| held-out campaign | A | B | C |
|---|---|---|---|
| `15_juillet` | 0.9621 | **0.9948** | 0.9753 |
| `22_juillet` | 0.9689 | 0.9321 | 0.9239 |
| `8_juillet` | **0.7707** | 0.8723 | 0.8830 |
| `OthmaneSalim` | 0.8847 | 0.9641 | **0.9968** |

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

| metric | A | B | target | met? |
|---|---|---|---|---|
| pooled ROC-AUC | 0.9010 | 0.9135 | **≥ 0.94** | ✗ |
| worst-campaign ROC-AUC | 0.7707 | 0.8723 | **≥ 0.85** | ✅ |
| pooled F1 @val threshold | 85.10 % | 83.56 % | **≥ 90 %** | ✗ |
| cross-campaign F1 std | 7.55 % | 7.52 % | **≤ 4 %** | ✗ |
| recall **and** specificity | 86.4 / 85.9 | 88.9 / 79.9 | **both ≥ 88 %** | ✗ |

Report pooled **and** per-campaign. A mean alone hides the weak campaign.

---

## 7. Next phases

### R1 — clean reference (**do this first**, ~17 min)
Correct `fs`, no DG stack, β = 1.0 (stops rewarding a non-firing model), patience 25
(survives the LR restarts). Isolates the `fs` fix and becomes the valid comparison point.

```bash
python train.py --model arcfaultnet_v2 --mode groupkfold --group-level campaign --data-dir combined_dataset_2048/combined_dataset_2048/ --fs 102400 --n-fft 128 --hop-length 64 --deep-clf --fusion-mode cross_attention --monitor val_fbeta --fbeta 1.0 --epochs 200 --patience 25 --seed 42
```

### R2 — DG stack on top of R1 (valid A/B, ~17 min)
Identical to R1 **plus** the domain-generalization levers (0 extra parameters, all already
wired). Judge against R1, not against B or C.

```bash
python train.py --model arcfaultnet_v2 --mode groupkfold --group-level campaign --data-dir combined_dataset_2048/combined_dataset_2048/ --fs 102400 --n-fft 128 --hop-length 64 --deep-clf --fusion-mode cross_attention --monitor val_fbeta --fbeta 1.0 --epochs 200 --patience 25 --seed 42 --group-dro --coral-weight 0.5 --strong-aug --channel-dropout 0.2 --use-pos-weight --dg-balanced-sampler
```

**Results placeholder** — fill in when the runs land:

| | R1 (fs fixed, no DG) | R2 (fs fixed + DG) |
|---|---|---|
| pooled ROC-AUC | | |
| mean per-fold ROC-AUC | | |
| pooled F1 @0.5 / @val | | |
| pooled recall / specificity | | |
| worst-campaign AUC | | |
| mean−pooled AUC gap (drift) | | |
| best epochs per fold | | |

**Decision rule after R2:**
- drift gap shrinks **and** worst-campaign AUC rises → keep the DG stack, go to Phase 3.
- no change → CORAL/DRO are not the lever here; go to Phase 4 (diagnose) then Phase D2.

### Phase 3 — generalization-oriented regularization (params only, paper-safe)
- **SWA** (stochastic weight averaging over the last epochs) — *to build, ~15 lines*;
  one of the most reliable generalization gains.
- `--weight-decay 1e-3` (from 5e-4); higher classifier dropout.
- **Seed ensemble within each fold** (3 seeds, average probabilities) — directly attacks
  the 7.5 % cross-fold std; fully honest per fold.
- Replace `CosineAnnealingWarmRestarts` with a smoother schedule — the restarts cause the
  val crashes seen at epochs 10/20/30 and destabilise early stopping.
- Monitor: prefer **`val_pr_auc`** (§5) or β = 1.0. **β = 0.5 is retired** — §4.3.

### Phase 4 — diagnose the residual gap
- **Why is `8_juillet` still the weakest** (AUC 0.87–0.88)? Compare its I-descriptor and
  STFT statistics, load mix, SNR and `arc_ratio` distribution against the other three.
  If it occupies a region the others do not cover → DG/augmentation is right. If it is a
  labeling/segmentation artifact → no model change will fix it.
- **Re-examine `22_juillet`** — the only campaign that got *worse* from A → B
  (0.9689 → 0.9321) and the one that collapsed to 62.9 % recall in C.
- Verify the `2 kHz–100 kHz` gate limits are still sensible now that Nyquist is 51.2 kHz
  (`freq_max_hz=100000` is clamped, so the gate keeps 63 of 65 bins — i.e. almost no
  band selection is happening any more; consider retuning `freq_min_hz`/`freq_max_hz`).

### Phase D2 — decision-layer adaptation (**declared transductive** — only if Phases 3–4 stall)
Since ranking is good (AUC 0.94) and the operating point is what fails (§3.4, 6.3 F1
points of headroom), and since val provably cannot fix it (§3.3), the remaining honest
options use the **unlabelled** target campaign:

- **AdaBN** — recompute BatchNorm running statistics on unlabelled test-campaign cycles.
  0 parameters, usually strong under domain shift.
- **Prior / quantile matching** — set the threshold so the predicted arc rate matches an
  assumed prevalence, estimated from unlabelled target data.
- **Multi-cycle aggregation** — decide over *k* consecutive périodes (median logit /
  majority vote) instead of one. Matches IEC 62606, which specifies detection within a
  number of half-cycles rather than a single one. Report the window length; never mix with
  per-cycle numbers.

⚠️ These change the claim from **inductive** to **transductive** generalization. Legitimate
and realistic for a commissioned AFDD, but it **must be declared as such** in the paper,
and **test labels must never be touched**.

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
