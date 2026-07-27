# Improving cross-campaign generalization — plan

Reference point: [baseline B1](baselines/arcssm_campaign_cv_v1.md) — ArcSSM,
leave-one-campaign-out, pooled 81.28 % / F1 79.63 %, worst campaign F1 65.46 %.

## 1. What the baseline numbers actually say

The drop from 98.5 % (random split) to 81.3 % (unseen campaign) is real and the
model is **not** deployable as it stands. But "the features do not transfer" is not
what the run shows — three measurements point elsewhere:

1. **Ranking survives the shift.** Per-campaign AUC is 0.912 / 0.908 / 0.880 /
   0.996. Inside every unseen campaign the model still orders arc above normal;
   0.996 on the 2026 campaign is near-perfect separation.
2. **The decision boundary does not survive it.** Mean p(arc) on *arc* cycles ranges
   0.497 (8 juillet) to 0.973 (2026); on *normal* cycles 0.047 to 0.450. The whole
   score distribution slides per campaign, which is why fold 1 collapses into
   over-detection (recall 97 %, specificity 52 %) and fold 3 into under-detection
   (recall 49 %, specificity 100 %). Pooled AUC (0.887) is *below* the mean per-fold
   AUC (0.924) — mixing offset score scales destroys ranking that exists within each
   campaign.
3. **A per-campaign threshold recovers 89.55 %** pooled versus 81.28 % at a fixed
   0.5.

So the 17-point gap splits roughly into **8 points of calibration / score shift**
and **9 points of genuine representation shift**. Those need different fixes, and
the calibration half is the cheaper one.

A fourth observation drives the first experiment: early stopping fired on a
validation set drawn from the *training* campaigns, at val F1 90.6–99.1 %, while the
held-out campaign scored 65–88 %. The checkpoint was selected by a signal that does
not track what we are trying to measure.

Two framing notes for the report: a real AFDD does not decide on one 20 ms cycle in
isolation (IEC 62606 specifies detection within a number of half-cycles), and it is
commissioned in one installation rather than shipped with a universal threshold.
Both facts are usable — see interventions D and E.

## 2. Interventions, in the order I would run them

Each is one training run with the same protocol command, then
`eval_groupcv.py` + `compare_groupcv.py` against B1.

### A. Measure the noise floor first (no change to the model)

Re-run B1 with 2–3 different seeds (`--seed 43/44`). Without this, any delta below
~2–3 points is uninterpretable. Cost: ~2 h per seed. This is not optional overhead
— fold-level metrics here swing by tens of points.

### B. Fix checkpoint selection — validate on an unseen campaign

`--val-mode group` already does this: it holds a whole training campaign out for
validation, so early stopping optimizes cross-campaign F1 instead of in-campaign F1.
The cost is one campaign's worth of training data (train drops to ~2 campaigns).
The alternative, if that proves too expensive, is to drop early stopping and fix a
training budget (e.g. 25 epochs) chosen once, so no in-domain signal picks the
checkpoint at all.

Cost: one run. Expected: better worst-campaign F1, possibly lower pooled accuracy.

### C. Stronger, physically-motivated augmentation

The current augmentation is 0.005·std Gaussian noise — far too weak to simulate a
different bench. Add, on the raw cycle before deriving the 4 channels:

- coloured/pink noise at randomised SNR (sensor and mains noise floor);
- random band-limiting and a random first-order high-pass (sensor bandwidth and
  coupling differ per bench);
- mains-frequency jitter (±0.5 Hz) via resampling, random time shift, polarity flip;
- **background-load mixing**: add a scaled *normal* cycle from another campaign to an
  arc cycle, keeping the arc label — this directly synthesises "same arc, different
  load mix", which is the dominant difference between campaigns;
- turn on the knob that already exists: `--channel-dropout 0.3`, so the classifier
  cannot lean on one descriptor whose statistics happen to shift.

Cost: one implementation pass in `dataset.py:_augment_temporal` + one run per
variant. Expected: the largest robustness gain per unit of effort.

### D. Train for domain invariance

On the 128-d embedding, in increasing order of complexity:

1. **campaign-balanced batches** (sample each training campaign equally);
2. **GroupDRO** — optimise the worst-campaign loss instead of the average;
3. **CORAL / MMD** alignment of embedding statistics across campaigns;
4. **domain-adversarial head** (gradient reversal predicting the campaign ID).

Start with 1 + 2: they are a sampler and a loss wrapper, roughly 40 lines, and they
target exactly the failure mode (one campaign dominating the learned score scale).

### E. Decision-layer adaptation — cash in the 8 calibration points

Since ranking is good, the threshold is what fails. Two honest options:

- **Unsupervised per-installation calibration.** Estimate the score distribution on
  *unlabelled* cycles from the target campaign and set the threshold at a fixed
  quantile of the normal-dominated traffic (or z-score the logits by a robust
  location/scale of a rolling window). Fit it only on unlabelled target data, report
  it as test-time adaptation — that is legitimate and is what a commissioned AFDD
  can genuinely do. Upper bound from B1's diagnostics: ~89.5 % pooled.
- **Multi-cycle aggregation.** Decide over a window of k consecutive cycles (median
  logit / majority vote) instead of one cycle. This matches how the standard defines
  detection and averages away per-cycle score noise. Report it as a different
  decision unit, with the window length stated, never mixed with per-cycle numbers.

### F. Capacity and regularisation

359 553 parameters is a lot for 1622 independent alternances. Try `n_layers=2`,
`d_model=96`, higher weight decay, higher `block_dropout`; also run the
`arcssm_selective` variant under the same protocol. Cheap, and a smaller model often
transfers better under domain shift.

### G. The real fix: more benches

Three of the four campaigns are the same IJL bench; the only genuinely different
setup (2026) is also the easiest fold (AUC 0.996). No training trick substitutes for
a fifth and sixth campaign on different installations, electrodes and load mixes. If
new acquisition is possible before the deadline, this outranks everything above.
Keep the 368-cycle 2026 hold-out (`eval_holdout.py`) untouched as a final one-shot
check.

## 3. How each experiment is judged

- **Primary**: worst-campaign F1 (a change that only lifts campaigns that already
  worked has not improved generalization).
- **Secondary**: pooled F1/AUC, and the spread across campaigns.
- **Significance**: McNemar p-value from `compare_groupcv.py` — the runs are paired
  on the same 10 860 cycles.
- **Sanity**: the gap to the random-split ceiling should shrink from both ends; if
  the ceiling number also collapses, the change broke the model rather than
  regularising it.

Record every adopted change as a new baseline file next to
[B1](baselines/arcssm_campaign_cv_v1.md), so the report can show the progression
B1 → B2 → B3 with what changed and what it bought.
