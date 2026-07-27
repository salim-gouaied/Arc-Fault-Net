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

## 1b. If time is short, this is the order

| # | Action | GPU cost | Status |
|---|---|---|---|
| 1 | E1/E2 decision-layer analysis | **none** — post-hoc on saved scores | **done**, see E |
| 2 | B checkpoint-selection check | **none** — used saved best/last | **done**, effect ±4 F1 → skip `--val-mode group` |
| 3 | C augmentation + `--channel-dropout 0.3` | 1 run (~2 h) | the one run worth doing |
| 4 | D campaign-balanced batches + GroupDRO | 1 run (~2 h) | next, if time |
| 5 | A seed noise floor | 2 runs (~4 h) | needed before believing small deltas |
| 6 | F smaller model | 1 run | only if C and D disappoint |

## 2. Interventions, in the order I would run them

Each is one training run with the same protocol command, then
`eval_groupcv.py` + `compare_groupcv.py` against B1.

### A. Measure the noise floor first (no change to the model)

Re-run B1 with 2–3 different seeds (`--seed 43/44`). Without this, any delta below
~2–3 points is uninterpretable. Cost: ~2 h per seed. This is not optional overhead
— fold-level metrics here swing by tens of points.

### B. Checkpoint selection — measured, and **deprioritised**

`--val-mode group` holds a whole training campaign out for validation, so early
stopping would optimise cross-campaign F1 instead of in-campaign F1. Before spending
a 2 h run on it, the size of the prize was measured on B1's own checkpoints: each
fold saved both `best_fold_k.pt` (selected on in-domain val F1) and `last_fold_k.pt`
(10 epochs later, patience exhausted). Evaluated on the held-out campaign:

| Campaign | F1 best → last | AUC best → last |
|---|---|---|
| 15_juillet | 77.23 → 74.09 | 0.9119 → **0.9453** |
| 22_juillet | 87.52 → 83.19 | 0.9076 → **0.9215** |
| 8_juillet | 65.46 → 61.14 | 0.8804 → 0.7863 |
| 2026 | 86.02 → **90.30** | 0.9963 → 0.9899 |

Checkpoint choice moves F1 by ±3–4 points in both directions — the same order as
expected seed noise, and much smaller than the 17-point generalization gap. Note that
on two campaigns the *ranking* (AUC) keeps improving while F1 at 0.5 degrades, which
is again the calibration effect, not overfitting in the usual sense.

Verdict: `--val-mode group` is not worth a run under time pressure. It costs 25–30 %
of the training data to chase a ±4-point effect. If you want the cheap version of
this idea, drop early stopping and fix the budget at ~25 epochs so no in-domain
signal picks the checkpoint.

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

### E. Decision-layer adaptation — measured on B1's saved scores, no retraining

Both options below were evaluated directly on `oof_predictions.npz`; the numbers are
measurements, not estimates.

**E1. Per-installation (commissioning) threshold** — set the threshold at the 99th
percentile of the target campaign's own arc-free cycles, i.e. a 1 % false-alarm rate
by construction, which is what an installer can calibrate on site:

| Campaign | recall @ 1 % FPR (single cycle) |
|---|---|
| 15_juillet | 6.5 % |
| 22_juillet | 6.1 % |
| 8_juillet | 68.3 % |
| 2026 | 98.9 % |

**This kills the "calibration is 8 free points" idea for the deployment regime.**
The ~89.5 % oracle-threshold figure is a *balanced-accuracy* operating point; at the
low false-alarm rate an AFDD actually needs, two of four campaigns collapse. The
cause is visible in the score distribution: on 15_juillet the arc windows sit at
logit ≈ 3.27 (p ≈ 0.963) and the 99th percentile of *normal* is 3.71 (p ≈ 0.976) —
the model is saturated near p ≈ 0.97 for both classes, and AUC 0.91 is measuring
ordering inside that squashed band.

**E2. Multi-cycle decision** — median logit over k consecutive cycles of the same
alternance, threshold recalibrated on arc-free windows of the same length:

| Campaign | k=1 | k=3 | k=5 | k=9 | k=15 |
|---|---|---|---|---|---|
| 15_juillet | 6.5 % | 2.8 % | 2.7 % | **96.4 %** | 100 % |
| 22_juillet | 6.1 % | 1.8 % | 1.0 % | **95.2 %** | 94.8 % |
| 8_juillet | 68.3 % | 65.3 % | 67.0 % | 68.1 % | 52.0 % |
| 2026 | 98.9 % | — (1 cycle per recording) | | | |

(recall at a 1 % window false-alarm rate; controlled against the same alternance
subset at k=1, so this is aggregation, not sample selection.)

The jump at k=9 is real but **knife-edge**: averaging pulls the normal tail from
logit 3.63 down to 2.89 while the arc bulk stays at 3.28, so recall flips from 3 %
to 96 % over a 0.4-logit move. Worth reporting, but it is a symptom of score
saturation, not evidence of a robust detector. It becomes trustworthy only once the
class distributions are actually separated — which is what C and D are for.

Note also that k=9 is ~180 ms of mains, which is compatible with IEC 62606 detection
times, so this is a legitimate decision unit provided the window length is stated
and per-cycle and per-window numbers are never mixed.

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
