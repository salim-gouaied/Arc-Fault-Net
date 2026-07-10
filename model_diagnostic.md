# Model Diagnostic: The Model Is NOT Learning What We Think

## The Smoking Gun: Per-Channel Ablation

I zeroed out each input channel one at a time and measured how the model's prediction changed. If the model was truly learning multi-channel arc signatures, removing any single channel should cause a moderate drop. Here's what actually happens:

| Sample | Full Prob | w/o I(t) | w/o \|ΔI\| | w/o TKEO | w/o RMS | w/o STFT |
|---|---|---|---|---|---|---|
| **FP #1** (96.1%) | 96.1% | **97.1%** ↑ | 96.1% = | 96.1% = | **5.4%** ↓↓↓ | **6.6%** ↓↓↓ |
| **FP #2** (90.7%) | 90.7% | **97.0%** ↑ | 90.7% = | 90.8% = | **5.4%** ↓↓↓ | **6.5%** ↓↓↓ |
| **FP #3** (88.8%) | 88.8% | **97.3%** ↑ | 88.7% = | 88.9% = | **4.2%** ↓↓↓ | **6.5%** ↓↓↓ |
| **TP best** (96.8%) | 96.8% | **97.3%** ↑ | 96.8% = | 96.8% = | **4.6%** ↓↓↓ | **7.5%** ↓↓↓ |
| **TN best** (4.6%) | 4.6% | **57.6%** ↑↑↑ | 4.5% = | 4.6% = | 4.3% = | 5.5% = |

> [!CAUTION]
> **The model is making ALL its decisions based on exactly ONE channel: the RMS sliding window.** Removing RMS drops every prediction to ~5%. Removing |ΔI| or TKEO changes NOTHING. Removing I(t) actually INCREASES confidence.

## What This Means

### The Diagnosis

1. **|ΔI| and TKEO are completely ignored.** Zeroing them out changes the prediction by 0.0-0.1%. The model has learned to bypass them entirely. Your carefully designed derivative and energy features contribute **zero** to the decision.

2. **I(t) is counter-productive.** Removing it makes the model MORE confident in its prediction (96.1% → 97.1% for FP#1, 96.8% → 97.3% for TP). The raw current waveform is actually introducing noise that slightly confuses the model.

3. **RMS is the sole decision-maker in the 1D branch.** Zeroing RMS drops everything to ~5%. The model has collapsed its entire 4-channel temporal representation down to: "is the RMS envelope shape consistent with what I've seen in arcs?"

4. **STFT is the only other contributor.** Removing STFT drops predictions to ~6-7%. So the model is essentially: `prediction ≈ f(RMS) + f(STFT)` — the other 3 channels are dead weight.

5. **The TN row reveals the catastrophe.** For the TN sample, removing I(t) makes the model jump from 4.6% → 57.6%. This means I(t) is the only thing *preventing* this normal sample from being classified as arc. The model has learned a bizarre inverse dependency where I(t) acts as a "suppress arc prediction" signal.

### Why This Explains the FPs

Now the FP#3 mystery is solved:
- FP#3 has a **perfectly normal RMS envelope** (smooth sinusoidal half-wave)
- But the model has learned that "any RMS envelope with this amplitude range = arc"
- It's not detecting arc-specific *patterns* in the RMS; it's just thresholding on amplitude/shape characteristics that happen to overlap between normal and arc samples from similar loads

**The model has NOT learned the multi-channel correlation you designed.** It found a shortcut: RMS amplitude + STFT overall energy level. This shortcut works for 98.8% of samples but fails on the edge cases where normal cycles happen to have similar RMS/STFT magnitudes as arc cycles.

## Gradient Saliency Maps

### FP #3 — What the Model Looks At on a Normal Cycle It Calls "Arc"

![Gradient saliency for FP#3 — notice how the gradient is concentrated almost entirely on the RMS channel](/home/manip/.gemini/antigravity/brain/7b8b2b80-b7de-4887-a499-b896da24feaa/grad_fp3.png)

### True Positive — What the Model Looks At on a Real Arc

![Gradient saliency for best TP — same pattern: RMS dominates, other channels barely register](/home/manip/.gemini/antigravity/brain/7b8b2b80-b7de-4887-a499-b896da24feaa/grad_tp.png)

### True Negative — What the Model Looks At on a Normal Cycle

![Gradient saliency for best TN — the model barely engages any channel](/home/manip/.gemini/antigravity/brain/7b8b2b80-b7de-4887-a499-b896da24feaa/grad_tn.png)

## Root Cause: Why the Model Takes This Shortcut

The training process found the **path of least resistance**. Here's why:

1. **RMS is the easiest feature to separate.** Arc fault cycles tend to have different RMS amplitudes/envelopes than normal cycles because the arc resistance changes the load's effective impedance. The model discovered this early in training and over-optimized on it.

2. **The loss function doesn't penalize shortcutting.** BCEWithLogitsLoss only cares about the final correct/incorrect answer. If RMS alone gets 98% accuracy, there's no gradient pressure to also learn |ΔI| patterns or TKEO correlations — the loss is already near-zero.

3. **Cross-attention amplifies the shortcut.** The cross-attention mechanism is supposed to align temporal and spectral features. But since the temporal branch is dominated by RMS, the attention just learns: "does the RMS shape correlate with the STFT energy level?" — which is a trivial, non-discriminative correlation.

## What Needs to Change

To force the model to learn the multi-channel signature you designed, consider these approaches:

### Option A: Channel Dropout During Training
Randomly zero out 1-2 temporal channels per batch during training. This forces the model to learn from ALL channels because it can never rely on a single one always being present.

```python
# In train_one_epoch, before forward pass:
if training:
    mask = torch.ones(4, device=x1.device)
    drop_channels = random.sample(range(4), k=random.randint(0, 2))
    for ch in drop_channels:
        mask[ch] = 0.0
    x1 = x1 * mask.view(1, 4, 1)
```

### Option B: Per-Channel Normalization
Currently, the 4 channels have very different scales (I(t) ranges ±1.5, TKEO is ~0.01, |ΔI| is ~0.02). The model naturally gravitates toward the highest-magnitude channel. Per-channel BatchNorm or standardization would equalize them.

### Option C: Auxiliary Per-Channel Losses
Add auxiliary classification heads after each branch/channel to enforce that each channel independently learns arc-discriminative features, then fuse them.

### Option D: Input-Level Data Augmentation
Add Gaussian noise, random scaling, or time-warping to the RMS and STFT channels specifically. This would destroy the model's ability to rely on simple amplitude thresholds and force it to learn structural patterns.

> [!IMPORTANT]
> **The core issue is not the architecture — it's the training dynamics.** The model has the capacity to learn multi-channel correlations (the cross-attention is structurally capable). But the optimizer found a shortcut that satisfies the loss function without needing the other channels. The fix must come from the training procedure (dropout, augmentation, auxiliary losses) to block the shortcut.
