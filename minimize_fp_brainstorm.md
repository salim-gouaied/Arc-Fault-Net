# Minimizing False Positives: Brainstorm & Strategy

## Current Situation (threshold = 0.5)

| | Predicted Normal | Predicted Arc |
|---|---|---|
| **True Normal** | TN = 868 | **FP = 5** |
| **True Arc** | FN = 14 | TP = 743 |

## The 5 False Positive Probabilities

These are the confidence scores the model assigned to the 5 normal samples it misclassified as arcs:

| FP # | Model Confidence | Interpretation |
|---|---|---|
| 1 | **96.1%** | Model is extremely confident — this sample genuinely looks like an arc |
| 2 | **90.7%** | Very high confidence — likely a noisy/ambiguous sample |
| 3 | **88.8%** | High confidence — same category |
| 4 | 56.3% | Borderline — easily fixable with threshold |
| 5 | 51.4% | Borderline — easily fixable with threshold |

> [!IMPORTANT]
> **3 out of 5 FPs are high-confidence (>85%).** Simple threshold tuning can only eliminate the 2 borderline ones. The top 3 are samples where the model genuinely "believes" it sees an arc. These require deeper strategies.

## Threshold Sweep Results

| Threshold | FP | FN | Precision | Recall | Trade-off |
|---|---|---|---|---|---|
| 0.50 | **5** | 14 | 99.33% | 98.15% | Current |
| 0.60 | **3** | 16 | 99.60% | 97.89% | Kills 2 borderline FPs, +2 FN |
| 0.90 | **2** | 20 | 99.73% | 97.36% | Kills 1 more, +6 FN total |
| 0.95 | **1** | 35 | 99.86% | 95.38% | Only 1 FP left, but +21 FN |

---

## Strategy 1: Threshold Tuning (Easiest — Deployment-Level)

**What:** Raise the decision threshold from 0.5 to 0.60.

**Result:** FP drops from 5 → 3. Zero code changes, zero retraining.

**Limitation:** Cannot kill the 3 high-confidence FPs (96%, 91%, 89%). These are samples where the current waveform genuinely mimics arc behavior (e.g., inrush currents from motor startups, switching transients).

---

## Strategy 2: Multi-Cycle Consensus (Best for Real Deployment)

**What:** In real-world deployment, don't trigger an arc fault alarm on a single 20ms cycle. Instead, require **N consecutive positive predictions** (e.g., N=3 or N=5) before declaring an arc.

**Why this works:** Real arc faults are persistent — they last for many consecutive cycles. A normal transient (motor startup, light switch) typically lasts only 1-2 cycles. By requiring 3+ consecutive positives, you filter out isolated false alarms.

**Implementation:**
```python
class ArcFaultDetector:
    def __init__(self, model, threshold=0.6, n_consensus=3):
        self.model = model
        self.threshold = threshold
        self.n_consensus = n_consensus
        self.consecutive_positives = 0
    
    def predict_cycle(self, x1, x2):
        prob = torch.sigmoid(self.model(x1, x2)).item()
        if prob >= self.threshold:
            self.consecutive_positives += 1
        else:
            self.consecutive_positives = 0
        
        # Only trigger alarm after N consecutive positive cycles
        alarm = self.consecutive_positives >= self.n_consensus
        return alarm, prob
```

**Expected impact:** If the 3 high-confidence FPs are isolated (not from consecutive cycles in the same recording), this eliminates them entirely. Combined with threshold=0.6, you likely reach **FP = 0** in practice.

> [!TIP]
> This is the approach used in the **IEC 62606** standard for AFDDs (Arc Fault Detection Devices). The standard explicitly allows a detection latency of several cycles to reduce nuisance tripping.

---

## Strategy 3: Asymmetric Loss Function (Retraining Required)

**What:** Replace `BCEWithLogitsLoss` with a custom loss that penalizes FP much more heavily than FN.

**Focal Loss with asymmetric weighting:**
```python
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, fp_weight=5.0, fn_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.fp_weight = fp_weight  # Heavy penalty for FP
        self.fn_weight = fn_weight  # Light penalty for FN
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal modulation
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = (1 - p_t) ** self.gamma * bce
        
        # Asymmetric weighting: penalize FP (target=0, pred=1) more
        weights = torch.where(targets == 1, self.fn_weight, self.fp_weight)
        return (weights * focal).mean()
```

**Why this works:** During training, whenever the model predicts "arc" on a normal sample, the gradient is 5x stronger than when it misses an actual arc. The model learns to be extremely cautious about saying "arc."

---

## Strategy 4: Investigate the 3 Hard FPs (Data-Level)

**What:** Extract and visually inspect those 3 high-confidence FP samples. They might be:
- **Mislabeled data** — genuinely arc-fault cycles that were labeled as normal
- **Switching transients** — inrush currents that physically mimic arc signatures (high dI/dt + broadband noise)
- **Measurement artifacts** — sensor noise or clipping

**If mislabeled:** Correcting the labels improves both training and evaluation.
**If switching transients:** These are the hardest cases in arc fault detection. The physical signals are genuinely similar. This is where Strategy 2 (multi-cycle consensus) becomes essential.

---

## Strategy 5: Two-Stage Classification (Nuclear Option)

**What:** Train a second, precision-focused model that only evaluates samples the first model flags as positive.

```
Stage 1 (High Recall): threshold=0.3 → catches all arcs + some FPs
Stage 2 (High Precision): Only runs on Stage 1 positives → kills FPs
```

**Why this could work:** The second model can be trained specifically on the "hard" samples near the decision boundary. It could use different features (e.g., zero-crossing rate, crest factor) that help distinguish switching transients from true arcs.

**Downside:** Doubles inference cost and architectural complexity. Only justified if the other strategies fail.

---

## Recommended Action Plan

| Priority | Strategy | FP Reduction | Effort |
|---|---|---|---|
| 1 | Threshold → 0.60 | 5 → 3 | Zero (config change) |
| 2 | Multi-cycle consensus (N=3) | 3 → ~0 | Low (deployment wrapper) |
| 3 | Inspect 3 hard FPs | May fix labels | Medium |
| 4 | Asymmetric loss | Retrain needed | High |
| 5 | Two-stage model | Guaranteed | Very High |

> [!NOTE]
> **Strategies 1+2 combined are almost certainly sufficient** to reach FP=0 in real deployment. The 3 high-confidence FPs are likely isolated transient events, not sustained arc-like behavior across multiple consecutive cycles. Multi-cycle consensus is both the simplest and most physically justified approach for an AFDD system.
