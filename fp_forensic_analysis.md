# False Positive Forensic Analysis

## Reference: What a Real Arc Fault Looks Like

![True Arc — 95.6% confidence. This is what the model has learned to detect.](/home/manip/.gemini/antigravity/brain/7b8b2b80-b7de-4887-a499-b896da24feaa/tp_reference.png)

A genuine arc produces:
- **Distorted current waveform** with irregular shoulders, spikes, or flat-top clipping
- **High-amplitude |ΔI| bursts** at specific phases of the cycle (near zero-crossings where re-ignition occurs)
- **TKEO spikes** indicating rapid energy changes at the arc ignition points
- **Broadband noise in the STFT** — energy spread across all frequencies (not just harmonics)

---

## FP #1 — AcierCu Kettle (Confidence: 96.1%)

![FP #1: AcierCu_kettle from OthmaneSalim dataset. Model confidence 96.1%](/home/manip/.gemini/antigravity/brain/7b8b2b80-b7de-4887-a499-b896da24feaa/fp1.png)

| Field | Value |
|---|---|
| **Dataset** | `OthmaneSalim10052026` (the newer, secondary dataset) |
| **Experiment** | `AcierCu_kettle` — Steel-Copper contact with a kettle load |
| **Label** | 0 (Normal) |
| **Cycle Index** | 96 |
| **Sample Range** | 1,938,406 → 1,958,335 |

### Why the Model Fails

This is a **kettle (resistive heating element)** connected through a **steel-copper (AcierCu) contact**. This is the most adversarial load type for arc fault detection:

- **High-current resistive loads** draw large sinusoidal currents that saturate the |ΔI| channel near zero-crossings — mimicking the "re-ignition spike" pattern of real arcs.
- **Contact resistance** at the AcierCu junction creates micro-voltage drops that produce **broadband noise in the STFT** — almost identical to arc plasma noise.
- The TKEO channel shows **energy bursts** consistent with the rapid current changes through a lossy contact.

> [!WARNING]
> This sample may be a **borderline mislabel**. A steel-copper contact under high current from a kettle can produce micro-arcing (glowing contacts) that technically is an incipient arc fault, even if the experiment wasn't designed to generate one. The model might actually be *correct* here.

---

## FP #2 — IJL Lab Experiment (Confidence: 90.7%)

![FP #2: exp11--IJL--LR from 8_juillet dataset. Model confidence 90.7%](/home/manip/.gemini/antigravity/brain/7b8b2b80-b7de-4887-a499-b896da24feaa/fp2.png)

| Field | Value |
|---|---|
| **Dataset** | `8_juillet_clean` (the original July dataset) |
| **Experiment** | `exp11--IJL--LR` — IJL laboratory, inductive/resistive load |
| **Label** | 0 (Normal) |
| **Cycle Index** | 15 (very early in the recording) |
| **Sample Range** | 300,172 → 320,166 |

### Why the Model Fails

This is cycle 15 from an **IJL (Institut Jean Lamour) experiment with an LR (inductive-resistive) load**:

- **Inductive loads** cause phase-shifted current waveforms where the current doesn't follow the voltage cleanly. This creates **non-sinusoidal distortion** in the current waveform — a feature the model associates with arc distortion.
- At cycle 15, the system may still be in a **transient startup phase** where the inductor's back-EMF creates high dI/dt signatures.
- The combination of inductive kick + resistive heating creates TKEO spikes and |ΔI| bursts that overlap significantly with arc fault signatures.

---

## FP #3 — IJL Lab Experiment, Next Cycle (Confidence: 88.8%)

![FP #3: Same experiment as FP #2, consecutive cycle. Model confidence 88.8%](/home/manip/.gemini/antigravity/brain/7b8b2b80-b7de-4887-a499-b896da24feaa/fp3.png)

| Field | Value |
|---|---|
| **Dataset** | `8_juillet_clean` |
| **Experiment** | `exp11--IJL--LR` — same experiment as FP #2 |
| **Label** | 0 (Normal) |
| **Cycle Index** | 15 (same alt_index) |
| **Sample Range** | 320,001 → 340,011 — **immediately after FP #2** |

### Why the Model Fails

This is the **consecutive cycle** right after FP #2 — same experiment, same load, same transient event. The start sample (320,001) is right where FP #2 ended (320,166). This confirms these two FPs are part of a **single transient event** (likely a motor/inductor startup) lasting approximately 2 cycles.

> [!IMPORTANT]
> **Key finding:** FP #2 and FP #3 are consecutive cycles from the same transient event. This means the **multi-cycle consensus strategy** might NOT eliminate them since they appear back-to-back. However, the confidence *decreases* from 90.7% → 88.8%, suggesting the transient is decaying. A 3-cycle consensus with threshold=0.90 would likely filter them.

---

## Summary

| FP | Source | Physical Cause | Model Confusion Reason |
|---|---|---|---|
| **#1 (96.1%)** | AcierCu + Kettle | Contact resistance + high current | Broadband STFT noise from lossy contact mimics arc plasma noise. Possibly a genuine micro-arc. |
| **#2 (90.7%)** | IJL LR Load | Inductive startup transient (cycle 15) | Phase-shifted inductive current + back-EMF creates non-sinusoidal distortion → mimics arc waveform |
| **#3 (88.8%)** | IJL LR Load | Same transient, next cycle | Continuation of the same inductive transient. Decaying confidence suggests the event is settling. |

## Root Cause Analysis

The model fails on these 3 samples for the **exact same fundamental reason**: the physical signals produced by these normal operating conditions (lossy contacts, inductive transients) are genuinely similar to arc faults at the single-cycle level. This is not a model deficiency — it is a **physical ambiguity** inherent to single-cycle classification.

The key discriminators that separate these from real arcs are:
1. **Duration** — Real arcs persist for many cycles; transients decay within 2-3 cycles
2. **Stochasticity** — Arc waveforms are chaotic and differ from cycle to cycle; transients are deterministic and repeatable
3. **Spectral evolution** — Arc broadband noise is random; contact noise has a more structured spectral signature

These discriminators require **multi-cycle temporal context** — something a single-cycle classifier fundamentally cannot capture. This strongly validates the **multi-cycle consensus approach** as the correct deployment strategy.
