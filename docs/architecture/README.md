# Arc-FaultNet — architecture documentation

This directory documents the **model architecture** of Arc-FaultNet
module by module, together with research-paper-style diagrams. Each
diagram is a clean PNG (no Mermaid) produced by
[`gen_diagrams.py`](gen_diagrams.py) using `matplotlib`.

## How to read this folder

* Start with the **end-to-end approach** to get the big picture, then
  jump to the **model architecture** (the main diagram).
* Each module has its own diagram + dedicated markdown describing:
  * what the module does,
  * how it is implemented (with line-anchored code references), and
  * what is the **scientific contribution** of the work (reused vs.
    original).

## Table of contents

| # | Module | Diagram | Markdown |
|---|--------|---------|----------|
| 0 | End-to-end approach | [diagram](diagrams/00_overall_approach.png) | [modules/00_overall_approach.md](modules/00_overall_approach.md) |
| 1 | **Arc-FaultNet — main architecture** | [diagram](diagrams/01_model_architecture.png) | [modules/01_model_architecture.md](modules/01_model_architecture.md) |
| 2 | Branch 1D — temporal feature extractor | [diagram](diagrams/02_branch1d.png) | [modules/02_branch1d.md](modules/02_branch1d.md) |
| 3 | Parametric Gabor convolution (`ParametricConv1d`) | [diagram](diagrams/03_parametric_gabor.png) | [modules/03_parametric_gabor.md](modules/03_parametric_gabor.md) |
| 4 | Branch 2D — spectral feature extractor | [diagram](diagrams/04_branch2d.png) | [modules/04_branch2d.md](modules/04_branch2d.md) |
| 5 | Joint Attention — cross-branch CAM + SAM | [diagram](diagrams/05_joint_attention.png) | [modules/05_joint_attention.md](modules/05_joint_attention.md) |
| 6 | Channel Attention Module (CAM) | [diagram](diagrams/06_channel_attention.png) | [modules/06_channel_attention.md](modules/06_channel_attention.md) |
| 7 | Spatial / Temporal Attention Module (SAM) | [diagram](diagrams/07_spatial_attention.png) | [modules/07_spatial_attention.md](modules/07_spatial_attention.md) |
| 8 | Classifier head | [diagram](diagrams/08_classifier_head.png) | [modules/08_classifier_head.md](modules/08_classifier_head.md) |
| 9 | Data pipeline | [diagram](diagrams/09_data_pipeline.png) | [modules/09_data_pipeline.md](modules/09_data_pipeline.md) |
| 10 | **Layer-by-layer node view** (whole network, from input to output) | [diagram](diagrams/10_network_nodes.png) | section in [modules/01_model_architecture.md](modules/01_model_architecture.md#5-layer-by-layer-node-view) |

### Supplementary / analysis figures

These figures do not introduce new modules; they give the reader
**alternative views** of the same architecture and dataset, of the
kind that typically accompany a deep-learning paper.

| # | Figure | Diagram | Markdown |
|---|--------|---------|----------|
| 11 | Tensor-shape flow (cuboid view across both branches) | [diagram](diagrams/11_tensor_flow.png) | [modules/11_tensor_flow.md](modules/11_tensor_flow.md) |
| 12 | Receptive-field cascade of Branch 1D, in real time | [diagram](diagrams/12_receptive_field_cascade.png) | [modules/12_receptive_field.md](modules/12_receptive_field.md) |
| 13 | What the model sees: real `exp13` cycles in time and STFT | [diagram](diagrams/13_input_examples.png) | [modules/13_input_examples.md](modules/13_input_examples.md) |
| 14 | Three-zone arc-ratio histogram (labeling oracle) | [diagram](diagrams/14_arc_ratio_histogram.png) | [modules/14_arc_ratio_histogram.md](modules/14_arc_ratio_histogram.md) |
| 15 | Parameter budget (live counts from the real model) | [diagram](diagrams/15_param_budget.png) | [modules/15_param_budget.md](modules/15_param_budget.md) |
| 16 | Parametric Gabor filter atlas + $(f_0,\sigma)$ scatter | [diagram](diagrams/16_gabor_atlas.png) | [modules/16_gabor_atlas.md](modules/16_gabor_atlas.md) |

## Summary of the scientific contribution

| Pillar | Where it lives | Reused / Original |
|--------|----------------|-------------------|
| Three-zone arc-ratio labeling on `C2`; `C2` discarded from model inputs | [data pipeline](modules/09_data_pipeline.md) | **Original** |
| Parametric Gabor convolutions with learnable $(f_0, \sigma)$ initialised in physical units | [Branch 1D](modules/02_branch1d.md) + [Parametric Gabor](modules/03_parametric_gabor.md) | Inspired by SincNet / MC-VSAttn — **re-applied** to arc faults at 1 MHz |
| **Dual-branch design** (raw 1D temporal + restricted 2–100 kHz STFT) | [Branch 1D](modules/02_branch1d.md) + [Branch 2D](modules/04_branch2d.md) | **Original** |
| Frequency-band restriction to 2–100 kHz motivated by arc physics | [Branch 2D](modules/04_branch2d.md) | **Original** |
| Joint Attention (CAM + SAM on joint context) with **clean per-branch split / projections** for diagnostic traceability | [Joint Attention](modules/05_joint_attention.md) | Adapted from MC-VSAttn — **redesigned** for diagnostic clarity |

## Out of scope (for now)

The following pieces of the project are *not* documented here, as
requested:

* the Leave-One-Charge-Out cross-validation splitter
  (`LeaveOneChargeOutSplitter`) — to be discussed when generalisation
  experiments are run;
* the ablation framework (`ablation.py`) — will be activated to
  quantify each pillar's contribution;
* `mini_evaluate.py`, `runs/`, `colab_plotter.py`.

## Regenerating the diagrams

```bash
/home/top/miniconda3/bin/python docs/architecture/gen_diagrams.py
```

This rewrites every `*.png` in `docs/architecture/diagrams/`. All
figures are saved at 220 DPI with a tight bounding box, suitable for
inclusion in LaTeX or Word manuscripts.
