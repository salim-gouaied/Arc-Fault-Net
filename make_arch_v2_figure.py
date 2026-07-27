import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ---- palette (ML-paper style) ----
TEMP  = ("#2C6DB5", "#E6EFFA")   # temporal branch
SPEC  = ("#1E8C74", "#E1F3EE")   # spectral branch
NEU   = ("#445168", "#EBEEF3")   # input / neutral
ATTN  = ("#C8891B", "#FBEFCF")   # attention accents
FUSE  = ("#7A46B8", "#EDE4F9")   # fusion
OUT   = ("#C0392B", "#FBE4E1")   # output
FEAT  = ("#667085", "#F1F3F6")   # feature map

fig, ax = plt.subplots(figsize=(20.7, 9.4), dpi=200)
ax.set_xlim(0, 20.7); ax.set_ylim(0, 9.4); ax.axis("off")

def node(x, y, w, h, title, details, shape, cmain, cfill, tsize=12, dsize=9.3):
    ax.add_patch(FancyBboxPatch((x+0.06, y-0.08), w, h,
        boxstyle="round,pad=0,rounding_size=0.14", fc="0.72", ec="none",
        alpha=0.35, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.14", fc=cfill, ec=cmain,
        lw=2.0, zorder=2))
    ax.add_patch(Rectangle((x+0.03, y+0.15), 0.14, h-0.30, fc=cmain,
        ec="none", zorder=3))
    ax.text(x+0.34, y+h-0.27, title, ha="left", va="top", fontsize=tsize,
        fontweight="bold", color=cmain, zorder=4)
    ax.text(x+0.34, y+h-0.30-0.34, details, ha="left", va="top",
        fontsize=dsize, color="#222222", zorder=4, linespacing=1.4)
    if shape:
        ax.text(x+w-0.16, y+0.15, shape, ha="right", va="bottom",
            fontsize=8.8, style="italic", color=cmain, zorder=4)

def frame(x0, y0, x1, y1, label, cmain):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=0,rounding_size=0.18", fc=cmain, ec=cmain,
        lw=1.3, alpha=0.06, zorder=0))
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=0,rounding_size=0.18", fc="none", ec=cmain,
        lw=1.3, ls=(0, (5, 3)), alpha=0.55, zorder=0))
    ax.text(x0+0.18, y1-0.05, label, ha="left", va="top", fontsize=10.5,
        fontweight="bold", color=cmain, alpha=0.9, zorder=1)

def arrow(p1, p2, rad=0.0, color="#3d4450"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=17,
        lw=1.9, color=color, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2, zorder=5))

H = 1.55
yT = 6.35            # temporal row bottom
yS = 1.55            # spectral row bottom
yM = 4.6 - H/2       # middle row bottom (input/fusion/head/output)
cT = yT + H/2; cS = yS + H/2; cM = yM + H/2

# ---- branch frames ----
frame(2.95, yT-0.30, 12.15, yT+H+0.30, "TEMPORAL BRANCH", TEMP[0])
frame(2.95, yS-0.30, 12.15, yS+H+0.30, "SPECTRAL BRANCH", SPEC[0])

# ---- input ----
node(0.35, yM, 2.35, H, "Input",
     "Line current  $I(t)$\nsingle 50 Hz cycle @ 102.4 kHz",
     "(1 × 2048)", NEU[0], NEU[1], tsize=12, dsize=9.0)

# ---- temporal branch ----
node(3.10, yT, 3.05, H, "Derived descriptors",
     "$[\\,I_{norm},\\ |\\Delta I|,\\ \\mathrm{TKEO},\\ \\mathrm{RMS}_{slide}\\,]$\nphysics-informed channels",
     "(4 × 2048)", TEMP[0], TEMP[1], dsize=9.0)
node(6.40, yT, 3.55, H, "Temporal encoder (1-D CNN)",
     "3 × [ Conv1d $k$=16/8/4 + BN + GELU + SE ]\nch. 32 → 64 → 128,  MaxPool ×4",
     None, TEMP[0], TEMP[1], dsize=8.5)
node(10.20, yT+0.22, 1.80, H-0.44, "Feature map",
     "temporal", "(128 × 64)", FEAT[0], FEAT[1], tsize=10.5, dsize=8.6)

# ---- spectral branch ----
node(3.10, yS, 3.05, H, "STFT spectrogram",
     "log-power of $I(t)$\n$n_{fft}$=128, hop=64",
     "(1 × 65 × 31)", SPEC[0], SPEC[1], dsize=9.0)
node(6.40, yS, 3.55, H, "Spectral encoder (2-D CNN)",
     "FreqGate + 3 × [ Conv2d 3×3 + BN + GELU + SE ]\nfrequency-only pooling",
     None, SPEC[0], SPEC[1], dsize=8.5)
node(10.20, yS+0.22, 1.80, H-0.44, "Feature map",
     "spectral", "(128 × 64)", FEAT[0], FEAT[1], tsize=10.5, dsize=8.6)

# ---- fusion ----
node(12.40, yM, 3.20, H, "Sequential Cross-Attention",
     "bidirectional Q/K/V\n4 heads,  $d_k$=32\ntemporal ⇄ spectral",
     "→ (128)", FUSE[0], FUSE[1], tsize=10.5, dsize=9.0)

# ---- head + output ----
node(15.95, yM, 2.35, H, "Classifier head",
     "FC 128 → 64 → 1\n(+ dropout)",
     "(1)", NEU[0], NEU[1], dsize=9.0)
node(18.55, yM+0.18, 1.75, H-0.36, "Output",
     "$\\sigma(\\cdot)$", "$P(\\mathrm{arc})\\in[0,1]$", OUT[0], OUT[1],
     tsize=11.5, dsize=9.5)

# ---- arrows ----
arrow((2.70, cM+0.15), (3.10, cT-0.10), rad=0.28, color=TEMP[0])
arrow((2.70, cM-0.15), (3.10, cS+0.10), rad=-0.28, color=SPEC[0])
arrow((6.15, cT), (6.40, cT), color=TEMP[0])
arrow((9.95, cT), (10.20, cT), color=TEMP[0])
arrow((6.15, cS), (6.40, cS), color=SPEC[0])
arrow((9.95, cS), (10.20, cS), color=SPEC[0])
arrow((12.00, cT-0.10), (12.40, cM+0.30), rad=-0.28, color=TEMP[0])
arrow((12.00, cS+0.10), (12.40, cM-0.30), rad=0.28, color=SPEC[0])
arrow((15.60, cM), (15.95, cM), color=FUSE[0])
arrow((18.30, cM), (18.55, cM), color="#3d4450")

# ---- derivative callout (in the central gap, pointing up to |dI|) ----
cx0, cy0, cw, chh = 3.70, 3.62, 6.7, 1.02
ax.add_patch(FancyBboxPatch((cx0, cy0), cw, chh,
    boxstyle="round,pad=0.10,rounding_size=0.12", fc=ATTN[1], ec=ATTN[0],
    lw=1.5, zorder=6))
ax.text(cx0+0.28, cy0+chh-0.30,
    "$|\\Delta I| = |I[n]-I[n-1]|$   —   derivative of $I(t)$",
    ha="left", va="center", fontsize=11, fontweight="bold",
    color=ATTN[0], zorder=7)
ax.text(cx0+0.28, cy0+0.34,
    "core load-invariant arc cue: captures re-ignition transients\ncommon to all load types → drives generalization",
    ha="left", va="center", fontsize=9.0, color="#5a4310", zorder=7,
    linespacing=1.35)
ax.add_patch(FancyArrowPatch((4.65, cy0+chh+0.02), (4.65, yT-0.02),
    arrowstyle="-|>", mutation_scale=15, lw=1.7, color=ATTN[0],
    ls=(0, (4, 2)), zorder=7))

# ---- title + footnote ----
ax.text(0.35, 9.15, "Arc-FaultNet V2 — Baseline Architecture",
    ha="left", va="top", fontsize=17, fontweight="bold", color="#1a1a1a")
ax.text(0.35, 8.66,
    "Dual-branch (temporal + spectral) single-cycle detector with sequential cross-attention fusion",
    ha="left", va="top", fontsize=11, color="#555555")

ax.text(0.35, 0.62,
    "Attention modules — SE channel attention (inside each conv block) and Sequential Cross-Attention (fusion) — are detailed in the following figures.",
    ha="left", va="center", fontsize=9.2, color="#444444")
ax.text(0.35, 0.28,
    "Base model ≈ 0.3M parameters  ·  input decimated to 102.4 kHz  ·  arc voltage used only for offline labeling (never a model input).",
    ha="left", va="center", fontsize=9.2, color="#444444")

plt.savefig("/home/top/Arc-Fault-Net/arcfaultnet_v2_architecture.png",
    bbox_inches="tight", facecolor="white", pad_inches=0.25)
print("saved")
