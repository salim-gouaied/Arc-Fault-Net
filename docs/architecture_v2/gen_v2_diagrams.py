#!/usr/bin/env python3
"""
Arc-FaultNet V2 — architecture diagram generator.
====================================================

Regenerates the V2 figures that reference the temporal front-end channels,
after channel 1 was changed from the intra-cycle |dI| to Dowalla's INTER-cycle
residual  residu_k = I_k - I_{k-1}  (see dataset.py::_derive_i_channels).

Only the figures impacted by that change are produced here:

  00_global_technical.png   — system overview (channel 1 label + FUTURE block)
  01_frontend_channels.png  — Stage-0 front-end detail (channel 1 = residual)
  04_multicycle_future.png  — multi-cycle hooks (Stage-1 ΔI now implemented)
  07_data_pipeline_real.png — end-to-end real-data flow (channel 1 curve)
  08_why_best.png           — per-scale justification (residual = inter-cycle)
  11_system_simplified.png  — whole-system flow (V2 only, no multi-cycle)

Run with an interpreter that has matplotlib + numpy + scipy, e.g.:
    /home/top/miniconda3/bin/python docs/architecture_v2/gen_v2_diagrams.py

Real waveforms are loaded from the exp12 LeCroy CSVs when present; otherwise a
synthetic fallback cycle is used so the script still runs anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


# ──────────────────────────────────────────────────────────────────────────
#  Paths & constants
# ──────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
OUT = HERE / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

PROJECT = HERE.parent.parent
DATA_DIR_RAW = PROJECT / "data" / "drive-download-20260525T152045Z-3-001"
EXP_TAG = "exp12--IJL--LR--00023"

FS_RAW = 1_000_000     # raw LeCroy sampling rate (Hz)
F0 = 50                # mains frequency (Hz)
SPC = FS_RAW // F0     # samples per 50 Hz cycle (20000)
TARGET_M = 2048        # decimated cycle length used by the model
V_TH = 10.0            # arc-voltage threshold on C2 (V), oracle for labelling

DPI = 150

_REAL_CACHE: dict = {}


# ──────────────────────────────────────────────────────────────────────────
#  Shared visual style (mirrors the original V2 diagram set)
# ──────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

COL = {
    "input":     "#cfe2f3",   # light blue   — input
    "frontend":  "#b6d7a8",   # green        — front-end / derived channels
    "temporal":  "#f9cb9c",   # light orange — temporal branch
    "spectral":  "#f6b26b",   # orange       — spectral branch
    "fusion":    "#b4a7d6",   # purple       — cross-attention fusion
    "embed":     "#9fc5e8",   # blue         — embedding
    "tree":      "#f4a582",   # salmon       — tree head / decision
    "future":    "#e8e8e8",   # grey         — future (dashed)
    "delta":     "#ead1dc",   # mauve        — inter-cycle residual (channel 1)
    "text":      "#1a1a1a",
    "muted":     "#7a7a7a",
    "accent":    "#1155cc",   # heading blue
    "arc":       "#b22222",   # arc / trip red
}

# per-channel fill colours for the 4 temporal channels
CH_FILL = ["#b6d7a8", "#ead1dc", "#f9cb9c", "#f6b26b"]
CH_EDGE = ["#6aa84f", "#a64d79", "#e69138", "#bf6a16"]


def _round(ax, x, y, w, h, fc, ec=None, lw=1.4, rounding=0.02, ls="-",
           alpha=1.0, z=2):
    """Draw a rounded rectangle in data (axes-fraction) coordinates."""
    if ec is None:
        ec = _darken(fc)
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
        alpha=alpha, mutation_aspect=1.0, zorder=z,
    )
    ax.add_patch(p)
    return p


def _darken(hexc, f=0.62):
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    return (r / 255 * f, g / 255 * f, b / 255 * f)


def _txt(ax, x, y, s, size=9, weight="normal", color=None, ha="center",
         va="center", style="normal", z=4):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
            style=style, color=color or COL["text"], zorder=z)


def _arrow(ax, p0, p1, color=None, lw=1.6, style="-|>", rad=0.0, z=3,
           ls="-", mut=14):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=mut,
        linewidth=lw, color=color or COL["muted"],
        connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=z,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    return a


def _badge(ax, x, y, s, color=None, fc="white", char_w=0.0095, pad=0.022,
           hh=0.014):
    """Small pill showing a tensor shape, e.g. (B,128,D)."""
    half = max(0.03, len(s) * char_w / 2 + pad)
    _round(ax, x - half, y - hh, 2 * half, 2 * hh,
           fc=fc, ec=color or COL["muted"], lw=1.0, rounding=0.012, z=5)
    _txt(ax, x, y, s, size=7.2, color=color or COL["muted"], z=6)


def _new_ax(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ──────────────────────────────────────────────────────────────────────────
#  Real-data loading + channel derivation (mirror of dataset.py)
# ──────────────────────────────────────────────────────────────────────────
def _parse_lecroy(path: Path) -> np.ndarray:
    """Read the value column of a LeCroy CSV (5-line header)."""
    vals = np.loadtxt(path, delimiter=",", skiprows=5, usecols=1)
    return np.asarray(vals, dtype=np.float64)


def _load_real_cycles() -> dict:
    """Return real (or synthetic-fallback) cycles incl. the arc's PREVIOUS cycle.

    Keys: real, I_arc, I_arc_prev, I_normal, arc_ratio, n_cycles.
    The previous cycle is the contiguous 50 Hz cycle just before the arc cycle,
    which is exactly what dataset.py uses for residu_k = I_k - I_{k-1}.
    """
    if _REAL_CACHE:
        return _REAL_CACHE

    c1p = DATA_DIR_RAW / f"C1--{EXP_TAG}.csv"
    c2p = DATA_DIR_RAW / f"C2--{EXP_TAG}.csv"
    c3p = DATA_DIR_RAW / f"C3--{EXP_TAG}.csv"

    out = {"real": False}
    if c1p.exists() and c2p.exists() and c3p.exists():
        try:
            from scipy import signal as sp
            c1 = _parse_lecroy(c1p)
            c2 = _parse_lecroy(c2p)
            c3 = _parse_lecroy(c3p)
            v = c1.astype(np.float64); v -= v.mean()
            sos = sp.butter(4, [40, 60], btype="bandpass", fs=FS_RAW, output="sos")
            vf = sp.sosfiltfilt(sos, v)
            s = np.sign(vf)
            zc = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]

            def cycle(a):
                seg = c3[a:a + SPC]
                k = len(seg) // TARGET_M
                seg = seg[:k * TARGET_M].reshape(TARGET_M, k).mean(axis=1)
                return seg.astype(np.float64)

            best_arc, best_r, best_prev = None, -1.0, None
            normal = None
            for i in range(len(zc) - 1):
                a, b = zc[i], zc[i + 1]
                if not (SPC * 0.92 <= b - a <= SPC * 1.08):
                    continue
                r = float(np.mean(np.abs(c2[a:b]) > V_TH))
                if r > best_r:
                    best_r, best_arc = r, a
                    best_prev = zc[i - 1] if i >= 1 else a
                if r < 0.01 and normal is None:
                    normal = a
            if best_arc is not None and normal is not None:
                out = {
                    "real": True,
                    "I_arc": cycle(best_arc),
                    "I_arc_prev": cycle(best_prev),
                    "I_normal": cycle(normal),
                    "arc_ratio": best_r,
                    "n_cycles": int(len(zc) - 1),
                }
        except Exception as exc:  # pragma: no cover
            print(f"  (real-data load failed: {exc}; using synthetic)")

    if not out.get("real"):
        t = np.linspace(0, 1, TARGET_M)
        base = 12.0 * np.sin(2 * np.pi * t)
        arc = base.copy()
        zc_reg = (t > 0.46) & (t < 0.54); arc[zc_reg] *= 0.2     # zero-crossing dip
        burst = (t > 0.18) & (t < 0.34)
        arc[burst] += 4.0 * np.sin(2 * np.pi * 60 * t[burst]) * np.hanning(int(burst.sum()))
        rng = np.random.default_rng(0)
        arc = arc + 0.3 * rng.standard_normal(TARGET_M)
        arc_prev = base + 0.05 * rng.standard_normal(TARGET_M)  # clean previous cycle
        out = {"real": False, "I_arc": arc, "I_arc_prev": arc_prev,
               "I_normal": base, "arc_ratio": 0.98, "n_cycles": 50}

    _REAL_CACHE.update(out)
    return _REAL_CACHE


def _derive4(i_sig: np.ndarray, i_prev: np.ndarray | None = None):
    """NumPy mirror of dataset._derive_i_channels (visualisation only).

    Channel 1 is the Dowalla INTER-cycle residual residu_k = I_k - I_{k-1}
    (each cycle normalised by its OWN RMS). When i_prev is None the residual is
    zero (first cycle of a recording), matching dataset.py.
    """
    rms = np.sqrt(np.mean(i_sig ** 2) + 1e-12)
    i_norm = i_sig / rms
    if i_prev is not None:
        rms_p = np.sqrt(np.mean(i_prev ** 2) + 1e-12)
        residu = i_norm - (i_prev[:len(i_norm)] / rms_p)
    else:
        residu = np.zeros_like(i_norm)
    core = i_norm[1:-1] ** 2 - i_norm[:-2] * i_norm[2:]
    tkeo = np.concatenate([core[:1], core, core[-1:]])
    w = max(2, len(i_norm) // 4)
    pad = np.pad(i_norm ** 2, (w // 2, w - 1 - w // 2), mode="reflect")
    rms_slide = np.sqrt(np.convolve(pad, np.ones(w) / w, mode="valid") + 1e-12)
    return i_norm, residu, tkeo, rms_slide


def _stft_logpower(i_sig: np.ndarray, n_fft=128, hop=64):
    """Log-power STFT magnitude (matches dataset STFT settings) for a picture."""
    win = np.hanning(n_fft)
    n = len(i_sig)
    cols = []
    for s0 in range(0, n - n_fft + 1, hop):
        seg = i_sig[s0:s0 + n_fft] * win
        cols.append(np.abs(np.fft.rfft(seg)) ** 2)
    S = np.array(cols).T
    return np.log1p(S)


def _mini(ax, sig, color, lw=0.7, fill=False):
    """Plot a small inline waveform inside an inset axes."""
    ax.plot(np.arange(len(sig)), sig, color=color, lw=lw)
    if fill:
        ax.fill_between(np.arange(len(sig)), sig, sig.min(), color=color, alpha=0.18)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#888"); sp.set_linewidth(0.8)


# Channel metadata shared across figures (single source of truth)
CHANNELS = [
    ("0", "I_norm", "raw waveform / RMS", "global cycle shape"),
    ("1", "derived |ΔI_k|", "I_k − I_(k−1)", "discontinuités locales: fronts raides de l'arc"),
    ("2", "TKEO(I)", "I[n]² − I[n−1]·I[n+1]",
     "instantaneous energy: sub-cycle ignition / extinction"),
    ("3", "RMS_slide(I)", "sliding RMS over M/4",
     "amplitude envelope: flat shoulder / current dip"),
]


# ──────────────────────────────────────────────────────────────────────────
#  Figure builders are imported from the companion module
# ──────────────────────────────────────────────────────────────────────────
from _v2_figures import (
    diagram_frontend_channels,
    diagram_data_pipeline_real,
    diagram_global_technical,
    diagram_multicycle_future,
    diagram_param_budget_v2,
    diagram_system_simplified,
    diagram_why_best,
)


def main():
    data = _load_real_cycles()
    tag = "REAL exp12" if data.get("real") else "synthetic fallback"
    print(f"Arc-FaultNet V2 diagram generator — data source: {tag}")

    ctx = dict(
        COL=COL, CH_FILL=CH_FILL, CH_EDGE=CH_EDGE, CHANNELS=CHANNELS,
        OUT=OUT, DPI=DPI, data=data,
        _round=_round, _txt=_txt, _arrow=_arrow, _badge=_badge,
        _new_ax=_new_ax, _mini=_mini, _derive4=_derive4,
        _stft_logpower=_stft_logpower, _darken=_darken,
    )

    diagram_global_technical(ctx)
    diagram_frontend_channels(ctx)
    diagram_multicycle_future(ctx)
    diagram_data_pipeline_real(ctx)
    diagram_why_best(ctx)
    diagram_system_simplified(ctx)
    diagram_param_budget_v2(ctx)

    print(f"Done. PNGs written to {OUT}")


if __name__ == "__main__":
    main()
