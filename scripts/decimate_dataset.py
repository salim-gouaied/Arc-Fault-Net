#!/usr/bin/env python3
"""
DECIMATE DATASET — 20 000 → 2 048 points per alternance
=========================================================
Applies scipy.signal.resample_poly (FIR anti-aliasing filter)
to downsample every alternance from 1 MHz to ~102.4 kHz.

Usage:
    python scripts/decimate_dataset.py

Input:  combined_dataset/  (X_multi.npy, y.npy, metadata.csv, config.json)
Output: combined_dataset_2048/ (same files, decimated)
        + QA plots in combined_dataset_2048/qa/
"""

import numpy as np
import shutil
import json
import sys
from pathlib import Path
from scipy.signal import resample_poly

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'combined_dataset'
DST_DIR = PROJECT_ROOT / 'combined_dataset_2048'

# ── Decimation parameters ────────────────────────────────
# 20000 / 2048 = 625/64  →  resample_poly(x, up=64, down=625)
UP   = 64
DOWN = 625
FS_ORIG = 1_000_000       # Hz
FS_NEW  = FS_ORIG * UP / DOWN   # = 102 400 Hz
TARGET_LEN = 2048


def decimate_array(X: np.ndarray) -> np.ndarray:
    """
    Decimate X of shape (N, C, 20000) → (N, C, 2048).
    Uses resample_poly which applies an anti-aliasing FIR filter internally.
    """
    N, C, L = X.shape
    assert L == 20000, f"Expected 20000 pts, got {L}"

    X_dec = np.empty((N, C, TARGET_LEN), dtype=np.float32)

    for i in range(N):
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  Decimating sample {i+1}/{N} ...", flush=True)
        for c in range(C):
            x_dec = resample_poly(X[i, c], UP, DOWN).astype(np.float32)
            # resample_poly may produce TARGET_LEN ± 1 due to rounding;
            # enforce exact length
            if len(x_dec) > TARGET_LEN:
                x_dec = x_dec[:TARGET_LEN]
            elif len(x_dec) < TARGET_LEN:
                x_dec = np.pad(x_dec, (0, TARGET_LEN - len(x_dec)))
            X_dec[i, c] = x_dec

    return X_dec


def plot_qa_overlay(X_orig, X_dec, y, label_val, tag, qa_dir, n_samples=3):
    """Overlay original (thin grey) vs decimated (colored) for QA."""
    indices = np.where(y == label_val)[0]
    np.random.seed(42)
    picks = np.random.choice(indices, size=min(n_samples, len(indices)), replace=False)

    channel_names = ['V_ligne', 'I(t)']
    fig, axes = plt.subplots(len(picks), 2, figsize=(16, 4 * len(picks)))
    if len(picks) == 1:
        axes = axes.reshape(1, -1)

    t_orig = np.arange(20000) / FS_ORIG * 1000   # ms
    t_dec  = np.arange(TARGET_LEN) / FS_NEW * 1000  # ms

    for row, idx in enumerate(picks):
        for col in range(2):
            ax = axes[row, col]
            ax.plot(t_orig, X_orig[idx, col], color='silver', lw=0.4,
                    label=f'Original (20000 pts, 1 MHz)', alpha=0.8)
            ax.plot(t_dec, X_dec[idx, col], color='tab:blue' if col == 0 else 'tab:orange',
                    lw=0.8, label=f'Decimated (2048 pts, 102.4 kHz)')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel(channel_names[col])
            ax.set_title(f'Sample {idx} — {channel_names[col]}  '
                         f'({"arc" if label_val == 1 else "normal"})')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'Decimation QA — {tag}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(qa_dir / f'decimation_qa_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → decimation_qa_{tag}.png")


def plot_spectrum_comparison(X_orig, X_dec, y, qa_dir, n_samples=3):
    """Compare FFT magnitude spectrum before and after decimation."""
    indices = np.where(y == 1)[0]  # use arc samples
    np.random.seed(123)
    picks = np.random.choice(indices, size=min(n_samples, len(indices)), replace=False)

    fig, axes = plt.subplots(len(picks), 1, figsize=(14, 4 * len(picks)))
    if len(picks) == 1:
        axes = [axes]

    for row, idx in enumerate(picks):
        ax = axes[row]
        # Channel 1 = I(t)
        sig_orig = X_orig[idx, 1]
        sig_dec  = X_dec[idx, 1]

        # FFT
        fft_orig = np.abs(np.fft.rfft(sig_orig))
        fft_dec  = np.abs(np.fft.rfft(sig_dec))

        freq_orig = np.fft.rfftfreq(len(sig_orig), d=1/FS_ORIG) / 1000  # kHz
        freq_dec  = np.fft.rfftfreq(len(sig_dec),  d=1/FS_NEW)  / 1000  # kHz

        ax.semilogy(freq_orig, fft_orig + 1e-10, color='silver', lw=0.5,
                    label='Original (0–500 kHz)', alpha=0.7)
        ax.semilogy(freq_dec, fft_dec + 1e-10, color='tab:red', lw=0.8,
                    label='Decimated (0–51.2 kHz)')
        ax.axvline(x=51.2, color='green', ls='--', lw=1, alpha=0.6,
                   label='New Nyquist (51.2 kHz)')
        ax.axvspan(2, 50, alpha=0.08, color='yellow', label='Arc band (2–50 kHz)')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('|FFT|')
        ax.set_title(f'Sample {idx} — I(t) Spectrum (arc)')
        ax.legend(fontsize=8)
        ax.set_xlim([0, 100])
        ax.grid(True, alpha=0.3)

    fig.suptitle('Spectrum Comparison — Original vs Decimated', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(qa_dir / 'spectrum_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → spectrum_comparison.png")


def main():
    print("=" * 60)
    print("DECIMATE DATASET: 20 000 → 2 048 pts/alternance")
    print(f"  Method: resample_poly(up={UP}, down={DOWN})")
    print(f"  fs: {FS_ORIG:,} Hz → {FS_NEW:,.0f} Hz")
    print(f"  Anti-aliasing: FIR filter (Kaiser, automatic)")
    print("=" * 60)

    # ── Load source ───────────────────────────────────────
    print(f"\nLoading {SRC_DIR} ...")
    X_orig = np.load(SRC_DIR / 'X_multi.npy')
    y = np.load(SRC_DIR / 'y.npy')
    print(f"  X shape: {X_orig.shape}, dtype: {X_orig.dtype}")
    print(f"  y shape: {y.shape}")

    # ── Decimate ──────────────────────────────────────────
    print(f"\nDecimating {X_orig.shape[0]} samples ...")
    X_dec = decimate_array(X_orig)
    print(f"  Output shape: {X_dec.shape}")
    print(f"  Size reduction: {X_orig.nbytes / 1024**2:.0f} MB → {X_dec.nbytes / 1024**2:.0f} MB")

    # ── Save ──────────────────────────────────────────────
    DST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {DST_DIR} ...")

    np.save(DST_DIR / 'X_multi.npy', X_dec)
    print(f"  Saved X_multi.npy ({X_dec.nbytes / 1024**2:.1f} MB)")

    shutil.copy2(SRC_DIR / 'y.npy', DST_DIR / 'y.npy')
    print(f"  Copied y.npy")

    shutil.copy2(SRC_DIR / 'metadata.csv', DST_DIR / 'metadata.csv')
    print(f"  Copied metadata.csv")

    # Copy charges if exists
    if (SRC_DIR / 'charges.npy').exists():
        shutil.copy2(SRC_DIR / 'charges.npy', DST_DIR / 'charges.npy')
    if (SRC_DIR / 'charge_map.json').exists():
        shutil.copy2(SRC_DIR / 'charge_map.json', DST_DIR / 'charge_map.json')

    # Updated config
    with open(SRC_DIR / 'config.json') as f:
        config = json.load(f)

    config['FS'] = int(FS_NEW)
    config['SAMPLES_PER_CYCLE'] = TARGET_LEN
    config['X_shape'] = list(X_dec.shape)
    config['decimation'] = {
        'original_fs': FS_ORIG,
        'original_samples': 20000,
        'method': 'scipy.signal.resample_poly',
        'up': UP,
        'down': DOWN,
        'anti_aliasing': 'FIR Kaiser (automatic)',
    }

    with open(DST_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json (FS={int(FS_NEW)}, SAMPLES_PER_CYCLE={TARGET_LEN})")

    # ── QA Plots ──────────────────────────────────────────
    qa_dir = DST_DIR / 'qa'
    qa_dir.mkdir(exist_ok=True)
    print(f"\nGenerating QA plots ...")

    plot_qa_overlay(X_orig, X_dec, y, label_val=1, tag='arc', qa_dir=qa_dir)
    plot_qa_overlay(X_orig, X_dec, y, label_val=0, tag='normal', qa_dir=qa_dir)
    plot_spectrum_comparison(X_orig, X_dec, y, qa_dir)

    print(f"\n{'='*60}")
    print(f"DONE — Decimated dataset saved to: {DST_DIR}")
    print(f"  Samples: {X_dec.shape[0]}")
    print(f"  Shape:   {X_dec.shape}")
    print(f"  fs:      {FS_NEW:,.0f} Hz")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
