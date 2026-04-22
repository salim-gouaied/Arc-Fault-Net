#!/usr/bin/env python3
"""
ARC FAULT DETECTION — STEP 2: Build Multi-Channel Matrix
=========================================================
Extends step1 to produce 2-channel input for Arc-FaultNet.

Input  : Raw CSV files from Teledyne LeCroy oscilloscope
Output : X_multi.npy   — (N, 2, 20000) multi-channel windows
         y.npy         — (N,) binary labels {0, 1}
         charges.npy   — (N,) charge type indices for leave-one-out CV
         charge_map.json — mapping from charge name to index
         metadata.csv  — per-sample metadata

Channel mapping:
  Channel 0 : V_ligne (C1) — mains voltage, phase reference
  Channel 1 : I       (C3) — line current, load-dependent

NOTE: V_arc (C2) is used ONLY for labeling (oracle signal) and is NOT
      included as a model input. In a real deployment, arc voltage cannot
      be measured because the location of the arc is unknown in advance.
      The model must detect arcs solely from V_ligne and I.
"""

import numpy as np
import pandas as pd
from scipy import signal as sp
from pathlib import Path
import re
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────
DATA_DIR   = Path('/home/top/PFE/OthmaneSalim11032026')
OUTPUT_DIR = Path('/home/top/PFE/labeled_dataset')

FS                 = 1_000_000   # Sampling rate (Hz)
F0                 = 50          # Mains frequency (Hz)
SAMPLES_PER_CYCLE  = FS // F0    # 20 000 samples per full cycle
ZC_TOLERANCE       = 0.08        # ±8% tolerance on expected period

V_TH   = 10.0   # Arc voltage threshold on C2 (Volts)
R_LOW  = 0.05   # Ratio threshold for label=0 (normal)
R_HIGH = 0.95   # Ratio threshold for label=1 (arc)

HEADER_LINES = 5

# ─────────────────────────────────────────────────────
#  STEP 0: Group CSV files into experiments
# ─────────────────────────────────────────────────────

def group_experiments(data_dir: Path) -> dict:
    """Match C1/C2/C3 files that share the same experiment suffix."""
    files = list(data_dir.glob('*.csv'))
    groups = defaultdict(dict)

    for f in sorted(files):
        name = f.name
        m = re.match(r'^(C[123])EE (.+)$', name)
        if not m:
            continue
        channel = m.group(1)
        suffix  = m.group(2)
        groups[suffix][channel] = f

    complete = {}
    for suffix, chans in groups.items():
        if all(c in chans for c in ['C1', 'C2', 'C3']):
            m = re.search(r'Arc_([^+]+)\+(.+?)\d{5}\.csv$', suffix)
            if m:
                arc_load = m.group(1)
                bg_loads = m.group(2).split('+')
            else:
                arc_load = 'unknown'
                bg_loads = []
            
            # Create a charge identifier combining arc_load and bg_loads
            charge_id = f"{arc_load}_{'+'.join(bg_loads)}"
            
            complete[suffix] = {
                'c1': chans['C1'],
                'c2': chans['C2'],
                'c3': chans['C3'],
                'arc_load': arc_load,
                'bg_loads': bg_loads,
                'charge_id': charge_id,
                'name': suffix
            }

    return complete


# ─────────────────────────────────────────────────────
#  CSV Parsing
# ─────────────────────────────────────────────────────

def parse_csv(filepath: Path) -> np.ndarray:
    """Parse LeCroy CSV export, returns amplitude array."""
    data = pd.read_csv(
        filepath,
        skiprows=HEADER_LINES,
        header=0,
        names=['Time', 'Ampl'],
        dtype={'Ampl': np.float32},
        usecols=['Ampl'],
        engine='c'
    )
    return data['Ampl'].values


# ─────────────────────────────────────────────────────
#  Zero Crossing Detection (on C1)
# ─────────────────────────────────────────────────────

def detect_zero_crossings(v: np.ndarray) -> np.ndarray:
    """Detect positive-going zero crossings on voltage signal C1."""
    v = v.astype(np.float64)
    v = v - np.mean(v)

    sos = sp.butter(4, [40, 60], btype='bandpass', fs=FS, output='sos')
    v_filt = sp.sosfiltfilt(sos, v)

    signs = np.sign(v_filt)
    crossings = np.where((signs[:-1] <= 0) & (signs[1:] > 0))[0]

    if len(crossings) < 2:
        return np.array([], dtype=int)

    tol = int(SAMPLES_PER_CYCLE * ZC_TOLERANCE)

    validated = [crossings[0]]
    for idx in crossings[1:]:
        spacing = idx - validated[-1]
        if abs(spacing - SAMPLES_PER_CYCLE) <= tol:
            validated.append(idx)
        elif spacing < SAMPLES_PER_CYCLE - tol:
            continue
        else:
            validated.append(idx)

    return np.array(validated, dtype=int)


# ─────────────────────────────────────────────────────
#  Arc Ratio Computation
# ─────────────────────────────────────────────────────

def compute_arc_ratios(c2: np.ndarray, zc_indices: np.ndarray) -> list:
    """Compute arc_ratio for each cycle using C2 as oracle."""
    results = []
    n = len(zc_indices)
    
    for i in range(n - 1):
        start = zc_indices[i]
        end   = zc_indices[i + 1]
        seg_len = end - start

        if abs(seg_len - SAMPLES_PER_CYCLE) > SAMPLES_PER_CYCLE * ZC_TOLERANCE:
            continue

        c2_seg = c2[start:end]
        n_arc  = np.sum(np.abs(c2_seg) > V_TH)
        ratio  = float(n_arc) / float(len(c2_seg))

        results.append({
            'start': int(start),
            'end':   int(end),
            'ratio': ratio
        })

    return results


# ─────────────────────────────────────────────────────
#  Per-channel normalization
# ─────────────────────────────────────────────────────

def normalize_segment(seg: np.ndarray) -> np.ndarray:
    """Z-score normalization per segment."""
    mean = np.mean(seg)
    std  = np.std(seg)
    if std < 1e-9:
        return seg - mean
    return (seg - mean) / std


# ─────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────

def build_multichannel_matrix():
    """
    Build 2-channel labeled matrix for Arc-FaultNet.

    Only V_ligne (C1) and I (C3) are included as model inputs.
    V_arc (C2) is used internally for labeling only and then discarded.

    Outputs:
      X_multi.npy   : (N, 2, 20000) - 2 channels per cycle (V_ligne, I)
      y.npy         : (N,) - binary labels
      charges.npy   : (N,) - charge type index per sample
      charge_map.json : charge name -> index mapping
      metadata.csv  : full metadata per sample
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = group_experiments(DATA_DIR)
    print(f"Found {len(experiments)} complete experiments")

    # Build charge mapping
    unique_charges = sorted(set(exp['charge_id'] for exp in experiments.values()))
    charge_map = {name: idx for idx, name in enumerate(unique_charges)}
    print(f"Found {len(unique_charges)} unique charge configurations:")
    for name, idx in charge_map.items():
        print(f"  [{idx:2d}] {name}")

    # Collect all data
    X_list = []
    y_list = []
    charge_list = []
    meta_rows = []

    n_label0 = 0
    n_label1 = 0
    n_excluded = 0

    for i, (exp_name, exp) in enumerate(sorted(experiments.items())):
        print(f"  [{i+1:02d}/{len(experiments)}] {exp['charge_id']:<30s}", end=' ')

        # Parse all 3 channels
        c1 = parse_csv(exp['c1'])  # V_ligne
        c2 = parse_csv(exp['c2'])  # V_arc
        c3 = parse_csv(exp['c3'])  # I

        # Detect zero crossings on C1
        zc = detect_zero_crossings(c1)
        ratios = compute_arc_ratios(c2, zc)

        charge_idx = charge_map[exp['charge_id']]
        n_kept = 0

        for alt_idx, alt in enumerate(ratios):
            ratio = alt['ratio']
            start = alt['start']
            end   = alt['end']

            # Three-zone labeling
            if ratio <= R_LOW:
                label = 0
                n_label0 += 1
            elif ratio >= R_HIGH:
                label = 1
                n_label1 += 1
            else:
                n_excluded += 1
                continue

            # Extract segments from model input channels only (C1 and C3).
            # C2 (V_arc) is intentionally excluded: it is the oracle signal
            # used for labeling and is not available at inference time.
            c1_seg = c1[start:end].astype(np.float32)
            c3_seg = c3[start:end].astype(np.float32)

            # Pad or truncate to exact length
            segments = []
            for seg in [c1_seg, c3_seg]:
                seg_len = len(seg)
                if seg_len < SAMPLES_PER_CYCLE:
                    seg = np.pad(seg, (0, SAMPLES_PER_CYCLE - seg_len), mode='edge')
                elif seg_len > SAMPLES_PER_CYCLE:
                    seg = seg[:SAMPLES_PER_CYCLE]
                # Normalize each channel independently
                seg = normalize_segment(seg)
                segments.append(seg)

            # Stack to (2, 20000)
            x_multi = np.stack(segments, axis=0).astype(np.float32)

            X_list.append(x_multi)
            y_list.append(label)
            charge_list.append(charge_idx)
            n_kept += 1

            meta_rows.append({
                'exp_name':     exp_name,
                'arc_load':     exp['arc_load'],
                'bg_loads':     '+'.join(exp['bg_loads']),
                'charge_id':    exp['charge_id'],
                'charge_idx':   charge_idx,
                'alt_index':    alt_idx,
                'arc_ratio':    round(ratio, 5),
                'label':        label,
                'start_sample': start,
                'end_sample':   end
            })

        print(f"→ {n_kept} samples kept")

    # Assemble final arrays
    X = np.stack(X_list, axis=0)  # (N, 2, 20000)
    y = np.array(y_list, dtype=np.int64)
    charges = np.array(charge_list, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    print(f"\n{'='*50}")
    print(f"DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"  Total samples    : {len(y):>5d}")
    print(f"  Label 0 (normal) : {n_label0:>5d}  ({100*n_label0/len(y):.1f}%)")
    print(f"  Label 1 (arc)    : {n_label1:>5d}  ({100*n_label1/len(y):.1f}%)")
    print(f"  Excluded         : {n_excluded:>5d}")
    print(f"  Matrix shape     : X{X.shape}")
    print(f"  Memory (X)       : {X.nbytes / 1e6:.1f} MB")
    print(f"  Unique charges   : {len(unique_charges)}")

    # Print samples per charge
    print(f"\n  Samples per charge configuration:")
    for charge_name, charge_idx in charge_map.items():
        n_samples = np.sum(charges == charge_idx)
        n_arc = np.sum((charges == charge_idx) & (y == 1))
        n_normal = np.sum((charges == charge_idx) & (y == 0))
        print(f"    [{charge_idx:2d}] {charge_name:<30s}: {n_samples:4d} ({n_normal:3d}N/{n_arc:3d}A)")

    # Save outputs
    np.save(OUTPUT_DIR / 'X_multi.npy', X)
    np.save(OUTPUT_DIR / 'y.npy', y)
    np.save(OUTPUT_DIR / 'charges.npy', charges)
    meta.to_csv(OUTPUT_DIR / 'metadata.csv', index=False)

    with open(OUTPUT_DIR / 'charge_map.json', 'w') as f:
        json.dump(charge_map, f, indent=2)

    config = {
        'V_TH': V_TH,
        'R_LOW': R_LOW,
        'R_HIGH': R_HIGH,
        'FS': FS,
        'F0': F0,
        'SAMPLES_PER_CYCLE': SAMPLES_PER_CYCLE,
        'n_channels': 2,
        'channel_names': ['V_ligne', 'I'],
        'n_samples': int(len(y)),
        'n_label0': int(n_label0),
        'n_label1': int(n_label1),
        'n_excluded': int(n_excluded),
        'n_charges': len(unique_charges),
        'X_shape': list(X.shape),
    }
    with open(OUTPUT_DIR / 'config_multi.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SAVED TO {OUTPUT_DIR}")
    print(f"{'='*50}")
    print(f"  X_multi.npy     {X.shape}  [channels: V_ligne, I]")
    print(f"  y.npy           {y.shape}")
    print(f"  charges.npy     {charges.shape}")
    print(f"  metadata.csv    {meta.shape}")
    print(f"  charge_map.json ({len(charge_map)} charges)")
    print(f"  config_multi.json")

    return X, y, charges, charge_map, meta


if __name__ == '__main__':
    print("=" * 60)
    print("ARC-FAULTNET — MULTI-CHANNEL DATA PIPELINE")
    print("=" * 60)
    
    X, y, charges, charge_map, meta = build_multichannel_matrix()
    
    print("\n=== DONE ===")
    print("Next step: python train.py")
