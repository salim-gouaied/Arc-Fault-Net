#!/usr/bin/env python3
"""
ARC FAULT DETECTION — STEP 4: Prepare OthmaneSalim10052026 Dataset
===================================================================
Prepares the new minimalist dataset for model testing.

Naming convention: C{1,2,3}{ExperimentName}{NNNNN}.csv
  e.g. C1AcierCu_AspiRouge00000.csv
       C2AcierCu_AspiRouge00000.csv
       C3AcierCu_AspiRouge00000.csv

Input  : CSV files from OthmaneSalim10052026/
Output : X_multi.npy   — (N, 2, 20000) multi-channel windows [V_ligne, I]
         y.npy         — (N,) binary labels {0, 1}
         metadata.csv  — per-sample metadata
         config.json   — pipeline parameters

Channel mapping (same as original pipeline):
  Channel 0 : V_ligne (C1) — mains voltage
  Channel 1 : I       (C3) — line current

NOTE: V_arc (C2) is used ONLY for labeling (oracle) and is NOT included
      as a model input.
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
DATA_DIR   = Path('/home/manip/pfe_salim_gouaied/Arc-Fault-Net/data/OthmaneSalim10052026')
OUTPUT_DIR = Path('/home/manip/pfe_salim_gouaied/Arc-Fault-Net/TestModel/prepared_data')

FS                = 1_000_000   # Sampling rate (Hz)
F0                = 50          # Mains frequency (Hz)
SAMPLES_PER_CYCLE = FS // F0    # 20 000 samples per full cycle
ZC_TOLERANCE      = 0.08        # ±8% tolerance on expected period

V_TH   = 10.0   # Arc voltage threshold on C2 (Volts)
R_LOW  = 0.05   # Ratio threshold for label=0 (normal)
R_HIGH = 0.95   # Ratio threshold for label=1 (arc)

HEADER_LINES  = 5
MIN_FILE_SIZE = 10_000   # Skip corrupt files smaller than this (bytes)


# ─────────────────────────────────────────────────────
#  STEP 0: Group CSV files into C1+C2+C3 triplets
# ─────────────────────────────────────────────────────

def group_experiments(data_dir: Path) -> dict:
    """
    Match C1/C2/C3 files that share the same experiment name and file number.

    Naming convention: C{1,2,3}{ExperimentName}{NNNNN}.csv
      Example: C1AcierCu_AspiRouge00000.csv
               C2AcierCu_AspiRouge00000.csv
               C3AcierCu_AspiRouge00000.csv

    Returns dict: { key: {c1, c2, c3, exp_name, file_num} }
    """
    files = list(data_dir.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    groups = defaultdict(dict)

    for f in sorted(files):
        if f.stat().st_size < MIN_FILE_SIZE:
            print(f"  SKIP (tiny file): {f.name} ({f.stat().st_size} bytes)")
            continue

        name = f.name
        # Pattern: C{1,2,3} followed by experiment name then 5-digit file number
        # e.g. C1AcierCu_AspiRouge00000.csv
        m = re.match(r'^C([123])(.+?)(\d{5})\.csv$', name)
        if not m:
            print(f"  SKIP (unrecognized name): {f.name}")
            continue

        channel_num = m.group(1)   # "1", "2", or "3"
        exp_name    = m.group(2)   # "AcierCu_AspiRouge"
        file_num    = m.group(3)   # "00000"

        key = f"{exp_name}{file_num}"
        groups[key][f'C{channel_num}'] = f
        groups[key]['_exp_name'] = exp_name
        groups[key]['_file_num'] = file_num

    # Keep only complete triplets (C1 + C2 + C3)
    complete = {}
    n_incomplete = 0
    for key, chans in groups.items():
        if all(c in chans for c in ['C1', 'C2', 'C3']):
            complete[key] = {
                'c1':      chans['C1'],
                'c2':      chans['C2'],
                'c3':      chans['C3'],
                'exp_name': chans['_exp_name'],
                'file_num': chans['_file_num'],
            }
        else:
            found = [c for c in ['C1', 'C2', 'C3'] if c in chans]
            print(f"  SKIP (incomplete triplet {found}): {key}")
            n_incomplete += 1

    if n_incomplete:
        print(f"  → {n_incomplete} incomplete triplets skipped")

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
#  Arc Ratio Computation (using C2 as oracle)
# ─────────────────────────────────────────────────────

def compute_arc_ratios(c2: np.ndarray, zc_indices: np.ndarray) -> list:
    """Compute arc_ratio for each cycle using C2 as oracle."""
    results = []
    n = len(zc_indices)

    for i in range(n - 1):
        start   = zc_indices[i]
        end     = zc_indices[i + 1]
        seg_len = end - start

        if abs(seg_len - SAMPLES_PER_CYCLE) > SAMPLES_PER_CYCLE * ZC_TOLERANCE:
            continue

        c2_seg = c2[start:end]
        n_arc  = np.sum(np.abs(c2_seg) > V_TH)
        ratio  = float(n_arc) / float(len(c2_seg))

        results.append({'start': int(start), 'end': int(end), 'ratio': ratio})

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

def prepare_dataset():
    """
    Build labeled dataset from OthmaneSalim10052026 CSV files.

    Outputs (saved to OUTPUT_DIR):
      X_multi.npy   : (N, 2, 20000) — [V_ligne (C1), I (C3)]
      y.npy         : (N,)          — binary labels {0=normal, 1=arc}
      metadata.csv  : per-sample metadata
      config.json   : pipeline parameters
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data source : {DATA_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print()

    # Step 0: group files
    experiments = group_experiments(DATA_DIR)
    print(f"Found {len(experiments)} complete C1+C2+C3 triplets\n")

    if len(experiments) == 0:
        print("ERROR: No valid experiments found. Check DATA_DIR and file naming.")
        return None

    # Collect all data
    X_list    = []
    y_list    = []
    meta_rows = []

    n_label0   = 0
    n_label1   = 0
    n_excluded = 0
    n_errors   = 0

    total = len(experiments)
    for i, (key, exp) in enumerate(sorted(experiments.items())):
        print(f"  [{i+1:03d}/{total}] {exp['exp_name']}{exp['file_num']}", end=' ')

        try:
            c1 = parse_csv(exp['c1'])   # V_ligne
            c2 = parse_csv(exp['c2'])   # V_arc (oracle for labeling only)
            c3 = parse_csv(exp['c3'])   # I

            zc = detect_zero_crossings(c1)
            if len(zc) < 2:
                print("→ SKIP (no valid zero crossings)")
                n_errors += 1
                continue

            ratios = compute_arc_ratios(c2, zc)
            if len(ratios) == 0:
                print("→ SKIP (no valid alternances)")
                n_errors += 1
                continue

        except Exception as e:
            print(f"→ ERROR: {e}")
            n_errors += 1
            continue

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

            # Extract C1 and C3 only (C2 excluded from model input)
            c1_seg = c1[start:end].astype(np.float32)
            c3_seg = c3[start:end].astype(np.float32)

            # Pad or truncate to exact SAMPLES_PER_CYCLE length
            segments = []
            for seg in [c1_seg, c3_seg]:
                seg_len = len(seg)
                if seg_len < SAMPLES_PER_CYCLE:
                    seg = np.pad(seg, (0, SAMPLES_PER_CYCLE - seg_len), mode='edge')
                elif seg_len > SAMPLES_PER_CYCLE:
                    seg = seg[:SAMPLES_PER_CYCLE]
                seg = normalize_segment(seg)
                segments.append(seg)

            x_multi = np.stack(segments, axis=0).astype(np.float32)   # (2, 20000)

            X_list.append(x_multi)
            y_list.append(label)
            n_kept += 1

            meta_rows.append({
                'exp_name':     exp['exp_name'],
                'file_num':     exp['file_num'],
                'alt_index':    alt_idx,
                'arc_ratio':    round(ratio, 5),
                'label':        label,
                'start_sample': start,
                'end_sample':   end,
            })

        print(f"→ {n_kept} samples kept")

    if len(X_list) == 0:
        print("\nERROR: No samples produced. Check data paths and file formats.")
        return None

    # Assemble final arrays
    X    = np.stack(X_list, axis=0)             # (N, 2, 20000)
    y    = np.array(y_list, dtype=np.int64)     # (N,)
    meta = pd.DataFrame(meta_rows)

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY — OthmaneSalim10052026")
    print(f"{'='*60}")
    print(f"  Total samples    : {len(y):>6d}")
    print(f"  Label 0 (normal) : {n_label0:>6d}  ({100*n_label0/len(y):.1f}%)")
    print(f"  Label 1 (arc)    : {n_label1:>6d}  ({100*n_label1/len(y):.1f}%)")
    print(f"  Excluded (mixed) : {n_excluded:>6d}")
    print(f"  Errors/skipped   : {n_errors:>6d}")
    print(f"  Matrix shape     : X{X.shape}")
    print(f"  Memory (X)       : {X.nbytes / 1e6:.1f} MB")

    # Save outputs
    np.save(OUTPUT_DIR / 'X_multi.npy', X)
    np.save(OUTPUT_DIR / 'y.npy',       y)
    meta.to_csv(OUTPUT_DIR / 'metadata.csv', index=False)

    config = {
        'dataset':          'OthmaneSalim10052026',
        'V_TH':             V_TH,
        'R_LOW':            R_LOW,
        'R_HIGH':           R_HIGH,
        'FS':               FS,
        'F0':               F0,
        'SAMPLES_PER_CYCLE': SAMPLES_PER_CYCLE,
        'n_channels':       2,
        'channel_names':    ['V_ligne (C1)', 'I (C3)'],
        'n_samples':        int(len(y)),
        'n_label0':         int(n_label0),
        'n_label1':         int(n_label1),
        'n_excluded':       int(n_excluded),
        'n_errors':         int(n_errors),
        'X_shape':          list(X.shape),
        'data_dir':         str(DATA_DIR),
        'output_dir':       str(OUTPUT_DIR),
    }
    with open(OUTPUT_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SAVED TO {OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"  X_multi.npy   {X.shape}  [channels: V_ligne, I]")
    print(f"  y.npy         {y.shape}")
    print(f"  metadata.csv  {meta.shape}")
    print(f"  config.json")

    return X, y, meta


if __name__ == '__main__':
    print("=" * 60)
    print("ARC-FAULTNET — DATA PREP: OthmaneSalim10052026")
    print("=" * 60)

    result = prepare_dataset()

    if result is not None:
        print("\n=== DONE ===")
        print("Next step: python TestModel/run_test.py")
    else:
        print("\n=== FAILED ===")
