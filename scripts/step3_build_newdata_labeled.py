#!/usr/bin/env python3
"""
ARC FAULT DETECTION — STEP 3: Build Labeled Matrix from New Dataset
====================================================================
Adapted from step2_build_multichannel.py for the new CSV data
(8_juillet_clean, 15_juillet_clean, 22_juillet_clean).

Key differences from step2:
  - New file naming convention: C{1,2,3}--exp{NN}--IJL--LR--{NNNNN}.csv
  - Multiple source directories (one per date)
  - No charge metadata available → no charge mapping
  - Corrupt/tiny files are skipped automatically

Input  : CSV files from 3 clean directories
Output : X_multi.npy   — (N, 2, 20000) multi-channel windows [V_ligne, I]
         y.npy         — (N,) binary labels {0, 1}
         metadata.csv  — per-sample metadata
         config_multi.json — pipeline parameters

Channel mapping (same as step2):
  Channel 0 : V_ligne (C1) — mains voltage, phase reference
  Channel 1 : I       (C3) — line current, load-dependent

NOTE: V_arc (C2) is used ONLY for labeling (oracle) and is NOT
      included as a model input.
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
DATA_DIRS = [
    Path('/home/manip/pfe_salim_gouaied/Arc-Fault-Net/data/DataSet/8_juillet_clean'),
    Path('/home/manip/pfe_salim_gouaied/Arc-Fault-Net/data/DataSet/15_juillet_clean'),
    Path('/home/manip/pfe_salim_gouaied/Arc-Fault-Net/data/DataSet/22_juillet_clean'),
]
OUTPUT_DIR = Path('/home/manip/pfe_salim_gouaied/Arc-Fault-Net/labeled_dataset')

FS                 = 1_000_000   # Sampling rate (Hz)
F0                 = 50          # Mains frequency (Hz)
SAMPLES_PER_CYCLE  = FS // F0    # 20 000 samples per full cycle
ZC_TOLERANCE       = 0.08        # ±8% tolerance on expected period

V_TH   = 10.0   # Arc voltage threshold on C2 (Volts)
R_LOW  = 0.05   # Ratio threshold for label=0 (normal)
R_HIGH = 0.95   # Ratio threshold for label=1 (arc)

HEADER_LINES = 5
MIN_FILE_SIZE = 10_000  # Skip corrupt files smaller than this (bytes)


# ─────────────────────────────────────────────────────
#  STEP 0: Group CSV files into experiments
# ─────────────────────────────────────────────────────

def group_experiments(data_dirs: list) -> dict:
    """
    Match C1/C2/C3 files that share the same experiment suffix.
    Scans multiple directories and groups by (dir_name, suffix_number).

    New naming convention: C{1,2,3}--exp{NN}--IJL--LR--{NNNNN}.csv

    Returns dict: { unique_key: {c1, c2, c3, source_dir, exp_id, file_num} }
    """
    all_groups = {}

    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"  WARNING: {data_dir} does not exist, skipping")
            continue

        dir_name = data_dir.name
        files = list(data_dir.glob('*.csv'))
        groups = defaultdict(dict)

        for f in sorted(files):
            # Skip corrupt/tiny files
            if f.stat().st_size < MIN_FILE_SIZE:
                print(f"  SKIP (tiny): {f.name} ({f.stat().st_size} bytes)")
                continue

            name = f.name
            # Parse: C{1,2,3}--exp{NN}--IJL--LR--{NNNNN}.csv
            m = re.match(r'^C(\d)--(.+?)--(\d{5})\.csv$', name)
            if not m:
                continue
            channel_num = m.group(1)    # "1", "2", or "3"
            exp_part    = m.group(2)    # "exp11--IJL--LR" etc.
            file_num    = m.group(3)    # "00000"

            channel = f'C{channel_num}'
            suffix  = f'{exp_part}--{file_num}'

            groups[suffix][channel] = f
            groups[suffix]['_exp_part'] = exp_part
            groups[suffix]['_file_num'] = file_num

        # Keep only complete triplets (C1 + C2 + C3)
        for suffix, chans in groups.items():
            if all(c in chans for c in ['C1', 'C2', 'C3']):
                unique_key = f'{dir_name}__{suffix}'
                all_groups[unique_key] = {
                    'c1': chans['C1'],
                    'c2': chans['C2'],
                    'c3': chans['C3'],
                    'source_dir': dir_name,
                    'exp_id': chans.get('_exp_part', ''),
                    'file_num': chans.get('_file_num', ''),
                    'name': suffix,
                }

    return all_groups


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

def build_newdata_labeled():
    """
    Build 2-channel labeled matrix from the new dataset.

    Same labeling logic as step2 (zero-crossing on C1, oracle C2,
    three-zone labeling), but:
      - Scans 3 clean directories
      - No charge metadata extraction
      - Skips corrupt files automatically

    Outputs:
      X_multi.npy    : (N, 2, 20000) - 2 channels per cycle (V_ligne, I)
      y.npy          : (N,) - binary labels
      metadata.csv   : full metadata per sample
      config_multi.json : pipeline parameters
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = group_experiments(DATA_DIRS)
    print(f"Found {len(experiments)} complete experiments (C1+C2+C3 triplets)")
    for d in DATA_DIRS:
        dir_name = d.name
        n = sum(1 for v in experiments.values() if v['source_dir'] == dir_name)
        print(f"  {dir_name}: {n} experiments")

    # Collect all data
    X_list = []
    y_list = []
    meta_rows = []

    n_label0 = 0
    n_label1 = 0
    n_excluded = 0
    n_errors = 0

    total = len(experiments)
    for i, (exp_key, exp) in enumerate(sorted(experiments.items())):
        src = exp['source_dir']
        fnum = exp['file_num']
        print(f"  [{i+1:03d}/{total}] {src}/{fnum}", end=' ')

        try:
            # Parse all 3 channels
            c1 = parse_csv(exp['c1'])  # V_ligne
            c2 = parse_csv(exp['c2'])  # V_arc (oracle)
            c3 = parse_csv(exp['c3'])  # I

            # Detect zero crossings on C1
            zc = detect_zero_crossings(c1)
            if len(zc) < 2:
                print(f"→ SKIP (no valid zero crossings)")
                n_errors += 1
                continue

            ratios = compute_arc_ratios(c2, zc)
            if len(ratios) == 0:
                print(f"→ SKIP (no valid alternances)")
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

            # Extract segments from C1 and C3 only (C2 excluded from model input)
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
            n_kept += 1

            meta_rows.append({
                'source_dir':   src,
                'exp_id':       exp['exp_id'],
                'file_num':     fnum,
                'alt_index':    alt_idx,
                'arc_ratio':    round(ratio, 5),
                'label':        label,
                'start_sample': start,
                'end_sample':   end,
            })

        print(f"→ {n_kept} samples kept")

    if len(X_list) == 0:
        print("\nERROR: No samples were produced. Check data paths and file formats.")
        return

    # Assemble final arrays
    X = np.stack(X_list, axis=0)  # (N, 2, 20000)
    y = np.array(y_list, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    print(f"\n{'='*50}")
    print(f"DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"  Total samples    : {len(y):>5d}")
    print(f"  Label 0 (normal) : {n_label0:>5d}  ({100*n_label0/len(y):.1f}%)")
    print(f"  Label 1 (arc)    : {n_label1:>5d}  ({100*n_label1/len(y):.1f}%)")
    print(f"  Excluded         : {n_excluded:>5d}")
    print(f"  Errors/skipped   : {n_errors:>5d}")
    print(f"  Matrix shape     : X{X.shape}")
    print(f"  Memory (X)       : {X.nbytes / 1e6:.1f} MB")

    # Print per-directory breakdown
    print(f"\n  Samples per source directory:")
    for d in DATA_DIRS:
        dir_name = d.name
        mask = meta['source_dir'] == dir_name
        n_total = mask.sum()
        n_arc = ((meta['label'] == 1) & mask).sum()
        n_normal = ((meta['label'] == 0) & mask).sum()
        print(f"    {dir_name:<25s}: {n_total:4d} ({n_normal:3d}N/{n_arc:3d}A)")

    # Save outputs
    np.save(OUTPUT_DIR / 'X_multi.npy', X)
    np.save(OUTPUT_DIR / 'y.npy', y)
    meta.to_csv(OUTPUT_DIR / 'metadata.csv', index=False)

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
        'n_errors': int(n_errors),
        'source_dirs': [str(d) for d in DATA_DIRS],
        'X_shape': list(X.shape),
    }
    with open(OUTPUT_DIR / 'config_multi.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SAVED TO {OUTPUT_DIR}")
    print(f"{'='*50}")
    print(f"  X_multi.npy       {X.shape}  [channels: V_ligne, I]")
    print(f"  y.npy             {y.shape}")
    print(f"  metadata.csv      {meta.shape}")
    print(f"  config_multi.json")

    return X, y, meta


if __name__ == '__main__':
    print("=" * 60)
    print("ARC-FAULTNET — NEW DATA LABELING PIPELINE")
    print("=" * 60)

    result = build_newdata_labeled()

    if result is not None:
        print("\n=== DONE ===")
        print("Next step: python sanity_check.py")
        print("Then:      python train.py --mode single --epochs 50 --batch-size 32")
    else:
        print("\n=== FAILED ===")
