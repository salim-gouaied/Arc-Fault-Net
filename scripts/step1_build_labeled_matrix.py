#!/usr/bin/env python3
"""
ARC FAULT DETECTION — STEP 1: Build Labeled Matrix
====================================================
Input  : Raw CSV files from Teledyne LeCroy oscilloscope
         (C1=Voltage, C2=Arc voltage oracle, C3=Current)
Output : X_raw.npy     — (M, 20000) labeled alternances from C3
         y.npy          — (M,)       binary labels {0, 1}
         metadata.csv   — (M,)       experiment info per alternance
         all_ratios.npy — (~1300,)   arc_ratio for ALL alternances
                                     (including excluded ones → for
                                      histogram calibration of R_low/R_high)

Critical design decisions (see discussion notes):
  - Segmentation on C1 (voltage), NOT C3 (current)
  - Labeling oracle = C2 with fixed V_th = 10V
  - Three-zone exclusion: R_low < ratio < R_high → discarded
  - C2 is NEVER saved as model input
  - Metadata preserved for load-wise cross-validation
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
#  CONFIGURATION — edit these if needed
# ─────────────────────────────────────────────────────
DATA_DIR   = Path('/home/top/PFE/OthmaneSalim11032026')
OUTPUT_DIR = Path('/home/top/PFE/labeled_dataset')

FS                 = 1_000_000   # Sampling rate (Hz)
F0                 = 50          # Mains frequency (Hz)
SAMPLES_PER_CYCLE  = FS // F0    # 20 000 samples per full cycle
ZC_TOLERANCE       = 0.08        # ±8% tolerance on expected period

V_TH   = 10.0   # Arc voltage threshold on C2 (Volts) — fixed, validated
R_LOW  = None   # Will be calibrated from histogram (or override below)
R_HIGH = None   # Same

# Override R_LOW / R_HIGH manually if you already know them:
# R_LOW  = 0.05
# R_HIGH = 0.95

HEADER_LINES = 5   # Lines to skip at top of each CSV file

# ─────────────────────────────────────────────────────
#  STEP 0: Group CSV files into experiments
# ─────────────────────────────────────────────────────

def group_experiments(data_dir: Path) -> dict:
    """
    Match C1/C2/C3 files that share the same experiment suffix.
    Returns dict: { experiment_id: {c1, c2, c3, arc_load, bg_loads} }
    """
    files = list(data_dir.glob('*.csv'))
    groups = defaultdict(dict)

    for f in sorted(files):
        name = f.name
        # Extract channel prefix (C1EE, C2EE, C3EE) and shared suffix
        m = re.match(r'^(C[123])EE (.+)$', name)
        if not m:
            continue
        channel = m.group(1)   # "C1", "C2", or "C3"
        suffix  = m.group(2)   # shared part: "GraphCu Arc_AspiRouge+...00000.csv"
        groups[suffix][channel] = f

    # Keep only complete triplets
    complete = {}
    for suffix, chans in groups.items():
        if all(c in chans for c in ['C1', 'C2', 'C3']):
            # Parse load info from filename
            m = re.search(r'Arc_([^+]+)\+(.+?)\d{5}\.csv$', suffix)
            if m:
                arc_load = m.group(1)
                bg_loads = m.group(2).split('+')
            else:
                arc_load = 'unknown'
                bg_loads = []
            complete[suffix] = {
                'c1': chans['C1'],
                'c2': chans['C2'],
                'c3': chans['C3'],
                'arc_load': arc_load,
                'bg_loads': bg_loads,
                'name': suffix
            }

    return complete


# ─────────────────────────────────────────────────────
#  STEP 1: Parse a single CSV file
# ─────────────────────────────────────────────────────

def parse_csv(filepath: Path) -> np.ndarray:
    """
    Parse LeCroy CSV export.
    Skips HEADER_LINES lines, then reads 'Time, Ampl' columns.
    Returns amplitude array as float32 (saves memory).
    """
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
#  STEP 2: Detect zero crossings on C1 (voltage)
# ─────────────────────────────────────────────────────

def detect_zero_crossings(v: np.ndarray) -> np.ndarray:
    """
    Detect positive-going zero crossings on the voltage signal C1.

    Why C1 and not C3?
      C1 (mains voltage) is stable and load-independent.
      C3 (current) changes phase and shape with load and arc events.

    Process:
      1. Remove DC offset (subtract mean)
      2. Bandpass filter around 50 Hz to remove HF noise and DC drift
      3. Find indices where signal crosses zero from negative to positive
      4. Validate spacing ≈ SAMPLES_PER_CYCLE (20 000) ± ZC_TOLERANCE

    Returns: array of validated zero-crossing indices (positive-going only).
             Each consecutive pair (ZC[i], ZC[i+1]) defines one alternance.
    """
    # 1. Remove DC offset (C1 may have a DC bias from the scope probe)
    v = v.astype(np.float64)
    v = v - np.mean(v)

    # 2. Bandpass filter (40–60 Hz) to isolate the 50 Hz fundamental
    #    This removes HF arc noise that could create spurious crossings
    sos = sp.butter(4, [40, 60], btype='bandpass', fs=FS, output='sos')
    v_filt = sp.sosfiltfilt(sos, v)

    # 3. Detect positive-going zero crossings
    #    sign goes from <= 0 to > 0
    signs = np.sign(v_filt)
    crossings = np.where((signs[:-1] <= 0) & (signs[1:] > 0))[0]

    if len(crossings) < 2:
        return np.array([], dtype=int)

    # 4. Validate: positive-going crossings appear every FULL PERIOD = 20 000 samples
    #    (one positive-going crossing per complete cycle)
    tol = int(SAMPLES_PER_CYCLE * ZC_TOLERANCE)  # ±8% of 20 000

    validated = [crossings[0]]
    for idx in crossings[1:]:
        spacing = idx - validated[-1]
        if abs(spacing - SAMPLES_PER_CYCLE) <= tol:
            # Correct spacing — valid crossing
            validated.append(idx)
        elif spacing < SAMPLES_PER_CYCLE - tol:
            # Too close — spurious crossing, skip it
            continue
        else:
            # Gap too large (missed one cycle) — accept anyway as next anchor
            validated.append(idx)

    return np.array(validated, dtype=int)


# ─────────────────────────────────────────────────────
#  STEP 3: Compute arc_ratio per alternance
# ─────────────────────────────────────────────────────

def compute_arc_ratios(c2: np.ndarray,
                       zc_indices: np.ndarray,
                       v_th: float = V_TH) -> list:
    """
    For each full cycle (alternance), compute the fraction of samples
    where C2 > V_th (i.e., arc is active at that instant).

    One alternance = ZC[i] to ZC[i+2]  (two half-periods = full cycle)

    Returns list of dicts: [{start, end, ratio}, ...]
    """
    results = []

    # Each consecutive pair of positive-going ZC defines one full cycle alternance
    # ZC[i] → ZC[i+1]  =  one complete period  (positive-going to next positive-going)
    n = len(zc_indices)
    for i in range(0, n - 1, 1):
        start = zc_indices[i]
        end   = zc_indices[i + 1]

        seg_len = end - start

        # Sanity check: segment length should be close to SAMPLES_PER_CYCLE
        if abs(seg_len - SAMPLES_PER_CYCLE) > SAMPLES_PER_CYCLE * ZC_TOLERANCE:
            continue   # skip malformed alternance

        c2_seg  = c2[start:end]
        n_arc   = np.sum(np.abs(c2_seg) > v_th)
        ratio   = float(n_arc) / float(len(c2_seg))

        results.append({
            'start': int(start),
            'end':   int(end),
            'ratio': ratio
        })

    return results


# ─────────────────────────────────────────────────────
#  STEP 4: Calibrate R_low and R_high from histogram
# ─────────────────────────────────────────────────────

def calibrate_thresholds(all_ratios: np.ndarray,
                         percentile_normal: float = 99.0,
                         percentile_arc:    float = 1.0) -> tuple:
    """
    Calibrate R_low and R_high from the full histogram of arc_ratios.

    Method:
      - group_normal = all ratios below 0.5 (clearly normal side)
      - group_arc    = all ratios above 0.5 (clearly arc side)
      - R_low  = percentile_normal of group_normal  (e.g., 99th percentile)
      - R_high = percentile_arc    of group_arc     (e.g., 1st percentile)

    Expected result: bimodal distribution with most ratios near 0 or near 1.
    The valley between the two peaks contains only transition alternances.
    """
    group_normal = all_ratios[all_ratios < 0.5]
    group_arc    = all_ratios[all_ratios >= 0.5]

    if len(group_normal) == 0 or len(group_arc) == 0:
        print("  WARNING: could not separate normal/arc groups. Using defaults.")
        return 0.05, 0.95

    r_low  = float(np.percentile(group_normal, percentile_normal))
    r_high = float(np.percentile(group_arc,    percentile_arc))

    print(f"\n  Calibration result:")
    print(f"    group_normal: {len(group_normal)} alternances")
    print(f"    group_arc:    {len(group_arc)} alternances")
    print(f"    R_low  (99th pct of normal group) = {r_low:.4f}")
    print(f"    R_high (1st  pct of arc group)    = {r_high:.4f}")
    print(f"    Gap = {r_high - r_low:.4f}  (should be > 0.3 for clean data)")

    if r_high - r_low < 0.3:
        print("  WARNING: gap is small — check V_th calibration")

    return r_low, r_high


# ─────────────────────────────────────────────────────
#  STEP 5: Normalize a C3 segment
# ─────────────────────────────────────────────────────

def normalize_segment(seg: np.ndarray) -> np.ndarray:
    """
    Per-alternance z-score normalization on the current signal C3.

    Why per-alternance?
      Different experiments have different load amplitudes (1A vs 20A).
      The model should learn the ARC SHAPE, not the absolute current level.
      Normalizing per alternance removes amplitude dependence.
    """
    mean = np.mean(seg)
    std  = np.std(seg)
    if std < 1e-9:
        return seg - mean   # flat signal, just center
    return (seg - mean) / std


# ─────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────

def build_labeled_matrix(r_low_override=None, r_high_override=None):
    """
    Full pipeline:
      1. Group experiments (26 triplets C1/C2/C3)
      2. For each experiment: detect ZC → compute arc_ratios → extract C3 segments
      3. Calibrate R_low/R_high from histogram of all arc_ratios
      4. Apply three-zone labeling
      5. Save X_raw, y, metadata, all_ratios
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = group_experiments(DATA_DIR)
    print(f"Found {len(experiments)} complete experiments (C1+C2+C3 triplets)")

    # ── Pass 1: collect all arc_ratios (for histogram / calibration) ──
    print("\n=== PASS 1: Computing arc_ratios for all alternances ===")

    all_ratios_list = []
    pass1_data      = []   # store parsed data to avoid re-reading in pass 2

    for i, (exp_name, exp) in enumerate(sorted(experiments.items())):
        print(f"  [{i+1:02d}/{len(experiments)}] {exp['arc_load']:<20s}", end=' ')

        c1 = parse_csv(exp['c1'])
        c2 = parse_csv(exp['c2'])
        c3 = parse_csv(exp['c3'])

        zc      = detect_zero_crossings(c1)
        ratios  = compute_arc_ratios(c2, zc)

        n_alt = len(ratios)
        r_vals = [r['ratio'] for r in ratios]
        print(f"→ {n_alt} alternances  |  "
              f"ratio range: [{min(r_vals):.3f}, {max(r_vals):.3f}]")

        for r in ratios:
            all_ratios_list.append(r['ratio'])

        pass1_data.append({
            'exp_name':  exp_name,
            'arc_load':  exp['arc_load'],
            'bg_loads':  exp['bg_loads'],
            'c3':        c3,
            'ratios':    ratios
        })

    all_ratios = np.array(all_ratios_list, dtype=np.float32)
    print(f"\n  Total alternances across all experiments: {len(all_ratios)}")

    # ── Calibrate or use overrides ──
    if r_low_override is not None and r_high_override is not None:
        r_low  = r_low_override
        r_high = r_high_override
        print(f"\n  Using manual thresholds: R_low={r_low}, R_high={r_high}")
    else:
        r_low, r_high = calibrate_thresholds(all_ratios)

    # ── Pass 2: apply three-zone labeling, extract C3 segments ──
    print(f"\n=== PASS 2: Labeling with R_low={r_low:.4f}, R_high={r_high:.4f} ===")

    X_list    = []
    y_list    = []
    meta_rows = []

    n_label0  = 0
    n_label1  = 0
    n_excluded = 0

    for exp_data in pass1_data:
        c3     = exp_data['c3']
        ratios = exp_data['ratios']

        for alt_idx, alt in enumerate(ratios):
            ratio = alt['ratio']
            start = alt['start']
            end   = alt['end']

            # Three-zone decision
            if ratio <= r_low:
                label = 0
                n_label0 += 1
            elif ratio >= r_high:
                label = 1
                n_label1 += 1
            else:
                n_excluded += 1
                continue   # ← DISCARD ambiguous transition alternance

            # Extract and normalize C3 segment
            c3_seg = c3[start:end].astype(np.float32)

            # Pad or truncate to exact SAMPLES_PER_CYCLE length
            seg_len = len(c3_seg)
            if seg_len < SAMPLES_PER_CYCLE:
                c3_seg = np.pad(c3_seg,
                                (0, SAMPLES_PER_CYCLE - seg_len),
                                mode='edge')
            elif seg_len > SAMPLES_PER_CYCLE:
                c3_seg = c3_seg[:SAMPLES_PER_CYCLE]

            c3_norm = normalize_segment(c3_seg).astype(np.float32)

            X_list.append(c3_norm)
            y_list.append(label)
            meta_rows.append({
                'exp_name':      exp_data['exp_name'],
                'arc_load':      exp_data['arc_load'],
                'bg_loads':      '+'.join(exp_data['bg_loads']),
                'alt_index':     alt_idx,
                'arc_ratio':     round(ratio, 5),
                'label':         label,
                'start_sample':  start,
                'end_sample':    end
            })

    # ── Assemble final matrix ──
    X = np.stack(X_list, axis=0)   # (M, 20000)
    y = np.array(y_list, dtype=np.int8)
    meta = pd.DataFrame(meta_rows)

    print(f"\n=== LABELING SUMMARY ===")
    print(f"  Label 0 (normal) :  {n_label0:>5d} alternances")
    print(f"  Label 1 (arc)    :  {n_label1:>5d} alternances")
    print(f"  Excluded         :  {n_excluded:>5d} alternances  "
          f"({100*n_excluded/(n_label0+n_label1+n_excluded):.1f}%)")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL KEPT       :  {len(y):>5d} alternances")
    print(f"  Class balance    :  {100*n_label0/len(y):.1f}% normal  /  "
          f"{100*n_label1/len(y):.1f}% arc")
    print(f"  Matrix shape     :  X{X.shape}, y{y.shape}")
    print(f"  Memory (X)       :  {X.nbytes / 1e6:.1f} MB")

    # ── Save outputs ──
    np.save(OUTPUT_DIR / 'X_raw.npy',      X)
    np.save(OUTPUT_DIR / 'y.npy',           y)
    np.save(OUTPUT_DIR / 'all_ratios.npy',  all_ratios)
    meta.to_csv(OUTPUT_DIR / 'metadata.csv', index=False)

    # Save config used
    config = {
        'V_TH':            V_TH,
        'R_LOW':           float(r_low),
        'R_HIGH':          float(r_high),
        'FS':              FS,
        'F0':              F0,
        'SAMPLES_PER_CYCLE': SAMPLES_PER_CYCLE,
        'HEADER_LINES':    HEADER_LINES,
        'n_experiments':   len(experiments),
        'n_label0':        int(n_label0),
        'n_label1':        int(n_label1),
        'n_excluded':      int(n_excluded),
        'X_shape':         list(X.shape),
    }
    with open(OUTPUT_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n=== SAVED TO {OUTPUT_DIR} ===")
    print(f"  X_raw.npy     {X.shape}  — C3 segments, normalized")
    print(f"  y.npy         {y.shape}   — binary labels")
    print(f"  metadata.csv  {meta.shape}  — per-sample metadata")
    print(f"  all_ratios.npy({len(all_ratios)},) — full ratio histogram data")
    print(f"  config.json             — pipeline configuration")

    return X, y, meta, all_ratios, r_low, r_high


# ─────────────────────────────────────────────────────
#  STEP 6: Print ratio histogram (text-based)
# ─────────────────────────────────────────────────────

def print_ratio_histogram(all_ratios: np.ndarray,
                          r_low: float, r_high: float,
                          bins: int = 20):
    """
    ASCII histogram of arc_ratio distribution.
    Helps visually confirm bimodal distribution and correct R_low/R_high.
    """
    print(f"\n=== ARC RATIO HISTOGRAM ===")
    print(f"  (R_low={r_low:.3f}  R_high={r_high:.3f}  V_th={V_TH}V)\n")

    counts, edges = np.histogram(all_ratios, bins=bins, range=(0, 1))
    max_count = max(counts)
    bar_width = 40

    for i, (cnt, edge) in enumerate(zip(counts, edges[:-1])):
        center     = edge + (edges[1] - edges[0]) / 2
        bar_len    = int(cnt / max_count * bar_width)
        bar        = '█' * bar_len

        # Color coding
        if center <= r_low:
            zone = 'NORMAL'
        elif center >= r_high:
            zone = 'ARC   '
        else:
            zone = 'EXCL. '

        print(f"  {center:.2f} | {bar:<{bar_width}} | {cnt:>4d}  [{zone}]")

    print(f"\n  Bimodal gap between {r_low:.3f} and {r_high:.3f}: "
          f"{sum((all_ratios > r_low) & (all_ratios < r_high))} alternances excluded")


# ─────────────────────────────────────────────────────
#  STEP 7: Print load breakdown (for load-wise CV)
# ─────────────────────────────────────────────────────

def print_load_breakdown(meta: pd.DataFrame):
    """
    Print the number of alternances per arc_load.
    Critical for planning load-wise cross-validation splits.
    """
    print(f"\n=== LOAD BREAKDOWN (for load-wise CV) ===")
    breakdown = meta.groupby('arc_load')['label'].value_counts().unstack(fill_value=0)
    breakdown.columns = ['label_0 (normal)', 'label_1 (arc)']
    breakdown['TOTAL'] = breakdown.sum(axis=1)
    print(breakdown.to_string())
    print(f"\n  → Use 'arc_load' column as grouping key for K-Fold CV")
    print(f"    Never put the same arc_load in both train AND validation set")


# ─────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("ARC FAULT DETECTION — LABELING PIPELINE")
    print("=" * 60)

    X, y, meta, all_ratios, r_low, r_high = build_labeled_matrix(
        r_low_override=None,   # set to e.g. 0.05 to override calibration
        r_high_override=None   # set to e.g. 0.95 to override calibration
    )

    print_ratio_histogram(all_ratios, r_low, r_high)
    print_load_breakdown(meta)

    print("\n=== DONE ===")
    print("Next step: load X_raw.npy and y.npy for feature engineering")
    print("           (3-channel 1D input + STFT spectrogram)")
