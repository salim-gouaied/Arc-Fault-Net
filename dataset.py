#!/usr/bin/env python3
"""
ARC FAULT DETECTION — PyTorch Dataset
======================================
Loads multi-channel data and computes STFT on-the-fly for the 2D branch.

Features:
  - Loads X_multi.npy (N, 2, 20000), y.npy, charges.npy
      Channel 0: V_ligne (C1) — mains voltage
      Channel 1: I       (C3) — line current
      V_arc (C2) is NOT included — oracle signal used only for labeling.
  - Computes log-power STFT spectrogram on-the-fly
  - Provides leave-one-charge-out cross-validation splits
  - GPU-friendly: STFT computed with torch.stft
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
import json
import csv
from typing import Tuple, List, Optional


class ArcFaultDataset(Dataset):
    """
    PyTorch Dataset for arc fault detection.
    
    Returns per sample:
      x_1d : (2, 20000) - raw signals for 1D branch  [V_ligne, I]
      x_2d : (2, n_freq, n_time) - STFT spectrogram for 2D branch  [V_ligne, I]
      label: scalar - binary label (0=normal, 1=arc)
      charge_idx: scalar - charge configuration index
    """
    
    def __init__(
        self,
        data_dir: str = '/home/manip/pfe_salim_gouaied/Arc-Fault-Net/labeled_dataset',
        n_fft: int = 512,
        hop_length: int = 256,
        compute_stft: bool = True,
        device: str = 'cpu',
        training: bool = False,
        channel_mode: str = 'raw2',
        strong_augment: bool = False
    ):
        """
        Args:
            data_dir: Path to labeled_dataset directory
            n_fft: FFT size for STFT
            hop_length: Hop size for STFT
            compute_stft: If True, compute STFT on-the-fly. If False, return only 1D.
            device: Device for STFT computation ('cpu' or 'cuda')
            channel_mode: Front-end representation for the 1D branch.
                'raw2'      -> V1 behaviour: 2 raw channels [V_ligne, I],
                               STFT computed on both channels.
                'i_derived4'-> Arc-FaultNet V2 front-end: 4 physically
                               complementary channels derived from I(t) only
                               [I, |ΔI|, TKEO(I), RMS_slide(I)]; STFT
                               computed on I(t) only (1 spectral channel).
                               |ΔI| is the sample-to-sample discrete derivative.
                               V(t) is used outside the model (segmentation) and never fed in.
        """
        self.data_dir = Path(data_dir)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.compute_stft = compute_stft
        self.device = device
        self.training = training  # Controls augmentation
        # Cross-campaign robustness augmentation (see _augment_temporal_strong).
        # OFF by default so every earlier run stays reproducible.
        self.strong_augment = strong_augment
        self._donor_pool = None  # set via set_donor_pool() — MUST exclude test data
        if channel_mode not in ('raw2', 'i_derived4'):
            raise ValueError(f"channel_mode must be 'raw2' or 'i_derived4', got {channel_mode!r}")
        self.channel_mode = channel_mode
        
        # Load data
        self.X = np.load(self.data_dir / 'X_multi.npy')  # (N, 2, seq_len) — [V_ligne, I]
        self.y = np.load(self.data_dir / 'y.npy')        # (N,)
        
        self.n_samples = len(self.y)
        self.n_channels = self.X.shape[1]
        self.seq_len = self.X.shape[2]
        
        # Read sampling frequency from config.json (auto-detect 1 MHz vs 102.4 kHz)
        config_path = self.data_dir / 'config.json'
        channel_names = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            self.fs = cfg.get('FS', 1_000_000)
            channel_names = cfg.get('channel_names')
        else:
            self.fs = 1_000_000

        # Locate the current channel (I) — it carries the arc signature.
        # Default to the conventional [V_ligne, I] ordering (I = last channel).
        if channel_names and 'I' in channel_names:
            self.i_channel = channel_names.index('I')
        else:
            self.i_channel = self.n_channels - 1

        # Number of channels the 1D branch will actually receive
        self.out_channels = 4 if self.channel_mode == 'i_derived4' else self.n_channels

        
        # Load charges (optional — may not exist for new datasets)
        charges_path = self.data_dir / 'charges.npy'
        charge_map_path = self.data_dir / 'charge_map.json'
        
        if charges_path.exists() and charge_map_path.exists():
            self.charges = np.load(charges_path)
            # Check for size mismatch (stale file from a previous dataset)
            if len(self.charges) != self.n_samples:
                print(f"  WARNING: charges.npy size ({len(self.charges)}) != X size ({self.n_samples})")
                print(f"           → Using dummy charges (single group). Use --mode single for training.")
                self.charges = np.zeros(self.n_samples, dtype=np.int64)
                self.charge_map = {'unknown': 0}
            else:
                with open(charge_map_path, 'r') as f:
                    self.charge_map = json.load(f)
        else:
            print(f"  INFO: No charges.npy found — using dummy charges (single group)")
            self.charges = np.zeros(self.n_samples, dtype=np.int64)
            self.charge_map = {'unknown': 0}
        
        self.n_charges = len(self.charge_map)
        
        # Precompute STFT window
        self.window = torch.hann_window(n_fft)
        
        # Expected STFT output shape
        self.n_freq = n_fft // 2 + 1  # 257 for n_fft=512
        self.n_time = (self.seq_len - n_fft) // hop_length + 1  # ~78 for seq_len=20000
        
        print(f"ArcFaultDataset loaded:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Sampling freq: {self.fs:,} Hz  (seq_len={self.seq_len})")
        print(f"  Channel mode: {self.channel_mode}  ->  1D branch in_channels={self.out_channels}")
        if self.channel_mode == 'i_derived4':
            print(f"    derived from I (channel {self.i_channel}): [I, |ΔI|, TKEO, RMS_slide]; STFT on I only")
        print(f"  STFT shape per channel: ({self.n_freq}, {self.n_time})  [n_fft={n_fft}, hop={hop_length}]")
        print(f"  Charges: {self.n_charges}")
        print(f"  Label distribution: {np.sum(self.y==0)} normal, {np.sum(self.y==1)} arc")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get raw data
        # .clone() is REQUIRED: X is float32, so .float() would return a view sharing
        # memory with self.X and the in-place augmentation below would permanently
        # corrupt the dataset array (and leak augmented samples into eval reads).
        x_raw = torch.from_numpy(self.X[idx]).float().clone()  # (n_channels, seq_len) — [V_ligne, I]
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        charge_idx = torch.tensor(self.charges[idx], dtype=torch.long)

        # Apply temporal augmentation on the RAW signal first (physical realism)
        if self.training:
            x_raw = (self._augment_temporal_strong(x_raw) if self.strong_augment
                     else self._augment_temporal(x_raw))

        if self.channel_mode == 'i_derived4':
            # ── V2 front-end: 4 channels derived from I(t) only ──────────
            i_sig = x_raw[self.i_channel]                  # (seq_len,)
            x_1d = self._derive_i_channels(i_sig)          # (4, seq_len)
            stft_src = i_sig.unsqueeze(0)                  # (1, seq_len) — STFT of I only
        else:
            # ── V1 front-end: raw channels [V_ligne, I] ──────────────────
            x_1d = x_raw                                    # (n_channels, seq_len)
            stft_src = x_raw                                # STFT on all channels

        # Compute STFT for 2D branch
        if self.compute_stft:
            x_2d = self._compute_stft(stft_src)            # (C_spec, n_freq, n_time)
            if self.training:
                x_2d = self._augment_spectrogram(x_2d)
        else:
            x_2d = torch.zeros(1)  # placeholder

        return x_1d, x_2d, label, charge_idx

    def _derive_i_channels(self, i_sig: torch.Tensor) -> torch.Tensor:
        """
        Build the 4 physically-complementary channels from I(t) (Arc-FaultNet V2).

        All channels share length M (= seq_len) and are normalised by the RAW
        cycle's RMS so that their RELATIVE magnitudes stay physically meaningful
        (load-invariant) — never use a global dataset normalisation here.

        Channels:
          0. I_norm          : raw waveform                       (global shape)
          1. |dI|            : |sample-to-sample derivative|      (local discontinuities / arc spikes)
          2. TKEO(I)         : I[n]^2 - I[n-1]*I[n+1]             (instantaneous energy, sub-cycle ignition/extinction)
          3. RMS_slide(I)    : sliding RMS over a M/4 window      (amplitude envelope: flat shoulder / current dip)
        """
        M = i_sig.shape[0]
        rms = torch.sqrt(torch.mean(i_sig ** 2) + 1e-12)
        i_norm = i_sig / rms

        # 1. |dI| — abs of discrete derivative, right-padded by 1 to keep length M
        d = i_norm[1:] - i_norm[:-1]
        abs_di = torch.cat([d.abs(), d.abs()[-1:]], dim=0)         # (M,)

        # 2. TKEO — I[n]^2 - I[n-1]*I[n+1], edges padded by replication (length M)
        tkeo_core = i_norm[1:-1] ** 2 - i_norm[:-2] * i_norm[2:]   # (M-2,)
        tkeo = torch.cat([tkeo_core[:1], tkeo_core, tkeo_core[-1:]], dim=0)  # (M,)

        # 3. RMS_slide — centered sliding RMS over window M/4 (reflect-padded → length M)
        win = max(2, M // 4)
        sq = (i_norm ** 2).unsqueeze(0).unsqueeze(0)               # (1,1,M)
        pad_l = win // 2
        pad_r = win - 1 - pad_l
        sq_pad = F.pad(sq, (pad_l, pad_r), mode='reflect')
        kernel = torch.ones(1, 1, win, device=i_sig.device) / win
        rms_slide = torch.sqrt(F.conv1d(sq_pad, kernel) + 1e-12).squeeze(0).squeeze(0)  # (M,)

        return torch.stack([i_norm, abs_di, tkeo, rms_slide], dim=0)  # (4, M)
    
    def _augment_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Light temporal augmentation preserving physical realism.
        - Amplitude scaling: uniform(0.95, 1.05) per channel
        - Additive Gaussian noise: N(0, 0.005 * std(channel))
        """
        n_channels = x.shape[0]
        for c in range(n_channels):
            # Amplitude scaling
            scale = 0.95 + 0.1 * torch.rand(1).item()
            x[c] = x[c] * scale
            # Small Gaussian noise
            noise_std = 0.005 * x[c].std().item()
            if noise_std > 0:
                x[c] = x[c] + torch.randn_like(x[c]) * noise_std
        return x
    
    def set_donor_pool(self, allowed_indices: np.ndarray):
        """
        Restrict the background-load donors of the strong augmentation to these
        indices (only their normal-labelled cycles are used).

        Call this with the TRAINING indices of the current fold before building the
        dataloaders. Without it the donors would be drawn from the whole dataset,
        which would pull the held-out campaign's signals into training — unlabelled,
        but still leakage.
        """
        allowed = np.asarray(allowed_indices)
        self._donor_pool = allowed[self.y[allowed] == 0]
        if len(self._donor_pool) == 0:
            raise ValueError("donor pool is empty — no normal cycles in the training split")

    def _augment_temporal_strong(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cross-campaign robustness augmentation, applied to the RAW cycle.

        Each transform simulates something that genuinely differs between two
        acquisition benches, which is what the leave-one-campaign-out folds punish:

          - pink (1/f) noise at a randomised SNR      -> different noise floor
          - spectral tilt                             -> different sensor response
          - band limiting                             -> different sensor bandwidth
          - mains-frequency jitter (±0.5 Hz)          -> grid frequency drift
          - circular time shift                       -> segmentation offset
          - half-cycle shift + polarity flip          -> mains phase invariance
          - background-load mixing                    -> different loads on the line
            (currents of parallel loads add, so summing a normal cycle onto this one
             is the physically correct operation; the arc label is unchanged)

        Amplitude scaling is deliberately absent: the i_derived4 front-end already
        normalises each cycle by its RMS, so a gain change is a no-op there.
        """
        M = x.shape[1]

        # ── mains-frequency jitter: ±1% time scale, then crop/pad back to M ──
        if torch.rand(1).item() < 0.5:
            factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.02
            new_len = max(8, int(round(M * factor)))
            y = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=new_len, mode='linear', align_corners=False
            ).squeeze(0)
            if new_len >= M:
                x = y[:, :M]
            else:
                x = torch.cat([y, y[:, :M - new_len]], dim=1)

        # ── circular time shift (segmentation offset) ──
        if torch.rand(1).item() < 0.5:
            x = torch.roll(x, shifts=int(torch.randint(-M // 20, M // 20 + 1, (1,)).item()), dims=1)

        # ── half-cycle shift + polarity flip (same mains phase, opposite half) ──
        if torch.rand(1).item() < 0.5:
            x = -torch.roll(x, shifts=M // 2, dims=1)

        # ── frequency-domain shaping: tilt + band limit ──
        if torch.rand(1).item() < 0.7:
            X = torch.fft.rfft(x, dim=1)
            n_bins = X.shape[1]
            freq = torch.arange(n_bins, dtype=torch.float32)
            # Sensor response: a mild gain ramp, 1.0 at DC to (1 ± 0.3) at Nyquist.
            # A power-law tilt was tried first and rejected — it attenuates the 50 Hz
            # fundamental ~2.5x, which after per-cycle RMS normalisation doubles the
            # energy of the |dI| and TKEO descriptors instead of perturbing them.
            delta = (torch.rand(1).item() - 0.5) * 0.6
            shape = 1.0 + delta * (freq / max(n_bins - 1, 1))
            # band limit: attenuate above a random cutoff (50-100% of the band)
            cutoff = int(n_bins * (0.5 + 0.5 * torch.rand(1).item()))
            if cutoff < n_bins:
                roll_off = torch.ones(n_bins)
                roll_off[cutoff:] = torch.linspace(1.0, 0.05, n_bins - cutoff)
                shape = shape * roll_off
            x = torch.fft.irfft(X * shape.unsqueeze(0), n=M, dim=1)

        # ── background-load mixing (parallel load on the same line) ──
        if self._donor_pool is not None and torch.rand(1).item() < 0.5:
            j = int(self._donor_pool[torch.randint(len(self._donor_pool), (1,)).item()])
            donor = torch.from_numpy(self.X[j]).float()
            if donor.shape == x.shape:
                donor = torch.roll(donor, shifts=int(torch.randint(0, M, (1,)).item()), dims=1)
                for c in range(x.shape[0]):
                    rms_x = torch.sqrt(torch.mean(x[c] ** 2) + 1e-12)
                    rms_d = torch.sqrt(torch.mean(donor[c] ** 2) + 1e-12)
                    # mix level between -20 dB and -6 dB relative to this cycle
                    ratio = 10 ** (-(6.0 + 14.0 * torch.rand(1).item()) / 20.0)
                    x[c] = x[c] + donor[c] * (rms_x / rms_d) * ratio

        # ── pink (1/f) noise at randomised SNR (30-55 dB) ──
        n_bins = M // 2 + 1
        noise = torch.fft.irfft(
            torch.fft.rfft(torch.randn(x.shape[0], M), dim=1)
            / torch.sqrt(torch.arange(n_bins, dtype=torch.float32) + 1.0).unsqueeze(0),
            n=M, dim=1)
        for c in range(x.shape[0]):
            snr_db = 30.0 + 25.0 * torch.rand(1).item()
            rms_x = torch.sqrt(torch.mean(x[c] ** 2) + 1e-12)
            rms_n = torch.sqrt(torch.mean(noise[c] ** 2) + 1e-12)
            x[c] = x[c] + noise[c] * (rms_x / rms_n) * 10 ** (-snr_db / 20.0)

        return x

    def _augment_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Light frequency masking on STFT spectrograms.
        Masks 1-3 consecutive frequency bins with channel mean.
        """
        n_channels, n_freq, n_time = spec.shape
        # Random number of bins to mask (1-3)
        mask_width = torch.randint(1, 4, (1,)).item()
        # Random start position (leave margins)
        if n_freq > mask_width + 2:
            start = torch.randint(1, n_freq - mask_width - 1, (1,)).item()
            for c in range(n_channels):
                channel_mean = spec[c].mean()
                spec[c, start:start + mask_width, :] = channel_mean
        return spec
    
    def _compute_stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-power STFT spectrogram for all channels.
        
        Args:
            x: (n_channels, seq_len) input signal
        
        Returns:
            spec: (n_channels, n_freq, n_time) log-power spectrogram
        """
        n_channels = x.shape[0]
        specs = []
        
        for c in range(n_channels):
            # STFT: returns complex tensor (n_freq, n_time)
            stft = torch.stft(
                x[c],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                return_complex=True
            )
            
            # Power spectrogram
            power = stft.abs().pow(2)
            
            # Log scale (add small epsilon for numerical stability)
            log_power = torch.log(power + 1e-10)
            
            specs.append(log_power)
        
        return torch.stack(specs, dim=0)  # (n_channels, n_freq, n_time)
    
    def get_charge_indices(self, charge_idx: int) -> np.ndarray:
        """Get all sample indices belonging to a specific charge configuration."""
        return np.where(self.charges == charge_idx)[0]
    
    def get_charge_name(self, charge_idx: int) -> str:
        """Get charge name from index."""
        for name, idx in self.charge_map.items():
            if idx == charge_idx:
                return name
        return f"unknown_{charge_idx}"


class LeaveOneChargeOutSplitter:
    """
    Cross-validation splitter that holds out one charge configuration at a time.
    
    This is the proper evaluation protocol for testing generalization
    to unseen electrical loads.
    """
    
    def __init__(self, dataset: ArcFaultDataset):
        self.dataset = dataset
        self.n_charges = dataset.n_charges
        self.charge_map = dataset.charge_map
    
    def __iter__(self):
        """
        Yields (train_indices, test_indices) for each fold.
        Each fold holds out one charge configuration for testing.
        """
        for test_charge_idx in range(self.n_charges):
            test_indices = self.dataset.get_charge_indices(test_charge_idx)
            train_indices = np.concatenate([
                self.dataset.get_charge_indices(c)
                for c in range(self.n_charges)
                if c != test_charge_idx
            ])
            
            yield train_indices, test_indices
    
    def __len__(self):
        return self.n_charges
    
    def get_fold_name(self, fold_idx: int) -> str:
        """Get the name of the held-out charge for a fold."""
        return self.dataset.get_charge_name(fold_idx)


def create_dataloaders(
    dataset: ArcFaultDataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from indices.
    
    Args:
        dataset: ArcFaultDataset instance
        train_indices: Indices for training
        test_indices: Indices for testing
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Fraction of train set to use for validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split train into train/val
    np.random.shuffle(train_indices)
    n_val = int(len(train_indices) * val_split)
    val_indices = train_indices[:n_val]
    train_indices = train_indices[n_val:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_random_split_loaders(
    dataset: ArcFaultDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with random split.
    NOTE: This is NOT the proper evaluation for generalization testing.
          Use LeaveOneChargeOutSplitter for proper evaluation.
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    
    n_train = int(len(dataset) * train_ratio)
    n_val = int(len(dataset) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return create_dataloaders(
        dataset,
        np.concatenate([train_indices, val_indices]),
        test_indices,
        batch_size,
        num_workers,
        val_split=n_val / (n_train + n_val)
    )


# ─────────────────────────────────────────────────────
#  Test
# ─────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing ArcFaultDataset...")
    
    # Check if data exists
    data_dir = Path('/home/manip/pfe_salim_gouaied/Arc-Fault-Net/labeled_dataset')
    if not (data_dir / 'X_multi.npy').exists():
        print(f"\nData not found at {data_dir}")
        print("Run: python scripts/step2_build_multichannel.py")
        exit(1)
    
    # Load dataset
    dataset = ArcFaultDataset(data_dir=str(data_dir))
    
    # Test single sample
    x_1d, x_2d, label, charge_idx = dataset[0]
    print(f"\nSingle sample:")
    print(f"  x_1d shape: {x_1d.shape}")
    print(f"  x_2d shape: {x_2d.shape}")
    print(f"  label: {label.item()}")
    print(f"  charge_idx: {charge_idx.item()}")
    
    # Test leave-one-charge-out splitter
    print(f"\nLeave-one-charge-out splits:")
    splitter = LeaveOneChargeOutSplitter(dataset)
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter):
        charge_name = splitter.get_fold_name(fold_idx)
        train_labels = dataset.y[train_idx]
        test_labels = dataset.y[test_idx]
        print(f"  Fold {fold_idx}: test on '{charge_name}'")
        print(f"    Train: {len(train_idx)} samples ({np.sum(train_labels==0)}N/{np.sum(train_labels==1)}A)")
        print(f"    Test:  {len(test_idx)} samples ({np.sum(test_labels==0)}N/{np.sum(test_labels==1)}A)")
    
    # Test dataloader
    print(f"\nTesting DataLoader...")
    train_loader, val_loader, test_loader = get_random_split_loaders(
        dataset, batch_size=32, num_workers=0
    )
    
    batch = next(iter(train_loader))
    x_1d_batch, x_2d_batch, labels, charges = batch
    print(f"  Batch x_1d: {x_1d_batch.shape}")
    print(f"  Batch x_2d: {x_2d_batch.shape}")
    print(f"  Batch labels: {labels.shape}")
    
    print("\n=== Dataset tests passed ===")
