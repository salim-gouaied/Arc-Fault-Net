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
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
import json
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
        data_dir: str = '/home/top/PFE/labeled_dataset',
        n_fft: int = 512,
        hop_length: int = 256,
        compute_stft: bool = True,
        device: str = 'cpu'
    ):
        """
        Args:
            data_dir: Path to labeled_dataset directory
            n_fft: FFT size for STFT
            hop_length: Hop size for STFT
            compute_stft: If True, compute STFT on-the-fly. If False, return only 1D.
            device: Device for STFT computation ('cpu' or 'cuda')
        """
        self.data_dir = Path(data_dir)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.compute_stft = compute_stft
        self.device = device
        
        # Load data
        self.X = np.load(self.data_dir / 'X_multi.npy')  # (N, 2, 20000) — [V_ligne, I]
        self.y = np.load(self.data_dir / 'y.npy')        # (N,)
        self.charges = np.load(self.data_dir / 'charges.npy')  # (N,)
        
        # Load charge mapping
        with open(self.data_dir / 'charge_map.json', 'r') as f:
            self.charge_map = json.load(f)
        
        self.n_samples = len(self.y)
        self.n_channels = self.X.shape[1]
        self.seq_len = self.X.shape[2]
        self.n_charges = len(self.charge_map)
        
        # Precompute STFT window
        self.window = torch.hann_window(n_fft)
        
        # Expected STFT output shape
        self.n_freq = n_fft // 2 + 1  # 257 for n_fft=512
        self.n_time = (self.seq_len - n_fft) // hop_length + 1  # ~78 for seq_len=20000
        
        print(f"ArcFaultDataset loaded:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Input shape: {self.X.shape}")
        print(f"  STFT shape per channel: ({self.n_freq}, {self.n_time})")
        print(f"  Charges: {self.n_charges}")
        print(f"  Label distribution: {np.sum(self.y==0)} normal, {np.sum(self.y==1)} arc")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get raw data
        x_1d = torch.from_numpy(self.X[idx]).float()  # (2, 20000)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        charge_idx = torch.tensor(self.charges[idx], dtype=torch.long)
        
        # Compute STFT for 2D branch
        if self.compute_stft:
            x_2d = self._compute_stft(x_1d)  # (2, n_freq, n_time)
        else:
            x_2d = torch.zeros(1)  # placeholder
        
        return x_1d, x_2d, label, charge_idx
    
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
    data_dir = Path('/home/top/PFE/labeled_dataset')
    if not (data_dir / 'X_multi.npy').exists():
        print(f"\nData not found at {data_dir}")
        print("Run: python step2_build_multichannel.py")
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
