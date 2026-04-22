#!/usr/bin/env python3
"""
ARC-FAULTNET — Model Architecture
==================================
Dual-branch CNN with Joint Attention, inspired by MC-VSAttn.

Architecture:
  Input: 2 signals x 20000 samples  [V_ligne (C1), I (C3)]
  NOTE: V_arc (C2) is excluded — it is the oracle signal used only for
        labeling. At inference time, arc voltage is not measurable.
    |
    +---> Branch 1D (temporal) ---> F_L (128 x D)
    |       - ParametricConv1d layers (Gabor filters)
    |
    +---> STFT --> Branch 2D (spectral) ---> F_H (128 x D)
    |       - Conv2d layers
    |
    +---> Joint Attention (CAM + SAM crossed)
    |       - CAM: which filters matter?
    |       - SAM: which time positions matter?
    |       - Cross-attention for mutual guidance
    |
    +---> Classifier --> P(arc)

Key innovations from MC-VSAttn:
  - Parametric Gabor filters with learnable f0, sigma
  - Joint Attention with crossed CAM/SAM
  
Original contribution (Arc-FaultNet):
  - Dual-branch (1D + 2D STFT) instead of single 1D branch
  - Cross-branch attention fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ═══════════════════════════════════════════════════════
#  PARAMETRIC GABOR FILTER LAYER
# ═══════════════════════════════════════════════════════

class ParametricConv1d(nn.Module):
    """
    Parametric convolution layer with learnable Gabor-like filters.
    
    Each filter is defined by:
      - f0: center frequency (learned)
      - sigma: temporal width (learned)
    
    Filter formula (simplified from MC-VSAttn, alpha=0):
      psi(t) = exp(-t^2 / (2*sigma^2)) * cos(2*pi*f0*t)
    
    This is a Gabor filter - an oscillation windowed by a Gaussian.
    Physically interpretable: f0 targets a frequency, sigma controls duration.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 64,
        stride: int = 1,
        padding: int = 0,
        fs: float = 1_000_000,  # Sampling frequency
        f0_init_range: Tuple[float, float] = (100, 50000),  # Hz
        sigma_init_range: Tuple[float, float] = (0.0001, 0.001)  # seconds
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.fs = fs
        
        # Learnable parameters: f0 and sigma for each (out, in) filter pair
        # Initialize f0 uniformly in log space
        f0_log_min = math.log(f0_init_range[0])
        f0_log_max = math.log(f0_init_range[1])
        f0_init = torch.exp(
            torch.rand(out_channels, in_channels) * (f0_log_max - f0_log_min) + f0_log_min
        )
        self.f0 = nn.Parameter(f0_init)
        
        # Initialize sigma uniformly in log space
        sigma_log_min = math.log(sigma_init_range[0])
        sigma_log_max = math.log(sigma_init_range[1])
        sigma_init = torch.exp(
            torch.rand(out_channels, in_channels) * (sigma_log_max - sigma_log_min) + sigma_log_min
        )
        self.sigma = nn.Parameter(sigma_init)
        
        # Time axis for filter generation (centered at 0)
        t = torch.linspace(
            -kernel_size / (2 * fs),
            kernel_size / (2 * fs),
            kernel_size
        )
        self.register_buffer('t', t)
        
        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def _generate_filters(self) -> torch.Tensor:
        """
        Generate Gabor filter kernels from learned parameters.
        
        Returns:
            filters: (out_channels, in_channels, kernel_size)
        """
        # t: (kernel_size,)
        # f0, sigma: (out_channels, in_channels)
        
        t = self.t.view(1, 1, -1)  # (1, 1, K)
        f0 = self.f0.unsqueeze(-1)  # (O, I, 1)
        sigma = torch.abs(self.sigma.unsqueeze(-1)) + 1e-8  # (O, I, 1), ensure positive
        
        # Gaussian envelope
        gaussian = torch.exp(-t ** 2 / (2 * sigma ** 2))
        
        # Oscillation
        oscillation = torch.cos(2 * math.pi * f0 * t)
        
        # Gabor filter
        filters = gaussian * oscillation  # (O, I, K)
        
        # Normalize to unit L2 norm
        filters = F.normalize(filters, p=2, dim=-1)
        
        return filters
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, length)
        
        Returns:
            y: (batch, out_channels, length')
        """
        filters = self._generate_filters()
        return F.conv1d(x, filters, self.bias, self.stride, self.padding)


# ═══════════════════════════════════════════════════════
#  BRANCH 1D - TEMPORAL
# ═══════════════════════════════════════════════════════

class Branch1D(nn.Module):
    """
    Temporal branch using ParametricConv1d layers.
    
    Architecture:
      Layer 1: ParametricConv1d(2, 32, k=64) + BN + ReLU + MaxPool(4)
      Layer 2: ParametricConv1d(32, 64, k=32) + BN + ReLU + MaxPool(4)
      Layer 3: ParametricConv1d(64, 128, k=16) + BN + ReLU + AdaptiveAvgPool(D)
    
    Output: F_L with shape (batch, 128, D)
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        kernel_sizes: Tuple[int, int, int] = (64, 32, 16),
        output_dim: int = 64,  # D in the plan
        use_parametric: bool = True
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.use_parametric = use_parametric
        
        dims = [in_channels] + list(hidden_dims)
        
        layers = []
        for i in range(3):
            if use_parametric:
                conv = ParametricConv1d(
                    dims[i], dims[i+1],
                    kernel_size=kernel_sizes[i],
                    padding=kernel_sizes[i] // 2
                )
            else:
                conv = nn.Conv1d(
                    dims[i], dims[i+1],
                    kernel_size=kernel_sizes[i],
                    padding=kernel_sizes[i] // 2
                )
            
            layers.append(conv)
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
            
            if i < 2:
                layers.append(nn.MaxPool1d(4))
        
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, 20000)  — [V_ligne, I]
        
        Returns:
            F_L: (batch, 128, D)
        """
        x = self.features(x)  # (batch, 128, ~1250)
        x = self.pool(x)      # (batch, 128, D)
        return x


# ═══════════════════════════════════════════════════════
#  BRANCH 2D - SPECTRAL
# ═══════════════════════════════════════════════════════

class Branch2D(nn.Module):
    """
    Spectral branch for STFT spectrogram input.

    Frequency range restricted to 2–100 kHz:
      - Arc faults generate characteristic broadband HF noise in this band.
      - Below 2 kHz: dominated by load harmonics (50 Hz fundamental + harmonics)
        which are load-specific and hinder generalization.
      - Above 100 kHz: above the useful arc noise band at 1 MHz sampling rate;
        mostly quantization noise and electromagnetic interference.

    With n_fft=512 @ fs=1 MHz → bin resolution = 1953 Hz/bin:
      freq_bin_low  =  1  (≈ 1.95 kHz)
      freq_bin_high = 52  (≈ 101.6 kHz)
      → keeps 51 frequency bins out of 257

    Architecture:
      Freq. slice: x[:, :, freq_bin_low:freq_bin_high, :]  →  (B, 3, 51, T)
      Layer 1: Conv2d(3,  32, 3×3) + BN + ReLU + MaxPool(2×2)
      Layer 2: Conv2d(32, 64, 3×3) + BN + ReLU + MaxPool(2×2)
      Layer 3: Conv2d(64,128, 3×3) + BN + ReLU + AdaptiveAvgPool

    Input:  STFT spectrogram (batch, 3, n_freq, n_time) — full 257 bins
    Output: F_H with shape (batch, 128, D)
    """

    # Frequency slice constants (computed for n_fft=512, fs=1 MHz)
    FREQ_BIN_LOW  =  1   #  ≈   2 kHz
    FREQ_BIN_HIGH = 52   #  ≈ 100 kHz  (exclusive upper bound)

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        output_dim: int = 64,
        fs: float = 1_000_000,
        n_fft: int = 512,
        freq_min_hz: float = 2_000,
        freq_max_hz: float = 100_000
    ):
        super().__init__()

        self.output_dim = output_dim

        # Compute frequency bin indices from physical Hz values
        bin_res = fs / n_fft  # Hz per bin
        self.freq_bin_low  = max(1, round(freq_min_hz / bin_res))
        self.freq_bin_high = min(n_fft // 2 + 1, round(freq_max_hz / bin_res) + 1)

        dims = [in_channels] + list(hidden_dims)

        layers = []
        for i in range(3):
            layers.append(nn.Conv2d(dims[i], dims[i+1], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

            if i < 2:
                layers.append(nn.MaxPool2d(2))

        self.features = nn.Sequential(*layers)

        # Adaptive pooling to get fixed size regardless of input shape
        self.pool = nn.AdaptiveAvgPool2d((1, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, n_freq, n_time) — full STFT spectrogram (257 freq bins)
                channels: [V_ligne, I]

        Returns:
            F_H: (batch, 128, D)
        """
        # Restrict to 2–100 kHz band: discard low-frequency load harmonics
        # and high-frequency noise above the useful arc signature band
        x = x[:, :, self.freq_bin_low:self.freq_bin_high, :]  # (B, 3, 51, T)

        x = self.features(x)   # (batch, 128, h', w')
        x = self.pool(x)       # (batch, 128, 1, D)
        x = x.squeeze(2)       # (batch, 128, D)
        return x


# ═══════════════════════════════════════════════════════
#  CHANNEL ATTENTION MODULE (CAM)
# ═══════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) from CBAM.
    
    Computes attention weights for each channel (filter) based on
    global average and max pooling followed by shared MLP.
    
    Formula:
      β = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, D)
        
        Returns:
            weights: (batch, channels, 1) - attention weights
        """
        # Global pooling
        avg_pool = x.mean(dim=-1)  # (batch, channels)
        max_pool = x.max(dim=-1)[0]  # (batch, channels)
        
        # Shared MLP
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        
        # Combine and sigmoid
        weights = torch.sigmoid(avg_out + max_out)  # (batch, channels)
        
        return weights.unsqueeze(-1)  # (batch, channels, 1)


# ═══════════════════════════════════════════════════════
#  SPATIAL ATTENTION MODULE (SAM)
# ═══════════════════════════════════════════════════════

class SpatialAttention(nn.Module):
    """
    Spatial (Temporal) Attention Module (SAM).
    
    Uses self-attention mechanism with Q, K, V projections
    to weight different temporal positions.
    
    Formula:
      α = softmax(Q @ K^T / sqrt(d))
      output = α @ V
    """
    
    def __init__(self, channels: int, d_k: int = 64):
        super().__init__()
        
        self.d_k = d_k
        
        # Q, K, V projections (1x1 convolution equivalent)
        self.query = nn.Conv1d(channels, d_k, 1)
        self.key = nn.Conv1d(channels, d_k, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        
        self.scale = math.sqrt(d_k)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, D)
        
        Returns:
            output: (batch, channels, D) - attention-weighted features
        """
        batch, channels, D = x.shape
        
        # Compute Q, K, V
        Q = self.query(x)  # (batch, d_k, D)
        K = self.key(x)    # (batch, d_k, D)
        V = self.value(x)  # (batch, channels, D)
        
        # Attention scores
        scores = torch.bmm(Q.transpose(1, 2), K) / self.scale  # (batch, D, D)
        attn = F.softmax(scores, dim=-1)  # (batch, D, D)
        
        # Apply attention to values
        output = torch.bmm(V, attn.transpose(1, 2))  # (batch, channels, D)
        
        return output
    
    def get_attn_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the raw attention matrix α without applying it to V.
        Useful for visualization and interpretation.
        
        Args:
            x: (batch, channels, D)
        
        Returns:
            alpha: (batch, D, D) - row i gives the attention distribution
                   that position i places over all D positions.
        """
        Q = self.query(x)   # (batch, d_k, D)
        K = self.key(x)     # (batch, d_k, D)
        scores = torch.bmm(Q.transpose(1, 2), K) / self.scale  # (batch, D, D)
        return F.softmax(scores, dim=-1)  # (batch, D, D)


# ═══════════════════════════════════════════════════════
#  JOINT ATTENTION MODULE
# ═══════════════════════════════════════════════════════

class JointAttention(nn.Module):
    """
    Joint Attention Module - Cross-branch attention fusion.

    Key insight from MC-VSAttn: CAM and SAM should receive information
    from BOTH branches, not just their own. This allows:
      - CAM to select which channels matter — guided by the joint context
      - SAM to focus on which temporal positions matter — guided by the joint context

    Design (clean branch separation):
      F_concat = cat(F_L, F_H)                    # (B, 2C, D)

      CAM on joint context → split weights by branch:
        cam_w  = CAM(F_concat)                     # (B, 2C, 1)
        cam_L  = cam_w[:, :C, :]                   # (B, C, 1)  weights for temporal channels
        cam_H  = cam_w[:, C:, :]                   # (B, C, 1)  weights for spectral channels
        F_L_cam = F_L * cam_L                      # (B, C, D)
        F_H_cam = F_H * cam_H                      # (B, C, D)

      SAM on joint context → split output by branch:
        F_sam   = SAM(F_concat)                    # (B, 2C, D)
        F_L_sam = proj_sam_L(F_sam)                # (B, C, D)  Conv1d(2C→C, k=1)
        F_H_sam = proj_sam_H(F_sam)                # (B, C, D)  Conv1d(2C→C, k=1)

      Residual-style combination per branch:
        F_L_out = F_L_cam + F_L_sam                # (B, C, D)
        F_H_out = F_H_cam + F_H_sam                # (B, C, D)

      Final fusion:
        F_out = fusion(cat(F_L_out, F_H_out))      # (B, C, D)  Conv1d(2C→C, k=1)

    This preserves the identity of each branch throughout attention and only
    merges at the final fusion step, making the cross-branch guidance
    scientifically traceable (cam_L belongs to F_L, cam_H belongs to F_H).
    """

    def __init__(self, channels: int = 128, reduction: int = 8):
        super().__init__()

        self.C = channels   # single-branch channel count

        # CAM and SAM operate on the joint (2C) context
        self.cam = ChannelAttention(channels * 2, reduction)
        self.sam = SpatialAttention(channels * 2)

        # SAM output (2C) projected back to per-branch size (C) — one per branch
        self.proj_sam_L = nn.Conv1d(channels * 2, channels, 1)
        self.proj_sam_H = nn.Conv1d(channels * 2, channels, 1)

        # Final fusion of two C-dim branch outputs
        self.fusion = nn.Conv1d(channels * 2, channels, 1)

    def forward(self, F_L: torch.Tensor, F_H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_L: (batch, C, D) - features from temporal branch
            F_H: (batch, C, D) - features from spectral branch

        Returns:
            F_out: (batch, C, D) - fused features
        """
        F_concat = torch.cat([F_L, F_H], dim=1)    # (batch, 2C, D)

        # ── Channel Attention ─────────────────────────────────────────
        # Compute joint CAM weights, then assign first C → temporal, last C → spectral
        cam_w = self.cam(F_concat)                  # (batch, 2C, 1)
        cam_L = cam_w[:, :self.C, :]                # (batch, C, 1)
        cam_H = cam_w[:, self.C:, :]                # (batch, C, 1)
        F_L_cam = F_L * cam_L                       # (batch, C, D)
        F_H_cam = F_H * cam_H                       # (batch, C, D)

        # ── Spatial / Temporal Attention ──────────────────────────────
        # SAM sees the full joint context; its output is split into two streams
        F_sam   = self.sam(F_concat)                # (batch, 2C, D)
        F_L_sam = self.proj_sam_L(F_sam)            # (batch, C, D)
        F_H_sam = self.proj_sam_H(F_sam)            # (batch, C, D)

        # ── Residual combination per branch ───────────────────────────
        F_L_out = F_L_cam + F_L_sam                 # (batch, C, D)
        F_H_out = F_H_cam + F_H_sam                 # (batch, C, D)

        # ── Final fusion ──────────────────────────────────────────────
        return self.fusion(torch.cat([F_L_out, F_H_out], dim=1))  # (batch, C, D)


# ═══════════════════════════════════════════════════════
#  CLASSIFIER HEAD
# ═══════════════════════════════════════════════════════

class ClassifierHead(nn.Module):
    """
    Classification head: GAP -> FC -> Sigmoid
    
    Binary classification for arc detection.
    """
    
    def __init__(self, in_channels: int = 128, hidden_dim: int = 64):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, D)
        
        Returns:
            logits: (batch,) - raw logits for BCE loss
        """
        x = self.gap(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        x = self.fc(x)  # (batch, 1)
        return x.squeeze(-1)  # (batch,)


# ═══════════════════════════════════════════════════════
#  ARC-FAULTNET - FULL MODEL
# ═══════════════════════════════════════════════════════

class ArcFaultNet(nn.Module):
    """
    Arc-FaultNet: Dual-Branch CNN with Joint Attention.
    
    Architecture:
      1. Branch 1D processes raw temporal signals
      2. Branch 2D processes STFT spectrograms
      3. Joint Attention fuses features with cross-branch guidance
      4. Classifier outputs arc probability
    
    Inspired by MC-VSAttn, extended with:
      - STFT spectral branch (original contribution)
      - Cross-branch attention for charge-invariant detection
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        output_dim: int = 64,
        use_parametric: bool = True,
        use_joint_attention: bool = True
    ):
        super().__init__()
        
        self.use_joint_attention = use_joint_attention
        
        # Branch 1D - Temporal
        self.branch_1d = Branch1D(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_parametric=use_parametric
        )
        
        # Branch 2D - Spectral
        self.branch_2d = Branch2D(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        # Joint Attention
        if use_joint_attention:
            self.joint_attn = JointAttention(
                channels=hidden_dims[-1],
                reduction=8
            )
        else:
            # Simple concatenation + projection
            self.joint_attn = nn.Conv1d(hidden_dims[-1] * 2, hidden_dims[-1], 1)
        
        # Classifier
        self.classifier = ClassifierHead(
            in_channels=hidden_dims[-1],
            hidden_dim=64
        )
    
    def forward(
        self,
        x_1d: torch.Tensor,
        x_2d: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_1d: (batch, 2, 20000) - raw signals [V_ligne, I]
            x_2d: (batch, 2, n_freq, n_time) - STFT spectrograms [V_ligne, I]
        
        Returns:
            logits: (batch,) - raw logits for BCEWithLogitsLoss
        """
        # Extract features from both branches
        F_L = self.branch_1d(x_1d)  # (batch, 128, D)
        F_H = self.branch_2d(x_2d)  # (batch, 128, D)
        
        # Fuse with attention
        if self.use_joint_attention:
            F_out = self.joint_attn(F_L, F_H)  # (batch, 128, D)
        else:
            F_concat = torch.cat([F_L, F_H], dim=1)
            F_out = self.joint_attn(F_concat)
        
        # Classify
        logits = self.classifier(F_out)  # (batch,)
        
        return logits
    
    def get_attention_maps(
        self,
        x_1d: torch.Tensor,
        x_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate features and all attention maps for visualization.

        Returns:
            F_L      : (batch, 128, D)  — temporal branch features
            F_H      : (batch, 128, D)  — spectral branch features
            F_out    : (batch, 128, D)  — fused output features
            cam_w    : (batch, 256, 1)  — joint CAM weights β per channel
                         cam_w[:, :128, :] → weights applied to F_L (temporal)
                         cam_w[:, 128:, :] → weights applied to F_H (spectral)
            sam_alpha: (batch, D, D)    — SAM attention matrix α
                         row i = attention weight that position i gives to all D positions
        """
        F_L = self.branch_1d(x_1d)  # (batch, 128, D)
        F_H = self.branch_2d(x_2d)  # (batch, 128, D)

        if self.use_joint_attention:
            F_concat = torch.cat([F_L, F_H], dim=1)  # (batch, 256, D)

            # CAM weights — β ∈ (0, 1) per channel
            cam_w = self.joint_attn.cam(F_concat)     # (batch, 256, 1)

            # SAM attention matrix — α[i, j] = weight pos i gives to pos j
            sam_alpha = self.joint_attn.sam.get_attn_weights(F_concat)  # (batch, D, D)

            F_out = self.joint_attn(F_L, F_H)         # (batch, 128, D)
        else:
            # Fallback for no-attention variant
            F_concat = torch.cat([F_L, F_H], dim=1)
            F_out = self.joint_attn(F_concat)
            cam_w = torch.ones(F_L.shape[0], 256, 1, device=F_L.device)
            sam_alpha = torch.eye(F_L.shape[2], device=F_L.device).unsqueeze(0).expand(F_L.shape[0], -1, -1)

        return F_L, F_H, F_out, cam_w, sam_alpha


# ═══════════════════════════════════════════════════════
#  ABLATION VARIANTS
# ═══════════════════════════════════════════════════════

class ArcFaultNet_1DOnly(nn.Module):
    """Ablation: Only temporal branch, no STFT."""
    
    def __init__(self, in_channels: int = 2, use_parametric: bool = True):
        super().__init__()
        self.branch = Branch1D(in_channels=in_channels, use_parametric=use_parametric)
        self.classifier = ClassifierHead(in_channels=128)
    
    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor = None) -> torch.Tensor:
        F = self.branch(x_1d)
        return self.classifier(F)


class ArcFaultNet_NoAttention(nn.Module):
    """Ablation: Dual-branch but simple concatenation instead of attention."""
    
    def __init__(self, in_channels: int = 2, use_parametric: bool = True):
        super().__init__()
        self.branch_1d = Branch1D(in_channels=in_channels, use_parametric=use_parametric)
        self.branch_2d = Branch2D(in_channels=in_channels)
        self.fusion = nn.Conv1d(256, 128, 1)
        self.classifier = ClassifierHead(in_channels=128)
    
    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> torch.Tensor:
        F_L = self.branch_1d(x_1d)
        F_H = self.branch_2d(x_2d)
        F = self.fusion(torch.cat([F_L, F_H], dim=1))
        return self.classifier(F)


class ArcFaultNet_StandardConv(nn.Module):
    """Ablation: Standard Conv1d instead of ParametricConv1d."""
    
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.model = ArcFaultNet(
            in_channels=in_channels,
            use_parametric=False,
            use_joint_attention=True
        )
    
    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> torch.Tensor:
        return self.model(x_1d, x_2d)


class ArcFaultNet_IndependentCBAM(nn.Module):
    """Ablation: CBAM applied independently to each branch (no cross-attention)."""
    
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.branch_1d = Branch1D(in_channels=in_channels)
        self.branch_2d = Branch2D(in_channels=in_channels)
        
        # Independent attention per branch
        self.cam_1d = ChannelAttention(128)
        self.sam_1d = SpatialAttention(128)
        self.cam_2d = ChannelAttention(128)
        self.sam_2d = SpatialAttention(128)
        
        self.fusion = nn.Conv1d(256, 128, 1)
        self.classifier = ClassifierHead(in_channels=128)
    
    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> torch.Tensor:
        F_L = self.branch_1d(x_1d)
        F_H = self.branch_2d(x_2d)
        
        # Independent attention
        F_L = F_L * self.cam_1d(F_L)
        F_L = self.sam_1d(F_L)
        F_H = F_H * self.cam_2d(F_H)
        F_H = self.sam_2d(F_H)
        
        F = self.fusion(torch.cat([F_L, F_H], dim=1))
        return self.classifier(F)


class BaselineCNN(nn.Module):
    """Ablation: Simple CNN baseline without attention or parametric filters."""
    
    def __init__(self, in_channels: int = 2):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, 64, padding=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, 32, padding=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, 16, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor = None) -> torch.Tensor:
        x = self.conv(x_1d)
        x = x.squeeze(-1)
        return self.fc(x).squeeze(-1)


# ═══════════════════════════════════════════════════════
#  MODEL FACTORY
# ═══════════════════════════════════════════════════════

def get_model(model_name: str, in_channels: int = 2) -> nn.Module:
    """
    Factory function to get model by name.
    
    Available models:
      - arcfaultnet: Full Arc-FaultNet
      - 1d_only: Only temporal branch
      - no_attention: No joint attention
      - standard_conv: Standard Conv1d instead of parametric
      - independent_cbam: Independent CBAM per branch
      - baseline_cnn: Simple CNN baseline
    """
    models = {
        'arcfaultnet': lambda: ArcFaultNet(in_channels=in_channels),
        '1d_only': lambda: ArcFaultNet_1DOnly(in_channels=in_channels),
        'no_attention': lambda: ArcFaultNet_NoAttention(in_channels=in_channels),
        'standard_conv': lambda: ArcFaultNet_StandardConv(in_channels=in_channels),
        'independent_cbam': lambda: ArcFaultNet_IndependentCBAM(in_channels=in_channels),
        'baseline_cnn': lambda: BaselineCNN(in_channels=in_channels),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()


# ═══════════════════════════════════════════════════════
#  TEST
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Testing Arc-FaultNet components...")
    
    batch_size = 4
    seq_len = 20000
    n_channels = 2  # V_ligne (C1) + I (C3) — V_arc excluded
    n_freq = 257
    n_time = 78
    
    # Test inputs
    x_1d = torch.randn(batch_size, n_channels, seq_len)
    x_2d = torch.randn(batch_size, n_channels, n_freq, n_time)
    
    print(f"\nInput shapes:")
    print(f"  x_1d: {x_1d.shape}")
    print(f"  x_2d: {x_2d.shape}")
    
    # Test ParametricConv1d
    print(f"\nParametricConv1d:")
    pconv = ParametricConv1d(2, 32, kernel_size=64, padding=32)
    y = pconv(x_1d)
    print(f"  Input: {x_1d.shape} -> Output: {y.shape}")
    print(f"  Learned f0 range: [{pconv.f0.min().item():.1f}, {pconv.f0.max().item():.1f}] Hz")
    
    # Test Branch1D
    print(f"\nBranch1D:")
    branch1d = Branch1D()
    F_L = branch1d(x_1d)
    print(f"  Input: {x_1d.shape} -> Output: {F_L.shape}")
    
    # Test Branch2D
    print(f"\nBranch2D:")
    branch2d = Branch2D()
    F_H = branch2d(x_2d)
    print(f"  Input: {x_2d.shape} -> Output: {F_H.shape}")
    
    # Test JointAttention
    print(f"\nJointAttention:")
    joint_attn = JointAttention()
    F_out = joint_attn(F_L, F_H)
    print(f"  F_L: {F_L.shape}, F_H: {F_H.shape} -> F_out: {F_out.shape}")
    
    # Test full model
    print(f"\nArcFaultNet (full model):")
    model = ArcFaultNet()
    logits = model(x_1d, x_2d)
    print(f"  Output logits: {logits.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    
    # Test ablation variants
    print(f"\nAblation variants:")
    for name in ['1d_only', 'no_attention', 'standard_conv', 'independent_cbam', 'baseline_cnn']:
        m = get_model(name)
        out = m(x_1d, x_2d)
        n_p = sum(p.numel() for p in m.parameters())
        print(f"  {name}: output={out.shape}, params={n_p:,}")
    
    print("\n=== All tests passed ===")
