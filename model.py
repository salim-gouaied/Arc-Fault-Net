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
        sigma_init_range: Tuple[float, float] = (0.0001, 0.001),  # seconds
        use_amplitude: bool = False
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
        
        # Optional learnable amplitude per filter
        if use_amplitude:
            self.amplitude = nn.Parameter(torch.ones(out_channels, in_channels))
        else:
            self.amplitude = None
    
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
        
        # Apply learnable amplitude if enabled
        if self.amplitude is not None:
            filters = self.amplitude.unsqueeze(-1) * filters
        
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
#  SQUEEZE-AND-EXCITATION BLOCK
# ═══════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for 1-D and 2-D feature maps.
    
    Learns channel-wise attention weights via global pooling + FC layers.
    Helps the network focus on the most informative feature channels.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:       # 1-D features: (B, C, L)
            gap = x.mean(-1)
        else:                  # 2-D features: (B, C, H, W)
            gap = x.mean([-2, -1])
        w = self.fc(gap)
        for _ in range(x.dim() - 2):
            w = w.unsqueeze(-1)
        return x * w


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
        use_parametric: bool = True,
        use_se: bool = False,
        se_reduction: int = 8,
        use_amplitude: bool = False,
        fs: float = 1_000_000
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
                    padding=kernel_sizes[i] // 2,
                    fs=fs,
                    use_amplitude=use_amplitude
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
            
            if use_se:
                layers.append(SEBlock(dims[i+1], reduction=se_reduction))
            
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
        freq_max_hz: float = 100_000,
        use_se: bool = False,
        se_reduction: int = 8
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

            if use_se:
                layers.append(SEBlock(dims[i+1], reduction=se_reduction))

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
    
    def __init__(self, channels: int, d_k: int = 32):
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
    
    def __init__(self, in_channels: int = 128, hidden_dim: int = 64, deep: bool = False):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        if deep:
            self.fc = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
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
        use_joint_attention: bool = True,
        use_se: bool = False,
        se_reduction: int = 8,
        use_amplitude: bool = False,
        deep_classifier: bool = False,
        classifier_hidden: int = 64,
        fs: float = 1_000_000,
        n_fft: int = 512
    ):
        super().__init__()
        
        self.use_joint_attention = use_joint_attention
        
        # Branch 1D - Temporal
        self.branch_1d = Branch1D(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_parametric=use_parametric,
            use_se=use_se,
            se_reduction=se_reduction,
            use_amplitude=use_amplitude,
            fs=fs
        )
        
        # Branch 2D - Spectral
        self.branch_2d = Branch2D(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            fs=fs,
            n_fft=n_fft,
            use_se=use_se,
            se_reduction=se_reduction
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
            hidden_dim=classifier_hidden,
            deep=deep_classifier
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
    
    def __init__(self, in_channels: int = 2, use_parametric: bool = True,
                 use_se: bool = False, se_reduction: int = 8,
                 use_amplitude: bool = False, deep_classifier: bool = False,
                 classifier_hidden: int = 64, fs: float = 1_000_000):
        super().__init__()
        self.branch = Branch1D(in_channels=in_channels, use_parametric=use_parametric,
                               use_se=use_se, se_reduction=se_reduction,
                               use_amplitude=use_amplitude, fs=fs)
        self.classifier = ClassifierHead(in_channels=128, hidden_dim=classifier_hidden, deep=deep_classifier)
    
    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor = None) -> torch.Tensor:
        F = self.branch(x_1d)
        return self.classifier(F)


class ArcFaultNet_NoAttention(nn.Module):
    """Ablation: Dual-branch but simple concatenation instead of attention."""
    
    def __init__(self, in_channels: int = 2, use_parametric: bool = True,
                 use_se: bool = False, se_reduction: int = 8,
                 use_amplitude: bool = False, deep_classifier: bool = False,
                 classifier_hidden: int = 64, fs: float = 1_000_000,
                 n_fft: int = 512):
        super().__init__()
        self.branch_1d = Branch1D(in_channels=in_channels, use_parametric=use_parametric,
                                  use_se=use_se, se_reduction=se_reduction,
                                  use_amplitude=use_amplitude, fs=fs)
        self.branch_2d = Branch2D(in_channels=in_channels, use_se=use_se,
                                  se_reduction=se_reduction, fs=fs, n_fft=n_fft)
        self.fusion = nn.Conv1d(256, 128, 1)
        self.classifier = ClassifierHead(in_channels=128, hidden_dim=classifier_hidden, deep=deep_classifier)
    
    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> torch.Tensor:
        F_L = self.branch_1d(x_1d)
        F_H = self.branch_2d(x_2d)
        F = self.fusion(torch.cat([F_L, F_H], dim=1))
        return self.classifier(F)


class ArcFaultNet_StandardConv(nn.Module):
    """Ablation: Standard Conv1d instead of ParametricConv1d."""
    
    def __init__(self, in_channels: int = 2,
                 use_se: bool = False, se_reduction: int = 8,
                 use_amplitude: bool = False, deep_classifier: bool = False,
                 classifier_hidden: int = 64, fs: float = 1_000_000,
                 n_fft: int = 512):
        super().__init__()
        self.model = ArcFaultNet(
            in_channels=in_channels,
            use_parametric=False,
            use_joint_attention=True,
            use_se=use_se,
            se_reduction=se_reduction,
            use_amplitude=use_amplitude,
            deep_classifier=deep_classifier,
            classifier_hidden=classifier_hidden,
            fs=fs,
            n_fft=n_fft
        )
    
    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> torch.Tensor:
        return self.model(x_1d, x_2d)


class ArcFaultNet_IndependentCBAM(nn.Module):
    """Ablation: CBAM applied independently to each branch (no cross-attention)."""
    
    def __init__(self, in_channels: int = 2,
                 use_se: bool = False, se_reduction: int = 8,
                 use_amplitude: bool = False, deep_classifier: bool = False,
                 classifier_hidden: int = 64, fs: float = 1_000_000,
                 n_fft: int = 512):
        super().__init__()
        self.branch_1d = Branch1D(in_channels=in_channels, use_se=use_se, se_reduction=se_reduction,
                                  use_amplitude=use_amplitude, fs=fs)
        self.branch_2d = Branch2D(in_channels=in_channels, use_se=use_se, se_reduction=se_reduction,
                                  fs=fs, n_fft=n_fft)
        
        # Independent attention per branch
        self.cam_1d = ChannelAttention(128)
        self.sam_1d = SpatialAttention(128)
        self.cam_2d = ChannelAttention(128)
        self.sam_2d = SpatialAttention(128)
        
        self.fusion = nn.Conv1d(256, 128, 1)
        self.classifier = ClassifierHead(in_channels=128, hidden_dim=classifier_hidden, deep=deep_classifier)
    
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


# ═══════════════════════════════════════════════════════════════════════
#  ARC-FAULTNET V2 — single-cycle adaptation
# ═══════════════════════════════════════════════════════════════════════
#
#  Design notes (see ablation_results/ArcFaultNet_V2 spec):
#    * Input is a SINGLE 50 Hz cycle (M samples, e.g. 2048 @ 102.4 kHz).
#    * The temporal branch consumes 4 physically-derived channels built from
#      I(t) only: [I, |ΔI|, TKEO(I), RMS_slide(I)] — produced in dataset.py.
#      |ΔI| is the sample-to-sample discrete derivative, which is
#      zero for stable (non-arc) waveforms and highlights arc perturbations.
#    * Gabor / ParametricConv1d is intentionally REMOVED: the arc is aperiodic
#      and impulsive, so plain Conv1d filters are the correct prior.
#    * The spectral branch is revised: a learnable soft FrequencyGate replaces
#      the hard frequency slice, and asymmetric pooling compresses time while
#      preserving frequency resolution; 4 frequency groups are kept.
#    * Cross-branch fusion uses separate, mutually-conditioned channel gates
#      (RevisedCrossAttention) instead of V1's single joint-CAM split.
#    * Stage 5 (XGBoost/RandomForest on the 128-d embedding) is handled OUTSIDE
#      this module by train_xgb_head.py — this model exposes the embedding via
#      forward(..., return_embedding=True).
#
#  The inter-cycle stages of the full V2 spec (delta encoding across cycles,
#  Dowalla per-cycle-pair scalars, BiGRU temporal reasoning, IEC ALS counter)
#  require a MULTI-cycle dataset (B, N, M) which does not exist yet, so they
#  are deliberately out of scope here. Hooks are documented for a future
#  multi-cycle dataset.

class FrequencyGate(nn.Module):
    """
    Learnable soft frequency attention applied on the spectrogram.

    Replaces the hard ``[FREQ_BIN_LOW:FREQ_BIN_HIGH]`` slice of V1. The gate
    operates on the frequency axis only (kernel (3,1)) and emphasises the bands
    that actually carry arc information for the current load, instead of a fixed
    hand-picked band.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, n_freq, n_time)
        return x * self.gate(x)


class SpectralBranchV2(nn.Module):
    """
    Revised spectral branch (Sub-Branch C of the V2 spec).

    FrequencyGate -> Conv2d stack with ASYMMETRIC pooling (time compressed,
    frequency preserved) -> keep 4 frequency groups -> project to (B, 128, D).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        output_dim: int = 64,
        freq_groups: int = 4,
        use_se: bool = False,
        se_reduction: int = 8,
        use_freq_gate: bool = True
    ):
        super().__init__()
        self.output_dim = output_dim
        self.freq_groups = freq_groups

        c0, c1, c2 = hidden_dims
        self.freq_gate = FrequencyGate(in_channels) if use_freq_gate else None

        # Build blocks with optional SE attention after each conv
        block1_layers = [
            nn.Conv2d(in_channels, c0, kernel_size=3, padding=1),
            nn.BatchNorm2d(c0), nn.GELU(),
        ]
        if use_se:
            block1_layers.append(SEBlock(c0, reduction=se_reduction))
        block1_layers.append(nn.MaxPool2d(kernel_size=(2, 1)))  # freq only (preserve time)
        self.block1 = nn.Sequential(*block1_layers)

        block2_layers = [
            nn.Conv2d(c0, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1), nn.GELU(),
        ]
        if use_se:
            block2_layers.append(SEBlock(c1, reduction=se_reduction))
        block2_layers.append(nn.MaxPool2d(kernel_size=(2, 1)))  # freq only (preserve time)
        self.block2 = nn.Sequential(*block2_layers)

        block3_layers = [
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2), nn.GELU(),
        ]
        if use_se:
            block3_layers.append(SEBlock(c2, reduction=se_reduction))
        self.block3 = nn.Sequential(*block3_layers)

        # Keep `freq_groups` frequency bands and `output_dim` time positions
        self.adaptive = nn.AdaptiveAvgPool2d((freq_groups, output_dim))
        # Collapse the (C * freq_groups) channels back to C
        self.proj = nn.Conv1d(c2 * freq_groups, c2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_spec, n_freq, n_time) — log-power STFT of I(t) (C_spec=1)
        Returns:
            (B, 128, D)
        """
        if self.freq_gate is not None:
            x = self.freq_gate(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)              # (B, C, f', t')
        x = self.adaptive(x)           # (B, C, freq_groups, D)
        b, c, g, d = x.shape
        x = x.reshape(b, c * g, d)     # (B, C*freq_groups, D)
        return self.proj(x)            # (B, C, D)


class DescriptorChannelAttention(nn.Module):
    """
    Channel attention for the temporal branch — distinct from Squeeze-and-Excitation.

    Rationale (arc physics + generalization).
    The temporal branch is fed physically-derived channels
    ``[I_norm, |ΔI|, TKEO, RMS_slide]``. The arc-discriminative content lives in the
    transient, load-*independent* descriptors — the re-ignition discontinuities in
    ``|ΔI|`` and ``TKEO`` — which appear as short, sparse PEAKS in time. A pure
    average-pooling squeeze (as used by :class:`SEBlock`) dilutes those peaks over the
    whole cycle and can miss them. This module therefore squeezes each channel with
    BOTH average- and max-pooling and combines the two through a shared bottleneck MLP,
    so a channel that is quiet on average but spikes sharply (the arc signature) can
    still be amplified.

    Because the input is already per-cycle normalized, the recalibration is computed
    from the *shape* of the channels rather than their absolute amplitude; the learned
    weighting therefore transfers across loads — it emphasizes the descriptors carrying
    the arc signature regardless of the connected appliance and suppresses channels
    dominated by load-specific low-frequency content. This is the mechanism by which
    channel attention is intended to contribute to cross-load generalization and
    false-positive reduction, complementing the sequential cross-attention that fuses
    the temporal and spectral branches.

    Works on both the raw descriptor channels ``(B, C_in, M)`` at the branch input and
    the deeper feature maps ``(B, C, T)`` after each convolution block.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        avg = x.mean(dim=-1)                              # (B, C) — sustained energy
        mx = x.amax(dim=-1)                               # (B, C) — transient peaks
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))   # (B, C) — channel gates
        return x * w.unsqueeze(-1)


class TemporalBranchV2(nn.Module):
    """
    Temporal branch for the 4 derived channels (Sub-Branch B style).

    Plain Conv1d (NO Gabor) with GELU — the filters learn arc-specific delta
    shapes directly from data without imposing a frequency-oscillation prior.

    Channel attention (:class:`DescriptorChannelAttention`, ON by default) is applied
    first to the raw derived channels — acting as a learned, per-sample descriptor
    selector that emphasizes the load-invariant arc descriptors — and again after each
    convolution block to recalibrate the learned feature channels. It is the temporal
    branch's channel-attention mechanism and is independent of the optional SE blocks.

    Optional SE blocks (``use_se``) add an extra average-pooled channel recalibration
    after each conv layer.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        kernel_sizes: Tuple[int, int, int] = (16, 8, 4),
        output_dim: int = 64,
        use_se: bool = False,
        se_reduction: int = 8,
        use_channel_attn: bool = True,
        ca_reduction: int = 8
    ):
        super().__init__()
        self.output_dim = output_dim
        self.use_channel_attn = use_channel_attn
        dims = [in_channels] + list(hidden_dims)
        layers = []
        # Input-level descriptor attention: learned per-sample weighting of the raw
        # [I_norm, |ΔI|, TKEO, RMS_slide] channels (interpretable descriptor selector).
        if use_channel_attn:
            layers.append(DescriptorChannelAttention(in_channels, reduction=ca_reduction))
        for i in range(3):
            layers += [
                nn.Conv1d(dims[i], dims[i + 1],
                          kernel_size=kernel_sizes[i],
                          padding=kernel_sizes[i] // 2),
                nn.BatchNorm1d(dims[i + 1]),
                nn.GELU(),
            ]
            if use_se:
                layers.append(SEBlock(dims[i + 1], reduction=se_reduction))
            # Feature-level channel attention after each block (peak-aware, always on).
            if use_channel_attn:
                layers.append(DescriptorChannelAttention(dims[i + 1], reduction=ca_reduction))
            if i < 2:
                layers.append(nn.MaxPool1d(4))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, M) -> (B, 128, D)
        return self.pool(self.features(x))


class RevisedCrossAttention(nn.Module):
    """
    Cross-branch fusion (Stage 4 of the V2 spec).

    Fixes the V1 CAM channel-ordering ambiguity: instead of splitting one joint
    CAM into [:C]/[C:], each branch gets its OWN channel gate that is conditioned
    on BOTH branches' global summaries, then the two gated vectors are fused.
    """

    def __init__(self, channels: int = 128):
        super().__init__()
        self.cam_temporal = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.ReLU(inplace=True),
            nn.Linear(channels, channels), nn.Sigmoid()
        )
        self.cam_spectral = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.ReLU(inplace=True),
            nn.Linear(channels, channels), nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.GELU()
        )

    def forward(self, f_temporal: torch.Tensor, f_spectral: torch.Tensor) -> torch.Tensor:
        # f_temporal, f_spectral: (B, C)
        joint = torch.cat([f_temporal, f_spectral], dim=-1)   # (B, 2C)
        f_t = f_temporal * self.cam_temporal(joint)
        f_s = f_spectral * self.cam_spectral(joint)
        return self.fusion(torch.cat([f_t, f_s], dim=-1))     # (B, C)


class SequentialCrossAttention(nn.Module):
    """
    True Q/K/V cross-attention operating on sequential features BEFORE GAP.

    Unlike RevisedCrossAttention (which operates on GAP'd vectors), this module
    receives full (B, C, T) feature maps from both branches and computes
    position-to-position attention:

      - Temporal branch queries attend to spectral branch keys/values
      - Spectral branch queries attend to temporal branch keys/values
      - Results are fused and GAP'd to produce (B, C)

    This is the standard cross-attention formulation:
      Q_t = W_q(F_temporal),  K_s = W_k(F_spectral),  V_s = W_v(F_spectral)
      α = softmax(Q_t @ K_s^T / √d_k)
      output_t = α @ V_s

    Bidirectional: both branches attend to each other, then fused.
    """

    def __init__(self, channels: int = 128, d_k: int = 32, n_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.d_k = d_k
        self.n_heads = n_heads
        self.head_dim = d_k // n_heads
        assert d_k % n_heads == 0, f"d_k ({d_k}) must be divisible by n_heads ({n_heads})"

        # Temporal → attends to Spectral (Q from temporal, K/V from spectral)
        self.q_temporal = nn.Conv1d(channels, d_k, 1)
        self.k_spectral = nn.Conv1d(channels, d_k, 1)
        self.v_spectral = nn.Conv1d(channels, channels, 1)

        # Spectral → attends to Temporal (Q from spectral, K/V from temporal)
        self.q_spectral = nn.Conv1d(channels, d_k, 1)
        self.k_temporal = nn.Conv1d(channels, d_k, 1)
        self.v_temporal = nn.Conv1d(channels, channels, 1)

        self.scale = math.sqrt(self.head_dim)

        # Layer norms for stable training
        self.norm_t = nn.LayerNorm(channels)
        self.norm_s = nn.LayerNorm(channels)

        # Fusion: merge the two cross-attended streams
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU()
        )

    def _cross_attn(self, Q_proj, K_proj, V_proj, Q_input, K_input):
        """
        Compute multi-head cross-attention.

        Args:
            Q_proj, K_proj, V_proj: projection layers
            Q_input: (B, C, T_q) - query source
            K_input: (B, C, T_k) - key/value source

        Returns:
            output: (B, C, T_q) - attended features
        """
        B = Q_input.shape[0]
        T_q = Q_input.shape[2]
        T_k = K_input.shape[2]

        Q = Q_proj(Q_input)   # (B, d_k, T_q)
        K = K_proj(K_input)   # (B, d_k, T_k)
        V = V_proj(K_input)   # (B, C, T_k)

        # Reshape for multi-head: (B, n_heads, head_dim, T)
        Q = Q.view(B, self.n_heads, self.head_dim, T_q)
        K = K.view(B, self.n_heads, self.head_dim, T_k)

        # Attention scores: (B, n_heads, T_q, T_k)
        scores = torch.einsum('bndt,bndk->bntk', Q, K) / self.scale
        attn = F.softmax(scores, dim=-1)  # normalize over key positions

        # Reshape V for multi-head application
        # V is (B, C, T_k), we need to split C into n_heads groups
        C = V.shape[1]
        head_c = C // self.n_heads
        V_mh = V.view(B, self.n_heads, head_c, T_k)  # (B, n_heads, head_c, T_k)

        # Apply attention: (B, n_heads, head_c, T_q)
        out = torch.einsum('bntk,bnck->bnct', attn, V_mh)

        # Merge heads back: (B, C, T_q)
        out = out.reshape(B, C, T_q)

        return out

    def forward(self, f_temporal: torch.Tensor, f_spectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_temporal:  (B, C, T) - sequential features from temporal branch
            f_spectral:  (B, C, T) - sequential features from spectral branch

        Returns:
            embedding: (B, C) - fused embedding (GAP applied internally)
        """
        # Bidirectional cross-attention
        # Temporal queries attend to spectral keys/values
        t_attended = self._cross_attn(
            self.q_temporal, self.k_spectral, self.v_spectral,
            f_temporal, f_spectral
        )  # (B, C, T)

        # Spectral queries attend to temporal keys/values
        s_attended = self._cross_attn(
            self.q_spectral, self.k_temporal, self.v_temporal,
            f_spectral, f_temporal
        )  # (B, C, T)

        # Residual connection + LayerNorm
        t_out = self.norm_t((f_temporal + t_attended).transpose(1, 2)).transpose(1, 2)  # (B, C, T)
        s_out = self.norm_s((f_spectral + s_attended).transpose(1, 2)).transpose(1, 2)  # (B, C, T)

        # GAP over time dimension
        t_emb = t_out.mean(dim=-1)   # (B, C)
        s_emb = s_out.mean(dim=-1)   # (B, C)

        # Fuse
        return self.fusion(torch.cat([t_emb, s_emb], dim=-1))  # (B, C)


class SimpleConcatFusion(nn.Module):
    """
    Ablation baseline: simple concatenation + linear projection.

    No attention, no gating — just concat the GAP'd branch vectors and
    project to the embedding dimension. The simplest possible fusion.
    """

    def __init__(self, channels: int = 128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU()
        )

    def forward(self, f_temporal: torch.Tensor, f_spectral: torch.Tensor) -> torch.Tensor:
        # f_temporal, f_spectral: (B, C)
        return self.fusion(torch.cat([f_temporal, f_spectral], dim=-1))  # (B, C)


class ArcFaultNetV2(nn.Module):
    """
    Arc-FaultNet V2 (single-cycle adaptation).

      4 derived channels (B,4,M) ── TemporalBranchV2 ─┐
                                                       ├─ RevisedCrossAttention ─ FC head ─ logit
      STFT of I(t) (B,1,F,T) ───── SpectralBranchV2 ──┘

    The fused 128-d vector is the embedding consumed later by the tree head
    (Stage 5). Use ``forward(..., return_embedding=True)`` to retrieve it.
    """

    def __init__(
        self,
        in_channels: int = 4,
        spec_in_channels: int = 1,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        output_dim: int = 64,
        freq_groups: int = 4,
        classifier_hidden: int = 64,
        dropout: float = 0.3,
        use_se: bool = False,
        se_reduction: int = 8,
        deep_classifier: bool = False,
        fusion_mode: str = 'gated',
        use_freq_gate: bool = True,
        use_channel_attn: bool = True,
        ca_reduction: int = 8
    ):
        super().__init__()
        C = hidden_dims[-1]
        self.fusion_mode = fusion_mode

        self.temporal = TemporalBranchV2(
            in_channels=in_channels, hidden_dims=hidden_dims, output_dim=output_dim,
            use_se=use_se, se_reduction=se_reduction,
            use_channel_attn=use_channel_attn, ca_reduction=ca_reduction
        )
        self.spectral = SpectralBranchV2(
            in_channels=spec_in_channels, hidden_dims=hidden_dims,
            output_dim=output_dim, freq_groups=freq_groups,
            use_se=use_se, se_reduction=se_reduction,
            use_freq_gate=use_freq_gate
        )

        # Fusion mechanism selection
        if fusion_mode == 'cross_attention':
            self.cross_attn = SequentialCrossAttention(channels=C)
        elif fusion_mode == 'concat':
            self.cross_attn = SimpleConcatFusion(channels=C)
        else:  # 'gated' (default, backward-compatible)
            self.cross_attn = RevisedCrossAttention(channels=C)

        # Classifier head — deep variant adds BN + heavier dropout for stability
        if deep_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(C, classifier_hidden),
                nn.BatchNorm1d(classifier_hidden),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(classifier_hidden, classifier_hidden // 2),
                nn.BatchNorm1d(classifier_hidden // 2),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(classifier_hidden // 2, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(C, classifier_hidden), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, 1)
            )

    def extract_embedding(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> torch.Tensor:
        """Return the fused 128-d embedding (input to Stage 5)."""
        if self.fusion_mode == 'cross_attention':
            # SequentialCrossAttention operates on (B, C, T) and does GAP internally
            f_t_seq = self.temporal(x_1d)    # (B, C, T)
            f_s_seq = self.spectral(x_2d)    # (B, C, T)
            return self.cross_attn(f_t_seq, f_s_seq)   # (B, C)
        else:
            # Gated and Concat operate on GAP'd (B, C) vectors
            f_t = self.temporal(x_1d).mean(dim=-1)     # (B, C)
            f_s = self.spectral(x_2d).mean(dim=-1)     # (B, C)
            return self.cross_attn(f_t, f_s)           # (B, C)

    def forward(
        self,
        x_1d: torch.Tensor,
        x_2d: torch.Tensor,
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x_1d: (B, 4, M)        — derived channels [I, |ΔI|, TKEO, RMS_slide]
            x_2d: (B, 1, F, T)     — log-power STFT of I(t)
            return_embedding: if True, also return the fused 128-d embedding
        Returns:
            logits: (B,)           — raw logits for BCEWithLogitsLoss
            (optionally) embedding: (B, 128)
        """
        emb = self.extract_embedding(x_1d, x_2d)   # (B, C)
        logits = self.classifier(emb).squeeze(-1)  # (B,)
        if return_embedding:
            return logits, emb
        return logits


class SpectralBranchV2_NoGate(nn.Module):
    """
    Ablation variant of SpectralBranchV2 without the learnable FrequencyGate.
    Applies Conv2d directly on the raw STFT spectrogram to ensure absolutely
    no attention mechanisms are present.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        output_dim: int = 64,
        freq_groups: int = 4
    ):
        super().__init__()
        self.output_dim = output_dim
        self.freq_groups = freq_groups

        c0, c1, c2 = hidden_dims

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c0, kernel_size=3, padding=1),
            nn.BatchNorm2d(c0), nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),           # freq only (preserve time)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1), nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),           # freq only (preserve time)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2), nn.GELU(),
        )
        self.adaptive = nn.AdaptiveAvgPool2d((freq_groups, output_dim))
        self.proj = nn.Conv1d(c2 * freq_groups, c2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NO freq_gate applied here!
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive(x)
        b, c, g, d = x.shape
        x = x.reshape(b, c * g, d)
        return self.proj(x)


# ═══════════════════════════════════════════════════════
#  ARC-FAULTNET V2 — ABLATION VARIANTS
# ═══════════════════════════════════════════════════════

class ArcFaultNetV2_NoAttention(nn.Module):
    """
    Ablation: Dual-branch V2 WITHOUT cross-attention.

    Both branches extract features independently; their GAP vectors are
    concatenated and fed to a simple FC fusion layer — no channel gating,
    no mutual conditioning.  This answers: *does the attention mechanism
    add anything over naive concatenation?*
    """

    def __init__(
        self,
        in_channels: int = 4,
        spec_in_channels: int = 1,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        output_dim: int = 64,
        freq_groups: int = 4,
        classifier_hidden: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        C = hidden_dims[-1]

        self.temporal = TemporalBranchV2(
            in_channels=in_channels, hidden_dims=hidden_dims, output_dim=output_dim
        )
        self.spectral = SpectralBranchV2_NoGate(
            in_channels=spec_in_channels, hidden_dims=hidden_dims,
            output_dim=output_dim, freq_groups=freq_groups
        )
        # Simple concat + linear (no gating)
        self.fusion = nn.Sequential(
            nn.Linear(C * 2, C), nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(C, classifier_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1)
        )

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor,
                return_embedding: bool = False) -> torch.Tensor:
        f_t = self.temporal(x_1d).mean(dim=-1)   # (B, C)
        f_s = self.spectral(x_2d).mean(dim=-1)   # (B, C)
        emb = self.fusion(torch.cat([f_t, f_s], dim=-1))  # (B, C)
        logits = self.classifier(emb).squeeze(-1)
        if return_embedding:
            return logits, emb
        return logits


class ArcFaultNetV2_NoChanGate(nn.Module):
    """
    Ablation: Cross-attention WITHOUT channel gating.

    The two branch vectors are concatenated and fused through a single FC
    layer (like NoAttention), but preceded by a shared self-attention over
    the concatenated sequence — keeping the spatial/temporal attention but
    removing the per-branch channel gate (sigmoid conditioning).

    This answers: *is it specifically the channel gating that matters, or
    just having any fusion mechanism?*
    """

    def __init__(
        self,
        in_channels: int = 4,
        spec_in_channels: int = 1,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        output_dim: int = 64,
        freq_groups: int = 4,
        classifier_hidden: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        C = hidden_dims[-1]

        self.temporal = TemporalBranchV2(
            in_channels=in_channels, hidden_dims=hidden_dims, output_dim=output_dim
        )
        self.spectral = SpectralBranchV2(
            in_channels=spec_in_channels, hidden_dims=hidden_dims,
            output_dim=output_dim, freq_groups=freq_groups
        )
        # Fusion WITHOUT gating: just concat → 2-layer MLP (same param count as gated)
        self.fusion = nn.Sequential(
            nn.Linear(C * 2, C * 2), nn.GELU(),
            nn.Linear(C * 2, C), nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(C, classifier_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1)
        )

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor,
                return_embedding: bool = False) -> torch.Tensor:
        f_t = self.temporal(x_1d).mean(dim=-1)
        f_s = self.spectral(x_2d).mean(dim=-1)
        emb = self.fusion(torch.cat([f_t, f_s], dim=-1))
        logits = self.classifier(emb).squeeze(-1)
        if return_embedding:
            return logits, emb
        return logits


class ArcFaultNetV2_TemporalOnly(nn.Module):
    """
    Ablation: ONLY the temporal branch (no spectral / STFT).

    The spectral branch is completely removed.  The temporal features
    go through GAP → FC head directly.

    This answers: *what does the spectral branch add?*
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        output_dim: int = 64,
        classifier_hidden: int = 64,
        dropout: float = 0.3,
        use_se: bool = False,
        se_reduction: int = 8,
        deep_classifier: bool = False
    ):
        super().__init__()
        C = hidden_dims[-1]

        self.temporal = TemporalBranchV2(
            in_channels=in_channels, hidden_dims=hidden_dims, output_dim=output_dim,
            use_se=use_se, se_reduction=se_reduction
        )
        if deep_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(C, classifier_hidden),
                nn.BatchNorm1d(classifier_hidden),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(classifier_hidden, classifier_hidden // 2),
                nn.BatchNorm1d(classifier_hidden // 2),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(classifier_hidden // 2, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(C, classifier_hidden), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, 1)
            )

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor = None,
                return_embedding: bool = False) -> torch.Tensor:
        emb = self.temporal(x_1d).mean(dim=-1)   # (B, C)
        logits = self.classifier(emb).squeeze(-1)
        if return_embedding:
            return logits, emb
        return logits


class ArcFaultNetV2_SpectralOnly(nn.Module):
    """
    Ablation: ONLY the spectral branch (no temporal / 1D convolutions).

    The temporal branch is completely removed.  The spectral features
    go through GAP → FC head directly.

    This answers: *what does the temporal branch add?*
    """

    def __init__(
        self,
        spec_in_channels: int = 1,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        output_dim: int = 64,
        freq_groups: int = 4,
        classifier_hidden: int = 64,
        dropout: float = 0.3,
        use_se: bool = False,
        se_reduction: int = 8,
        deep_classifier: bool = False,
        use_freq_gate: bool = True
    ):
        super().__init__()
        C = hidden_dims[-1]

        self.spectral = SpectralBranchV2(
            in_channels=spec_in_channels, hidden_dims=hidden_dims,
            output_dim=output_dim, freq_groups=freq_groups,
            use_se=use_se, se_reduction=se_reduction,
            use_freq_gate=use_freq_gate
        )
        if deep_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(C, classifier_hidden),
                nn.BatchNorm1d(classifier_hidden),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(classifier_hidden, classifier_hidden // 2),
                nn.BatchNorm1d(classifier_hidden // 2),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(classifier_hidden // 2, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(C, classifier_hidden), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, 1)
            )

    def forward(self, x_1d: torch.Tensor = None, x_2d: torch.Tensor = None,
                return_embedding: bool = False) -> torch.Tensor:
        # x_2d is required; x_1d is ignored
        emb = self.spectral(x_2d).mean(dim=-1)   # (B, C)
        logits = self.classifier(emb).squeeze(-1)
        if return_embedding:
            return logits, emb
        return logits


class ArcFaultNetV2_BaselineCNN(nn.Module):
    """
    Ablation: Classic CNN baseline for the V2 data format.

    Same convolutional backbone as TemporalBranchV2 (Conv1d × 3 + BN + GELU
    + pooling) followed by GAP → FC.  NO spectral branch, NO attention,
    NO frequency gating.

    This answers: *is the full V2 architecture better than a plain CNN
    with equivalent convolutional capacity?*
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dims: Tuple[int, int, int] = (32, 64, 128),
        kernel_sizes: Tuple[int, int, int] = (16, 8, 4),
        classifier_hidden: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        dims = [in_channels] + list(hidden_dims)
        layers = []
        for i in range(3):
            layers += [
                nn.Conv1d(dims[i], dims[i + 1],
                          kernel_size=kernel_sizes[i],
                          padding=kernel_sizes[i] // 2),
                nn.BatchNorm1d(dims[i + 1]),
                nn.GELU(),
            ]
            if i < 2:
                layers.append(nn.MaxPool1d(4))
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*layers)

        C = hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(C, classifier_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1)
        )

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor = None,
                return_embedding: bool = False) -> torch.Tensor:
        emb = self.features(x_1d).squeeze(-1)    # (B, C)
        logits = self.classifier(emb).squeeze(-1)
        if return_embedding:
            return logits, emb
        return logits


# ═══════════════════════════════════════════════════════
#  MODEL FACTORY
# ═══════════════════════════════════════════════════════

def get_model(
    model_name: str,
    in_channels: int = 2,
    use_se: bool = False,
    se_reduction: int = 8,
    use_amplitude: bool = False,
    deep_classifier: bool = False,
    fs: float = 1_000_000,
    n_fft: int = 512,
    **kwargs
) -> nn.Module:
    """
    Factory function to get model by name.
    
    Available models:
      - arcfaultnet: Full Arc-FaultNet
      - 1d_only: Only temporal branch
      - no_attention: No joint attention
      - standard_conv: Standard Conv1d instead of parametric
      - independent_cbam: Independent CBAM per branch
      - baseline_cnn: Simple CNN baseline
    
    Enhancement flags (applied to arcfaultnet only):
      - use_se: Add Squeeze-and-Excitation blocks after each conv layer
      - use_amplitude: Add learnable amplitude to Gabor filters
      - deep_classifier: Use deeper classifier head with BatchNorm
    
    Signal parameters:
      - fs: Sampling frequency in Hz (default 1 MHz, use 102400 for decimated)
      - n_fft: FFT size for the 2D spectral branch
    """
    # SSM-only track (Track B). Lazy import avoids a circular import
    # (model_ssm imports building blocks from this module).
    if model_name in ('arcssm', 'arcssm_selective'):
        from model_ssm import ArcSSMNet
        return ArcSSMNet(
            in_channels=4,
            deep_classifier=deep_classifier,
            selective=(model_name == 'arcssm_selective'),
            backbone=kwargs.get('ssm_backbone', 's4d'),
            n_layers=kwargs.get('ssm_layers', 4),
            fas_k=kwargs.get('fas_k', 0),
            fas_channels=kwargs.get('fas_channels', (1, 2)),
            use_voltage=kwargs.get('use_voltage', False),
        )

    models = {
        'arcfaultnet': lambda: ArcFaultNet(
            in_channels=in_channels,
            use_se=use_se,
            se_reduction=se_reduction,
            use_amplitude=use_amplitude,
            deep_classifier=deep_classifier,
            fs=fs,
            n_fft=n_fft
        ),
        '1d_only': lambda: ArcFaultNet_1DOnly(
            in_channels=in_channels, use_se=use_se, se_reduction=se_reduction,
            use_amplitude=use_amplitude, deep_classifier=deep_classifier,
            fs=fs
        ),
        'no_attention': lambda: ArcFaultNet_NoAttention(
            in_channels=in_channels, use_se=use_se, se_reduction=se_reduction,
            use_amplitude=use_amplitude, deep_classifier=deep_classifier,
            fs=fs, n_fft=n_fft
        ),
        'standard_conv': lambda: ArcFaultNet_StandardConv(
            in_channels=in_channels, use_se=use_se, se_reduction=se_reduction,
            use_amplitude=use_amplitude, deep_classifier=deep_classifier,
            fs=fs, n_fft=n_fft
        ),
        'independent_cbam': lambda: ArcFaultNet_IndependentCBAM(
            in_channels=in_channels, use_se=use_se, se_reduction=se_reduction,
            use_amplitude=use_amplitude, deep_classifier=deep_classifier,
            fs=fs, n_fft=n_fft
        ),
        'baseline_cnn': lambda: BaselineCNN(in_channels=in_channels),
        # Arc-FaultNet V2 (single-cycle): 4 I-derived temporal channels + revised
        # spectral branch (STFT of I only) + cross-attention fusion.
        'arcfaultnet_v2': lambda: ArcFaultNetV2(
            in_channels=4, spec_in_channels=1,
            use_se=use_se, se_reduction=se_reduction,
            deep_classifier=deep_classifier,
            fusion_mode=kwargs.get('fusion_mode', 'gated'),
            use_channel_attn=kwargs.get('use_channel_attn', True)
        ),
        # ── V2 ablation variants ──
        'v2_no_attention':   lambda: ArcFaultNetV2_NoAttention(in_channels=4, spec_in_channels=1),
        'v2_no_chan_gate':   lambda: ArcFaultNetV2_NoChanGate(in_channels=4, spec_in_channels=1),
        'v2_temporal_only':  lambda: ArcFaultNetV2_TemporalOnly(in_channels=4),
        'v2_spectral_only':  lambda: ArcFaultNetV2_SpectralOnly(spec_in_channels=1),
        'v2_baseline_cnn':   lambda: ArcFaultNetV2_BaselineCNN(in_channels=4),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()


# ═══════════════════════════════════════════════════════
#  AUTO-DETECT ARCHITECTURE FROM CHECKPOINT
# ═══════════════════════════════════════════════════════

def build_model_from_checkpoint(ckpt_path, device='cpu', fs: float = 1_000_000, n_fft: int = 512):
    """
    Auto-detect model architecture from checkpoint state_dict keys
    and reconstruct the exact model used during training.
    
    This handles all combinations of:
      - SE blocks (use_se)
      - Learnable amplitude (use_amplitude)
      - Deep classifier (deep_classifier)
    
    Args:
        ckpt_path: Path to .pt checkpoint file
        device: Device to load model onto
        fs: Sampling frequency (Hz) — pass dataset.fs for correct Gabor init
        n_fft: FFT size used during training STFT
    
    Returns:
        model: Loaded model in eval mode
    """
    sd = torch.load(ckpt_path, map_location='cpu')

    # Detect if V2
    if 'temporal.features.0.weight' in sd or 'cross_attn.fusion.0.weight' in sd:
        print("  Detected: ArcFaultNetV2 architecture")
        use_se = 'temporal.features.3.fc.0.weight' in sd
        deep_classifier = 'classifier.4.weight' in sd or 'classifier.1.running_mean' in sd
        # Auto-detect fusion_mode from state_dict keys
        if 'cross_attn.q_temporal.weight' in sd:
            fusion_mode = 'cross_attention'
        elif 'cross_attn.cam_temporal.0.weight' in sd:
            fusion_mode = 'gated'
        else:
            fusion_mode = 'concat'
        print(f"  fusion_mode={fusion_mode}, use_se={use_se}, deep_classifier={deep_classifier}")
        model = ArcFaultNetV2(
            in_channels=4, spec_in_channels=1,
            use_se=use_se, deep_classifier=deep_classifier,
            fusion_mode=fusion_mode
        )
        model.load_state_dict(sd)
        model.to(device).eval()
        return model

    # Detect hidden_dims from fusion weight shape [C, 2C, 1]
    C = sd['joint_attn.fusion.weight'].shape[0]

    # Detect hidden_dims[0] from first ParametricConv1d f0 shape
    C0 = sd['branch_1d.features.0.f0'].shape[0]
    C1 = C0 * 2
    hidden_dims = (C0, C1, C)

    # Detect optional features from state_dict keys
    use_amplitude   = 'branch_1d.features.0.amplitude' in sd
    use_se          = 'branch_1d.features.3.fc.0.weight' in sd
    d_k             = sd['joint_attn.sam.query.weight'].shape[0]
    clf_hidden      = sd['classifier.fc.0.weight'].shape[0]
    deep_classifier = 'classifier.fc.4.weight' in sd

    # Auto-detect SE reduction from checkpoint
    if use_se:
        se_reduced = sd['branch_1d.features.3.fc.0.weight'].shape[0]
        se_reduction = C0 // se_reduced
    else:
        se_reduction = 8

    print(f"  Detected: hidden_dims={hidden_dims}, C={C}, d_k={d_k}")
    print(f"  amplitude={use_amplitude}, SE={use_se}, deep_clf={deep_classifier}")
    print(f"  fs={fs:,} Hz, n_fft={n_fft}")

    model = ArcFaultNet(
        in_channels=2,
        hidden_dims=hidden_dims,
        output_dim=64,
        use_parametric=True,
        use_joint_attention=True,
        use_se=use_se,
        se_reduction=se_reduction,
        use_amplitude=use_amplitude,
        deep_classifier=deep_classifier,
        classifier_hidden=clf_hidden,
        fs=fs,
        n_fft=n_fft
    )
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


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
