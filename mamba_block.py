"""
mamba_block.py — Pure-PyTorch selective state-space model (Mamba / S6) blocks
for the Arc-FaultNet SSM track.

Reference
---------
Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
(2023), arXiv:2312.00752. This module implements Algorithm 2 (SSM + Selection,
the "S6" recurrence) and the simplified Mamba block of Figure 3.

Why a pure-PyTorch implementation?
----------------------------------
This is a *pure PyTorch* selective scan (no CUDA kernel, no ``mamba-ssm``
dependency): the recurrence is unrolled in a Python loop over the sequence
length. It is slower than the fused hardware-aware kernel described in the
paper (Section 3.3), but it runs on CPU / any GPU, has no fragile build step,
and is easy to read and debug. That is exactly what we want to first answer the
question "does a selective SSM work on the arc-fault problem at all?" before
committing to the fused kernel.

Design choices specific to Arc-FaultNet
---------------------------------------
* The arc-fault classifier consumes a *fixed* window (e.g. 2048 samples of one
  50 Hz cycle), not an infinite stream, so both time directions are available.
  We therefore default to a **bidirectional** block (one SSM forward, one on the
  reversed sequence), which is the accuracy-oriented recipe (cf. Vision Mamba).
  Set ``bidirectional=False`` for the causal variant that matches a real-time /
  embedded AFDD deployment (constant time and memory per step).
* ``MambaTemporalBranch`` is a **drop-in replacement** for the Conv1d
  ``TemporalBranchV2`` in ``model.py``: same input/output contract
  ``(B, C_in, M) -> (B, C_out, T)``, so the rest of Arc-FaultNet V2 (STFT
  branch, cross-attention fusion, classifier) is reused unchanged and any
  performance delta is attributable to Conv-vs-SSM temporal modelling.

Shape convention (matching the paper)
--------------------------------------
    B = batch,  L = sequence length,  D = d_inner (expanded model dim),
    N = d_state.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    The selective-scan recurrence (Algorithm 2, discretised with ZOH).

        Ā_t   = exp(Δ_t · A)                        (discretised state matrix)
        h_t   = Ā_t · h_{t-1} + (Δ_t · B_t) · u_t
        y_t   = C_t · h_t + D · u_t

    Because Δ, B and C are functions of the input (see :class:`SelectiveSSM`),
    the dynamics are *time-varying* — this is the "selection" mechanism that
    lets the model filter irrelevant samples and retain the arc signature.

    Args:
        u:     (B, D, L)   input sequence (post depthwise-conv, post-SiLU)
        delta: (B, D, L)   input-dependent step size Δ (strictly positive)
        A:     (D, N)      state matrix (negative real; parameterised as -exp)
        B:     (B, N, L)   input-dependent input matrix
        C:     (B, N, L)   input-dependent output matrix
        D:     (D,)        skip / direct feedthrough

    Returns:
        y:     (B, D, L)
    """
    batch, d_inner, L = u.shape
    d_state = A.shape[1]

    # Discretise. Shapes broadcast to (B, D, L, N):
    #   deltaA   = exp(Δ · A)
    #   deltaB_u = (Δ · B) · u          (Euler approximation of B̄, as in the
    #                                    reference minimal Mamba implementations)
    # A is (D, N); broadcast as (1, D, 1, N) so deltaA[b,d,l,n] = exp(Δ[b,d,l]·A[d,n]).
    deltaA = torch.exp(delta.unsqueeze(-1) * A[None, :, None, :])      # (B, D, L, N)
    deltaB_u = (
        delta.unsqueeze(-1)                                          # (B, D, L, 1)
        * B.permute(0, 2, 1).unsqueeze(1)                            # (B, 1, L, N)
        * u.unsqueeze(-1)                                            # (B, D, L, 1)
    )                                                                # (B, D, L, N)

    h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]                   # (B, D, N)
        y_t = torch.einsum("bdn,bn->bd", h, C[:, :, t])              # (B, D)
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)                                      # (B, D, L)
    y = y + u * D.view(1, -1, 1)
    return y


class SelectiveSSM(nn.Module):
    """
    One Mamba block core (Figure 3, right), unidirectional / causal.

        in_proj -> [x, z]
        x  -> causal depthwise Conv1d -> SiLU -> selective SSM
        y  = SSM(x) * SiLU(z)          (gating)
        out = out_proj(y)

    The pre-norm and residual connection are handled by the wrapping
    :class:`BiMambaBlock`.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank

        # Project input to the expanded dimension, twice (main path x + gate z).
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Causal depthwise convolution over time (local context before the SSM).
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # The selection mechanism: Δ, B, C are all produced from the input.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A initialised as S4D-Real: A_n = -(n) for n = 1..N, shared idea across
        # channels (the paper's default real-valued init, Section 3.6).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))          # store log for stability
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        _, L, _ = x.shape

        x_and_z = self.in_proj(x)                        # (B, L, 2*d_inner)
        x_in, z = x_and_z.chunk(2, dim=-1)               # each (B, L, d_inner)

        # Causal depthwise conv (crop the right pad to keep it causal).
        x_in = x_in.transpose(1, 2)                      # (B, d_inner, L)
        x_in = self.conv1d(x_in)[..., :L]
        x_in = x_in.transpose(1, 2)                      # (B, L, d_inner)
        x_in = F.silu(x_in)

        # Input-dependent Δ, B, C (the "selection").
        x_dbl = self.x_proj(x_in)                        # (B, L, dt_rank + 2N)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))                # (B, L, d_inner), > 0

        A = -torch.exp(self.A_log)                       # (d_inner, d_state), < 0

        y = selective_scan(
            u=x_in.transpose(1, 2),                      # (B, d_inner, L)
            delta=dt.transpose(1, 2),                    # (B, d_inner, L)
            A=A,
            B=B.transpose(1, 2),                         # (B, d_state, L)
            C=C.transpose(1, 2),                         # (B, d_state, L)
            D=self.D,
        )                                                # (B, d_inner, L)

        y = y.transpose(1, 2)                            # (B, L, d_inner)
        y = y * F.silu(z)                                # gate
        return self.out_proj(y)                          # (B, L, d_model)


class BiMambaBlock(nn.Module):
    """
    Pre-norm residual Mamba block, optionally bidirectional.

    ``out = x + SSM_fwd(norm(x)) [+ SSM_bwd(norm(x) reversed) reversed]``
    """

    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        norm: bool = True,
        **ssm_kwargs,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.fwd = SelectiveSSM(d_model, **ssm_kwargs)
        self.bwd = SelectiveSSM(d_model, **ssm_kwargs) if bidirectional else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        h = self.norm(x)
        out = self.fwd(h)
        if self.bwd is not None:
            out = out + self.bwd(h.flip(1)).flip(1)
        return x + out


class MambaTemporalBranch(nn.Module):
    """
    Selective-SSM temporal branch — drop-in replacement for ``TemporalBranchV2``.

    Contract (identical to the Conv1d branch it replaces):
        input  : (B, in_channels, M)     e.g. (B, 4, 2048), the derived channels
                 [I, |ΔI|, TKEO(I), RMS_slide(I)]
        output : (B, out_channels, T)     e.g. (B, 128, T'), fed to the fusion GAP

    Pipeline:
        strided Conv1d "patch embedding"  (M -> L' tokens, C_in -> d_model)
        -> LayerNorm
        -> n_layers x BiMambaBlock
        -> LayerNorm
        -> Linear projection  (d_model -> out_channels)
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 128,
        d_model: int = 64,
        n_layers: int = 4,
        patch_stride: int = 8,
        patch_kernel: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        if patch_kernel is None:
            patch_kernel = patch_stride * 2 + 1

        # "Patch embedding": downsample the long raw window into a shorter token
        # sequence. Keeps cost low and brings the length close to the feature
        # resolution the fusion stage expects (the SSM is still O(L), the stride
        # just reduces the constant).
        self.embed = nn.Conv1d(
            in_channels,
            d_model,
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_kernel // 2,
        )
        self.embed_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model,
                    bidirectional=bidirectional,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, M)
        x = self.embed(x)                    # (B, d_model, L')
        x = x.transpose(1, 2)                # (B, L', d_model)
        x = self.embed_norm(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)
        x = self.proj(x)                     # (B, L', out_channels)
        return x.transpose(1, 2)             # (B, out_channels, L')  == (B, 128, T)


if __name__ == "__main__":
    # Sanity check: run where torch is installed (Colab / GPU box).
    torch.manual_seed(0)

    B, C_in, M = 2, 4, 2048
    x = torch.randn(B, C_in, M)

    for bidir in (True, False):
        branch = MambaTemporalBranch(
            in_channels=C_in, out_channels=128, d_model=64, n_layers=4,
            patch_stride=8, bidirectional=bidir,
        )
        y = branch(x)
        n_params = sum(p.numel() for p in branch.parameters())
        tag = "bidirectional" if bidir else "causal"
        print(f"[{tag:13s}] in {tuple(x.shape)} -> out {tuple(y.shape)} | "
              f"params = {n_params:,}")
        assert y.shape[0] == B and y.shape[1] == 128, "output contract broken"

    # Gradient flow check (overfit-ability precondition).
    branch = MambaTemporalBranch()
    y = branch(x).mean()
    y.backward()
    n_grad = sum(1 for p in branch.parameters() if p.grad is not None)
    print(f"[grad check ] {n_grad} tensors received gradients — OK")
