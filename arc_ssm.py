"""
arc_ssm.py — Diagonal complex state-space model (S4D) core for the Arc-FaultNet
SSM-only track ("ArcSSM").

Rationale (engineering, not a copy of Mamba)
--------------------------------------------
For arc-fault detection we replace the whole front-end (temporal *and* spectral)
with a single SSM. The SSM must therefore behave as BOTH a memory machine and a
frequency analyser (it stands in for the STFT branch that is removed). The right
tool for that is the **classic S4/S4D lineage with a complex diagonal state**:

  * A *complex* eigenvalue a = r·e^{iθ} makes each state dimension a **damped
    resonator** — it decays at rate r (memory) while rotating at frequency θ
    (a learnable band-pass filter). A bank of these is a learnable, data-driven
    spectral analyser: exactly what stands in for the STFT.
  * Mamba defaults to a *real* state (chosen for text, where "frequency" is
    meaningless). The Mamba paper itself notes complex helps for continuous /
    perceptual signals — which is what an electrical current waveform is.
  * Staying LTI (non-selective) lets us compute the SSM as a **global
    convolution via FFT** — O(L log L), fully parallel, pure PyTorch, no custom
    CUDA kernel — and deploy it on hardware as a fixed IIR filter bank (the
    embedded constraint).

Selection (the Mamba S6 idea, input-dependent Δ) is available as an *ablation*
flag (``selective=True``). It makes the dynamics time-varying, so it drops to a
sequential recurrent scan (slower) — we turn it on only to measure whether
content-adaptivity improves arc-signature detection, rather than assuming it.

References: Gu, Goel & Ré, S4 (2022); Gu et al., S4D "On the Parameterization
and Initialization of Diagonal State Space Models" (2022); Gu & Dao, Mamba
(2023, arXiv:2312.00752).

Shapes: B = batch, H = d_model (independent SSM channels), N = d_state,
L = sequence length.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_conv_fft(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Depthwise *causal* convolution of ``u`` (B, H, L) with a per-channel kernel
    ``K`` (H, L), computed by FFT. Zero-padding to 2L makes it linear (not
    circular); cropping to the first L samples enforces causality:
        y[b,h,t] = Σ_{l=0}^{t} K[h,l] · u[b,h,t-l]
    """
    L = u.size(-1)
    n = 2 * L
    Uf = torch.fft.rfft(u, n=n, dim=-1)          # (B, H, n//2+1)
    Kf = torch.fft.rfft(K, n=n, dim=-1)          # (H,   n//2+1)
    y = torch.fft.irfft(Uf * Kf, n=n, dim=-1)[..., :L]
    return y


class S4DKernel(nn.Module):
    """
    Parameters of H independent diagonal complex SSMs (N states each) and the
    SSM convolution kernel they generate.

        A_{h,n} = -exp(logA_re) + i·A_im            (stable: Re(A) < 0)
        Ā       = exp(Δ · A)                          (ZOH discretisation, B ≡ 1)
        B̄       = (Ā - 1) / A
        K[h,l]  = Σ_n Re( C_{h,n} · B̄_{h,n} · Ā_{h,n}^l )

    ``A_im`` is initialised to π·n (S4D-Lin): a spread of resonator frequencies
    across the band, so the state acts as a learnable filter bank from the start.
    """

    def __init__(self, d_model: int, d_state: int = 64,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        H, N = d_model, d_state

        # Δ : one step size per channel, log-uniform init in [dt_min, dt_max].
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # A : S4D-Lin init. Real part = -1/2 (stored as log for positivity of the
        # magnitude), imaginary part = π·n (resonator frequencies).
        self.log_A_re = nn.Parameter(torch.log(0.5 * torch.ones(H, N)))
        A_im = math.pi * torch.arange(N, dtype=torch.float32).unsqueeze(0).repeat(H, 1)
        self.A_im = nn.Parameter(A_im)

        # C : complex output projection, stored as two real tensors.
        self.C_re = nn.Parameter(torch.randn(H, N) * (0.5 ** 0.5))
        self.C_im = nn.Parameter(torch.randn(H, N) * (0.5 ** 0.5))

    def A(self) -> torch.Tensor:
        """Complex diagonal state matrix, (H, N)."""
        return -torch.exp(self.log_A_re) + 1j * self.A_im

    def C(self) -> torch.Tensor:
        """Complex output vector, (H, N)."""
        return torch.complex(self.C_re, self.C_im)

    def forward(self, L: int) -> torch.Tensor:
        """Materialise the real convolution kernel K of shape (H, L)."""
        dt = torch.exp(self.log_dt).unsqueeze(-1)            # (H, 1)
        A = self.A()                                         # (H, N)
        Abar = torch.exp(dt * A)                             # (H, N)
        Bbar = (Abar - 1.0) / A                              # (H, N)   (B ≡ 1)
        CB = self.C() * Bbar                                 # (H, N)

        # Vandermonde Ā^l via exp(l·log Ā). l is integer, so the 2π phase
        # wrapping of the complex log is exact (exp(i·l·2πk) = 1).
        l = torch.arange(L, device=self.log_dt.device, dtype=torch.float32)
        vander = torch.exp(torch.log(Abar).unsqueeze(-1) * l)   # (H, N, L) complex
        K = torch.einsum("hn,hnl->hl", CB, vander).real         # (H, L)
        return K


class S4D(nn.Module):
    """
    One S4D sequence-mixing layer over H independent channels.

    ``selective=False`` (default): LTI, computed by FFT convolution — fast and
    parallel. ``selective=True``: input-dependent Δ (Mamba-style selection on Δ
    only), computed by a sequential complex recurrence — the ablation path.
    ``bidirectional=True`` runs a second, independent kernel over the reversed
    sequence and sums (valid because we classify a whole fixed window).
    """

    def __init__(self, d_model: int, d_state: int = 64,
                 bidirectional: bool = True, selective: bool = False,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        self.selective = selective

        self.kernel_fwd = S4DKernel(d_model, d_state, dt_min, dt_max)
        self.kernel_bwd = (
            S4DKernel(d_model, d_state, dt_min, dt_max) if bidirectional else None
        )
        self.D = nn.Parameter(torch.ones(d_model))            # skip connection

        if selective:
            # Input-dependent Δ (the selection). One projection per direction.
            self.dt_proj_fwd = nn.Linear(d_model, d_model)
            self.dt_proj_bwd = nn.Linear(d_model, d_model) if bidirectional else None

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, H, L)
        if not self.selective:
            y = _causal_conv_fft(u, self.kernel_fwd(u.size(-1)))
            if self.bidirectional:
                yb = _causal_conv_fft(torch.flip(u, [-1]), self.kernel_bwd(u.size(-1)))
                y = y + torch.flip(yb, [-1])
        else:
            y = self._selective_scan(u, self.kernel_fwd, self.dt_proj_fwd)
            if self.bidirectional:
                yb = self._selective_scan(
                    torch.flip(u, [-1]), self.kernel_bwd, self.dt_proj_bwd
                )
                y = y + torch.flip(yb, [-1])
        return y + u * self.D.unsqueeze(-1)

    def _selective_scan(self, u, kernel, dt_proj) -> torch.Tensor:
        """Time-varying-Δ recurrence with a complex state (ablation path)."""
        B_, H, L = u.shape
        A = kernel.A()                                        # (H, N)
        C = kernel.C()                                        # (H, N)
        base_dt = torch.exp(kernel.log_dt)                    # (H,)

        # Δ_t = softplus(W · u_t + base_dt) — content-dependent step size.
        dt = F.softplus(dt_proj(u.transpose(1, 2)) + base_dt)  # (B, L, H)
        dt = dt.transpose(1, 2)                                # (B, H, L)

        h = torch.zeros(B_, H, self.d_state, dtype=torch.cfloat, device=u.device)
        ys = []
        for t in range(L):
            dtt = dt[:, :, t].unsqueeze(-1)                    # (B, H, 1)
            Abar = torch.exp(dtt * A)                          # (B, H, N)
            Bbar = (Abar - 1.0) / A                            # (B, H, N)
            h = Abar * h + Bbar * u[:, :, t].unsqueeze(-1)
            ys.append(torch.einsum("bhn,hn->bh", h, C).real)   # (B, H)
        return torch.stack(ys, dim=-1)                         # (B, H, L)


class S4Block(nn.Module):
    """Pre-norm residual block: LayerNorm -> S4D -> GELU -> pointwise mix."""

    def __init__(self, d_model: int, d_state: int = 64,
                 bidirectional: bool = True, selective: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s4d = S4D(d_model, d_state, bidirectional, selective)
        self.act = nn.GELU()
        self.mix = nn.Linear(d_model, d_model)   # mixes across channels
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H)
        z = self.norm(x)
        z = self.s4d(z.transpose(1, 2)).transpose(1, 2)   # sequence mixing
        z = self.drop(self.mix(self.act(z)))              # channel mixing
        return x + z


if __name__ == "__main__":
    # Sanity check — run where torch is installed (Colab / GPU box).
    torch.manual_seed(0)
    B, H, N, L = 2, 16, 8, 64
    u = torch.randn(B, H, L)

    # (1) FFT kernel vs the recurrence must agree in the non-selective case.
    layer = S4D(H, N, bidirectional=False, selective=False)
    y_fft = layer(u)
    with torch.no_grad():
        base_dt = torch.exp(layer.kernel_fwd.log_dt)
        dt = base_dt.view(1, H, 1).expand(B, H, L)
        A, C = layer.kernel_fwd.A(), layer.kernel_fwd.C()
        h = torch.zeros(B, H, N, dtype=torch.cfloat)
        ys = []
        for t in range(L):
            Abar = torch.exp(dt[:, :, t].unsqueeze(-1) * A)
            Bbar = (Abar - 1.0) / A
            h = Abar * h + Bbar * u[:, :, t].unsqueeze(-1)
            ys.append(torch.einsum("bhn,hn->bh", h, C).real)
        y_rec = torch.stack(ys, -1) + u * layer.D.unsqueeze(-1)
    err = (y_fft - y_rec).abs().max().item()
    print(f"[FFT vs recurrence] max abs diff = {err:.2e}  (should be ~1e-5)")

    # (2) Block shapes + gradient flow.
    blk = S4Block(H, N, bidirectional=True, selective=False)
    x = torch.randn(B, L, H)
    out = blk(x)
    print(f"[S4Block] in {tuple(x.shape)} -> out {tuple(out.shape)}")
    out.mean().backward()
    print("[grad] backward OK")

    # (3) Selective (ablation) path runs and has grads.
    blk_sel = S4Block(H, N, bidirectional=True, selective=True)
    blk_sel(torch.randn(B, L, H)).mean().backward()
    print("[selective] scan + backward OK")
