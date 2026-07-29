"""
model_ssm.py — Arc-FaultNet SSM-only track ("ArcSSM").

This is Track B of the project: a deliberately different modelling approach to
the attention-centric ArcFaultNetV2 (Track A). Here a single diagonal complex
state-space stack (S4D, see ``arc_ssm.py``) replaces the ENTIRE V2 front-end —
both the Conv1d temporal branch AND the STFT spectral branch. The SSM does
everything: its complex resonator state acts as a learnable filter bank
(standing in for the STFT) while its recurrence provides the long-range memory
that motivated this track (pattern before the arc, the zero-crossing signature,
the persistence after onset).

Pipeline:
    x_1d (B, 4, M)   [I, |ΔI|, TKEO(I), RMS_slide(I)]
      -> Conv1d channel-embedding (stride 1: keeps full temporal/HF resolution)
      -> n_layers x S4Block   (bidirectional; selective optional)
      -> LayerNorm -> global average pool over time
      -> Linear -> 128-d embedding
      -> classifier -> logit

The STFT input ``x_2d`` is accepted but ignored, so the existing train.py /
evaluate.py / ablation harness (which always passes both tensors) drives this
model unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from arc_ssm import S4Block


def _build_classifier(C: int, hidden: int, dropout: float, deep: bool) -> nn.Module:
    """Same classifier head as ArcFaultNetV2 (kept identical for fair comparison)."""
    if deep:
        return nn.Sequential(
            nn.Linear(C, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )
    return nn.Sequential(
        nn.Linear(C, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


class ArcSSMNet(nn.Module):
    """
    SSM-only arc-fault detector.

    Args:
        in_channels:   number of derived temporal channels (4 for i_derived4).
        d_model:       SSM feature width H (independent channels).
        d_state:       number of complex states N per channel (resonators).
        n_layers:      number of stacked S4 blocks.
        bidirectional: run the SSM in both time directions (fixed-window
                       classification, so both are available).
        selective:     if True, use the Mamba-style input-dependent Δ (ablation;
                       slower recurrent scan). Default False = classic LTI S4D.
        embed_kernel:  kernel of the input Conv1d channel-embedding. Stride is 1,
                       so no temporal downsampling — the high-frequency arc
                       content is preserved (the whole point of dropping STFT).
    """

    def __init__(
        self,
        in_channels: int = 4,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        bidirectional: bool = True,
        selective: bool = False,
        embed_kernel: int = 7,
        classifier_hidden: int = 64,
        dropout: float = 0.3,
        block_dropout: float = 0.1,
        deep_classifier: bool = False,
        backbone: str = "s4d",
        fas_k: int = 0,
        fas_channels: tuple = (1, 2),
        use_voltage: bool = False,
        v_d_model: int = 96,
        v_d_state: int = 32,
        v_n_layers: int = 2,
        v_embed: int = 64,
        **_ignore,   # swallow fs / n_fft / use_se / fusion_mode passed by get_model
    ):
        super().__init__()
        if backbone not in ("s4d", "mamba"):
            raise ValueError(f"backbone must be s4d|mamba, got {backbone!r}")
        self.backbone = backbone

        # ── FAS (Feature Amplification Strategy, DCAMamba 2025) ────────────
        # Optional order-statistic front-end: per channel, keep the top-K and
        # bottom-K values over time (2K total) and discard the rest — turning
        # the cycle into a compact, TIME-ORDER-INVARIANT amplitude-distribution
        # descriptor. DCAMamba applies it to a DC current (no fundamental); in
        # AC the top/bottom of the raw 50 Hz sinusoid is just its ±peak, so we
        # apply FAS only to the FUNDAMENTAL-SUPPRESSED channels (default |dI|
        # and TKEO), where arc scatter — not the load sinusoid — dominates.
        self.fas_k = int(fas_k)
        self.fas_channels = list(fas_channels)
        enc_in = len(self.fas_channels) if self.fas_k > 0 else in_channels

        self.encoder = nn.Conv1d(
            enc_in, d_model, kernel_size=embed_kernel, padding=embed_kernel // 2
        )
        self.enc_act = nn.GELU()

        # ── sequence-mixing backbone ───────────────────────────────────────
        # s4d   : LTI diagonal-COMPLEX S4D — a resonator filter bank (spectral
        #         inductive bias, FFT-parallel). See arc_ssm.py.
        # mamba : selective S6 (real diagonal, input-dependent Δ/B/C, causal
        #         scan) — the DCAMamba backbone. Forced CAUSAL here: it halves
        #         the (pure-PyTorch) scan cost and matches a real-time AFDD;
        #         bidirectionality buys little on a fixed classification window.
        if backbone == "s4d":
            self.blocks = nn.ModuleList(
                [
                    S4Block(d_model, d_state, bidirectional, selective, block_dropout)
                    for _ in range(n_layers)
                ]
            )
        else:  # "mamba"
            from mamba_block import BiMambaBlock  # lazy: s4d path stays decoupled
            self.blocks = nn.ModuleList(
                [
                    BiMambaBlock(
                        d_model, bidirectional=False, d_state=16, d_conv=4, expand=2
                    )
                    for _ in range(n_layers)
                ]
            )

        self.norm = nn.LayerNorm(d_model)
        self.to_embed = nn.Linear(d_model, 128)

        # ── optional VOLTAGE branch (spectral, bench-invariant complement) ──
        # v(t)'s HF arc signature is (empirically) far more CONSISTENT across
        # benches than i(t)'s (AUC 0.70-0.79 on every campaign vs I's 0.63-0.90).
        # A lighter S4D branch on v_derived4 adds that complementary evidence;
        # the complex S4D resonators are exactly the spectral analyser it needs.
        # Current stays PRIMARY; voltage assists (mainly to stabilise specificity
        # on unseen benches). Fusion = concat of the two branch embeddings.
        self.use_voltage = use_voltage
        if use_voltage:
            self.encoder_v = nn.Conv1d(4, v_d_model, kernel_size=embed_kernel,
                                       padding=embed_kernel // 2)
            self.enc_act_v = nn.GELU()
            self.blocks_v = nn.ModuleList(
                [S4Block(v_d_model, v_d_state, bidirectional, False, block_dropout)
                 for _ in range(v_n_layers)])
            self.norm_v = nn.LayerNorm(v_d_model)
            self.to_embed_v = nn.Linear(v_d_model, v_embed)
        clf_in = 128 + (v_embed if use_voltage else 0)
        self.classifier = _build_classifier(
            clf_in, classifier_hidden, dropout, deep_classifier
        )

    def _apply_fas(self, x_1d: torch.Tensor) -> torch.Tensor:
        """FAS front-end: per selected channel, concat top-K and bottom-K over
        time → (B, len(fas_channels), 2K). Parameter-free (order statistics)."""
        x = x_1d[:, self.fas_channels, :]                          # (B, C', M)
        top = x.topk(self.fas_k, dim=-1, largest=True).values      # (B, C', K)
        bot = x.topk(self.fas_k, dim=-1, largest=False).values     # (B, C', K)
        return torch.cat([top, bot], dim=-1)                       # (B, C', 2K)

    def extract_embedding(
        self, x_1d: torch.Tensor, x_2d: torch.Tensor = None
    ) -> torch.Tensor:
        """Fused embedding (128, or 128+v_embed with use_voltage). x_1d is (B,4,M)
        for the current-only model, or (B,8,M) = [i_derived4 | v_derived4] when the
        voltage branch is on. Also consumable by a downstream tree head."""
        x_i = x_1d[:, :4, :] if self.use_voltage else x_1d
        # ── current branch (primary) ──
        xi = self._apply_fas(x_i) if self.fas_k > 0 else x_i
        xi = self.enc_act(self.encoder(xi))
        xi = xi.transpose(1, 2)
        for blk in self.blocks:
            xi = blk(xi)
        emb_i = self.to_embed(self.norm(xi).mean(dim=1))       # (B, 128)
        if not self.use_voltage:
            return emb_i
        # ── voltage branch (spectral, secondary) ──
        xv = self.enc_act_v(self.encoder_v(x_1d[:, 4:8, :]))
        xv = xv.transpose(1, 2)
        for blk in self.blocks_v:
            xv = blk(xv)
        emb_v = self.to_embed_v(self.norm_v(xv).mean(dim=1))   # (B, v_embed)
        return torch.cat([emb_i, emb_v], dim=-1)               # (B, 128+v_embed)

    def forward(
        self,
        x_1d: torch.Tensor,
        x_2d: torch.Tensor = None,
        return_embedding: bool = False,
    ):
        """
        Args:
            x_1d: (B, 4, M)  — derived channels [I, |ΔI|, TKEO, RMS_slide]
            x_2d:            — ignored (SSM-only; no STFT branch)
        Returns:
            logits: (B,)     — raw logits for BCEWithLogitsLoss
            (optionally) embedding: (B, 128)
        """
        emb = self.extract_embedding(x_1d)
        logits = self.classifier(emb).squeeze(-1)
        if return_embedding:
            return logits, emb
        return logits


if __name__ == "__main__":
    # Sanity check — run where torch is installed (Colab / GPU box).
    torch.manual_seed(0)
    B, M = 4, 2048
    x_1d = torch.randn(B, 4, M)

    for name, kw in [
        ("ArcSSM (S4D, bidir)      ", dict()),
        ("ArcSSM (S4D, causal)     ", dict(bidirectional=False)),
        ("ArcSSM (S4D selective abl)", dict(selective=True, n_layers=2)),
        ("ArcSSM (mamba, no FAS)   ", dict(backbone="mamba", n_layers=2)),
        ("ArcSSM (mamba + FAS K=256)", dict(backbone="mamba", n_layers=2, fas_k=256)),
        ("ArcSSM (I+V dual branch) ", dict(use_voltage=True)),
    ]:
        model = ArcSSMNet(**kw)
        xin = torch.randn(B, 8, M) if kw.get("use_voltage") else x_1d
        logits, emb = model(xin, None, return_embedding=True)
        n = sum(p.numel() for p in model.parameters())
        print(f"[{name}] logits {tuple(logits.shape)}  emb {tuple(emb.shape)}  "
              f"params = {n:,}")
        assert logits.shape == (B,)

    ArcSSMNet()(x_1d).mean().backward()
    print("[grad] backward OK")
