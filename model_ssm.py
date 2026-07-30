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
        **_ignore,   # swallow fs / n_fft / use_se / fusion_mode passed by get_model
    ):
        super().__init__()
        if backbone not in ("s4d", "mamba"):
            raise ValueError(f"backbone must be s4d|mamba, got {backbone!r}")
        self.backbone = backbone

        self.encoder = nn.Conv1d(
            in_channels, d_model, kernel_size=embed_kernel, padding=embed_kernel // 2
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
        self.classifier = _build_classifier(
            128, classifier_hidden, dropout, deep_classifier
        )

    def extract_embedding(
        self, x_1d: torch.Tensor, x_2d: torch.Tensor = None
    ) -> torch.Tensor:
        """Return the 128-d embedding (also consumable by a downstream tree head)."""
        x = self.enc_act(self.encoder(x_1d))     # (B, H, L)
        x = x.transpose(1, 2)                    # (B, L, H)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)                        # global average pool over time
        return self.to_embed(x)                  # (B, 128)

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
        ("ArcSSM (mamba backbone)  ", dict(backbone="mamba", n_layers=2)),
    ]:
        model = ArcSSMNet(**kw)
        logits, emb = model(x_1d, None, return_embedding=True)
        n = sum(p.numel() for p in model.parameters())
        print(f"[{name}] logits {tuple(logits.shape)}  emb {tuple(emb.shape)}  "
              f"params = {n:,}")
        assert logits.shape == (B,) and emb.shape == (B, 128)

    ArcSSMNet()(x_1d).mean().backward()
    print("[grad] backward OK")
