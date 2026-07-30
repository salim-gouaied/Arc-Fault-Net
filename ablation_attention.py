"""
ablation_attention.py — standalone attention-ablation study for Arc-FaultNet.

Written from scratch for the paper's attention-ablation section. It is deliberately
INDEPENDENT of the variant classes in model.py: a single configurable model exposes
the three attention mechanisms as switches, so every ablation differs from the full
model by exactly one component.

Switches
--------
  cross_attention : True  -> Sequential Cross-Attention (bidirectional multi-head Q/K/V)
                    False -> simple concatenation + linear fusion
  freq_gate       : True  -> learnable soft FrequencyGate over the full STFT
                    False -> fixed band-pass slice (2 kHz -> Nyquist), no gate
  channel_attn    : True  -> Descriptor Channel Attention (avg+max) in the temporal branch
                    False -> identity (plain Conv1d stack)
  deep_classifier : deep head with BatchNorm + progressive dropout (used for ALL variants here)

Study variants (each = full model minus one mechanism, plus the all-off baseline)
    full         cross=on  gate=on  dca=on
    no_xattn     cross=OFF gate=on  dca=on     (concat instead of cross-attention)
    no_freqgate  cross=on  gate=OFF dca=on     (fixed 2 kHz–Nyquist band)
    no_dca       cross=on  gate=on  dca=OFF    (plain conv temporal branch)
    none         cross=OFF gate=OFF dca=OFF    (no attention at all)

Data     : combined_dataset_2048 (i_derived4 front-end), STFT n_fft=128 / hop=64.
Protocol : random 70/15/15 split, AdamW 3e-4 / wd 5e-4, cosine warm restarts
           (T0=10, Tmult=2), BCEWithLogitsLoss + label smoothing 0.05, grad-clip 0.5,
           early stopping on validation F1 (NOT F-beta), deep classifier for all.

Usage
-----
  python ablation_attention.py --mode validate
  python ablation_attention.py --mode diagram --variant none
  python ablation_attention.py --mode train --variants none full --seeds 42 2 3 4 5
  python ablation_attention.py --mode plot
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Data loading only (not an ablation component).
from dataset import ArcFaultDataset


# ═══════════════════════════════════════════════════════════════════════
#  Building blocks
# ═══════════════════════════════════════════════════════════════════════

class DescriptorChannelAttention(nn.Module):
    """Peak-aware channel attention: avg AND max pooling through a shared MLP."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )

    def forward(self, x):                       # (B, C, T)
        avg = x.mean(dim=-1)                    # sustained energy
        mx = x.amax(dim=-1)                     # transient peaks
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w.unsqueeze(-1)


class TemporalBranch(nn.Module):
    """1-D conv stack over the 4 descriptors; DCA optional (input + after each block)."""

    def __init__(self, in_ch=4, dims=(32, 64, 128), ks=(16, 8, 4),
                 out_dim=64, channel_attn=True):
        super().__init__()
        d = [in_ch] + list(dims)
        layers = []
        if channel_attn:
            layers.append(DescriptorChannelAttention(in_ch))
        for i in range(3):
            layers += [nn.Conv1d(d[i], d[i + 1], ks[i], padding=ks[i] // 2),
                       nn.BatchNorm1d(d[i + 1]), nn.GELU()]
            if channel_attn:
                layers.append(DescriptorChannelAttention(d[i + 1]))
            if i < 2:
                layers.append(nn.MaxPool1d(4))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x):                        # (B, 4, M) -> (B, 128, out_dim)
        return self.pool(self.features(x))


class FrequencyGate(nn.Module):
    """Learnable soft mask along the frequency axis."""

    def __init__(self, in_ch=1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(3, 1), padding=(1, 0)), nn.Sigmoid()
        )

    def forward(self, x):                        # (B, C, F, T)
        return x * self.gate(x)


class SpectralBranch(nn.Module):
    """2-D conv stack over the log-power STFT. Frequency selection is either a
    learnable gate (freq_gate=True) or a fixed band-pass slice (freq_gate=False)."""

    def __init__(self, in_ch=1, dims=(32, 64, 128), out_dim=64, freq_groups=4,
                 freq_gate=True, band=(3, 65)):
        super().__init__()
        self.freq_gate = FrequencyGate(in_ch) if freq_gate else None
        self.band = None if freq_gate else band     # (low_bin, high_bin), inclusive-exclusive
        c0, c1, c2 = dims
        self.block1 = nn.Sequential(nn.Conv2d(in_ch, c0, 3, padding=1),
                                    nn.BatchNorm2d(c0), nn.GELU(), nn.MaxPool2d((2, 1)))
        self.block2 = nn.Sequential(nn.Conv2d(c0, c1, 3, padding=1),
                                    nn.BatchNorm2d(c1), nn.GELU(), nn.MaxPool2d((2, 1)))
        self.block3 = nn.Sequential(nn.Conv2d(c1, c2, 3, padding=1),
                                    nn.BatchNorm2d(c2), nn.GELU())
        self.adaptive = nn.AdaptiveAvgPool2d((freq_groups, out_dim))
        self.proj = nn.Conv1d(c2 * freq_groups, c2, kernel_size=1)

    def forward(self, x):                        # (B, 1, F, T) -> (B, 128, out_dim)
        if self.freq_gate is not None:
            x = self.freq_gate(x)
        elif self.band is not None:
            x = x[:, :, self.band[0]:self.band[1], :]
        x = self.block3(self.block2(self.block1(x)))
        x = self.adaptive(x)                      # (B, C, freq_groups, out_dim)
        b, c, g, d = x.shape
        x = x.reshape(b, c * g, d)
        return self.proj(x)                       # (B, C, out_dim)


class SequentialCrossAttention(nn.Module):
    """Bidirectional multi-head cross-attention on the (B, C, T) sequences before GAP."""

    def __init__(self, channels=128, d_k=32, n_heads=4):
        super().__init__()
        assert d_k % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_k // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.q_t = nn.Conv1d(channels, d_k, 1); self.k_s = nn.Conv1d(channels, d_k, 1)
        self.v_s = nn.Conv1d(channels, channels, 1)
        self.q_s = nn.Conv1d(channels, d_k, 1); self.k_t = nn.Conv1d(channels, d_k, 1)
        self.v_t = nn.Conv1d(channels, channels, 1)
        self.norm_t = nn.LayerNorm(channels); self.norm_s = nn.LayerNorm(channels)
        self.fusion = nn.Sequential(nn.Linear(channels * 2, channels), nn.GELU())

    def _attn(self, Qp, Kp, Vp, Qin, Kin):
        B, Tq, Tk = Qin.shape[0], Qin.shape[2], Kin.shape[2]
        Q = Qp(Qin).view(B, self.n_heads, self.head_dim, Tq)
        K = Kp(Kin).view(B, self.n_heads, self.head_dim, Tk)
        V = Vp(Kin); C = V.shape[1]
        Vmh = V.view(B, self.n_heads, C // self.n_heads, Tk)
        attn = F.softmax(torch.einsum('bndt,bndk->bntk', Q, K) / self.scale, dim=-1)
        return torch.einsum('bntk,bnck->bnct', attn, Vmh).reshape(B, C, Tq)

    def forward(self, f_t, f_s):                 # (B, C, T) each -> (B, C)
        t_att = self._attn(self.q_t, self.k_s, self.v_s, f_t, f_s)
        s_att = self._attn(self.q_s, self.k_t, self.v_t, f_s, f_t)
        t_out = self.norm_t((f_t + t_att).transpose(1, 2)).transpose(1, 2)
        s_out = self.norm_s((f_s + s_att).transpose(1, 2)).transpose(1, 2)
        return self.fusion(torch.cat([t_out.mean(-1), s_out.mean(-1)], dim=-1))


class ConcatFusion(nn.Module):
    """Ablation of cross-attention: GAP each branch, concatenate, linear-fuse."""

    def __init__(self, channels=128):
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(channels * 2, channels), nn.GELU())

    def forward(self, f_t, f_s):                 # (B, C, T) each -> (B, C)
        return self.fusion(torch.cat([f_t.mean(-1), f_s.mean(-1)], dim=-1))


def make_head(C=128, hidden=64, deep=True, dropout=0.3):
    if deep:
        return nn.Sequential(
            nn.Linear(C, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1))
    return nn.Sequential(nn.Linear(C, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


class AblationNet(nn.Module):
    """Dual-branch detector with switchable attention mechanisms, descriptor bank
    and spectral branch. Every study variant differs from `full` by one switch."""

    def __init__(self, cross_attention=True, freq_gate=True, channel_attn=True,
                 descriptors=True, spectral=True,
                 deep_classifier=True, in_ch=4, spec_ch=1, C=128, out_dim=64, band=(3, 65)):
        super().__init__()
        self.descriptors = descriptors
        self.use_spectral = spectral
        # descriptors=False -> feed the RMS-normalised current alone, dropping
        # [|dI|, TKEO, RMS_slide]. Same normalisation, one input channel.
        self.temporal = TemporalBranch(in_ch=in_ch if descriptors else 1,
                                       out_dim=out_dim, channel_attn=channel_attn)
        if spectral:
            self.spectral = SpectralBranch(in_ch=spec_ch, out_dim=out_dim,
                                           freq_gate=freq_gate, band=band)
            self.fuse = SequentialCrossAttention(C) if cross_attention else ConcatFusion(C)
        else:
            self.spectral = None
            self.fuse = None
        self.classifier = make_head(C, deep=deep_classifier)

    def forward(self, x_1d, x_2d, return_embedding=False):
        if not self.descriptors:
            x_1d = x_1d[:, :1]                   # keep channel 0 = normalised I only
        f_t = self.temporal(x_1d)                # (B, C, T)
        if self.use_spectral:
            f_s = self.spectral(x_2d)            # (B, C, T)
            z = self.fuse(f_t, f_s)              # (B, C)
        else:
            z = f_t.mean(-1)                     # GAP straight into the head
        logits = self.classifier(z).squeeze(-1)
        return (logits, z) if return_embedding else logits


# ═══════════════════════════════════════════════════════════════════════
#  Variant registry
# ═══════════════════════════════════════════════════════════════════════

_D = dict(cross_attention=True, freq_gate=True, channel_attn=True,
          descriptors=True, spectral=True)

VARIANTS = {
    'full':           {**_D},
    'no_xattn':       {**_D, 'cross_attention': False},
    'no_freqgate':    {**_D, 'freq_gate': False},
    'no_dca':         {**_D, 'channel_attn': False},
    'none':           {**_D, 'cross_attention': False, 'freq_gate': False,
                             'channel_attn': False},
    # Front-end / branch ablations
    'no_descriptors': {**_D, 'descriptors': False},
    'temporal_only':  {**_D, 'spectral': False},
}
VARIANT_LABEL = {
    'full': 'Full (all mechanisms)',
    'no_xattn': '– Cross-attn (concat)',
    'no_freqgate': '– Freq gate (fixed band)',
    'no_dca': '– Descriptor chan. attn',
    'none': 'None (no attention)',
    'no_descriptors': '– Descriptor bank (raw I only)',
    'temporal_only': '– Spectral branch (temporal only)',
}


def build_model(name, deep_classifier=True, band=(3, 65)):
    return AblationNet(deep_classifier=deep_classifier, band=band, **VARIANTS[name])


def band_bins(low_khz=2.0, fs=102_400.0, n_fft=128):
    """First STFT bin >= low_khz, up to Nyquist (exclusive high = n_fft//2+1)."""
    bin_hz = fs / n_fft
    low = int(math.ceil(low_khz * 1000.0 / bin_hz))
    high = n_fft // 2 + 1
    return low, high


# ═══════════════════════════════════════════════════════════════════════
#  Train / evaluate
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Removes the run-to-run spread observed when re-training the same seed.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _fbeta(precision, recall, beta):
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-8)


def monitor_score(m, monitor='val_fbeta', fbeta=0.5):
    """Model-selection score, matching train.py::_monitor_score."""
    if monitor == 'val_f1':
        return m['f1']
    if monitor == 'val_fbeta':
        return _fbeta(m['precision'], m['recall'], fbeta)
    raise ValueError(f"unknown monitor {monitor!r}")


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    tp = tn = fp = fn = 0
    for x1, x2, y, _ in loader:
        x1, x2 = x1.to(device), x2.to(device)
        p = torch.sigmoid(model(x1, x2)).cpu().numpy()
        pred = (p > threshold).astype(int)
        yy = y.numpy().astype(int)
        tp += int(((pred == 1) & (yy == 1)).sum())
        tn += int(((pred == 0) & (yy == 0)).sum())
        fp += int(((pred == 1) & (yy == 0)).sum())
        fn += int(((pred == 0) & (yy == 1)).sum())
    tot = max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1e-9); rec = tp / max(tp + fn, 1e-9)
    spec = tn / max(tn + fp, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(accuracy=(tp + tn) / tot, precision=prec, recall=rec,
                specificity=spec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def train_variant(name, dataset, seed, device, band, epochs=60, patience=30,
                  batch_size=64, lr=3e-4, wd=5e-4, grad_clip=0.5, label_smoothing=0.05,
                  monitor='val_fbeta', fbeta=0.5, augment=True, deterministic=False):
    set_seed(seed, deterministic)
    n = len(dataset)
    idx = np.random.permutation(n)
    ntr, nva = int(n * 0.7), int(n * 0.15)
    tr, va, te = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
    dl = lambda s, sh: DataLoader(Subset(dataset, s), batch_size=batch_size, shuffle=sh,
                                  num_workers=4, pin_memory=True, drop_last=sh)
    tr_l, va_l, te_l = dl(tr, True), dl(va, False), dl(te, False)

    model = build_model(name, deep_classifier=True, band=band).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    best_score, best_state, wait, best_ep = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        # Augmentation is a property of the shared dataset: on for training,
        # off for every evaluation pass (mirrors train.py).
        dataset.training = augment
        model.train()
        for x1, x2, y, _ in tr_l:
            x1, x2, y = x1.to(device), x2.to(device), y.float().to(device)
            ys = y * (1.0 - 2 * label_smoothing) + label_smoothing
            opt.zero_grad()
            loss = crit(model(x1, x2), ys)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        sched.step(ep)
        dataset.training = False
        vm = evaluate(model, va_l, device)
        score = monitor_score(vm, monitor, fbeta)
        if score > best_score:
            best_score, best_ep = score, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    dataset.training = False
    tm = evaluate(model, te_l, device)
    tm.update(variant=name, seed=int(seed), n_params=int(n_params), best_epoch=int(best_ep),
              monitor=monitor, fbeta=float(fbeta), augment=bool(augment),
              best_monitor_val=float(best_score))
    return tm


# ═══════════════════════════════════════════════════════════════════════
#  Architecture diagram
# ═══════════════════════════════════════════════════════════════════════

def draw_arch(name, path, band):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    cfg = VARIANTS[name]
    ON, OFF, NEU, OUT = "#2C6DB5", "#C0392B", "#445168", "#1E8C74"
    low_khz = round(band[0] * 102.4 / 128, 1)
    fig, ax = plt.subplots(figsize=(15, 6)); ax.set_xlim(0, 15); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color + "22", ec=color, lw=2))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=11,
                fontweight="bold", color=color)
        ax.text(x + w / 2, y + 0.33, sub, ha="center", va="center", fontsize=8.5, color="#333")

    def arrow(p1, p2):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15, lw=1.8, color="#555"))

    box(0.3, 2.5, 1.9, 1.4, "Input", "$I(t)$, 1 cycle\n2048 @ 102.4 kHz", NEU)
    # temporal
    dca_on = cfg['channel_attn']
    box(2.7, 4.0, 3.6, 1.6, "Temporal branch",
        ("Conv1d stack\n+ Descriptor Chan. Attn" if dca_on else "Conv1d stack\n(no channel attention)"),
        ON if dca_on else OFF)
    # spectral
    gate_on = cfg['freq_gate']
    box(2.7, 0.4, 3.6, 1.6, "Spectral branch",
        ("STFT + learnable\nFrequency Gate" if gate_on else f"STFT + fixed band\n{low_khz}–51.2 kHz (no gate)"),
        ON if gate_on else OFF)
    # fusion
    xa = cfg['cross_attention']
    box(6.8, 2.3, 3.4, 1.4, ("Sequential\nCross-Attention" if xa else "Simple\nConcatenation"),
        ("bidirectional Q/K/V" if xa else "GAP + concat + linear"), ON if xa else OFF)
    box(10.7, 2.4, 2.0, 1.2, "Deep head", "FC 128→64→32→1", NEU)
    box(13.1, 2.5, 1.6, 1.0, "Output", "$P(\\mathrm{arc})$", OUT)

    arrow((2.2, 3.5), (2.7, 4.6)); arrow((2.2, 2.9), (2.7, 1.2))
    arrow((6.3, 4.6), (6.8, 3.3)); arrow((6.3, 1.2), (6.8, 2.7))
    arrow((10.2, 3.0), (10.7, 3.0)); arrow((12.7, 3.0), (13.1, 3.0))
    ax.text(7.5, 5.6, f"Arc-FaultNet ablation — variant: {VARIANT_LABEL[name]}",
            ha="center", fontsize=14, fontweight="bold")
    red = [k for k, v in cfg.items() if not v]
    ax.text(7.5, 0.05, "red = ablated/removed" if red else "full model (all mechanisms active)",
            ha="center", fontsize=9, color="#777")
    plt.savefig(path, bbox_inches="tight", facecolor="white", dpi=200)
    print(f"saved {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Comparison plots
# ═══════════════════════════════════════════════════════════════════════

def plot_results(results_path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.load(open(results_path))
    order = [v for v in VARIANTS if any(r['variant'] == v for r in rows)]
    def agg(v, k):
        xs = [r[k] for r in rows if r['variant'] == v]
        return (np.mean(xs), np.std(xs)) if xs else (0, 0)

    # F1 / accuracy bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [VARIANT_LABEL[v] for v in order]
    f1m = [100 * agg(v, 'f1')[0] for v in order]; f1s = [100 * agg(v, 'f1')[1] for v in order]
    ax.bar(labels, f1m, yerr=f1s, capsize=4, color="#2C6DB5")
    ax.set_ylabel("Test F1 (%)"); ax.set_ylim(min(f1m) - 3, 100)
    ax.set_title("Attention ablation — F1 by variant (deep head, mean ± std over seeds)")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(Path(out_dir) / "ablation_f1.png", dpi=200); plt.close()

    # FP / FN bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order)); w = 0.38
    fpm = [agg(v, 'fp')[0] for v in order]; fnm = [agg(v, 'fn')[0] for v in order]
    ax.bar(x - w / 2, fpm, w, yerr=[agg(v, 'fp')[1] for v in order], capsize=4,
           label="False positives", color="#C0392B")
    ax.bar(x + w / 2, fnm, w, yerr=[agg(v, 'fn')[1] for v in order], capsize=4,
           label="False negatives", color="#C8891B")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count (test set)"); ax.legend()
    ax.set_title("Attention ablation — FP / FN by variant (mean over seeds)")
    plt.tight_layout(); plt.savefig(Path(out_dir) / "ablation_fp_fn.png", dpi=200); plt.close()
    print(f"saved plots to {out_dir}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['validate', 'diagram', 'train', 'plot'], required=True)
    ap.add_argument('--variant', default='none', choices=list(VARIANTS))
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 2, 3, 4, 5])
    ap.add_argument('--monitor', choices=['val_fbeta', 'val_f1'], default='val_fbeta',
                    help='model-selection metric (paper protocol: val_fbeta)')
    ap.add_argument('--fbeta', type=float, default=0.5,
                    help='beta for F-beta selection (paper protocol: 0.5)')
    ap.add_argument('--no-augment', action='store_true',
                    help='disable the training-time augmentation used by train.py')
    ap.add_argument('--deterministic', action='store_true',
                    help='cuDNN-deterministic kernels; removes run-to-run spread')
    ap.add_argument('--data-dir', default='combined_dataset_2048')
    ap.add_argument('--n-fft', type=int, default=128)
    ap.add_argument('--hop-length', type=int, default=64)
    ap.add_argument('--band-low-khz', type=float, default=2.0)
    ap.add_argument('--epochs', type=int, default=60)
    # 30 is the smallest patience that reproduces the no-early-stop checkpoint on
    # all 5 reference seeds: with T_0=10/T_mult=2 the third cosine cycle spans
    # epochs 31-70, so a shorter patience truncates it and selects from cycle 2.
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--out-dir', default='ablation_attention_results')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    band = band_bins(args.band_low_khz, fs=102_400.0, n_fft=args.n_fft)

    if args.mode == 'validate':
        print(f"fixed band-pass bins = {band}  (~{args.band_low_khz}-51.2 kHz)\n")
        print(f"{'variant':13s} {'params':>9}  cross / gate / dca")
        for v in VARIANTS:
            m = build_model(v, deep_classifier=True, band=band)
            n = sum(p.numel() for p in m.parameters())
            c = VARIANTS[v]
            print(f"{v:13s} {n:>9,}  {c['cross_attention']!s:5} {c['freq_gate']!s:5} {c['channel_attn']!s:5}")
        # forward-pass shape check
        m = build_model('none', band=band).eval()
        x1 = torch.randn(2, 4, 2048); x2 = torch.randn(2, 1, args.n_fft // 2 + 1, 31)
        with torch.no_grad():
            o = m(x1, x2)
        print(f"\nforward OK — logits shape {tuple(o.shape)} for dummy batch of 2")
        return

    if args.mode == 'diagram':
        draw_arch(args.variant, out / f"arch_{args.variant}.png", band)
        return

    if args.mode == 'plot':
        plot_results(out / 'results.json', out)
        return

    # train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  band={band}  variants={args.variants}  seeds={args.seeds}")
    ds = ArcFaultDataset(data_dir=args.data_dir, channel_mode='i_derived4',
                         n_fft=args.n_fft, hop_length=args.hop_length, compute_stft=True)
    res_path = out / 'results.json'
    rows = json.load(open(res_path)) if res_path.exists() else []
    for v in args.variants:
        for s in args.seeds:
            t0 = time.time()
            m = train_variant(v, ds, s, device, band,
                              epochs=args.epochs, patience=args.patience,
                              monitor=args.monitor, fbeta=args.fbeta,
                              augment=not args.no_augment,
                              deterministic=args.deterministic)
            m['seconds'] = round(time.time() - t0, 1)
            rows.append(m)
            json.dump(rows, open(res_path, 'w'), indent=2)
            print(f"[{v:12s} seed {s}] acc {100*m['accuracy']:.2f}  f1 {100*m['f1']:.2f}  "
                  f"FP {m['fp']:3d}  FN {m['fn']:3d}  ({m['seconds']}s)")
    print(f"\nresults -> {res_path}")


if __name__ == '__main__':
    main()
